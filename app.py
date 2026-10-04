"""
Flask application for the Indian government construction budget predictor.

One process, two pages (the dashboard and the building estimator), twenty HTTP routes and a
Server-Sent Events stream. The models are loaded once at start-up and every number the
browser displays travels through this file from the trained artifacts or from a stated
formula, so nothing on screen can be stale or invented.

Run:  py app.py        (then open the printed http://127.0.0.1:5000)

Windows note: the console codepage is cp1252 on a normal install, so stdout is
reconfigured below -- a rupee sign in the start-up banner would otherwise raise
UnicodeEncodeError before the server ever binds a port.
"""

import io
import json
import os
import queue
import shutil
import sys
import threading
import time

import pandas as pd
from flask import (Flask, Response, jsonify, render_template, request,
                   send_file, stream_with_context)
from werkzeug.exceptions import HTTPException

# Keep console output alive on a cp1252 terminal.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config                                      # noqa: E402
from data_layer import pipeline, schema, storage    # noqa: E402
from data_layer.schema import FEATURES              # noqa: E402
import model.train as trainer                       # noqa: E402
import model.route_engine as routes                 # noqa: E402
import model.planning as planning                   # noqa: E402
import model.planning_data as planning_data         # noqa: E402
import model.building as building                   # noqa: E402

TEAM = {
    "title": "Government Construction Project Budget Prediction Using Machine Learning",
    "subtitle": "Indian government schemes, FY 2025-26",
    "developers": ["Vishal Raghunath Chatur", "Mohsin Jabiulla Balgyar"],
    "guide": "Tapase Miss",
    "institution": "DBATU",
    "programme": "B.Tech in Artificial Intelligence & Data Science Engineering",
    "currency": "\u20b9",
    "credit_line": ("Developed by Vishal Raghunath Chatur and Mohsin Jabiulla Balgyar "
                    "| Guide: Tapase Miss | B.Tech in Artificial Intelligence & Data "
                    "Science Engineering, DBATU"),
    "data_note": ("Simulated Indian scheme dataset, FY 2025-26. Model figures are "
                  "measured from the trained model; the project rows are synthetic."),
}

MIN_RETRAIN_ROWS = 30
# Where an uploaded dataset is kept, so a restart serves the same rows the current
# metrics were computed from.
RETRAIN_ACTIVE = os.path.join(config.DATA_DIR, "retrain_active.csv")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ApiError(Exception):
    """A request the client got wrong. Becomes a 400 with field-level detail."""

    def __init__(self, message, fields=None, status=400):
        super().__init__(message)
        self.message = message
        self.fields = fields or {}
        self.status = status


# ---------------------------------------------------------------------------
# Live event broker
# ---------------------------------------------------------------------------


class EventBroker:
    """Publish/subscribe for the real-time dashboard. Each browser gets its own bounded
    queue, so a stalled client can neither block a prediction nor grow memory without
    limit. This is the single-process stand-in for a stream service; SPEC_INDIA.md
    section 8 names Kinesis/SNS as the production replacement."""

    def __init__(self, max_queue=64):
        self.max_queue = max_queue
        self._subscribers = []
        self._lock = threading.Lock()

    def subscribe(self):
        channel = queue.Queue(maxsize=self.max_queue)
        with self._lock:
            self._subscribers.append(channel)
        return channel

    def unsubscribe(self, channel):
        with self._lock:
            if channel in self._subscribers:
                self._subscribers.remove(channel)

    def publish(self, event):
        with self._lock:
            channels = list(self._subscribers)
        for channel in channels:
            try:
                channel.put_nowait(event)
            except queue.Full:
                pass  # drop for a client that is not reading, never delay the request

    @property
    def subscriber_count(self):
        with self._lock:
            return len(self._subscribers)


BROKER = EventBroker()


# ---------------------------------------------------------------------------
# Model service
# ---------------------------------------------------------------------------


class ModelService:
    """Holds the dataset plus both trained pipelines and answers every API question.

    A retrain swaps the contents of one dictionary under a lock, so a concurrent
    prediction either sees the old model completely or the new one completely.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.meta = None
        self.pipelines = {}
        self.frame = None
        self.train_frame = None
        self.encoded_train = None
        self.source = {}
        self.prediction_count = 0
        self.last_error = None

    # -- loading ----------------------------------------------------------

    def load(self, store_name=None, ref=None, rebuild_store=True):
        """Read the dataset through the storage adapter and load both models."""
        adapter = storage.build_adapter(store_name)
        raw = adapter.load(ref)
        frame = schema.add_derived_columns(schema.normalise_frame(raw))

        meta = trainer.load_meta()
        pipelines = {name: trainer.load_model(name) for name in ("knn", "rf")}

        with self._lock:
            self.frame = frame
            self.meta = meta
            self.pipelines = pipelines
            self.train_frame = self._training_frame(frame, meta)
            self.encoded_train = self._encode_train(pipelines["knn"], self.train_frame)
            self.source = {
                "store": adapter.name,
                "ref": str(ref or getattr(adapter, "url", "") or adapter.describe().get("bucket", "")
                          or config.LOCAL_DATASET),
                "rows": int(len(frame)),
                "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            }
            if rebuild_store:
                try:
                    pipeline.build_feature_store(adapter=adapter, source=ref, quiet=True)
                except Exception as exc:            # a broken store never stops the app
                    self.last_error = "feature store rebuild failed: %s" % exc
        return self.source

    @staticmethod
    def _training_frame(frame, meta):
        """Recover the exact training rows the model was fitted on.

        The split is deterministic (same seed, same frame order), so re-running it and
        taking the train half reproduces those rows. Needed for neighbour lookups and for
        reporting how many projects a prediction was compared against.
        """
        x = frame[FEATURES].copy()
        y = frame[schema.TARGET].copy()
        counts = y.value_counts()
        stratify = y if counts.min() >= 2 else None
        from sklearn.model_selection import train_test_split
        x_train, _ = train_test_split(
            x, test_size=schema_test_fraction(meta), random_state=meta["dataset"]["random_state"],
            stratify=stratify)
        return frame.loc[x_train.index].copy()

    @staticmethod
    def _encode_train(knn_pipeline, train_frame):
        return knn_pipeline.named_steps["preprocess"].transform(train_frame[FEATURES])

    # -- validation -------------------------------------------------------

    def vocabularies(self):
        return self.meta["vocabularies"]

    def validate(self, payload):
        """Turn a JSON body into clean feature values, or explain every problem at once."""
        payload = payload or {}
        errors = {}
        vocab = self.vocabularies()
        k_range = self.meta["model"]["k_range"]

        def text(field, label, allowed, required=True):
            value = str(payload.get(field, "") or "").strip()
            if not value:
                if required:
                    errors[field] = "%s is required." % label
                return None
            if value not in allowed:
                errors[field] = "'%s' is not a known %s." % (value, label.lower())
                return None
            return value

        scheme = text("scheme_name", "Scheme", vocab["scheme_name"])
        state = text("state", "State", vocab["state"])
        district = text("district", "District", vocab["district"])
        contractor = text("contractor_type", "Contractor type", vocab["contractor_type"])

        # A district that exists but not in the stated state is the classic data-entry
        # slip in this domain (Pune in Karnataka), so it gets its own message.
        if district and state:
            legal_states = vocab["district_states"].get(district, [])
            if legal_states and state not in legal_states:
                errors["district"] = ("%s is not a district of %s (it appears under: %s)."
                                      % (district, state, ", ".join(legal_states)))

        budget = self._number(payload, "sanctioned_budget_lakh", "Sanctioned budget (rupee lakh)",
                              errors, low=0.1, high=1_000_000)
        duration = self._number(payload, "duration_months", "Scheduled duration (months)",
                                errors, low=1, high=240, integer=True)

        k_value = payload.get("k", self.meta["model"]["default_k"])
        try:
            k = int(float(k_value))
        except (TypeError, ValueError):
            errors["k"] = "K must be a whole number."
            k = self.meta["model"]["default_k"]
        else:
            if not (k_range[0] <= k <= k_range[1]):
                errors["k"] = "K must be between %d and %d." % (k_range[0], k_range[1])
                k = self.meta["model"]["default_k"]

        model_key = str(payload.get("model") or "auto").strip().lower()
        if model_key == "auto":
            model_key = self.meta["comparison"]["winner"]
        if model_key not in ("knn", "rf"):
            errors["model"] = "Choose knn, rf or auto."
            model_key = self.meta["comparison"]["winner"]

        if errors:
            raise ApiError("The prediction request is not valid.", fields=errors)

        return {
            "scheme_name": scheme, "state": state, "district": district,
            "contractor_type": contractor, "sanctioned_budget_lakh": float(budget),
            "duration_months": int(duration), "k": k, "model": model_key,
        }

    @staticmethod
    def _number(payload, field, label, errors, low, high, integer=False):
        raw = payload.get(field)
        if raw in (None, ""):
            errors[field] = "%s is required." % label
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[field] = "%s must be a number." % label
            return None
        if value < low or value > high:
            errors[field] = "%s must be between %s and %s." % (label, low, high)
            return None
        if integer and float(value).is_integer() is False:
            errors[field] = "%s must be a whole number of months." % label
            return None
        return int(value) if integer else value

    # -- prediction -------------------------------------------------------

    def predict(self, payload):
        values = self.validate(payload)
        query = pd.DataFrame([[values[c] for c in FEATURES]], columns=FEATURES)

        pipeline_knn = self.pipelines["knn"]
        pipeline_rf = self.pipelines["rf"]
        chosen = pipeline_rf if values["model"] == "rf" else pipeline_knn

        encoded_query = pipeline_knn.named_steps["preprocess"].transform(query)

        k = values["k"]
        with self._lock:
            knn_model = pipeline_knn.named_steps["model"]
            original_k = knn_model.n_neighbors
            knn_model.n_neighbors = k
            try:
                knn_prediction = str(pipeline_knn.predict(query)[0])
                knn_proba = pipeline_knn.predict_proba(query)[0]
                distances, indices = knn_model.kneighbors(encoded_query, n_neighbors=k)
                knn_classes = [str(c) for c in pipeline_knn.classes_]
            finally:
                knn_model.n_neighbors = original_k

            rf_prediction = str(pipeline_rf.predict(query)[0])
            rf_proba = pipeline_rf.predict_proba(query)[0]
            rf_classes = [str(c) for c in pipeline_rf.classes_]

        def prob_yes(classes, proba):
            return float(proba[classes.index(schema.POSITIVE_CLASS)])

        knn_risk = prob_yes(knn_classes, knn_proba)
        rf_risk = prob_yes(rf_classes, rf_proba)
        active_risk = rf_risk if values["model"] == "rf" else knn_risk
        active_label = str(chosen.predict(query)[0])

        neighbour_rows = self._neighbours(indices[0], distances[0], k)
        yes_votes = sum(1 for row in neighbour_rows if row["over_budget"] == "Yes")
        no_votes = len(neighbour_rows) - yes_votes

        cost = self.expected_cost(values["scheme_name"], values["sanctioned_budget_lakh"], active_risk)
        explanation = self.build_explanation(values, active_label, active_risk, k,
                                             yes_votes, no_votes, neighbour_rows,
                                             values["model"])

        result = {
            "prediction": active_label,
            "verdict": ("OVER BUDGET" if active_label == "Yes" else "WITHIN BUDGET"),
            "risk_pct": round(100.0 * active_risk, 1),
            "risk_pct_by_model": {"knn": round(100.0 * knn_risk, 1),
                                  "rf": round(100.0 * rf_risk, 1)},
            "model": values["model"],
            "model_label": trainer.MODEL_LABELS[values["model"]],
            "k": k,
            "votes": {"yes": yes_votes, "no": no_votes, "total": len(neighbour_rows)},
            "neighbours": neighbour_rows,
            "inputs": values,
            "cost": cost,
            "explanation": explanation,
            "specific_advisory": self.procurement_advisory(values),
            "train_size": int(self.meta["dataset"]["train_size"]),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        }

        with self._lock:
            self.prediction_count += 1
            result["sequence"] = self.prediction_count

        BROKER.publish({"type": "prediction", "payload": self.brief(result)})
        return result

    def _neighbours(self, indices, distances, k):
        rows = []
        for rank, (position, distance) in enumerate(zip(indices, distances), start=1):
            source_row = self.train_frame.iloc[int(position)]
            rows.append({
                "rank": rank,
                "project_name": source_row["project_name"],
                "scheme_name": source_row["scheme_name"],
                "state": source_row["state"],
                "district": source_row["district"],
                "contractor_type": source_row["contractor_type"],
                "sanctioned_budget_lakh": round(float(source_row["sanctioned_budget_lakh"]), 1),
                "sanctioned_inr": schema.format_inr(source_row["sanctioned_budget_lakh"]),
                "actual_cost_lakh": round(float(source_row["actual_cost_lakh"]), 1),
                "actual_inr": schema.format_inr(source_row["actual_cost_lakh"]),
                "duration_months": int(source_row["duration_months"]),
                "over_budget": source_row[schema.TARGET],
                "distance": round(float(distance), 4),
            })
        return rows[:k]

    def expected_cost(self, scheme, sanctioned_lakh, risk_fraction):
        """Turn a risk probability into an expected spend, using the mean overrun of the
        projects in this scheme that actually overran (training rows only).

        This is a stated heuristic rather than a second trained model:
            expected_overrun_pct = risk x mean_overrun_pct_of_overruns
        """
        heuristic = self.meta["overrun_heuristic"]
        per_scheme = heuristic["per_scheme"].get(scheme)
        mean_pct = per_scheme["mean_overrun_pct"] if per_scheme else heuristic["overall_mean_pct"]
        expected_pct = round(risk_fraction * mean_pct, 2)
        expected_actual = sanctioned_lakh * (1.0 + expected_pct / 100.0)
        overrun = expected_actual - sanctioned_lakh
        return {
            "sanctioned_lakh": round(float(sanctioned_lakh), 1),
            "sanctioned_inr": schema.format_inr(sanctioned_lakh),
            "expected_actual_lakh": round(expected_actual, 1),
            "expected_actual_inr": schema.format_inr(expected_actual),
            "expected_overrun_lakh": round(overrun, 1),
            "expected_overrun_inr": schema.format_inr(overrun),
            "expected_overrun_pct": expected_pct,
            "scheme_mean_overrun_pct": mean_pct,
            "basis": "risk-weighted heuristic (not a regression model)",
        }

    def procurement_advisory(self, values):
        """The policy note the original paper highlighted: works let to a single
        department, or awarded without open bidding, carry a documented overrun penalty.
        Here the analogue is contractor execution."""
        if values["contractor_type"] == "Govt":
            return ("Department/PSU-executed work. In this dataset Govt-executed projects "
                    "overrun more often than contractor-executed ones, because land "
                    "clearance and utility shifting sit with the department itself.")
        return ("Contractor (EPC) execution. Private execution carries the lower overrun "
                "rate in this dataset.")

    def build_explanation(self, values, label, risk, k, yes_votes, no_votes, neighbours,
                          model_key):
        """Plain-English paragraph a student can read out in a viva."""
        share_pct = round(100.0 * risk, 1)
        verdict = "go OVER sanction" if label == "Yes" else "stay within sanction"
        train_size = self.meta["dataset"]["train_size"]

        if model_key == "rf":
            mechanism = ("The %s voted tree-by-tree over all %d training projects and "
                         "%.1f%% of the trees voted 'over budget'."
                         % (trainer.MODEL_LABELS["rf"], train_size, share_pct))
        else:
            mechanism = ("The KNN measured the Euclidean distance from this project to all "
                         "%d encoded training projects, kept the %d closest, and took a "
                         "majority vote. %d of those %d neighbours went over sanction and "
                         "%d did not." % (train_size, k, yes_votes, k, no_votes))

        matching = self.agreement_summary(neighbours)
        risk_word = ("high" if risk >= 0.66 else "moderate" if risk >= 0.34 else "low")

        cost = self.expected_cost(values["scheme_name"], values["sanctioned_budget_lakh"], risk)
        return (
            "For this %s project in %s, %s, scheduled %d months at %s, the %s says it will "
            "%s. %s Predicted cost-overrun risk: %s%% (%s). %s "
            "Risk-weighted expected spend: %s, about %s above sanction (%.1f%% of the "
            "estimate for %s overruns at all)."
            % (values["scheme_name"], values["district"], values["state"],
               values["duration_months"], schema.format_inr(values["sanctioned_budget_lakh"]),
               trainer.MODEL_LABELS[model_key], verdict, mechanism, risk_word, share_pct,
               matching, cost["expected_actual_inr"], cost["expected_overrun_inr"],
               round(risk * 100, 1), values["scheme_name"])
        )

    @staticmethod
    def agreement_summary(neighbours):
        """How many of the nearest neighbours share each input value -- the honest,
        inspectable version of 'why this answer'."""
        if not neighbours:
            return ""
        fields = ["scheme_name", "state", "district", "contractor_type"]
        labels = {"scheme_name": "scheme", "state": "state", "district": "district",
                  "contractor_type": "execution"}
        total = len(neighbours)
        parts = []
        for field in fields:
            same = sum(1 for row in neighbours if row[field] == neighbours[0][field])
            parts.append("%d/%d match on %s" % (same, total, labels[field]))
        return "Neighbour agreement: " + ", ".join(parts) + "."

    @staticmethod
    def brief(result):
        """Small shape for the live feed and the SSE stream."""
        return {
            "sequence": result["sequence"],
            "prediction": result["prediction"],
            "verdict": result["verdict"],
            "risk_pct": result["risk_pct"],
            "model": result["model_label"],
            "k": result["k"],
            "scheme_name": result["inputs"]["scheme_name"],
            "district": result["inputs"]["district"],
            "state": result["inputs"]["state"],
            "sanctioned_inr": result["cost"]["sanctioned_inr"],
            "expected_actual_inr": result["cost"]["expected_actual_inr"],
            "generated_at": result["generated_at"],
        }

    # -- reporting --------------------------------------------------------

    def metrics(self):
        meta = self.meta
        return {
            "models": meta["metrics"],
            "comparison": meta["comparison"],
            "k_curve": meta["k_curve"],
            "importances": meta["importances"],
            "dataset": meta["dataset"],
            "model": meta["model"],
            "overrun_heuristic": meta["overrun_heuristic"],
            "trained_at": meta["trained_at"],
            "source": self.source,
        }

    def meta_payload(self):
        meta = self.meta
        return {
            "team": TEAM,
            "vocabularies": meta["vocabularies"],
            "features": FEATURES,
            "categorical_features": schema.CATEGORICAL_FEATURES,
            "numeric_features": schema.NUMERIC_FEATURES,
            "target": schema.TARGET,
            "models": [{"key": key, "label": trainer.MODEL_LABELS[key]} for key in ("knn", "rf")],
            "default_model": meta["comparison"]["winner"],
            "default_k": meta["model"]["default_k"],
            "k_range": meta["model"]["k_range"],
            "currency": config.CURRENCY,
            "theme": config.THEME,
            "settings": config.describe(),
            "sources": storage.adapter_status(),
            "dataset": meta["dataset"],
            "model_spec": meta["model"],
            "comparison": meta["comparison"],
            "derived_columns": schema.DERIVED_COLUMNS,
        }

    def dataset_page(self, args):
        frame = self.frame
        try:
            page = max(1, int(args.get("page", 1)))
            size = min(100, max(5, int(args.get("size", 25))))
        except ValueError:
            raise ApiError("page and size must be whole numbers.")

        search = str(args.get("search", "") or "").strip().lower()
        filtered = frame
        if search:
            haystack = (filtered["project_name"].str.lower()
                        + " | " + filtered["state"].str.lower()
                        + " | " + filtered["district"].str.lower()
                        + " | " + filtered["scheme_name"].str.lower())
            filtered = filtered[haystack.str.contains(search, regex=False, na=False)]

        for field in ("scheme_name", "state", "contractor_type", schema.TARGET):
            wanted = str(args.get(field, "") or "").strip()
            if wanted:
                filtered = filtered[filtered[field] == wanted]

        sort = str(args.get("sort", "project_name"))
        allowed_sorts = {"project_name", "state", "district", "scheme_name",
                         "contractor_type",
                         "sanctioned_budget_lakh", "actual_cost_lakh", "duration_months",
                         "overrun_lakh", "overrun_pct", schema.TARGET}
        if sort not in allowed_sorts:
            raise ApiError("Unknown sort column.", fields={"sort": "choose one of: %s"
                                                            % ", ".join(sorted(allowed_sorts))})
        ascending = str(args.get("dir", "asc")) != "desc"
        ordered = filtered.sort_values(sort, ascending=ascending, na_position="last")

        total = int(len(ordered))
        start = (page - 1) * size
        window = ordered.iloc[start:start + size]

        rows = []
        for _, row in window.iterrows():
            rows.append({
                "project_name": row["project_name"],
                "scheme_name": row["scheme_name"],
                "project_type": row["project_type"],
                "state": row["state"],
                "district": row["district"],
                "contractor_type": row["contractor_type"],
                "sanctioned_budget_lakh": round(float(row["sanctioned_budget_lakh"]), 1),
                "sanctioned_inr": schema.format_inr(row["sanctioned_budget_lakh"]),
                "duration_months": int(row["duration_months"]),
                "actual_cost_lakh": round(float(row["actual_cost_lakh"]), 1),
                "actual_inr": schema.format_inr(row["actual_cost_lakh"]),
                "overrun_lakh": round(float(row["overrun_lakh"]), 1),
                "overrun_inr": schema.format_inr(row["overrun_lakh"]),
                "overrun_pct": round(float(row["overrun_pct"]), 2),
                "over_budget": row[schema.TARGET],
            })
        return {"rows": rows, "page": page, "size": size, "total": total,
                "pages": max(1, -(-total // size)), "filters": {"search": search}}

    def dashboard(self):
        frame = self.frame
        grouped_scheme = []
        for scheme, group in frame.groupby("scheme_name", sort=True):
            sanctioned = float(group["sanctioned_budget_lakh"].sum())
            actual = float(group["actual_cost_lakh"].sum())
            yes = int((group[schema.TARGET] == "Yes").sum())
            grouped_scheme.append({
                "scheme_name": str(scheme),
                "projects": int(len(group)),
                "sanctioned_lakh": round(sanctioned, 1),
                "sanctioned_inr": schema.format_inr(sanctioned),
                "actual_lakh": round(actual, 1),
                "actual_inr": schema.format_inr(actual),
                "overrun_lakh": round(actual - sanctioned, 1),
                "overrun_inr": schema.format_inr(actual - sanctioned),
                "overrun_pct": round(100.0 * (actual - sanctioned) / sanctioned, 2),
                "over_budget_projects": yes,
                "over_budget_share_pct": round(100.0 * yes / len(group), 1),
            })

        grouped_state = []
        for state, group in frame.groupby("state", sort=True):
            yes = int((group[schema.TARGET] == "Yes").sum())
            sanctioned = float(group["sanctioned_budget_lakh"].sum())
            actual = float(group["actual_cost_lakh"].sum())
            grouped_state.append({
                "state": str(state),
                "projects": int(len(group)),
                "over_budget": yes,
                "within_budget": int(len(group)) - yes,
                "over_budget_share_pct": round(100.0 * yes / len(group), 1),
                "sanctioned_lakh": round(sanctioned, 1),
                "sanctioned_inr": schema.format_inr(sanctioned),
                "actual_lakh": round(actual, 1),
                "actual_inr": schema.format_inr(actual),
                "districts": int(group["district"].nunique()),
            })
        grouped_state.sort(key=lambda item: -item["projects"])

        grouped_contractor = []
        for kind, group in frame.groupby("contractor_type", sort=True):
            yes = int((group[schema.TARGET] == "Yes").sum())
            grouped_contractor.append({
                "contractor_type": str(kind),
                "projects": int(len(group)),
                "over_budget": yes,
                "over_budget_share_pct": round(100.0 * yes / len(group), 1),
                "mean_overrun_pct": round(float(group.loc[group[schema.TARGET] == "Yes",
                                                          "overrun_pct"].mean()), 2),
            })

        top = frame.nlargest(10, "overrun_lakh")
        top_overruns = [{
            "project_name": row["project_name"],
            "state": row["state"],
            "district": row["district"],
            "scheme_name": row["scheme_name"],
            "sanctioned_inr": schema.format_inr(row["sanctioned_budget_lakh"]),
            "actual_inr": schema.format_inr(row["actual_cost_lakh"]),
            "overrun_inr": schema.format_inr(row["overrun_lakh"]),
            "overrun_pct": round(float(row["overrun_pct"]), 2),
        } for _, row in top.iterrows()]

        total_sanctioned = float(frame["sanctioned_budget_lakh"].sum())
        total_actual = float(frame["actual_cost_lakh"].sum())
        yes_total = int((frame[schema.TARGET] == "Yes").sum())

        return {
            "totals": {
                "projects": int(len(frame)),
                "states": int(frame["state"].nunique()),
                "districts": int(frame["district"].nunique()),
                "schemes": int(frame["scheme_name"].nunique()),
                "sanctioned_lakh": round(total_sanctioned, 1),
                "sanctioned_inr": schema.format_inr(total_sanctioned),
                "actual_lakh": round(total_actual, 1),
                "actual_inr": schema.format_inr(total_actual),
                "overrun_lakh": round(total_actual - total_sanctioned, 1),
                "overrun_inr": schema.format_inr(total_actual - total_sanctioned),
                "overrun_pct": round(100.0 * (total_actual - total_sanctioned) / total_sanctioned, 2),
                "over_budget": yes_total,
                "over_budget_share_pct": round(100.0 * yes_total / len(frame), 2),
            },
            "by_scheme": grouped_scheme,
            "by_state": grouped_state,
            "by_contractor": grouped_contractor,
            "top_overruns": top_overruns,
            "source": self.source,
            "predictions_served": self.prediction_count,
        }

    # -- retraining -------------------------------------------------------

    def retrain(self, frame, label):
        """Retrain both models on a new frame, write fresh artifacts, and reload.

        Everything is backed up first: if the new data trains but fails a validation
        check afterwards, the previous artifacts and in-memory models are restored, so a
        bad upload cannot leave the app serving nothing.
        """
        clean = schema.normalise_frame(frame)
        if len(clean) < MIN_RETRAIN_ROWS:
            raise ApiError("A retrain needs at least %d rows; this file has %d."
                           % (MIN_RETRAIN_ROWS, len(clean)), status=422)
        counts = clean[schema.TARGET].value_counts().to_dict()
        if len(counts) < 2:
            raise ApiError("A retrain needs both Yes and No rows; this file only has %s."
                           % ", ".join(sorted(counts)), status=422)

        backups = {}
        for name in ("knn", "rf", "meta"):
            path = trainer.MODEL_FILES[name] if name in trainer.MODEL_FILES else trainer.META_PATH
            if os.path.exists(path):
                backups[path] = path + ".bak"
                shutil.copy2(path, backups[path])

        previous = (self.meta, self.pipelines, self.frame, self.train_frame,
                    self.encoded_train, self.source)
        # Persist an upload under data/ and load the app from that same file, so the
        # rows on screen are always the rows the models were fitted on. Retraining on a
        # frame that then got forgotten would leave the metrics describing data nobody
        # can see.
        target_ref = None
        if not str(label).startswith("synthetic"):
            target_ref = RETRAIN_ACTIVE
            storage.LocalStore().put(clean, target_ref)
        else:
            # Resetting means the shipped CSV is authoritative again; a stale upload
            # left in data/ would suggest the opposite.
            _quiet_remove(RETRAIN_ACTIVE)
        try:
            meta = trainer.train_and_save(clean, data_path=target_ref, source_label=label)
            self.load(store_name="local", ref=target_ref, rebuild_store=False)
        except Exception as exc:
            for original, backup in backups.items():
                if os.path.exists(backup):
                    shutil.copy2(backup, original)
            (self.meta, self.pipelines, self.frame, self.train_frame,
             self.encoded_train, self.source) = previous
            for backup in backups.values():
                _quiet_remove(backup)
            raise ApiError("Retraining failed, so the previous model was restored: %s" % exc,
                           status=500)

        for backup in backups.values():
            _quiet_remove(backup)
        self.last_error = None
        return {"rows": int(len(clean)), "target_counts": meta["dataset"]["target_counts"],
                "comparison": meta["comparison"], "metrics": meta["metrics"],
                "source": self.source}


def schema_test_fraction(meta):
    return float(meta["dataset"].get("test_size_fraction", trainer.TEST_SIZE))


def _quiet_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


SERVICE = ModelService()


# ---------------------------------------------------------------------------
# Flask wiring
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False


def _json_safe(payload):
    """Replace float NaN/inf, which is not legal JSON and would silently become 'null'
    in the browser while still failing strictly."""
    return json.loads(json.dumps(payload, default=str, allow_nan=False))


@app.errorhandler(ApiError)
def handle_api_error(error):
    response = {"error": error.message}
    if error.fields:
        response["fields"] = error.fields
    return jsonify(response), error.status


@app.errorhandler(Exception)
def handle_unexpected(error):
    if isinstance(error, ApiError):
        return handle_api_error(error)
    if isinstance(error, HTTPException):
        # A 404 or a 405 is the client's mistake, not a server fault: keep the code.
        return jsonify({"error": error.description}), error.code
    SERVICE.last_error = "%s: %s" % (type(error).__name__, error)
    app.logger.exception("Unhandled error")
    return jsonify({"error": "The server could not complete the request.",
                    "detail": str(error)}), 500


@app.route("/")
def index():
    return render_template("index.html", TEAM=TEAM, SETTINGS=config.describe())


@app.route("/meta")
def meta_route():
    return jsonify(_json_safe(SERVICE.meta_payload()))


@app.route("/predict", methods=["POST"])
def predict_route():
    payload = request.get_json(silent=True) or request.form.to_dict()
    return jsonify(_json_safe(SERVICE.predict(payload)))


@app.route("/metrics")
def metrics_route():
    return jsonify(_json_safe(SERVICE.metrics()))


@app.route("/dataset")
def dataset_route():
    return jsonify(_json_safe(SERVICE.dataset_page(request.args)))


@app.route("/sample.csv")
def sample_route():
    """The 20-row teaching sample, served as a real CSV download so a grader can load it
    into the retrain box. The path is fixed here -- it never comes from the request."""
    return send_file(config.SAMPLE_DATASET, mimetype="text/csv", as_attachment=True,
                     download_name="sample_20_projects.csv")


@app.route("/dashboard")
def dashboard_route():
    return jsonify(_json_safe(SERVICE.dashboard()))


@app.route("/sources")
def sources_route():
    return jsonify(_json_safe({"sources": storage.adapter_status(),
                               "active": SERVICE.source,
                               "sample_dataset": config.SAMPLE_DATASET}))


@app.route("/sources", methods=["POST"])
def switch_source_route():
    """Switch the storage backend and reload. Nothing is ever written to a bucket here:
    this only reads."""
    payload = request.get_json(silent=True) or {}
    store_name = str(payload.get("store") or "local").strip().lower()
    ref = payload.get("ref")
    try:
        source = SERVICE.load(store_name=store_name, ref=ref)
    except storage.StorageError as exc:
        raise ApiError(str(exc), fields={"store": str(exc)})
    return jsonify(_json_safe({"source": source, "meta": SERVICE.meta_payload()}))


@app.route("/pipeline")
def pipeline_route():
    manifest_path = os.path.join(config.FEATURE_STORE_DIR, pipeline.MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        manifest = pipeline.build_feature_store(quiet=True)
    else:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    return jsonify(_json_safe({"manifest": manifest,
                               "store_dir": config.FEATURE_STORE_DIR,
                               "chunk_size": config.CHUNK_SIZE,
                               "engine": pipeline.ENGINE}))


@app.route("/pipeline/rebuild", methods=["POST"])
def rebuild_pipeline_route():
    storage.reset_feature_store()
    manifest = pipeline.build_feature_store(quiet=True)
    return jsonify(_json_safe({"manifest": manifest}))


@app.route("/route/cities")
def route_cities_route():
    """Everything the map needs: city coordinates, road specs, terrain/soil factors
    and the FY 2025-26 unit rates the cost model multiplies."""
    return jsonify(_json_safe({"catalogue": routes.city_catalogue(),
                               "credit_line": TEAM["credit_line"],
                               "note": ("Distances are great-circle x a corridor coefficient "
                                        "calibrated on published Indian road distances. "
                                        "Costs are a quantity take-off at assumed 2025-26 "
                                        "unit rates and times are distance divided by an "
                                        "assumed km-per-day speed band -- none of them are "
                                        "model outputs.")}))


@app.route("/route/samples")
def route_samples_route():
    """Budget, distance and completion-time series over the calibration routes,
    computed here and now by the same engine the predictor uses, so a chart cannot
    drift from the form."""
    road_type = str(request.args.get("road_type") or "rural_two_lane").strip()
    try:
        rows = routes.sample_routes(road_type=road_type)
    except routes.RouteError as exc:
        raise ApiError(exc.message, fields=exc.fields)
    return jsonify(_json_safe({"road_type": road_type, "rows": rows,
                               "corridor_coefficient": routes.CORRIDOR_FACTOR}))


@app.route("/locations")
def locations_route():
    """The four-level planning hierarchy: State -> District -> Taluka -> Village.

    With `?q=` this is the search box and returns matching locations. Without it the
    whole tree comes back along with the assumption tables the page prints beside the
    dropdowns. Taluka and village names are generated placeholders and the payload says
    so -- see `model/planning_data.py`, and the district coordinates are the real ones
    the route engine and the trained models already use.
    """
    query = str(request.args.get("q") or "").strip()
    if query:
        raw_limit = str(request.args.get("limit") or "").strip()
        try:
            limit = int(raw_limit) if raw_limit else 12
        except ValueError:
            raise ApiError("limit must be a whole number of results.",
                           fields={"limit": raw_limit})
        if not (1 <= limit <= 40):
            raise ApiError("limit must be between 1 and 40.", fields={"limit": raw_limit})
        return jsonify(_json_safe({"query": query,
                                   "results": planning_data.suggest(query, limit=limit)}))
    return jsonify(_json_safe({"catalogue": planning_data.catalogue(),
                               "credit_line": TEAM["credit_line"],
                               "note": ("District centres, terrain and soil are the real "
                                        "catalogue the route planner uses. Taluka and "
                                        "village names are generated placeholders "
                                        "seated around their district, and every "
                                        "distance is haversine between two of those "
                                        "coordinates multiplied by the calibrated "
                                        "corridor coefficient -- no distance is looked "
                                        "up in a table.")}))


@app.route("/building")
def building_page():
    """The second estimator: a government building, costed per square foot.

    Separate page, separate engine, separate JSON contract. Nothing on the dashboard
    changes because of it, and it does not ask the trained classifier anything -- the
    classifier knows sanctioned scheme projects, not panchayat buildings.
    """
    return render_template("building.html", TEAM=TEAM, SETTINGS=config.describe(),
                           BUILDING=building.catalogue())


@app.route("/predict-building", methods=["POST"])
def building_estimate():
    """Cost one government building from area, storeys, quality and location type.

    The whole answer is one multiplied line and a stated split of it, so the page can
    show both and a reader can check the arithmetic. There is no model call here: the
    classifier was trained on scheme budgets, and pretending a ₹ 40 lakh school block
    came out of it would be exactly the kind of number this project refuses to print.
    """
    payload = request.get_json(silent=True) or request.form.to_dict()
    values = _building_request(payload)
    try:
        result = building.estimate(**values)
    except routes.RouteError as exc:
        raise ApiError(exc.message, fields=exc.fields)
    result["credit_line"] = TEAM["credit_line"]
    return jsonify(_json_safe(result))


def _building_request(payload):
    """Type-check the building form. The location is context, not an input to the
    cost, so each level is only checked for existing -- down to the deepest one given."""
    errors = {}

    def text(name):
        return str(payload.get(name) or "").strip()

    def number(name, label):
        raw = text(name)
        if not raw:
            errors[name] = "%s is required." % label
            return None
        try:
            return float(raw)
        except ValueError:
            errors[name] = "%s must be a number." % label
            return None

    location = {key: text(key) for key in ("state", "district", "taluka", "village")}
    if any(location.values()):
        if not location["state"] or not location["district"]:
            raise ApiError("A building location needs at least a state and a district.",
                           fields={"state": location["state"] or "required",
                                   "district": location["district"] or "required"})
        def refuse(message, level):
            raise ApiError(message, fields={level: message})

        planning_data.matches(location["state"], location["district"],
                              location["taluka"] or None,
                              location["village"] or None, fail=refuse)

    area = number("area_sqft", "Built-up area (square feet)")
    values = {
        "state": location["state"] or None, "district": location["district"] or None,
        "taluka": location["taluka"] or None, "village": location["village"] or None,
        "building_type": text("building_type") or "gram_panchayat",
        "area_sqft": area,
        "floors": text("floors") or 1,
        "quality": text("quality") or "medium",
        "location_type": text("location_type") or "rural",
    }
    if errors:
        raise ApiError("The building request is not valid.", fields=errors)
    return values


def _building_location_error(message, level):
    """Turn a hierarchy miss into the field the form can mark red."""
    raise routes.RouteError(message, {"%s" % level: level})


@app.route("/predict-route", methods=["POST"])
def route_estimate():
    """Cost a road corridor between two Indian cities.

    Two numbers come back and they mean different things. `total_crore` is a
    bottom-up engineering estimate from quantity x unit rate. `model_check` is the
    trained classifier's own overrun probability for a sanctioned work of that size.
    The response labels which is which because a visitor should not have to guess
    whether a rupee figure came from data or from a formula.

    The corridor is named either by two city names or by two nested
    {state, district, taluka, village} locations from `/locations`. The hierarchy
    route takes the same engine -- only the length it is given, measured from the two
    coordinates, is different.
    """
    payload = request.get_json(silent=True) or request.form.to_dict()
    values = _route_request(payload)

    try:
        result = routes.estimate(**values["estimate"])
    except routes.RouteError as exc:
        raise ApiError(exc.message, fields=exc.fields)

    points = values["points"]
    if points:
        _relabel_for_hierarchy(result, points)
        result["route_path"] = routes.geodesic_between(
            (points["start"]["lat"], points["start"]["lng"]),
            (points["end"]["lat"], points["end"]["lng"]))
    else:
        result["route_path"] = routes.geodesic_path(values["start"], values["end"])
    result["model_check"] = _route_model_check(SERVICE, result["model_input"],
                                               values["contractor_type"],
                                               values["model"], values["k"])
    result["planning"] = _planning_view(result, values["estimate"].get("width_m"))
    # The contract the brief asked for: distance, budget, time, risk. Kept alongside
    # the full breakdown rather than instead of it.
    result["brief"] = {"distance": result["distance_km"],
                       "budget": result["total_rupees"],
                       "time_months": result["time_months"],
                       "risk": result["risk_percent"]["band"],
                       "risk_percent": result["risk_percent"]["percent"]}
    return jsonify(_json_safe(result))


def _relabel_for_hierarchy(result, points):
    """Re-label a take-off that was priced from a measured length, not a city pair.

    `route_engine.estimate` is keyed on the 56 district cities; a village corridor
    reaches it as a length. Everything the engine computed from that length stands, and
    the fields that would otherwise name a city are filled from the two points instead,
    including the state and district the trained classifier is asked about.
    """
    origin, destination = points["start"], points["end"]
    result["start"] = origin["label"]
    result["end"] = destination["label"]
    result["start_point"] = origin
    result["end_point"] = destination
    result["location_levels"] = [origin["level"], destination["level"]]
    result["states"] = [origin["state"], destination["state"]]
    result["calibrated_km"] = result["distance_km"]
    result["straight_km"] = planning.straight_line_km(
        (origin["lat"], origin["lng"]), (destination["lat"], destination["lng"]))
    result["distance_basis"] = (
        "haversine distance between the two selected hierarchy points x %.4f corridor "
        "coefficient (calibrated on %d published Indian road distances); the points are "
        "village-level entries from /locations, not the city catalogue"
        % (routes.CORRIDOR_FACTOR, len(routes.CALIBRATION_ROUTES)))
    result["model_input"]["state"] = origin["state"]
    result["model_input"]["district"] = origin["district"]


def _planning_view(result, width_m):
    """The same corridor priced by the brief's one-line formula, beside the take-off.

    Deliberately two numbers for one road: `total_crore` above is quantities x unit
    rates, `planning.total_crore` is distance x an assumed ₹ per km for the cross-section
    class. They disagree by design and each prints the arithmetic that made it.
    """
    view = planning.plan(distance_km=result["distance_km"],
                         road_type=result["road_type"],
                         terrain=result["terrain"], soil=result["soil"],
                         width_m=width_m)
    # The planning brief's literal contract, in the key names it asked for.
    view["brief"] = {"distance": view["distance_km"],
                     "budget": view["total_rupees"],
                     "time_months": view["time_months"],
                     "risk": view["risk"]["band"],
                     "breakdown": view["breakdown"]}
    return view


def _route_request(payload):
    """Pull and type-check the route form. Blanks mean "use the engine's default",
    which is what the auto-detect path needs."""
    errors = {}
    points = _route_points(payload)
    if points:
        origin, destination = points["start"], points["end"]
        start = end = None
    else:
        start = str(payload.get("start") or "").strip()
        end = str(payload.get("end") or "").strip()
        if not start:
            errors["start"] = "Choose a start city."
        if not end:
            errors["end"] = "Choose an end city."
        if errors:
            raise ApiError("A route needs two cities.", fields=errors)

    def optional_number(name, low, high, label):
        raw = payload.get(name)
        text_value = "" if raw is None else str(raw).strip()
        if not text_value:
            return None
        try:
            value = float(text_value)
        except ValueError:
            errors[name] = "%s must be a number." % label
            return None
        if not (low <= value <= high):
            errors[name] = "%s must be between %s and %s." % (label, low, high)
            return None
        return value

    road_type = str(payload.get("road_type") or "rural_two_lane").strip()
    terrain = str(payload.get("terrain") or "").strip()
    soil = str(payload.get("soil") or "").strip()
    contractor = str(payload.get("contractor_type") or "Private").strip().title()
    if road_type not in routes.ROAD_TYPES:
        errors["road_type"] = "'%s' is not a known road type." % road_type
    if terrain and terrain not in routes.TERRAIN_FACTORS:
        errors["terrain"] = "'%s' is not a known terrain." % terrain
    if soil and soil not in routes.SOIL_FACTORS:
        errors["soil"] = "'%s' is not a known soil type." % soil
    if contractor not in ("Govt", "Private"):
        errors["contractor_type"] = "Execution must be Govt or Private."

    width_m = optional_number("width_m", 3.0, 30.0, "Carriageway width (metres)")
    cost_per_km = optional_number("cost_per_km_crore", 0.01, 200.0,
                                  "Cost per km (₹ crore)")
    distance_override = optional_number("distance_km", 0.1, 4000.0, "Route length (km)")
    k_value = optional_number("k", 1, 31, "K")
    model_key = str(payload.get("model") or "auto").strip().lower()
    if model_key not in ("auto", "knn", "rf"):
        errors["model"] = "Choose knn, rf or auto."
    if errors:
        raise ApiError("The route request is not valid.", fields=errors)

    if points:
        # A village-to-village length is measured here, from the two coordinates the
        # hierarchy resolved to. A DPR length typed in the request still wins, exactly
        # as it does on the city path.
        try:
            measured = planning_data.between(origin, destination)
        except routes.RouteError as exc:
            raise ApiError(exc.message, fields=exc.fields)
        terrain = terrain or routes.worst_terrain(origin["terrain"],
                                                 destination["terrain"])
        soil = soil or routes.costlier_soil(origin["soil_type"],
                                            destination["soil_type"])
        estimate_args = {"start": None, "end": None,
                         "distance_km": distance_override or measured,
                         "road_type": road_type, "terrain": terrain, "soil": soil,
                         "width_m": width_m, "cost_per_km_crore": cost_per_km}
    else:
        estimate_args = {"start": start, "end": end, "road_type": road_type,
                         "terrain": terrain or None, "soil": soil or None,
                         "width_m": width_m, "cost_per_km_crore": cost_per_km}
        if distance_override:
            # A length typed in from a DPR wins over the calibrated one; the response
            # still carries the calibrated figure beside it.
            estimate_args["distance_km"] = distance_override
    return {"start": start, "end": end, "estimate": estimate_args,
            "contractor_type": contractor, "model": model_key, "points": points,
            "k": int(k_value) if k_value else 5}


def _route_points(payload):
    """The nested start/end objects of a hierarchy request, or None for a city request.

    Mixing the two forms is refused rather than guessed at: half a village and half a
    city would price a corridor whose ends the reader cannot name.
    """
    start_raw, end_raw = payload.get("start"), payload.get("end")
    nested = [spec for spec in (start_raw, end_raw) if isinstance(spec, dict)]
    if not nested:
        return None
    if len(nested) != 2:
        raise ApiError("Name both ends the same way: either two city names or two "
                       "{state, district, taluka, village} locations.",
                       fields={"start": "mixed location forms",
                               "end": "mixed location forms"})

    def text(spec, level):
        return str(spec.get(level) or "").strip()

    points = {}
    for side, spec in (("start", start_raw), ("end", end_raw)):
        try:
            points[side] = planning_data.point(text(spec, "state"), text(spec, "district"),
                                               text(spec, "taluka") or None,
                                               text(spec, "village") or None,
                                               prefix=side)
        except routes.RouteError as exc:
            raise ApiError(exc.message, fields=exc.fields)
    return points



def _route_model_check(service, model_input, contractor, model_key, k):
    """Ask the trained classifier about the same corridor.

    The classifier was trained on sanctioned projects up to a few thousand ₹ lakh, so
    a 6-lane expressway corridor can fall outside that range. Rather than quietly
    returning an extrapolated number, the answer says whether it is interpolating.
    """
    record = dict(model_input)
    record["contractor_type"] = contractor
    record["model"] = model_key
    record["k"] = k
    if service.meta is None:
        return {"available": False,
                "reason": "The trained models are not loaded yet, so no cross-check is "
                          "possible. The cost estimate above is unaffected."}
    try:
        prediction = service.predict(record)
    except ApiError as exc:
        return {"available": False, "reason": exc.message, "fields": exc.fields}

    max_budget = float(service.meta["vocabularies"].get("budget_max_lakh") or 0)
    sanctioned = float(record["sanctioned_budget_lakh"])
    inside = sanctioned <= max_budget
    return {
        "available": True,
        "model": prediction["model_label"],
        "verdict": prediction["verdict"],
        "risk_pct": prediction["risk_pct"],
        "expected_actual_lakh": prediction["cost"]["expected_actual_lakh"],
        "expected_actual_inr": prediction["cost"]["expected_actual_inr"],
        "neighbours": len(prediction["neighbours"]),
        "votes": prediction["votes"],
        "budget_max_lakh_in_training": max_budget,
        "within_training_range": inside,
        "note": ("Trained-model probability for a %s-executed work of %s sanctioned "
                 "under %s. %s" % (contractor, schema.format_inr(sanctioned),
                                   record["scheme_name"],
                                   "This sanction is inside the range the model was "
                                   "trained on." if inside else
                                   "This sanction is above every project in the "
                                   "training data, so the model is extrapolating: "
                                   "treat the percentage as an order-of-magnitude "
                                   "guide, not a probability.")),
    }


@app.route("/retrain", methods=["POST"])
def retrain_route():
    """Accept an uploaded CSV (multipart) or CSV text (JSON) and retrain on it.

    A malformed upload is the client's problem, so it comes back as a 400 that names the
    offending column rather than a 500 with a pandas traceback in it.
    """
    try:
        frame, label = _read_upload()
        result = SERVICE.retrain(frame, label)
    except ApiError:
        raise
    except storage.StorageError as exc:      # an unconfigured or unreachable backend
        raise ApiError(str(exc))
    except schema.DataError as exc:
        raise ApiError(str(exc))
    except ValueError as exc:            # pandas ParserError / EmptyDataError
        raise ApiError("This does not parse as a CSV file: %s" % exc)
    return jsonify(_json_safe(result))


def _read_upload():
    if "file" in request.files:
        handle = request.files["file"]
        raw = handle.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig", errors="replace")
        frame = pd.read_csv(io.StringIO(raw))
        return frame, "upload:%s" % (handle.filename or "clipboard.csv")

    payload = request.get_json(silent=True) or {}
    if payload.get("reset"):
        frame = storage.LocalStore().load(config.LOCAL_DATASET)
        return frame, "synthetic-india-2025-26"
    if payload.get("source") in ("local", "http", "s3"):
        frame = storage.build_adapter(payload["source"]).load(payload.get("ref"))
        return frame, "%s:%s" % (payload["source"], payload.get("ref") or "")
    if payload.get("csv_text"):
        return pd.read_csv(io.StringIO(str(payload["csv_text"]))), "upload:pasted-csv"
    raise ApiError("Provide a CSV file, a csv_text body, a source name, or {\"reset\": true}.")


@app.route("/stream")
def stream_route():
    """Server-Sent Events. Every prediction is broadcast, and a heartbeat carries the
    dashboard totals so an open page updates without reloading."""

    def generate():
        channel = BROKER.subscribe()
        try:
            hello = {"type": "hello", "source": SERVICE.source,
                     "subscribers": BROKER.subscriber_count,
                     "predictions_served": SERVICE.prediction_count}
            yield "event: hello\ndata: %s\n\n" % json.dumps(hello, default=str)
            last_beat = time.time()
            while True:
                try:
                    event = channel.get(timeout=5.0)
                    yield "event: prediction\ndata: %s\n\n" % json.dumps(event, default=str)
                except queue.Empty:
                    pass
                if time.time() - last_beat >= 15.0:
                    last_beat = time.time()
                    totals = SERVICE.dashboard()["totals"] if SERVICE.frame is not None else {}
                    beat = {"type": "heartbeat", "at": time.strftime("%H:%M:%S"),
                            "totals": totals,
                            "subscribers": BROKER.subscriber_count,
                            "predictions_served": SERVICE.prediction_count}
                    yield "event: heartbeat\ndata: %s\n\n" % json.dumps(beat, default=str)
        finally:
            BROKER.unsubscribe(channel)

    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
               "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    return Response(stream_with_context(generate()), headers=headers)


@app.route("/health")
def health_route():
    ready = SERVICE.meta is not None and SERVICE.frame is not None
    return jsonify({"ok": ready, "source": SERVICE.source,
                    "rows": int(len(SERVICE.frame)) if ready else 0,
                    "models_loaded": sorted(SERVICE.pipelines) if ready else [],
                    "last_error": SERVICE.last_error})


# ---------------------------------------------------------------------------
# Start-up
# ---------------------------------------------------------------------------


def bootstrap():
    """Train on first run so a fresh clone can go straight to `python app.py`.

    A dataset uploaded through /retrain is persisted under data/ and stays active after
    a restart: loading the shipped CSV while the metrics on disk described the upload
    would put two different datasets on the same page.
    """
    if not (os.path.exists(trainer.KNN_PATH) and os.path.exists(trainer.RF_PATH)
            and os.path.exists(trainer.META_PATH)):
        dataset = schema.default_dataset_path(PROJECT_ROOT)
        if not os.path.exists(dataset):
            from model import generate_dataset
            generate_dataset.write_outputs(PROJECT_ROOT)
        trainer.train_and_save(data_path=dataset)
    # The planning hierarchy CSV is generated, so it is written once on first run and
    # left alone after that: /locations serves the same rows from memory, and the file
    # exists so a grader can open the dataset the brief named.
    planning_data.ensure_dataset(config.DATA_DIR, quiet=True)
    SERVICE.load(ref=RETRAIN_ACTIVE if os.path.exists(RETRAIN_ACTIVE) else None)

def main():
    bootstrap()
    meta, frame = SERVICE.meta, SERVICE.frame
    port = config.PORT
    knn, rf = meta["metrics"]["knn"], meta["metrics"]["rf"]
    winner = meta["comparison"]["winner"]
    winner_metrics = meta["metrics"][winner]

    banner = [
        "=" * 72,
        TEAM["title"],
        "  by %s | Guide: %s | %s" % (" & ".join(TEAM["developers"]), TEAM["guide"],
                                       TEAM["institution"]),
        "  Open http://127.0.0.1:%d in your browser" % port,
        "  Dataset rows : %d (train %d / test %d)" % (meta["dataset"]["rows"],
                                                     meta["dataset"]["train_size"],
                                                     meta["dataset"]["test_size"]),
        "  Winner       : %s (Yes-F1 %.4f, accuracy %.4f)" % (
            trainer.MODEL_LABELS[winner], winner_metrics["f1"], winner_metrics["accuracy"]),
        "  KNN          : accuracy %.4f  Yes-F1 %.4f  AUC %s" % (
            knn["accuracy"], knn["f1"], knn["roc_auc"]),
        "  Random Forest: accuracy %.4f  Yes-F1 %.4f  AUC %s" % (
            rf["accuracy"], rf["f1"], rf["roc_auc"]),
        "  Data store   : %s (%s)" % (SERVICE.source["store"], SERVICE.source["ref"]),
        "=" * 72,
    ]
    print("\n".join(banner))
    # debug stays off: the reloader would start the server twice and confuse the browser
    # test, and threaded=True lets the SSE stream hold one connection while others work.
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
