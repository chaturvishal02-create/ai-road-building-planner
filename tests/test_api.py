"""Tests for the HTTP layer: every route in app.py, driven through a real Flask test
client against the real trained models.

Two rules hold this file together:
  * nothing is mocked. A number asserted here is a number the shipped app produced.
  * a test that changes global state (retrain, source switch) restores it inside the
    same test, so the suite can be run in any order.
"""

import io
import json
import math
import os

import pandas as pd
import pytest

import app as app_module
import config
from data_layer import pipeline, schema
from data_layer.schema import CSV_COLUMNS, FEATURES, TARGET

RUPEE = "₹"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def payload_for(vocab, index=0):
    """A valid /predict body built from the vocabularies themselves."""
    scheme = vocab["scheme_name"][index % len(vocab["scheme_name"])]
    state = vocab["state"][index % len(vocab["state"])]
    district = next((d for d, states in vocab["district_states"].items() if state in states))
    return {"scheme_name": scheme, "state": state, "district": district,
            "contractor_type": vocab["contractor_type"][index % 2],
            "sanctioned_budget_lakh": 250, "duration_months": 18,
            "k": 5, "model": "auto"}


def post_json(client, url, body):
    return client.post(url, json=body)


def csv_upload(frame, filename="upload.csv"):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return {"file": (io.BytesIO(buffer.getvalue().encode("utf-8")), filename)}


# ---------------------------------------------------------------------------
# Page and /meta
# ---------------------------------------------------------------------------


def test_index_serves_the_page_with_the_team_and_rupees(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/html")
    html = response.get_data(as_text=True)
    for wanted in ("Vishal Raghunath Chatur", "Mohsin Jabiulla Balgyar", "Tapase Miss",
                   "DBATU", RUPEE):
        assert wanted in html


def test_index_has_no_dollar_sign_left_from_the_previous_version(client):
    assert "$" not in client.get("/").get_data(as_text=True)


def test_meta_exposes_the_indian_schemes_and_states(meta_json):
    vocab = meta_json["vocabularies"]
    assert set(vocab["scheme_name"]) == {"PMGSY (Rural Roads)", "PMAY (Housing)",
                                         "Jal Jeevan Mission", "Smart Cities Mission"}
    for state in ("Maharashtra", "Karnataka", "Delhi", "Uttar Pradesh"):
        assert state in vocab["state"]
    for district in ("Pune", "Mumbai City", "Nashik", "Bengaluru Urban", "New Delhi"):
        assert district in vocab["district"]


def test_meta_vocabularies_agree_with_each_other(meta_json):
    vocab = meta_json["vocabularies"]
    assert set(vocab["district_states"]) == set(vocab["district"])
    for district, states in vocab["district_states"].items():
        assert states and all(state in vocab["state"] for state in states)
    assert vocab["budget_min_lakh"] > 0
    assert vocab["budget_max_lakh"] > vocab["budget_min_lakh"]
    assert vocab["target_values"] == list(schema.TARGET_VALUES)


def test_meta_names_the_features_the_model_actually_uses(meta_json):
    assert meta_json["features"] == FEATURES
    assert meta_json["categorical_features"] == schema.CATEGORICAL_FEATURES
    assert meta_json["numeric_features"] == schema.NUMERIC_FEATURES
    assert meta_json["target"] == TARGET
    assert meta_json["derived_columns"] == schema.DERIVED_COLUMNS
    # the answer column must never be an input column
    assert TARGET not in FEATURES and "actual_cost_lakh" not in FEATURES


def test_meta_carries_the_project_credit_block(meta_json):
    team = meta_json["team"]
    assert team["developers"] == ["Vishal Raghunath Chatur", "Mohsin Jabiulla Balgyar"]
    assert team["guide"] == "Tapase Miss"
    assert team["institution"] == "DBATU"
    assert team["currency"] == RUPEE
    assert "2025-26" in team["subtitle"]


def test_meta_declares_the_currency_and_theme(meta_json):
    assert meta_json["currency"]["symbol"] == RUPEE
    assert meta_json["currency"]["lakh_per_crore"] == 100
    assert meta_json["theme"]["saffron"] == "#FF9933"
    assert meta_json["theme"]["india_green"] == "#138808"


def test_meta_reports_the_model_choices_and_k_window(meta_json):
    assert [m["key"] for m in meta_json["models"]] == ["knn", "rf"]
    assert meta_json["default_model"] in ("knn", "rf")
    low, high = meta_json["k_range"]
    assert low == 1 and high >= 15
    assert low <= meta_json["default_k"] <= high


def test_meta_describes_the_dataset_it_was_trained_on(meta_json):
    dataset = meta_json["dataset"]
    assert dataset["rows"] == 700
    assert dataset["train_size"] + dataset["test_size"] == dataset["rows"]
    assert dataset["synthetic"] is True, "synthetic data must be labelled as such"
    assert dataset["features"] == FEATURES
    assert dataset["target_counts"]["Yes"] + dataset["target_counts"]["No"] == 700


def test_meta_settings_expose_flags_and_never_a_credential(meta_json):
    settings = meta_json["settings"]
    assert set(settings) == {"data_store", "local_dataset", "sample_dataset",
                             "http_url_configured", "s3_bucket_configured",
                             "s3_credentials_present", "allow_writes", "chunk_size",
                             "currency", "theme"}
    assert settings["allow_writes"] is False, "uploads to cloud storage stay opt-in"
    text = json.dumps(meta_json).lower()
    for leaked in ("secret_access_key", "aws_access_key_id", "akia"):
        assert leaked not in text


def test_meta_lists_every_storage_backend_with_a_reason(meta_json):
    sources = {entry["name"]: entry for entry in meta_json["sources"]}
    assert set(sources) == {"local", "http", "s3"}
    assert sources["local"]["active"] is True
    assert sources["local"]["configured"] is True
    assert sources["s3"]["configured"] is False
    assert sources["s3"]["reason"], "an inactive backend must say why"


def test_meta_json_contains_no_dollar_sign(meta_json):
    assert "$" not in json.dumps(meta_json)


# ---------------------------------------------------------------------------
# /predict — happy paths
# ---------------------------------------------------------------------------


def test_predict_accepts_a_valid_request(client, meta_json, valid_payload):
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 200
    result = response.get_json()
    assert result["prediction"] in schema.TARGET_VALUES
    assert result["verdict"] == ("OVER BUDGET" if result["prediction"] == "Yes"
                                 else "WITHIN BUDGET")
    assert 0.0 <= result["risk_pct"] <= 100.0
    assert result["k"] == valid_payload["k"]
    assert result["train_size"] == meta_json["dataset"]["train_size"]


def test_auto_model_resolves_to_the_published_winner(client, meta_json, valid_payload):
    valid_payload["model"] = "auto"
    result = post_json(client, "/predict", valid_payload).get_json()
    assert result["model"] == meta_json["default_model"]
    assert result["model_label"] == meta_json["comparison"]["winner_label"]


@pytest.mark.parametrize("model_key", ["knn", "rf", "KNN", "RF", "auto"])
def test_predict_accepts_every_model_name_caselessly(client, valid_payload, model_key):
    valid_payload["model"] = model_key
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 200
    assert response.get_json()["model"] in ("knn", "rf")


def test_predict_reports_the_risk_from_both_models(client, valid_payload):
    valid_payload["model"] = "knn"
    result = post_json(client, "/predict", valid_payload).get_json()
    assert set(result["risk_pct_by_model"]) == {"knn", "rf"}
    assert result["risk_pct"] == result["risk_pct_by_model"]["knn"]

    valid_payload["model"] = "rf"
    switched = post_json(client, "/predict", valid_payload).get_json()
    assert switched["risk_pct"] == switched["risk_pct_by_model"]["rf"]


def test_risk_percentages_are_real_probabilities(client, valid_payload):
    result = post_json(client, "/predict", valid_payload).get_json()
    for value in [result["risk_pct"]] + list(result["risk_pct_by_model"].values()):
        assert 0.0 <= value <= 100.0
        assert math.isfinite(value)


def test_predict_returns_exactly_k_neighbours(client, valid_payload):
    for k in (1, 5, 12, 31):
        valid_payload["k"] = k
        result = post_json(client, "/predict", valid_payload).get_json()
        assert len(result["neighbours"]) == k
        assert [row["rank"] for row in result["neighbours"]] == list(range(1, k + 1))
        assert result["votes"]["total"] == k
        assert result["votes"]["yes"] + result["votes"]["no"] == k


def test_neighbours_are_ordered_by_ascending_distance(client, valid_payload):
    result = post_json(client, "/predict", valid_payload).get_json()
    distances = [row["distance"] for row in result["neighbours"]]
    assert distances == sorted(distances)
    assert all(distance >= 0 for distance in distances)


def test_k_one_vote_is_exactly_its_single_neighbour(client, valid_payload):
    valid_payload["k"] = 1
    result = post_json(client, "/predict", valid_payload).get_json()
    neighbour = result["neighbours"][0]
    assert result["votes"]["yes"] == (1 if neighbour["over_budget"] == "Yes" else 0)
    assert result["votes"]["no"] == (1 if neighbour["over_budget"] == "No" else 0)


def test_a_bigger_k_reaches_further_but_keeps_the_closest_neighbours(client, valid_payload):
    valid_payload["k"] = 5
    small = post_json(client, "/predict", valid_payload).get_json()["neighbours"]
    valid_payload["k"] = 15
    large = post_json(client, "/predict", valid_payload).get_json()["neighbours"]
    assert {row["project_name"] for row in small} <= {row["project_name"] for row in large}
    assert large[4]["distance"] >= small[4]["distance"] - 1e-9


def test_neighbours_are_real_training_rows_with_rupee_amounts(client, valid_payload,
                                                              service):
    result = post_json(client, "/predict", valid_payload).get_json()
    known = set(service.train_frame["project_name"])
    for row in result["neighbours"]:
        assert row["project_name"] in known
        assert row["sanctioned_inr"].startswith(RUPEE)
        assert row["actual_inr"].startswith(RUPEE)
        assert row["over_budget"] in schema.TARGET_VALUES
        assert (row["actual_cost_lakh"] > row["sanctioned_budget_lakh"]) == \
            (row["over_budget"] == "Yes")


def test_vote_count_equals_the_neighbours_that_went_over(client, valid_payload):
    result = post_json(client, "/predict", valid_payload).get_json()
    yes = sum(1 for row in result["neighbours"] if row["over_budget"] == "Yes")
    assert result["votes"]["yes"] == yes


def test_expected_cost_follows_the_published_heuristic(client, valid_payload, metrics_json):
    valid_payload["sanctioned_budget_lakh"] = 400
    result = post_json(client, "/predict", valid_payload).get_json()
    cost = result["cost"]
    heuristic = metrics_json["overrun_heuristic"]
    scheme = heuristic["per_scheme"].get(valid_payload["scheme_name"])
    mean_pct = scheme["mean_overrun_pct"] if scheme else heuristic["overall_mean_pct"]

    assert cost["sanctioned_lakh"] == 400
    assert cost["scheme_mean_overrun_pct"] == mean_pct
    assert cost["expected_overrun_pct"] == pytest.approx(
        result["risk_pct"] / 100.0 * mean_pct, abs=0.05)
    assert cost["expected_actual_lakh"] == pytest.approx(
        400 * (1 + cost["expected_overrun_pct"] / 100.0), abs=0.2)
    assert "not a regression model" in cost["basis"]


def test_cost_amounts_are_formatted_in_rupees_not_dollars(client, valid_payload):
    cost = post_json(client, "/predict", valid_payload).get_json()["cost"]
    for key in ("sanctioned_inr", "expected_actual_inr", "expected_overrun_inr"):
        assert cost[key].startswith(RUPEE)
        assert "$" not in cost[key]


def test_explanation_reads_like_english_and_quotes_the_inputs(client, valid_payload):
    valid_payload.update({"scheme_name": "PMGSY (Rural Roads)", "state": "Maharashtra",
                          "district": "Nashik", "duration_months": 24,
                          "sanctioned_budget_lakh": 120})
    result = post_json(client, "/predict", valid_payload).get_json()
    text = result["explanation"]
    for wanted in ("PMGSY (Rural Roads)", "Nashik", "Maharashtra", "24 months",
                   RUPEE, "risk"):
        assert wanted.lower() in text.lower(), wanted
    assert "$" not in text


def test_explanation_names_the_mechanism_of_the_chosen_model(client, valid_payload):
    valid_payload["model"] = "knn"
    knn_text = post_json(client, "/predict", valid_payload).get_json()["explanation"]
    assert "Euclidean distance" in knn_text

    valid_payload["model"] = "rf"
    rf_text = post_json(client, "/predict", valid_payload).get_json()["explanation"]
    assert "tree-by-tree" in rf_text


def test_agreement_summary_counts_neighbours_per_field(client, valid_payload):
    result = post_json(client, "/predict", valid_payload).get_json()
    summary = result["explanation"]
    assert "Neighbour agreement:" in summary
    for label in ("scheme", "state", "district", "execution"):
        assert "match on %s" % label in summary


@pytest.mark.parametrize("contractor,phrase", [("Govt", "Department/PSU-executed"),
                                               ("Private", "Contractor (EPC) execution")])
def test_advisory_matches_the_execution_mode(client, valid_payload, contractor, phrase):
    valid_payload["contractor_type"] = contractor
    result = post_json(client, "/predict", valid_payload).get_json()
    assert result["inputs"]["contractor_type"] == contractor
    assert phrase in result["specific_advisory"]


def test_a_training_project_is_its_own_nearest_neighbour(client, training_row):
    """KNN at K=1 must return the label of the single closest training row. For a row
    that is in the training set that row is itself -- the cheapest honest check that the
    neighbour lookup, the encoding and the label column all line up."""
    row = training_row(3)
    truth = row.pop("true_label")
    row.update({"k": 1, "model": "knn"})
    result = post_json(client, "/predict", row).get_json()
    assert result["prediction"] == truth
    assert result["votes"]["total"] == 1
    assert result["neighbours"][0]["over_budget"] == truth
    assert result["neighbours"][0]["distance"] == pytest.approx(0.0, abs=1e-6)


def test_predictions_are_numbered_in_order(client, valid_payload):
    first = post_json(client, "/predict", valid_payload).get_json()["sequence"]
    second = post_json(client, "/predict", valid_payload).get_json()["sequence"]
    assert second == first + 1


def test_predict_accepts_a_form_encoded_body(client, valid_payload):
    response = client.post("/predict", data=valid_payload)
    assert response.status_code == 200
    assert response.get_json()["inputs"]["district"] == valid_payload["district"]


@pytest.mark.parametrize("index", range(8))
def test_predict_handles_a_request_from_every_vocabulary_combination(client, meta_json,
                                                                     index):
    """Not an exhaustive cross product (that would be thousands of requests) but every
    scheme, state and contractor type gets exercised."""
    body = payload_for(meta_json["vocabularies"], index)
    response = post_json(client, "/predict", body)
    assert response.status_code == 200, response.get_json()


# ---------------------------------------------------------------------------
# /predict — validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["scheme_name", "state", "district", "contractor_type",
                                   "sanctioned_budget_lakh", "duration_months"])
def test_predict_requires_every_input_field(client, valid_payload, field):
    del valid_payload[field]
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 400
    assert field in response.get_json()["fields"]


@pytest.mark.parametrize("field,value,message", [
    ("scheme_name", "Delhi Durbar Works", "not a known scheme"),
    ("state", "Atlantis", "not a known state"),
    ("district", "Atlantis", "not a known district"),
    ("contractor_type", "Foreign JV", "not a known contractor type"),
])
def test_predict_rejects_categories_the_model_never_saw(client, valid_payload,
                                                        field, value, message):
    valid_payload[field] = value
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 400
    assert message in response.get_json()["fields"][field]


def test_predict_rejects_a_district_from_the_wrong_state(client, valid_payload):
    """The classic entry slip: Pune is real, Maharashtra is real, the pair is not."""
    valid_payload.update({"state": "Karnataka", "district": "Pune"})
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 400
    error = response.get_json()["fields"]["district"]
    assert "Pune is not a district of Karnataka" in error
    assert "Maharashtra" in error, "the message must name where the district does live"


@pytest.mark.parametrize("budget", [0, -250, "twelve crore", 10 ** 9, 0.01, None])
def test_predict_rejects_impossible_budgets(client, valid_payload, budget):
    valid_payload["sanctioned_budget_lakh"] = budget
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 400
    assert "sanctioned_budget_lakh" in response.get_json()["fields"]


@pytest.mark.parametrize("budget", [250, "250", "250.5", 250.0])
def test_predict_accepts_a_budget_written_any_sensible_way(client, valid_payload, budget):
    valid_payload["sanctioned_budget_lakh"] = budget
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 200
    assert response.get_json()["inputs"]["sanctioned_budget_lakh"] == float(budget)


@pytest.mark.parametrize("duration", [0, -3, 400, "six months", 12.5])
def test_predict_rejects_impossible_durations(client, valid_payload, duration):
    valid_payload["duration_months"] = duration
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 400
    assert "duration_months" in response.get_json()["fields"]


@pytest.mark.parametrize("bad_k", [0, -1, 32, 999])
def test_predict_rejects_k_outside_the_sweep(client, meta_json, valid_payload, bad_k):
    valid_payload["k"] = bad_k
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 400
    low, high = meta_json["k_range"]
    assert "between %d and %d" % (low, high) in response.get_json()["fields"]["k"]


@pytest.mark.parametrize("bad_k", ["many", ""])
def test_predict_rejects_a_k_that_is_not_a_number(client, valid_payload, bad_k):
    valid_payload["k"] = bad_k
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 400
    assert "whole number" in response.get_json()["fields"]["k"]


@pytest.mark.parametrize("edge_k", [1, 31])
def test_predict_accepts_the_k_boundaries(client, meta_json, valid_payload, edge_k):
    assert [1, 31] == meta_json["k_range"], "keep this in step with the sweep"
    valid_payload["k"] = edge_k
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 200
    assert response.get_json()["k"] == edge_k


def test_predict_uses_the_default_k_when_none_is_sent(client, meta_json, valid_payload):
    del valid_payload["k"]
    result = post_json(client, "/predict", valid_payload).get_json()
    assert result["k"] == meta_json["default_k"]


def test_predict_rejects_an_unknown_model_name(client, valid_payload):
    valid_payload["model"] = "xgboost"
    response = post_json(client, "/predict", valid_payload)
    assert response.status_code == 400
    assert "knn, rf or auto" in response.get_json()["fields"]["model"]


def test_predict_reports_every_problem_at_once(client, valid_payload):
    valid_payload.update({"state": "Atlantis", "contractor_type": "", "duration_months": -1})
    fields = post_json(client, "/predict", valid_payload).get_json()["fields"]
    assert {"state", "contractor_type", "duration_months"} <= set(fields)


@pytest.mark.parametrize("body", [None, {}, "not json", []])
def test_predict_refuses_a_malformed_body(client, body):
    if body is None:
        response = client.post("/predict")
    elif isinstance(body, str):
        response = client.post("/predict", data=body, content_type="application/json")
    else:
        response = client.post("/predict", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_get_on_predict_is_not_allowed(client):
    assert client.get("/predict").status_code == 405


def test_validate_returns_python_types_the_model_can_consume(service, meta_json):
    values = service.validate(payload_for(meta_json["vocabularies"], 3))
    assert isinstance(values["sanctioned_budget_lakh"], float)
    assert isinstance(values["duration_months"], int)
    assert isinstance(values["k"], int)
    assert values["model"] in ("knn", "rf")


def test_validation_errors_surface_as_api_error(service, meta_json):
    with pytest.raises(app_module.ApiError) as caught:
        service.validate({"scheme_name": "Nope"})
    assert caught.value.status == 400
    assert "scheme_name" in caught.value.fields


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def test_metrics_serve_the_numbers_from_the_saved_artifacts(client, metrics_json, service):
    assert metrics_json["models"] == service.meta["metrics"]
    assert metrics_json["trained_at"] == service.meta["trained_at"]
    assert set(metrics_json["models"]) == {"knn", "rf"}


def test_metrics_have_no_foreign_paper_comparison_block(client, metrics_json):
    """v1 printed a Thai paper's figures next to the live ones; there is no such
    comparison in the India version."""
    text = json.dumps(metrics_json).lower()
    for banned in ("paper", "kusonkhum", "reported_", "thailand", "bangkok"):
        assert banned not in text


def test_metrics_report_both_models_beating_the_trivial_baseline(client, metrics_json):
    for name, scores in metrics_json["models"].items():
        baseline = scores["baseline_always_no_accuracy"]
        assert 0.4 < baseline < 0.75, "%s baseline %s" % (name, baseline)
        assert scores["accuracy"] > baseline, "%s is not better than always-No" % name
        assert scores["confusion_matrix"] and len(scores["confusion_matrix"]) == 2


def test_metrics_confusion_matrices_account_for_the_whole_test_set(client, metrics_json):
    for name, scores in metrics_json["models"].items():
        total = sum(sum(row) for row in scores["confusion_matrix"])
        assert total == scores["total"] == metrics_json["dataset"]["test_size"]
        assert scores["report"]["Yes"]["support"] + scores["report"]["No"]["support"] == total


def test_metrics_are_finite_after_json_round_trip(client, metrics_json):
    text = json.dumps(metrics_json)
    assert "NaN" not in text and "Infinity" not in text
    for scores in metrics_json["models"].values():
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"):
            value = scores[key]
            assert value is None or math.isfinite(value)


def test_metrics_expose_the_pipeline_description(client, metrics_json):
    model = metrics_json["model"]
    assert model["pipeline_steps"] == ["preprocess", "model"]
    assert model["distance_metric"] == "euclidean"
    assert model["random_state"] == 42
    assert model["encoded_width"] > len(FEATURES), "one-hot expansion must be visible"


def test_comparison_winner_is_decided_on_the_yes_class_f1(client, metrics_json):
    comparison = metrics_json["comparison"]
    f1s = {name: scores["f1"] for name, scores in metrics_json["models"].items()}
    assert comparison["winner"] == max(f1s, key=lambda name: f1s[name])
    assert "Yes" in comparison["rule"] or "over_budget" in comparison["rule"]
    assert {row["key"] for row in comparison["rows"]} == {"knn", "rf"}


def test_k_curve_spans_the_declared_range(client, metrics_json, meta_json):
    curve = metrics_json["k_curve"]
    assert curve["ks"] == list(range(meta_json["k_range"][0], meta_json["k_range"][1] + 1))
    assert curve["best_k"] not in meta_json["k_range"], "the sweep is truncated"
    assert curve["best_accuracy"] == max(curve["accuracies"])
    assert len(curve["points"]) == len(curve["ks"])


def test_importances_use_one_honest_method_per_model(client, metrics_json):
    importances = metrics_json["importances"]
    forest = importances["random_forest"]
    assert {row["feature"] for row in forest} == set(FEATURES)
    assert sum(row["importance_pct"] for row in forest) == pytest.approx(100.0, abs=0.5)

    knn = importances["knn"]
    assert "ablation" in knn["method"]
    assert {row["feature"] for row in knn["features"]} == set(FEATURES)


def test_overrun_heuristic_is_labelled_as_a_heuristic(client, metrics_json):
    heuristic = metrics_json["overrun_heuristic"]
    assert "not a regression model" in heuristic["note"]
    assert set(heuristic["per_scheme"]) == {"PMGSY (Rural Roads)", "PMAY (Housing)",
                                            "Jal Jeevan Mission", "Smart Cities Mission"}
    for entry in heuristic["per_scheme"].values():
        assert entry["sample_size"] > 0
        assert entry["mean_overrun_pct"] > 0, "an average overrun must be an overrun"


def test_metrics_source_block_names_the_file_behind_the_numbers(client, metrics_json):
    source = metrics_json["source"]
    assert source["store"] == "local"
    assert source["rows"] == 700
    assert source["ref"].endswith("india_projects_2025_26.csv")


# ---------------------------------------------------------------------------
# /dataset
# ---------------------------------------------------------------------------


def test_dataset_returns_a_page_of_the_real_table(client):
    body = client.get("/dataset").get_json()
    assert body["page"] == 1 and body["size"] == 25
    assert body["total"] == 700
    assert body["pages"] == 28
    assert len(body["rows"]) == 25
    assert set(body["rows"][0]) == {"project_name", "scheme_name", "project_type", "state",
                                    "district", "contractor_type", "sanctioned_budget_lakh",
                                    "sanctioned_inr", "duration_months", "actual_cost_lakh",
                                    "actual_inr", "overrun_lakh", "overrun_inr",
                                    "overrun_pct", "over_budget"}


def test_dataset_pages_never_repeat_a_row(client):
    seen = []
    for page in range(1, 6):
        rows = client.get("/dataset?page=%d&size=25" % page).get_json()["rows"]
        seen.extend(row["project_name"] for row in rows)
    assert len(seen) == 125 and len(set(seen)) == 125


@pytest.mark.parametrize("requested,expected", [(1, 5), (25, 25), (5000, 100)])
def test_dataset_page_size_is_clamped(client, requested, expected):
    body = client.get("/dataset?size=%d" % requested).get_json()
    assert body["size"] == expected
    assert len(body["rows"]) == min(expected, body["total"])


def test_dataset_page_beyond_the_end_returns_nothing_rather_than_lies(client):
    body = client.get("/dataset?page=999&size=25").get_json()
    assert body["rows"] == []
    assert body["total"] == 700


@pytest.mark.parametrize("args", [{"page": "abc"}, {"size": "abc"}, {"sort": "drop table"},
                                  {"sort": "actual_cost_lakh; DROP TABLE projects"}])
def test_dataset_rejects_nonsense_parameters(client, args):
    response = client.get("/dataset", query_string=args)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_dataset_search_finds_the_district(client):
    body = client.get("/dataset?search=nashik&size=100").get_json()
    assert 0 < body["total"] < 700
    assert all("nashik" in (row["district"] + row["project_name"] + row["state"]
                            + row["scheme_name"]).lower() for row in body["rows"])


def test_dataset_search_is_case_insensitive_and_substring_based(client):
    lower = client.get("/dataset?search=pmay").get_json()["total"]
    upper = client.get("/dataset?search=PMAY").get_json()["total"]
    assert lower == upper > 0


def test_dataset_search_with_no_hits_returns_an_empty_page(client):
    body = client.get("/dataset?search=zzzz-no-such-work").get_json()
    assert body["total"] == 0 and body["rows"] == []


def test_dataset_search_treats_regex_characters_as_text(client):
    body = client.get("/dataset?search=(Rural").get_json()
    assert body["total"] > 0
    assert all("PMGSY" in row["scheme_name"] for row in body["rows"])


@pytest.mark.parametrize("field,value", [
    ("scheme_name", "Jal Jeevan Mission"),
    ("state", "Karnataka"),
    ("contractor_type", "Private"),
    ("over_budget", "Yes"),
])
def test_dataset_column_filters(client, field, value):
    body = client.get("/dataset", query_string={field: value, "size": 100}).get_json()
    assert 0 < body["total"] < 700
    assert all(row[field] == value for row in body["rows"])


def test_dataset_filters_combine(client, meta_json):
    body = client.get("/dataset", query_string={"state": "Maharashtra",
                                                "over_budget": "Yes",
                                                "contractor_type": "Govt",
                                                "size": 100}).get_json()
    for row in body["rows"]:
        assert (row["state"], row["over_budget"], row["contractor_type"]) == \
            ("Maharashtra", "Yes", "Govt")


@pytest.mark.parametrize("column", sorted({"sanctioned_budget_lakh", "actual_cost_lakh",
                                           "duration_months", "overrun_lakh",
                                           "overrun_pct", "project_name", "state",
                                           "district", "scheme_name",
                                           "contractor_type", "over_budget"}))
def test_dataset_sorting_is_whitelisted_and_ordered(client, column):
    """Each direction is checked inside its own window: the ascending page holds the 100
    smallest rows and the descending page the 100 largest, so the two lists describe
    different rows and are only comparable to their own ordering."""
    ascending = client.get("/dataset?sort=%s&dir=asc&size=100" % column).get_json()["rows"]
    values = [row[column] for row in ascending]
    assert len(values) == 100
    assert values == sorted(values)

    descending = client.get("/dataset?sort=%s&dir=desc&size=100" % column).get_json()["rows"]
    reverse = [row[column] for row in descending]
    assert reverse == sorted(reverse, reverse=True)
    assert values[0] <= reverse[0]


def test_dataset_rows_are_internally_consistent(client):
    rows = client.get("/dataset?size=100").get_json()["rows"]
    for row in rows:
        assert row["sanctioned_inr"].startswith(RUPEE)
        assert "$" not in json.dumps(row)
        assert row["over_budget"] == ("Yes" if row["actual_cost_lakh"]
                                      > row["sanctioned_budget_lakh"] else "No")
        assert row["overrun_lakh"] == pytest.approx(
            row["actual_cost_lakh"] - row["sanctioned_budget_lakh"], abs=0.11)


def test_dataset_shows_the_four_brief_schemes_across_the_whole_table(client):
    schemes = set()
    for page in range(1, 8):
        rows = client.get("/dataset?page=%d&size=100" % page).get_json()["rows"]
        schemes.update(row["scheme_name"] for row in rows)
    assert schemes == {"PMGSY (Rural Roads)", "PMAY (Housing)", "Jal Jeevan Mission",
                       "Smart Cities Mission"}


# ---------------------------------------------------------------------------
# /dashboard
# ---------------------------------------------------------------------------


def test_dashboard_totals_match_the_loaded_frame(dashboard_json, dataset_frame):
    totals = dashboard_json["totals"]
    assert totals["projects"] == len(dataset_frame) == 700
    assert totals["sanctioned_lakh"] == pytest.approx(
        float(dataset_frame["sanctioned_budget_lakh"].sum()), abs=1.0)
    assert totals["actual_lakh"] == pytest.approx(
        float(dataset_frame["actual_cost_lakh"].sum()), abs=1.0)
    assert totals["overrun_lakh"] == pytest.approx(totals["actual_lakh"]
                                                   - totals["sanctioned_lakh"], abs=1.0)
    assert totals["over_budget"] == int((dataset_frame[TARGET] == "Yes").sum())
    assert totals["states"] == int(dataset_frame["state"].nunique())


def test_dashboard_breaks_the_money_down_by_scheme(dashboard_json, dataset_frame):
    by_scheme = dashboard_json["by_scheme"]
    assert {row["scheme_name"] for row in by_scheme} == set(dataset_frame["scheme_name"])
    assert sum(row["projects"] for row in by_scheme) == 700
    for row in by_scheme:
        group = dataset_frame[dataset_frame["scheme_name"] == row["scheme_name"]]
        assert row["projects"] == len(group)
        assert row["sanctioned_lakh"] == pytest.approx(
            float(group["sanctioned_budget_lakh"].sum()), abs=1.0)
        assert row["overrun_pct"] == pytest.approx(
            100.0 * (row["actual_lakh"] - row["sanctioned_lakh"]) / row["sanctioned_lakh"],
            abs=0.05)
        assert row["sanctioned_inr"].startswith(RUPEE)


def test_dashboard_state_distribution_is_ranked_by_project_volume(dashboard_json):
    states = dashboard_json["by_state"]
    counts = [row["projects"] for row in states]
    assert counts == sorted(counts, reverse=True)
    for state in ("Maharashtra", "Karnataka", "Delhi", "Uttar Pradesh"):
        assert state in {row["state"] for row in states}
    assert sum(row["projects"] for row in states) == 700
    for row in states:
        assert row["over_budget"] + row["within_budget"] == row["projects"]
        assert row["districts"] >= 1


def test_dashboard_compares_government_and_private_execution(dashboard_json):
    by_contractor = dashboard_json["by_contractor"]
    assert {row["contractor_type"] for row in by_contractor} == set(schema.CONTRACTOR_TYPES)
    assert sum(row["projects"] for row in by_contractor) == 700
    for row in by_contractor:
        assert 0 <= row["over_budget_share_pct"] <= 100


def test_dashboard_lists_the_ten_worst_overruns_in_rupees(dashboard_json, dataset_frame):
    """Ranked by rupees lost, which is what a finance note cares about -- the percentage
    is shown next to it but is not the ordering key."""
    top = dashboard_json["top_overruns"]
    expected = list(dataset_frame.nlargest(10, "overrun_lakh")["project_name"])
    assert [row["project_name"] for row in top] == expected
    for row in top:
        assert row["overrun_pct"] > 0
        assert row["overrun_inr"].startswith(RUPEE)
        assert row["actual_inr"].startswith(RUPEE)


def test_dashboard_carries_no_dollar_signs_and_no_invented_numbers(dashboard_json):
    text = json.dumps(dashboard_json)
    assert "$" not in text
    assert "NaN" not in text and "Infinity" not in text


def test_dashboard_counts_the_predictions_it_has_served(client, dashboard_json,
                                                        valid_payload):
    before = client.get("/dashboard").get_json()["predictions_served"]
    post_json(client, "/predict", valid_payload)
    assert client.get("/dashboard").get_json()["predictions_served"] == before + 1
    assert dashboard_json["source"]["store"] == "local"


# ---------------------------------------------------------------------------
# Storage backends and the pipeline
# ---------------------------------------------------------------------------


def test_health_reports_a_ready_service(client):
    body = client.get("/health").get_json()
    assert body["ok"] is True
    assert body["rows"] == 700
    assert body["models_loaded"] == ["knn", "rf"]
    assert body["source"]["store"] == "local"


def test_sources_get_describes_every_backend(client):
    body = client.get("/sources").get_json()
    assert {entry["name"] for entry in body["sources"]} == {"local", "http", "s3"}
    assert body["active"]["store"] == "local"
    assert body["sample_dataset"].endswith("sample_20_projects.csv")


def test_switching_to_local_reloads_the_same_data(client):
    response = client.post("/sources", json={"store": "local"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["source"]["rows"] == 700
    assert body["meta"]["dataset"]["rows"] == 700


@pytest.mark.parametrize("store,phrase", [
    ("carrier-pigeon", "unknown data store"),
    ("http", "HTTP_DATASET_URL"),
    ("s3", "S3_BUCKET"),
])
def test_an_unusable_backend_refuses_with_an_explanation(client, store, phrase):
    response = client.post("/sources", json={"store": store})
    assert response.status_code == 400
    message = response.get_json()["fields"]["store"]
    assert phrase in message


def test_a_failed_source_switch_leaves_the_app_running_the_original_data(client, service):
    client.post("/sources", json={"store": "s3"})
    assert service.source["store"] == "local"
    assert client.get("/health").get_json()["ok"] is True


def test_pipeline_route_publishes_the_manifest(client):
    body = client.get("/pipeline").get_json()
    manifest = body["manifest"]
    assert "Spark-shaped" in body["engine"]
    assert body["chunk_size"] == config.CHUNK_SIZE
    assert manifest["row_count"] == 700
    assert manifest["partition_column"] == "scheme_name"
    assert len(manifest["partitions"]) == 4
    assert sum(entry["rows"] for entry in manifest["partitions"].values()) == 700
    assert manifest["format"] == "csv"
    assert manifest["source_sha256"], "the manifest must fingerprint its input"


def test_pipeline_rebuild_rewrites_the_partitioned_store(client):
    response = client.post("/pipeline/rebuild")
    assert response.status_code == 200
    manifest = response.get_json()["manifest"]
    assert manifest["row_count"] == 700
    assert manifest["file_count"] == sum(entry["files"]
                                         for entry in manifest["partitions"].values())
    directories = [name for name in os.listdir(config.FEATURE_STORE_DIR)
                   if name.startswith("scheme_name=")]
    assert len(directories) == 4


def test_feature_store_on_disk_reads_back_to_the_same_rows(client):
    frame = pipeline.load_feature_store()
    assert len(frame) == 700
    assert set(frame["scheme_name"]) == {"PMGSY (Rural Roads)", "PMAY (Housing)",
                                         "Jal Jeevan Mission", "Smart Cities Mission"}


def test_sample_csv_downloads_as_a_file(client):
    response = client.get("/sample.csv")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/csv")
    assert "attachment" in response.headers["Content-Disposition"]
    text = response.get_data(as_text=True)
    frame = pd.read_csv(io.StringIO(text))
    assert list(frame.columns) == CSV_COLUMNS
    assert len(frame) == 20
    assert "$" not in text


def test_unknown_route_returns_a_json_404_not_a_500(client):
    response = client.get("/no-such-endpoint")
    assert response.status_code == 404
    assert "error" in response.get_json()


# ---------------------------------------------------------------------------
# /retrain — the destructive path, so it cleans up after itself
# ---------------------------------------------------------------------------


def balanced_upload_frame(dataset_frame, per_class=20):
    """A deliberately small but two-class dataset: enough rows to clear the 30-row
    floor, few enough that the retrain is quick."""
    yes = dataset_frame[dataset_frame[TARGET] == "Yes"].head(per_class)
    no = dataset_frame[dataset_frame[TARGET] == "No"].head(per_class)
    return pd.concat([yes, no])


def test_retrain_requires_a_body(client):
    response = client.post("/retrain", json={})
    assert response.status_code == 400
    assert "CSV file" in response.get_json()["error"]


def test_retrain_rejects_text_that_is_not_csv(client):
    response = client.post("/retrain", json={"csv_text": "this is not a table\nat all"})
    assert response.status_code == 400


def test_retrain_rejects_a_file_missing_contract_columns(client):
    frame = pd.DataFrame({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})
    response = client.post("/retrain", data=csv_upload(frame),
                           content_type="multipart/form-data")
    assert response.status_code == 400
    assert "missing required column" in response.get_json()["error"]


def test_retrain_rejects_an_unknown_target_value(client, dataset_frame):
    frame = balanced_upload_frame(dataset_frame)
    frame = frame.copy()
    frame[TARGET] = "maybe"
    response = client.post("/retrain", data=csv_upload(frame),
                           content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Yes or No" in response.get_json()["error"]


def test_retrain_requires_enough_rows(client, dataset_frame):
    frame = balanced_upload_frame(dataset_frame, per_class=5)
    response = client.post("/retrain", data=csv_upload(frame),
                           content_type="multipart/form-data")
    assert response.status_code == 422
    assert "at least %d rows" % app_module.MIN_RETRAIN_ROWS in response.get_json()["error"]


def test_retrain_requires_both_classes(client, dataset_frame):
    frame = dataset_frame[dataset_frame[TARGET] == "No"].head(40)
    response = client.post("/retrain", data=csv_upload(frame),
                           content_type="multipart/form-data")
    assert response.status_code == 422
    assert "both Yes and No" in response.get_json()["error"]


def test_retrain_rejects_a_blank_cell(client, dataset_frame):
    frame = balanced_upload_frame(dataset_frame).copy()
    frame.loc[frame.index[0], "duration_months"] = None
    response = client.post("/retrain", data=csv_upload(frame),
                           content_type="multipart/form-data")
    assert response.status_code == 400
    assert "blank" in response.get_json()["error"]


def test_failed_retrains_leave_the_serving_models_untouched(client):
    before = client.get("/health").get_json()
    client.post("/retrain", json={"csv_text": "a,b\n1,2"})
    client.post("/retrain", data=csv_upload(pd.DataFrame({"foo": [1]})))
    after = client.get("/health").get_json()
    assert after["rows"] == before["rows"] == 700
    assert after["models_loaded"] == ["knn", "rf"]
    assert client.get("/metrics").get_json()["dataset"]["rows"] == 700


def test_retraining_on_a_small_upload_then_resetting_restores_everything(client,
                                                                        dataset_frame,
                                                                        training_row,
                                                                        metrics_json):
    """The whole point of /retrain: a grader can paste their own 40 rows, see the
    numbers move, and press reset to get the shipped model back.

    The reset runs in a finally block because this is the one test that overwrites the
    artifacts every other test reads -- a failure here must not leave the repository
    trained on 40 rows.
    """
    frame = balanced_upload_frame(dataset_frame)
    response = client.post("/retrain", data=csv_upload(frame),
                           content_type="multipart/form-data")
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["rows"] == 40
    assert body["source"]["rows"] == 40
    assert body["source"]["ref"].endswith("retrain_active.csv")
    assert set(body["metrics"]) == {"knn", "rf"}
    try:
        # the app now really serves the small model
        assert client.get("/metrics").get_json()["dataset"]["rows"] == 40
        assert client.get("/health").get_json()["rows"] == 40
        assert os.path.exists(app_module.RETRAIN_ACTIVE)
        # one of the uploaded rows, asked back of the model that just learned it
        body = training_row()
        body.pop("true_label")
        prediction = post_json(client, "/predict", body)
        assert prediction.status_code == 200, prediction.get_json()
        assert prediction.get_json()["train_size"] == 32
    finally:
        reset = client.post("/retrain", json={"reset": True})

    assert reset.status_code == 200
    assert reset.get_json()["rows"] == 700
    restored = client.get("/metrics").get_json()
    assert restored["dataset"]["rows"] == 700
    assert restored["models"] == metrics_json["models"], \
        "a reset retrains from the same seeded CSV, so every metric must come back equal"
    assert client.get("/health").get_json()["source"]["ref"].endswith(
        "india_projects_2025_26.csv")
    assert not os.path.exists(app_module.RETRAIN_ACTIVE), \
        "a reset must not leave an upload behind as the active dataset"


def test_a_retrained_upload_survives_a_restart(tmp_path, monkeypatch):
    """data/retrain_active.csv is what makes a restart honest: the rows and the metrics
    have to come back describing the same dataset."""
    assert app_module.RETRAIN_ACTIVE.endswith("retrain_active.csv")
    monkeypatch.setattr(app_module, "RETRAIN_ACTIVE",
                        os.path.join(str(tmp_path), "retrain_active.csv"))
    loaded = {}
    monkeypatch.setattr(app_module.SERVICE, "load",
                        lambda store_name=None, ref=None, rebuild_store=True:
                        loaded.update({"ref": ref}))

    app_module.bootstrap()                      # no persisted upload -> shipped CSV
    assert loaded["ref"] is None

    marker = os.path.join(str(tmp_path), "retrain_active.csv")
    pd.DataFrame({"x": [1]}).to_csv(marker, index=False)
    app_module.bootstrap()                      # an upload is waiting -> serve it
    assert loaded["ref"] == marker


def test_retrain_reads_the_named_source_and_a_pasted_file(client, monkeypatch):
    """Only the upload-reading half of /retrain here: training is covered above."""
    seen = {}

    def fake_retrain(frame, label):
        seen["rows"] = int(len(frame))
        seen["label"] = label
        return {"ok": True}

    monkeypatch.setattr(app_module.SERVICE, "retrain", fake_retrain)

    assert client.post("/retrain", json={"source": "local"}).status_code == 200
    assert seen == {"rows": 700, "label": "local:"}

    sample = client.get("/sample.csv").get_data(as_text=True)
    client.post("/retrain", json={"csv_text": sample})
    assert seen == {"rows": 20, "label": "upload:pasted-csv"}


def test_retrain_from_an_unreachable_source_fails_cleanly(client):
    response = client.post("/retrain", json={"source": "http"})
    assert response.status_code == 400
    assert "HTTP_DATASET_URL" in response.get_json()["error"]
    assert client.get("/health").get_json()["rows"] == 700


# ---------------------------------------------------------------------------
# Live events
# ---------------------------------------------------------------------------


def test_broker_delivers_one_event_to_every_subscriber():
    broker = app_module.EventBroker()
    first, second = broker.subscribe(), broker.subscribe()
    assert broker.subscriber_count == 2
    broker.publish({"type": "prediction", "sequence": 1})
    assert first.get_nowait()["sequence"] == 1
    assert second.get_nowait()["sequence"] == 1
    assert first.empty()


def test_broker_drops_events_for_a_client_that_stops_reading():
    """A stalled browser tab must never slow a prediction down."""
    broker = app_module.EventBroker(max_queue=2)
    stalled = broker.subscribe()
    broker.publish({"n": 1})
    broker.publish({"n": 2})
    broker.publish({"n": 3})          # queue is full: dropped, not raised
    assert stalled.qsize() == 2


def test_broker_stops_tracking_an_unsubscribed_channel():
    broker = app_module.EventBroker()
    channel = broker.subscribe()
    broker.unsubscribe(channel)
    assert broker.subscriber_count == 0
    broker.unsubscribe(channel)       # idempotent
    assert broker.subscriber_count == 0


def test_stream_opens_with_a_hello_event(client):
    with client.open("/stream", buffered=False) as response:
        assert response.status_code == 200
        assert response.headers["Content-Type"].startswith("text/event-stream")
        assert response.headers["Cache-Control"] == "no-cache"
        chunk = next(iter(response.response))
    text = chunk.decode("utf-8")
    assert text.startswith("event: hello")
    payload = json.loads(text.split("data: ", 1)[1].strip())
    assert payload["source"]["store"] == "local"
    assert payload["predictions_served"] >= 1


def test_a_prediction_reaches_every_stream_subscriber(client, valid_payload):
    channel = app_module.BROKER.subscribe()
    try:
        result = post_json(client, "/predict", valid_payload).get_json()
        event = channel.get_nowait()
    finally:
        app_module.BROKER.unsubscribe(channel)
    assert event["type"] == "prediction"
    brief = event["payload"]
    assert brief["sequence"] == result["sequence"]
    assert brief["risk_pct"] == result["risk_pct"]
    assert brief["sanctioned_inr"].startswith(RUPEE)
    assert brief["model"] == result["model_label"]


def test_the_stream_leaves_no_orphan_subscribers(client):
    start = app_module.BROKER.subscriber_count
    for _ in range(3):
        with client.open("/stream", buffered=False) as response:
            next(iter(response.response))
    assert app_module.BROKER.subscriber_count == start
