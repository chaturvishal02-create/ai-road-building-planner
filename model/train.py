"""
Train the two models this project compares, then write the artifacts the web app reads.

WHAT THIS SCRIPT DOES
  1. Load data/india_projects_2025_26.csv (or a frame handed to it by /retrain).
  2. Build ONE preprocessor: one-hot for the category columns, standard-scaling for the
     money and time columns.
  3. Split 80/20, stratified on the target, seed 42 -> 560 train / 140 test.
  4. Fit KNeighborsClassifier and RandomForestClassifier on the same split, so the only
     difference between their numbers is the classifier itself.
  5. Score both, pick a winner on the OVER_BUDGET (Yes) F1 score, sweep K for the KNN,
     measure feature importance both ways, and fit the rupee heuristic that turns a risk
     percentage into an expected actual cost.
  6. Save model/india_knn.joblib, model/india_rf.joblib, model/model_meta.json.

WHY TWO MODELS
  KNN is the algorithm the project is named after and it gives an explainable
  "these are the five past projects that look like yours" answer. RandomForest is here
  because it handles mixed numeric+categorical data natively, tolerates imbalance with
  class_weight='balanced', and produces feature importances. Showing both, and saying
  which one won, is stronger than showing one.

WHY NOT XGBOOST
  XGBoost ships a large compiled binary. On this machine Windows Smart App Control has
  already blocked unsigned extension modules from newer scikit-learn and scipy wheels,
  so adding another native dependency risks an install that fails on a reviewer's laptop.
  RandomForest is part of scikit-learn and needs nothing extra. Noted in README too.

Run:  py model/train.py  [--data path/to.csv]
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_layer import schema  # noqa: E402
from data_layer.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET  # noqa: E402

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
KNN_PATH = os.path.join(MODEL_DIR, "india_knn.joblib")
RF_PATH = os.path.join(MODEL_DIR, "india_rf.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_K = 5
# Swept 1-31 rather than 1-25: accuracy plateaus from about K=19 onward, so a window
# that ends at the edge would leave the K-vs-accuracy chart looking truncated.
K_RANGE = (1, 31)

MODEL_FILES = {"knn": KNN_PATH, "rf": RF_PATH}
MODEL_LABELS = {"knn": "K-Nearest Neighbours", "rf": "Random Forest"}


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def build_preprocessor():
    """One-hot the categories, scale the numbers.

    Scaling is not optional here: sanctioned budgets run from a few tens of lakh to
    several thousand, and without standardising, one rupee column would dominate every
    distance the KNN computes while the 60-odd one-hot columns would barely register.
    """
    return ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("scale", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def build_estimator(name):
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=DEFAULT_K, weights="uniform", metric="euclidean")
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=5,
                                      class_weight="balanced", random_state=RANDOM_STATE,
                                      n_jobs=1)
    raise ValueError("unknown model name: %r" % name)


def build_pipeline(name):
    return Pipeline([("preprocess", build_preprocessor()), ("model", build_estimator(name))])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _positive_index(classes):
    """Which column of predict_proba() is the 'Yes' class."""
    return list(classes).index(schema.POSITIVE_CLASS)


def score_model(pipeline, x_test, y_test):
    """Every metric the UI shows, computed from the fitted pipeline. Nothing is
    hard-coded, so a retrain cannot leave a stale number behind."""
    predicted = pipeline.predict(x_test)
    proba = pipeline.predict_proba(x_test)
    classes = [str(c) for c in pipeline.classes_]
    pos = _positive_index(classes)

    accuracy = float(accuracy_score(y_test, predicted))
    precision = float(precision_score(y_test, predicted, pos_label=schema.POSITIVE_CLASS,
                                      zero_division=0))
    recall = float(recall_score(y_test, predicted, pos_label=schema.POSITIVE_CLASS,
                               zero_division=0))
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    yes_proba = proba[:, pos]
    try:
        auc = float(roc_auc_score(pd.Series(y_test).map({"Yes": 1, "No": 0}), yes_proba))
    except ValueError:                      # only one class present in the split
        auc = None
    try:
        cross_entropy = float(log_loss(pd.Series(y_test).map({"Yes": 1, "No": 0}), proba,
                                       labels=[0, 1]))
    except ValueError:
        cross_entropy = None

    report = _normalise_report(classification_report(
        y_test, predicted, labels=list(classes), output_dict=True, zero_division=0))

    matrix = confusion_matrix(y_test, predicted, labels=list(classes)).tolist()

    # The trivial comparator an examiner always asks about: what does "assume nothing
    # ever overruns" score? Stating it makes the accuracy figure honest.
    always_no = float(np.mean(np.asarray(y_test) == "No"))

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "log_loss": round(cross_entropy, 4) if cross_entropy is not None else None,
        "labels": classes,
        "confusion_matrix": matrix,
        "report": report,
        "baseline_always_no_accuracy": round(always_no, 4),
        "correct": int(sum(pd.Series(predicted).to_numpy() == pd.Series(y_test).to_numpy())),
        "total": int(len(y_test)),
    }


def _normalise_report(report):
    """sklearn keys F1 as 'f1-score'; the front end prefers 'f1'. Flatten to a stable
    shape so a template never has to know sklearn's spelling."""
    out = {}
    for key, value in report.items():
        if not isinstance(value, dict):
            continue
        out[key] = {
            "precision": round(float(value["precision"]), 4),
            "recall": round(float(value["recall"]), 4),
            "f1": round(float(value["f1-score"]), 4),
            "support": int(value["support"]),
        }
    return out


def compute_k_curve(x_train, y_train, x_test, y_test, k_range=K_RANGE):
    """Accuracy of the KNN for every K, refitting the whole pipeline each time so the
    preprocessor never sees a different split."""
    curve = []
    best = {"k": DEFAULT_K, "accuracy": -1.0}
    for k in range(k_range[0], k_range[1] + 1):
        pipe = build_pipeline("knn")
        pipe.named_steps["model"].set_params(n_neighbors=k)
        pipe.fit(x_train, y_train)
        acc = float(accuracy_score(y_test, pipe.predict(x_test)))
        curve.append({"k": k, "accuracy": round(acc, 6)})
        if acc > best["accuracy"]:
            best = {"k": k, "accuracy": round(acc, 6)}
    return {"ks": [row["k"] for row in curve],
            "accuracies": [row["accuracy"] for row in curve],
            "points": curve, "best_k": best["k"], "best_accuracy": best["accuracy"],
            "default_k": DEFAULT_K, "train_size": int(len(x_train)), "test_size": int(len(x_test))}


# ---------------------------------------------------------------------------
# Feature importance, one honest method per model
# ---------------------------------------------------------------------------


def _grouped_names(pipeline):
    """Encoded column names produced by the preprocessor, e.g. 'onehot__district_Pune'
    or 'scale__sanctioned_budget_lakh'."""
    transformer = pipeline.named_steps["preprocess"]
    return list(transformer.get_feature_names_out())


def rf_feature_importance(pipeline):
    """Random Forest impurity-based importances, summed back to the original columns.
    A categorical column that exploded into 59 dummy columns gets the total of all of
    them, which is the fair comparison against a single numeric column."""
    encoded = _grouped_names(pipeline)
    raw = pipeline.named_steps["model"].feature_importances_
    totals = {}
    for column, value in zip(encoded, raw):
        source = _origin_of(column, pipeline)
        totals[source] = totals.get(source, 0.0) + float(value)
    ordered = sorted(totals.items(), key=lambda item: -item[1])
    total = sum(v for _, v in ordered) or 1.0
    return [{"feature": name, "importance_pct": round(100.0 * value / total, 2)}
            for name, value in ordered]


def _origin_of(encoded_name, pipeline=None):
    """Map an encoded name back to the source column the UI should show."""
    tail = encoded_name.split("__", 1)[-1]
    for candidate in FEATURES:
        if tail == candidate or tail.startswith(candidate + "_"):
            return candidate
    return tail


def knn_ablation_importance(pipeline, x_test, y_test, x_train, y_train):
    """KNN has no native importance. This measures what happens to accuracy when one
    real column is flattened to its most common value in the training data: the bigger
    the drop, the more the neighbours were matching on that column. Simple, cheap, and
    defensible in a viva -- and it is reported as an ablation, not as forest-style
    importance, so the two numbers are never confused."""
    baseline = float(accuracy_score(y_test, pipeline.predict(x_test)))

    rows = []
    for feature in FEATURES:
        if not isinstance(x_test, pd.DataFrame) or feature not in x_test.columns:
            rows.append({"feature": feature, "accuracy_drop": None})
            continue
        perturbed = x_test.copy()
        if feature in CATEGORICAL_FEATURES:
            replacement = x_train[feature].mode(dropna=True).iloc[0]
        else:
            replacement = float(x_train[feature].median())
        perturbed[feature] = replacement
        dropped = float(accuracy_score(y_test, pipeline.predict(perturbed)))
        rows.append({"feature": feature,
                     "accuracy_with_blank": round(dropped, 6),
                     "accuracy_drop": round(baseline - dropped, 6)})
    present = [r for r in rows if r.get("accuracy_drop") is not None]
    present.sort(key=lambda r: -r["accuracy_drop"])
    return {"method": "ablation (replace one column with its training-set common value)",
            "baseline_accuracy": round(baseline, 6), "features": rows,
            "ranking": [r["feature"] for r in present]}


# ---------------------------------------------------------------------------
# Rupee heuristic: risk percentage -> expected actual cost
# ---------------------------------------------------------------------------


def overrun_heuristic(train_frame):
    """Mean and median overrun percentage of the projects that DID overrun, per scheme,
    taken from TRAINING rows only.

    The app converts a predicted risk into an expected spend with
        expected_overrun_pct = risk_fraction * mean_overrun_pct_of_overruns_in_scheme
    which is a stated heuristic, not a second trained model, and is labelled that way on
    screen."""
    frame = train_frame.copy()
    actual = pd.to_numeric(frame.get("actual_cost_lakh"), errors="coerce")
    frame["_pct"] = ((actual - frame["sanctioned_budget_lakh"])
                     / frame["sanctioned_budget_lakh"] * 100.0)
    over = frame[frame[TARGET] == schema.POSITIVE_CLASS]

    per_scheme = {}
    for scheme, group in over.groupby("scheme_name"):
        values = group["_pct"].dropna()
        if not len(values):
            continue
        per_scheme[str(scheme)] = {
            "mean_overrun_pct": round(float(values.mean()), 2),
            "median_overrun_pct": round(float(values.median()), 2),
            "sample_size": int(len(values)),
        }
    all_values = over["_pct"].dropna()
    return {
        "basis": "mean of (actual - sanctioned) / sanctioned over training projects that overran",
        "overall_mean_pct": round(float(all_values.mean()), 2) if len(all_values) else 0.0,
        "overall_median_pct": round(float(all_values.median()), 2) if len(all_values) else 0.0,
        "per_scheme": per_scheme,
        "note": "heuristic, not a regression model",
    }


# ---------------------------------------------------------------------------
# Vocabularies for validation
# ---------------------------------------------------------------------------


def build_vocabularies(train_frame):
    """Everything the API needs to validate an incoming prediction request. Built from
    the training rows so the app cannot accept a category the model never saw as anything
    other than 'unknown'."""
    district_state = {}
    for _, row in train_frame[["district", "state"]].drop_duplicates().iterrows():
        district_state.setdefault(str(row["district"]), []).append(str(row["state"]))

    return {
        "scheme_name": sorted(train_frame["scheme_name"].unique().tolist()),
        "state": sorted(train_frame["state"].unique().tolist()),
        "district": sorted(train_frame["district"].unique().tolist()),
        "contractor_type": sorted(train_frame["contractor_type"].unique().tolist()),
        "district_states": {k: sorted(set(v)) for k, v in sorted(district_state.items())},
        "budget_min_lakh": round(float(train_frame["sanctioned_budget_lakh"].min()), 1),
        "budget_max_lakh": round(float(train_frame["sanctioned_budget_lakh"].max()), 1),
        "duration_min": int(train_frame["duration_months"].min()),
        "duration_max": int(train_frame["duration_months"].max()),
        "target_values": list(schema.TARGET_VALUES),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def split(df, random_state=RANDOM_STATE):
    """One split, shared by both models. X stays a DataFrame so the ColumnTransformer can
    address columns by name and the ablation importances can perturb single columns."""
    frame = schema.normalise_frame(df)
    x = frame[FEATURES].copy()
    y = frame[TARGET].copy()
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None      # a one-class upload must not crash
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=random_state, stratify=stratify)
    return x_train, x_test, y_train, y_test, frame


def train_and_save(df=None, data_path=None, source_label="synthetic-india-2025-26"):
    """Fit both models, score them, write the artifacts. Returns the metadata dict."""
    started = time.time()
    if df is None:
        data_path = data_path or schema.default_dataset_path(PROJECT_ROOT)
        df = pd.read_csv(data_path, encoding="utf-8-sig")

    frame = schema.normalise_frame(df)
    data_path = data_path or schema.default_dataset_path(PROJECT_ROOT)

    x_train, x_test, y_train, y_test, frame = split(frame)

    fitted, metrics = {}, {}
    for name in ("knn", "rf"):
        pipeline = build_pipeline(name)
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        metrics[name] = score_model(pipeline, x_test, y_test)
        metrics[name]["model"] = MODEL_LABELS[name]

    winner = max(("knn", "rf"), key=lambda n: metrics[n]["f1"])

    k_curve = compute_k_curve(x_train, y_train, x_test, y_test)
    importances = {"random_forest": rf_feature_importance(fitted["rf"]),
                   "knn": knn_ablation_importance(fitted["knn"], x_test, y_test, x_train, y_train)}

    heuristic = overrun_heuristic(frame.loc[y_train.index])
    vocabularies = build_vocabularies(frame.loc[y_train.index])
    validation = schema.validate_dataset(frame)

    joblib.dump(fitted["knn"], KNN_PATH)
    joblib.dump(fitted["rf"], RF_PATH)

    transformer = fitted["knn"].named_steps["preprocess"]
    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 2),
        "project": {
            "title": "Government Construction Project Budget Prediction Using Machine Learning",
            "region": "India, FY 2025-26",
            "team": ["Vishal Raghunath Chatur", "Mohsin Jabiulla Balgyar"],
            "guide": "Tapase Miss",
            "institution": "DBATU",
            "programme": "B.Tech in Artificial Intelligence & Data Science Engineering",
        },
        "dataset": {
            "path": data_path,
            "source": source_label,
            "rows": int(len(frame)),
            "train_size": int(len(x_train)),
            "test_size": int(len(x_test)),
            "columns": list(frame.columns),
            "features": FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "target": TARGET,
            "target_counts": validation["target_counts"],
            "over_budget_share_pct": validation["over_budget_share_pct"],
            "test_size_fraction": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "synthetic": True,
        },
        "model": {
            "pipeline_steps": ["preprocess", "model"],
            "preprocessor": ["ColumnTransformer", "OneHotEncoder(handle_unknown='ignore')",
                            "StandardScaler"],
            "encoded_width": int(len(transformer.get_feature_names_out())),
            "default_k": DEFAULT_K,
            "k_range": list(K_RANGE),
            "distance_metric": "euclidean",
            "vote_rule": "uniform (majority)",
            "random_state": RANDOM_STATE,
        },
        "metrics": metrics,
        "comparison": {
            "winner": winner,
            "winner_label": MODEL_LABELS[winner],
            "rule": "higher F1 on the over_budget=Yes class (accuracy is inflated by the majority class)",
            "rows": [{"key": name, "label": MODEL_LABELS[name],
                      "accuracy": metrics[name]["accuracy"],
                      "precision": metrics[name]["precision"],
                      "recall": metrics[name]["recall"],
                      "f1": metrics[name]["f1"],
                      "roc_auc": metrics[name]["roc_auc"]} for name in ("knn", "rf")],
        },
        "k_curve": k_curve,
        "importances": importances,
        "overrun_heuristic": heuristic,
        "vocabularies": vocabularies,
    }

    with open(META_PATH, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    return meta


def load_meta(path=META_PATH):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model(name):
    path = MODEL_FILES.get(name)
    if not path:
        raise ValueError("unknown model %r (expected knn or rf)" % name)
    return joblib.load(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the KNN and Random Forest comparison.")
    parser.add_argument("--data", default=None, help="CSV to train on (default: data/india_projects_2025_26.csv)")
    parser.add_argument("--source", default="synthetic-india-2025-26", help="label stored in the metadata")
    args = parser.parse_args(argv)

    if args.data and not os.path.exists(args.data):
        print("No such file: %s" % args.data)
        return 2

    meta = train_and_save(data_path=args.data, source_label=args.source)
    lines = [
        "Trained in %.1fs on %d rows (train %d / test %d), %d encoded features" % (
            meta["elapsed_seconds"], meta["dataset"]["rows"], meta["dataset"]["train_size"],
            meta["dataset"]["test_size"], meta["model"]["encoded_width"]),
        "over_budget: %s" % meta["dataset"]["target_counts"],
    ]
    for name in ("knn", "rf"):
        m = meta["metrics"][name]
        lines.append("  %-14s acc %.4f  Yes-P %.4f  Yes-R %.4f  Yes-F1 %.4f  AUC %s" % (
            m["model"], m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"]))
        lines.append("  %-14s confusion %s  (always-No baseline %.4f)" % (
            "", m["confusion_matrix"], m["baseline_always_no_accuracy"]))
    lines.append("Winner by Yes-F1: %s" % meta["comparison"]["winner_label"])
    lines.append("Best K on this split: %d (accuracy %.4f)" % (
        meta["k_curve"]["best_k"], meta["k_curve"]["best_accuracy"]))
    lines.append("Top RF importances: %s" % ", ".join(
        "%s %.1f%%" % (r["feature"], r["importance_pct"]) for r in meta["importances"]["random_forest"][:4]))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
