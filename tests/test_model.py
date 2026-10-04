"""Tests for the dataset generator and the training pipeline.

Everything here is checked against the artifacts the app actually ships with, so a
test cannot pass on a mock while the real model is broken.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import config
from data_layer import schema
from data_layer.schema import FEATURES, TARGET
from model import generate_dataset as gen
import model.train as trainer


# ---------------------------------------------------------------------------
# The generated dataset
# ---------------------------------------------------------------------------


def test_row_count_and_columns_match_the_contract():
    frame = gen.generate()
    assert len(frame) == gen.ROW_COUNT == 700
    assert list(frame.columns) == schema.CSV_COLUMNS


def test_generator_is_deterministic():
    """Same seed, same rows — the README tells a grader they can reproduce the dataset."""
    first = gen.generate(seed=7)
    second = gen.generate(seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert not gen.generate(seed=8).equals(first)


def test_scheme_quotas_are_exact():
    counts = gen.generate()["scheme_name"].value_counts().to_dict()
    assert counts == gen.SCHEME_COUNTS
    assert sum(counts.values()) == 700


def test_overrun_rate_is_the_target_rate_exactly():
    frame = gen.generate()
    share = (frame[TARGET] == "Yes").mean() * 100
    assert abs(share - 42.0) < 0.6, "SPEC_INDIA.md section 3.4 targets 42% +/- 3%"


def test_label_agrees_with_the_two_money_columns():
    """The target must be pure arithmetic on the stored numbers, not a separate
    random draw — otherwise the app would show rows that contradict their own label."""
    frame = gen.generate()
    over = frame["actual_cost_lakh"] > frame["sanctioned_budget_lakh"]
    assert ((frame[TARGET] == "Yes") == over).all()


def test_every_state_district_pair_is_real():
    frame = gen.generate()
    allowed = {state: set(districts) for state, districts in gen.STATE_DISTRICTS.items()}
    for _, row in frame.iterrows():
        assert row["district"] in allowed[row["state"]], \
            "%s is not a district of %s" % (row["district"], row["state"])


def test_pune_mumbai_nashik_bangalore_and_delhi_are_present():
    """The districts the brief asked for specifically."""
    districts = set(gen.generate()["district"])
    for wanted in ("Pune", "Mumbai City", "Nashik", "Bengaluru Urban", "New Delhi"):
        assert wanted in districts


def test_states_the_brief_named_are_all_covered():
    states = set(gen.generate()["state"])
    for wanted in ("Maharashtra", "Karnataka", "Delhi", "Uttar Pradesh"):
        assert wanted in states


def test_project_names_are_unique_and_indian():
    frame = gen.generate()
    assert frame["project_name"].is_unique
    assert frame["project_name"].str.contains(r"\[Work [A-Z]{2}-").all()
    # no Thai leftovers from the previous version of this project
    for banned in ("Bangkok", "Thailand", "Kusonkhum", "Ratchaburi"):
        assert not frame["project_name"].str.contains(banned).any()


def test_amounts_and_durations_stay_in_a_built_range():
    frame = gen.generate()
    assert (frame["sanctioned_budget_lakh"] > 0).all()
    assert frame["sanctioned_budget_lakh"].between(10, 100_000).all()
    assert frame["duration_months"].between(6, 96).all()


def test_contractor_types_are_the_two_indian_modes():
    assert set(gen.generate()["contractor_type"]) == {"Govt", "Private"}


def test_project_types_map_to_the_four_central_schemes():
    frame = gen.generate()
    assert frame.groupby("scheme_name")["project_type"].nunique().eq(1).all()
    assert set(frame["project_type"]) == set(gen.SCHEME_PROJECT_TYPE.values())


def test_self_check_passes_on_the_generated_frame():
    report = gen.self_check(gen.generate())
    assert report["rows"] == 700
    assert report["yes"] + report["no"] == 700
    assert 39.0 <= report["yes_share_pct"] <= 45.0


def test_sample_csv_is_small_valid_and_indian():
    raw = pd.read_csv(config.SAMPLE_DATASET)
    assert len(raw) == 20
    assert list(raw.columns) == schema.CSV_COLUMNS
    clean = schema.normalise_frame(raw)
    assert set(clean["state"]) <= set(gen.STATE_DISTRICTS)
    assert (clean[TARGET] == "Yes").sum() >= 1
    assert (clean[TARGET] == "No").sum() >= 1


def test_shipped_dataset_matches_a_fresh_generation():
    """data/india_projects_2025_26.csv on disk must be the seeded output, so the
    README's 'reproduce it' instruction is true."""
    on_disk = pd.read_csv(config.LOCAL_DATASET)
    fresh = gen.generate()
    pd.testing.assert_frame_equal(on_disk.reset_index(drop=True),
                                  fresh.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_pipeline_shape_and_steps():
    pipeline = trainer.build_pipeline("knn")
    assert [name for name, _ in pipeline.steps] == ["preprocess", "model"]
    assert isinstance(pipeline.named_steps["model"], KNeighborsClassifier)
    assert isinstance(trainer.build_pipeline("rf").named_steps["model"], RandomForestClassifier)


def test_preprocessor_encodes_categories_and_scales_money():
    preprocessor = trainer.build_preprocessor()
    names = [name for name, _trans, _cols in preprocessor.transformers]
    assert names == ["onehot", "scale"], "one branch per column type, no third path"
    assert sorted(preprocessor.transformers[0][2]) == sorted(schema.CATEGORICAL_FEATURES)
    assert sorted(preprocessor.transformers[1][2]) == sorted(schema.NUMERIC_FEATURES)
    assert preprocessor.remainder == "drop", "the target and the name must never leak in"

    onehot = preprocessor.transformers[0][1]
    assert onehot.handle_unknown == "ignore", "an unseen district must not crash a prediction"
    assert onehot.sparse_output is False


def test_scaling_is_part_of_the_pipeline_not_a_separate_step():
    """If the scaler lived outside the Pipeline, a prediction would have to remember
    the training means by hand."""
    preprocessor = trainer.load_model("knn").named_steps["preprocess"]
    fitted = {name: trans for name, trans, *_ in preprocessor.transformers_}
    scale = fitted["scale"]
    assert hasattr(scale, "mean_"), "scaler must be fitted inside the pipeline"
    assert len(scale.mean_) == len(schema.NUMERIC_FEATURES)


def test_unknown_category_does_not_raise():
    """A district the model never saw has to encode, not explode -- the API lets an
    unseen category through as an 'unknown' row on purpose."""
    preprocessor = trainer.load_model("knn").named_steps["preprocess"]
    frame = pd.DataFrame([["Smart Cities Mission", "Maharashtra", "Nowhere City",
                           "Govt", 500.0, 24]],
                         columns=FEATURES)
    encoded = preprocessor.transform(frame)
    assert encoded.shape == (1, preprocessor.get_feature_names_out().size)


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def test_split_is_stratified_reproducible_and_disjoint(dataset_frame):
    x_train, x_test, y_train, y_test, _ = trainer.split(dataset_frame)
    assert len(x_train) == 560 and len(x_test) == 140
    assert abs((y_train == "Yes").mean() - (y_test == "Yes").mean()) < 0.02
    assert not set(x_train.index) & set(x_test.index)

    again = trainer.split(dataset_frame)
    pd.testing.assert_index_equal(x_train.index, again[0].index)
    pd.testing.assert_index_equal(x_test.index, again[1].index)


def test_vocabulary_comes_from_training_rows_only(dataset_frame):
    """If the vocabularies were built on the whole frame, a district that exists only
    in the test half would still be offered to the user — and the model would have
    never seen it."""
    x_train, x_test, _, _, frame = trainer.split(dataset_frame)
    vocab = trainer.build_vocabularies(frame.loc[x_train.index])
    test_only_states = set(x_test["state"]) - set(x_train["state"])
    for state in vocab["state"]:
        assert state in set(x_train["state"])
    assert set(vocab["state"]) == set(x_train["state"]) - test_only_states
    assert vocab["budget_min_lakh"] == pytest.approx(x_train["sanctioned_budget_lakh"].min())
    assert vocab["budget_max_lakh"] == pytest.approx(x_train["sanctioned_budget_lakh"].max())


def test_district_states_map_is_consistent(dataset_frame):
    x_train, _, _, _, frame = trainer.split(dataset_frame)
    vocab = trainer.build_vocabularies(frame.loc[x_train.index])
    for district, states in vocab["district_states"].items():
        assert district in vocab["district"]
        for state in states:
            assert state in vocab["state"]
            assert district in gen.STATE_DISTRICTS[state]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_model_counts_add_up_and_match_the_confusion_matrix(dataset_frame):
    x_train, x_test, y_train, y_test, _ = trainer.split(dataset_frame)
    pipeline = trainer.build_pipeline("rf").fit(x_train, y_train)
    scores = trainer.score_model(pipeline, x_test, y_test)

    tn, fp, fn, tp = np.array(scores["confusion_matrix"]).flatten()
    assert tn + fp + fn + tp == 140 == scores["total"]
    assert scores["correct"] == tn + tp
    assert scores["accuracy"] == pytest.approx((tn + tp) / 140, abs=1e-4)
    assert scores["precision"] == pytest.approx(tp / (tp + fp), abs=1e-4)
    assert scores["recall"] == pytest.approx(tp / (tp + fn), abs=1e-4)
    f1 = 2 * scores["precision"] * scores["recall"] / (scores["precision"] + scores["recall"])
    assert scores["f1"] == pytest.approx(f1, abs=1e-3)


def test_baseline_is_published_so_accuracy_cannot_be_bragged_about(dataset_frame):
    """42% of projects overrun, so 'always say No' scores 58%. A model near that
    number has learned nothing and the UI must show it."""
    x_train, x_test, y_train, y_test, _ = trainer.split(dataset_frame)
    scores = trainer.score_model(trainer.build_pipeline("rf").fit(x_train, y_train),
                                 x_test, y_test)
    majority = (y_test == "No").mean()
    assert scores["baseline_always_no_accuracy"] == pytest.approx(majority, abs=0.01)
    assert scores["accuracy"] > majority + 0.10, "the model must clearly beat always-No"


def test_both_models_learn_something(dataset_frame):
    x_train, x_test, y_train, y_test, _ = trainer.split(dataset_frame)
    for name in ("knn", "rf"):
        scores = trainer.score_model(trainer.build_pipeline(name).fit(x_train, y_train),
                                     x_test, y_test)
        assert scores["roc_auc"] > 0.7, "%s is barely better than a coin flip" % name
        assert scores["f1"] > 0.55


def test_metrics_are_finite_numbers():
    meta = trainer.load_meta()
    for name, scores in meta["metrics"].items():
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            value = scores[key]
            assert isinstance(value, float) and math.isfinite(value), (name, key)
            assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# K sweep, importances, heuristic
# ---------------------------------------------------------------------------


def test_k_curve_covers_the_whole_range_and_has_an_interior_best():
    meta = trainer.load_meta()
    curve = meta["k_curve"]
    assert curve["ks"] == list(range(trainer.K_RANGE[0], trainer.K_RANGE[1] + 1))
    assert len(curve["accuracies"]) == len(curve["ks"])
    assert curve["best_k"] not in trainer.K_RANGE, \
        "the best K sits on the edge of the sweep, so the chart is truncated"
    assert curve["best_accuracy"] == max(curve["accuracies"])
    assert curve["default_k"] == trainer.DEFAULT_K


def test_k_curve_at_default_k_equals_the_reported_knn_accuracy():
    """The K=5 point on the chart and the headline KNN number must be the same fit."""
    meta = trainer.load_meta()
    point = [row for row in meta["k_curve"]["points"] if row["k"] == trainer.DEFAULT_K][0]
    assert point["accuracy"] == pytest.approx(meta["metrics"]["knn"]["accuracy"])


def test_random_forest_importances_are_grouped_and_sum_to_one_hundred():
    meta = trainer.load_meta()
    rows = meta["importances"]["random_forest"]
    assert {row["feature"] for row in rows} == set(FEATURES)
    assert sum(row["importance_pct"] for row in rows) == pytest.approx(100.0, abs=0.5)


def test_knn_ablation_covers_every_feature_and_reports_negative_drops():
    """A negative drop means blanking the field helped — that is a finding, and the
    UI shows it rather than hiding it."""
    meta = trainer.load_meta()
    ablation = meta["importances"]["knn"]
    assert {row["feature"] for row in ablation["features"]} == set(FEATURES)
    for row in ablation["features"]:
        assert row["accuracy_drop"] == pytest.approx(
            ablation["baseline_accuracy"] - row["accuracy_with_blank"], abs=1e-4)


def test_overrun_heuristic_uses_training_overruns_only(dataset_frame):
    x_train, x_test, _, _, frame = trainer.split(dataset_frame)
    train_frame = frame.loc[x_train.index]
    heuristic = trainer.overrun_heuristic(train_frame)
    assert "not a regression model" in heuristic["note"]

    derived = schema.add_derived_columns(train_frame)
    training_yes = int((train_frame[TARGET] == "Yes").sum())
    assert sum(entry["sample_size"] for entry in heuristic["per_scheme"].values()) \
        == training_yes < int((frame[TARGET] == "Yes").sum()), \
        "test rows must not feed the heuristic"

    for scheme, entry in heuristic["per_scheme"].items():
        rows = derived[(derived["scheme_name"] == scheme) & (derived[TARGET] == "Yes")]
        assert len(rows) == entry["sample_size"]
        assert entry["mean_overrun_pct"] == pytest.approx(rows["overrun_pct"].mean(), abs=0.05)
        assert (rows["overrun_pct"] > 0).all(), "an 'overrun' with negative slip is a bug"


# ---------------------------------------------------------------------------
# Saved artifacts
# ---------------------------------------------------------------------------


def test_model_files_exist_and_load():
    for name in ("knn", "rf"):
        assert os.path.exists(trainer.MODEL_FILES[name])
        assert isinstance(trainer.load_model(name), Pipeline)


def test_winner_is_the_higher_yes_f1():
    meta = trainer.load_meta()
    f1s = {name: scores["f1"] for name, scores in meta["metrics"].items()}
    expected = max(f1s, key=lambda name: f1s[name])
    assert meta["comparison"]["winner"] == expected
    assert meta["comparison"]["winner_label"] == trainer.MODEL_LABELS[expected]
    assert "accuracy" in meta["comparison"]["rule"]


def test_meta_records_the_project_team_and_the_provenance():
    meta = trainer.load_meta()["project"]
    assert meta["team"] == ["Vishal Raghunath Chatur", "Mohsin Jabiulla Balgyar"]
    assert meta["guide"] == "Tapase Miss"
    assert meta["institution"] == "DBATU"
    assert "India" in meta["region"]


def test_meta_dataset_block_agrees_with_the_csv(dataset_frame):
    meta = trainer.load_meta()["dataset"]
    assert meta["rows"] == len(dataset_frame) == 700
    assert meta["train_size"] + meta["test_size"] == meta["rows"]
    assert meta["target_counts"]["Yes"] == int((dataset_frame[TARGET] == "Yes").sum())
    assert meta["features"] == FEATURES
    assert meta["random_state"] == trainer.RANDOM_STATE
    assert meta["source"] == "synthetic-india-2025-26"


def test_no_paper_comparison_survives_in_the_artifacts():
    """v1 reported a Thai paper's numbers next to the live ones. The India version has
    no external paper, so nothing here may claim one."""
    with open(trainer.META_PATH, encoding="utf-8") as handle:
        text = handle.read()
    for banned in ("paper", "Kusonkhum", "reported_", "thailand"):
        assert banned.lower() not in text.lower(), banned


def test_training_writes_fresh_artifacts_and_restores_them(dataset_frame, tmp_path, monkeypatch):
    """Retraining on a subset must work end to end — that is what /retrain does —
    without destroying the shipped models."""
    files = {name: trainer.MODEL_FILES[name] for name in ("knn", "rf")}
    monkeypatch.setattr(trainer, "KNN_PATH", str(tmp_path / "knn.joblib"))
    monkeypatch.setattr(trainer, "RF_PATH", str(tmp_path / "rf.joblib"))
    monkeypatch.setattr(trainer, "META_PATH", str(tmp_path / "meta.json"))
    monkeypatch.setattr(trainer, "MODEL_FILES", {"knn": trainer.KNN_PATH, "rf": trainer.RF_PATH})

    subset = dataset_frame.head(120)
    meta = trainer.train_and_save(subset, source_label="test-subset")
    assert meta["dataset"]["rows"] == 120
    assert os.path.exists(trainer.KNN_PATH) and os.path.exists(trainer.RF_PATH)
    assert meta["dataset"]["train_size"] == 96 and meta["dataset"]["test_size"] == 24
    # the shipped files were never touched
    assert all(os.path.exists(path) for path in files.values())
