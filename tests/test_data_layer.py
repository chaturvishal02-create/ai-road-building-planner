"""Tests for the data layer: the column contract, the storage adapters and the
chunked pipeline.

These run without Flask and without a trained model — the data layer is the
foundation everything else stands on, so it gets checked on its own.
"""

import json
import os

import pandas as pd
import pytest

import config
from data_layer import pipeline, schema, storage
from data_layer.schema import (CATEGORICAL_FEATURES, CSV_COLUMNS, FEATURES, NUMERIC_FEATURES,
                               TARGET, DataError)


def make_frame(**overrides):
    """A tiny valid frame, so a test can break exactly one rule."""
    rows = {
        "project_name": ["Work A", "Work B"],
        "scheme_name": ["PMGSY (Rural Roads)", "PMAY (Housing)"],
        "project_type": ["Road construction", "Housing"],
        "state": ["Maharashtra", "Bihar"],
        "district": ["Pune", "Patna"],
        "contractor_type": ["Govt", "Private"],
        "sanctioned_budget_lakh": [250.0, 90.0],
        "duration_months": [18, 12],
        "actual_cost_lakh": [280.0, 84.0],
        TARGET: ["Yes", "No"],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The column contract
# ---------------------------------------------------------------------------


def test_feature_list_excludes_identifiers_and_target():
    """A project name, a project type and the answer itself must never be features.
    project_type is 1-to-1 with scheme_name, so including both would double-count it."""
    assert "project_name" not in FEATURES
    assert "project_type" not in FEATURES
    assert "actual_cost_lakh" not in FEATURES
    assert TARGET not in FEATURES
    assert FEATURES == CATEGORICAL_FEATURES + NUMERIC_FEATURES


def test_csv_columns_cover_everything_the_ui_shows():
    for column in ("sanctioned_budget_lakh", "actual_cost_lakh", "duration_months",
                   "contractor_type", "scheme_name", "state", "district", TARGET):
        assert column in CSV_COLUMNS


def test_normalise_trims_whitespace_and_coerces_labels():
    frame = make_frame(state=["  Maharashtra  ", "Bihar"],
                       **{TARGET: ["y", "n"]})
    clean = schema.normalise_frame(frame)
    assert clean["state"].tolist() == ["Maharashtra", "Bihar"]
    assert clean[TARGET].tolist() == ["Yes", "No"]


def test_normalise_rejects_a_missing_feature_column():
    with pytest.raises(DataError, match="over_budget"):
        schema.normalise_frame(make_frame().drop(columns=[TARGET]))


def test_normalise_rejects_blank_and_non_numeric_budgets():
    with pytest.raises(DataError, match="sanctioned_budget_lakh"):
        schema.normalise_frame(make_frame(sanctioned_budget_lakh=["", 90.0]))
    with pytest.raises(DataError, match="sanctioned_budget_lakh"):
        schema.normalise_frame(make_frame(sanctioned_budget_lakh=["not a number", 90.0]))


def test_normalise_rejects_impossible_amounts_and_durations():
    with pytest.raises(DataError, match="positive"):
        schema.normalise_frame(make_frame(sanctioned_budget_lakh=[0.0, 90.0]))
    with pytest.raises(DataError, match="duration_months"):
        schema.normalise_frame(make_frame(duration_months=[300, 12]))


def test_derived_columns_are_arithmetic_not_assumptions():
    derived = schema.add_derived_columns(make_frame())
    assert derived.loc[0, "overrun_lakh"] == pytest.approx(30.0)
    assert derived.loc[0, "overrun_pct"] == pytest.approx(12.0)
    assert derived.loc[1, "overrun_lakh"] == pytest.approx(-6.0)
    assert derived.loc[1, "overrun_pct"] == pytest.approx(-6.67, abs=0.01)


def test_validate_dataset_reports_the_shape_the_ui_depends_on():
    report = schema.validate_dataset(make_frame())
    assert report["rows"] == 2
    assert report["target_counts"] == {"Yes": 1, "No": 1}
    assert report["over_budget_share_pct"] == 50.0
    assert report["scheme_counts"]["PMGSY (Rural Roads)"] == 1


def test_validate_dataset_flags_duplicate_project_names():
    report = schema.validate_dataset(make_frame(project_name=["Same", "Same"]))
    assert report["duplicate_project_names"] == 1


@pytest.mark.parametrize("lakh,expected", [
    (88.0, "₹ 88.0 lakh"),
    (99.5, "₹ 99.5 lakh"),
    (100.0, "₹ 1.00 crore"),            # 100 lakh is exactly 1 crore
    (250.0, "₹ 2.50 crore"),
    (22779.9, "₹ 227.80 crore"),
    (12_500_000.0, "₹ 125000 crore"),   # four-digit crores drop the decimals
    (-30.0, "-₹ 30.0 lakh"),
])
def test_format_inr_follows_the_lakh_crore_rule(lakh, expected):
    assert schema.format_inr(lakh) == expected


def test_format_inr_never_emits_a_dollar_sign():
    for value in (0, 1, 99.5, 100, 100000, -1):
        assert "$" not in schema.format_inr(value)


# ---------------------------------------------------------------------------
# Storage adapters
# ---------------------------------------------------------------------------


def test_local_store_loads_the_shipped_dataset(dataset_path):
    frame = storage.LocalStore().load(dataset_path)
    assert len(frame) == 700
    assert list(frame.columns) == CSV_COLUMNS


def test_local_store_refuses_to_read_outside_the_data_folder():
    """The dataset ref comes from an HTTP body, so it must not be able to name
    app.py, a .env file or anything outside data/."""
    store = storage.LocalStore()
    for ref in ("app.py", "config.py", "../app.py", "../../Windows/win.ini",
                os.path.join(config.PROJECT_ROOT, "app.py")):
        with pytest.raises(storage.StorageError):
            store.resolve(ref)


def test_local_store_resolves_paths_inside_data():
    resolved = storage.LocalStore().resolve("data/whatever.csv")
    assert resolved == os.path.normpath(os.path.join(config.DATA_DIR, "whatever.csv"))


def test_local_store_put_round_trips():
    target = os.path.join(config.DATA_DIR, "_test_round_trip.csv")
    try:
        storage.LocalStore().put(make_frame(), target)
        back = storage.LocalStore().load(target)
        assert len(back) == 2
        assert back.loc[0, "project_name"] == "Work A"
    finally:
        if os.path.exists(target):
            os.remove(target)
    assert not os.path.exists(target)


def test_local_store_missing_file_message_tells_you_what_to_run():
    with pytest.raises(storage.StorageError, match="generate_dataset"):
        storage.LocalStore().load("data/definitely_not_here.csv")


def test_local_store_iter_chunks_keeps_every_row():
    chunks = list(storage.LocalStore().iter_chunks(chunksize=120))
    assert sum(len(chunk) for chunk in chunks) == 700
    assert max(len(chunk) for chunk in chunks) <= 120
    assert len(chunks) == 6


def test_http_store_refuses_non_http_schemes():
    """A file:// or ftp:// 'data URL' would turn the adapter into a local file reader
    with a network-looking name."""
    with pytest.raises(storage.StorageError, match="http"):
        storage.HttpStore(url="file:///etc/passwd").load()


def test_http_store_is_read_only():
    with pytest.raises(storage.StorageError, match="read-only"):
        storage.HttpStore(url="https://example.invalid/x.csv").put(make_frame(), "x")


def test_http_store_without_a_url_explains_how_to_configure_it():
    store = storage.HttpStore(url="")
    assert store.describe()["configured"] is False
    assert "HTTP_DATASET_URL" in store.missing_reason()
    with pytest.raises(storage.StorageError, match="HTTP_DATASET_URL"):
        store.load()


def test_s3_store_refuses_to_build_without_configuration():
    """No bucket, no credentials, no connection attempt — and no crash at import."""
    with pytest.raises(storage.StorageError, match="S3_BUCKET"):
        storage.S3Store(bucket="")


def test_s3_store_refuses_without_credentials(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(storage.StorageError, match="credentials"):
        storage.S3Store(bucket="some-bucket")


def test_adapter_status_describes_all_three_backends_without_raising():
    report = storage.adapter_status()
    assert [entry["name"] for entry in report] == ["local", "http", "s3"]
    local = report[0]
    assert local["configured"] is True and local["active"] is True
    assert report[1]["configured"] is False and "HTTP_DATASET_URL" in report[1]["reason"]
    assert report[2]["configured"] is False and report[2]["reason"]


def test_build_adapter_rejects_unknown_names():
    with pytest.raises(storage.StorageError):
        storage.build_adapter("carrier-pigeon")


# ---------------------------------------------------------------------------
# Chunked pipeline
# ---------------------------------------------------------------------------


def test_sanitise_partition_value_is_filesystem_safe():
    assert pipeline.sanitise_partition_value("PMGSY (Rural Roads)") == "PMGSY_Rural_Roads"
    assert pipeline.sanitise_partition_value("Jal Jeevan Mission") == "Jal_Jeevan_Mission"
    assert "/" not in pipeline.sanitise_partition_value("a/b")
    assert pipeline.sanitise_partition_value("   ") == "unknown"


def test_map_partitions_applies_the_transform_to_every_chunk():
    chunks = [make_frame().iloc[:1], make_frame().iloc[1:]]
    out = pipeline.map_partitions(pipeline.enrich_partition, chunks)
    assert len(out) == 2
    assert "overrun_pct" in out.columns


def test_transform_source_matches_a_plain_read():
    streamed = pipeline.transform_source(chunksize=97)
    direct = pipeline.enrich_partition(pd.read_csv(config.LOCAL_DATASET))
    assert len(streamed) == len(direct) == 700
    assert streamed["overrun_pct"].sum() == pytest.approx(direct["overrun_pct"].sum())


def test_build_feature_store_writes_partitions_and_a_matching_manifest(tmp_path):
    manifest = pipeline.build_feature_store(out_dir=str(tmp_path), chunksize=200)
    assert manifest["row_count"] == 700
    assert manifest["partition_column"] == "scheme_name"
    assert manifest["engine"] == pipeline.ENGINE
    assert manifest["chunk_size"] == 200
    assert manifest["source_sha256"], "the manifest must fingerprint its input"

    directories = sorted(name for name in os.listdir(str(tmp_path))
                         if name.startswith("scheme_name="))
    assert len(directories) == 4, "one directory per central scheme"
    assert directories == sorted("scheme_name=" + key.split("=", 1)[1]
                                 for key in manifest["partitions"])

    with open(os.path.join(str(tmp_path), pipeline.MANIFEST_NAME), encoding="utf-8") as handle:
        on_disk = json.load(handle)
    assert on_disk["row_count"] == 700
    assert on_disk["file_count"] == manifest["file_count"]
    assert sum(entry["rows"] for entry in on_disk["partitions"].values()) == 700


def test_feature_store_round_trips_through_load(tmp_path):
    pipeline.build_feature_store(out_dir=str(tmp_path), chunksize=250)
    loaded = pipeline.load_feature_store(out_dir=str(tmp_path))
    assert len(loaded) == 700
    # the target and the derived columns must survive the trip through CSV
    assert set(loaded[TARGET]) == {"Yes", "No"}
    assert loaded["overrun_pct"].notna().all()
    assert loaded["overrun_pct"].sum() == pytest.approx(
        pipeline.transform_source()["overrun_pct"].sum(), abs=0.5)


def test_shipped_feature_store_is_partitioned_by_scheme():
    """The one that ships in data/feature_store must be readable and complete."""
    manifest_path = os.path.join(config.FEATURE_STORE_DIR, pipeline.MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        pipeline.build_feature_store()
    loaded = pipeline.load_feature_store()
    assert len(loaded) == 700
    assert set(loaded["scheme_name"]) == {"PMGSY (Rural Roads)", "PMAY (Housing)",
                                          "Jal Jeevan Mission", "Smart Cities Mission"}
