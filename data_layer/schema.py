"""
Single source of truth for the dataset contract.

The generator, the training script, the Flask API and the tests all import this, so a
column rename cannot silently leave one of them reading a stale name.
"""

import hashlib
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Column contract (see SPEC_INDIA.md section 3.1)
# ---------------------------------------------------------------------------

DISPLAY_ONLY = ["project_name", "project_type"]
CATEGORICAL_FEATURES = ["scheme_name", "state", "district", "contractor_type"]
NUMERIC_FEATURES = ["sanctioned_budget_lakh", "duration_months"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET = "over_budget"
TARGET_VALUES = ("Yes", "No")
POSITIVE_CLASS = "Yes"

# Derived by the pipeline, never by the generator.
DERIVED_COLUMNS = ["overrun_lakh", "overrun_pct"]

ALL_COLUMNS = DISPLAY_ONLY + [TARGET] + FEATURES + ["actual_cost_lakh"]
# Order written to disk (matches model.generate_dataset.ALL_COLUMNS).
CSV_COLUMNS = ["project_name", "scheme_name", "project_type", "state", "district",
               "contractor_type", "sanctioned_budget_lakh", "duration_months",
               "actual_cost_lakh", TARGET]

CONTRACTOR_TYPES = ["Govt", "Private"]


class DataError(ValueError):
    """Raised when a frame does not satisfy the contract above."""


def sha256_of_file(path, block=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_frame(df, copy=True):
    """Coerce dtypes and trim whitespace so a hand-edited or uploaded CSV behaves
    the same as a generated one. Does not invent missing values."""
    frame = df.copy() if copy else df
    frame.columns = [str(c).strip() for c in frame.columns]

    missing = [c for c in FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        raise DataError("dataset is missing required column(s): %s" % ", ".join(missing))

    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype(str).str.strip()

    frame[TARGET] = (frame[TARGET].astype(str).str.strip().str.title()
                     .replace({"Y": "Yes", "N": "No"}))
    bad_labels = sorted(set(frame[TARGET]) - set(TARGET_VALUES))
    if bad_labels:
        raise DataError("%s must be Yes or No, found: %s" % (TARGET, ", ".join(bad_labels)))

    for col in NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "actual_cost_lakh" not in frame.columns:
        frame["actual_cost_lakh"] = pd.NA
    else:
        frame["actual_cost_lakh"] = pd.to_numeric(frame["actual_cost_lakh"], errors="coerce")

    nulls = [c for c in FEATURES if frame[c].isna().any()]
    if nulls:
        raise DataError("these feature columns contain blank or non-numeric values: %s"
                        % ", ".join(nulls))
    if (frame["sanctioned_budget_lakh"] <= 0).any():
        raise DataError("sanctioned_budget_lakh must be positive (values are in rupee lakh)")
    if (frame["duration_months"] <= 0).any() or (frame["duration_months"] > 240).any():
        raise DataError("duration_months must be between 1 and 240")

    return frame


def add_derived_columns(frame):
    """Work the pipeline layer owns: how far the actual cost slipped, in rupee lakh
    and as a percentage of sanction."""
    out = frame.copy()
    if "actual_cost_lakh" in out.columns:
        actual = pd.to_numeric(out["actual_cost_lakh"], errors="coerce")
        out["overrun_lakh"] = (actual - out["sanctioned_budget_lakh"]).round(1)
        out["overrun_pct"] = (out["overrun_lakh"] / out["sanctioned_budget_lakh"] * 100).round(2)
    else:
        out["overrun_lakh"] = pd.NA
        out["overrun_pct"] = pd.NA
    return out


def validate_dataset(df):
    """Full structural check. Returns a report dict; raises DataError on anything a
    model or chart would silently mishandle."""
    frame = normalise_frame(df)
    report = {"rows": int(len(frame)), "columns": list(frame.columns)}

    if frame[FEATURES + [TARGET]].isna().any().any():
        raise DataError("no feature or target cell may be empty")

    duplicate_names = frame["project_name"].duplicated() if "project_name" in frame else pd.Series(False, index=frame.index)
    report["duplicate_project_names"] = int(duplicate_names.sum())

    counts = frame[TARGET].value_counts().to_dict()
    report["target_counts"] = {str(k): int(v) for k, v in counts.items()}
    yes = int(counts.get("Yes", 0))
    report["over_budget_share_pct"] = round(100.0 * yes / len(frame), 2) if len(frame) else 0.0
    report["scheme_counts"] = {str(k): int(v) for k, v in frame["scheme_name"].value_counts().items()}
    report["state_counts"] = {str(k): int(v) for k, v in frame["state"].value_counts().items()}
    report["contractor_counts"] = {str(k): int(v) for k, v in frame["contractor_type"].value_counts().items()}
    return report


def format_inr(lakh_value):
    """Format a rupee amount held in lakh: below 1 crore show lakh, at or above show
    crore (1 crore = 100 lakh). Kept server-side so CSV/PDF exports and tests agree
    with the browser's copy in static/js/app.js."""
    value = float(lakh_value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value < 100:
        return "%s\u20B9 %.1f lakh" % (sign, value)
    if value < 100000:
        return "%s\u20B9 %.2f crore" % (sign, value / 100.0)
    return "%s\u20B9 %.0f crore" % (sign, value / 100.0)


def default_dataset_path(project_root=None):
    root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "data", "india_projects_2025_26.csv")
