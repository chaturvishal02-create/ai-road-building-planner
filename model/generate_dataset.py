"""
Generate a simulated Indian government construction project dataset (FY 2025-26).

WHY SIMULATED: real project-level sanctioned-vs-actual cost pairs for 2025-26 are not
published anywhere as one clean downloadable file -- they are scattered across CAG audit
reports and dozens of state e-procurement portals. So this script builds a dataset that
follows the *structure* and the *documented patterns* of Indian scheme data (which
schemes exist, how big their projects are, who executes them, what overruns look like)
without pretending to be the real record. Every screen that shows these numbers labels
them simulated.

HOW A ROW IS BUILT
  1. scheme quota   -- fixed row counts per scheme (SCHEME_COUNTS)
  2. geography      -- state chosen by weight, district taken from that state's list,
                       so no row can have a district outside its state
  3. money and time -- budget drawn from a log-normal per scheme (many small works,
                       a few very large ones), duration from a per-scheme range
  4. contractor     -- Govt / Private probability per scheme
  5. target         -- a latent risk score (scheme + contractor + size + duration + noise)
                       is ranked WITHIN each scheme and the top N rows are marked Yes,
                       where N comes from OVER_BUDGET_BY_SCHEME. This gives the model a
                       real signal to learn and lands the overall overrun rate exactly.
  6. actual cost    -- Yes rows get a positive overrun %, No rows an underrun %, both
                       drawn around the scheme's typical magnitude.

Run:  py model/generate_dataset.py
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_layer import schema  # noqa: E402  the column contract lives there

# ---------------------------------------------------------------------------
# Column contract -- imported rather than re-declared, so the generator, the
# trainer, the API and the tests cannot drift apart.
#
# Columns the model sees. `project_type` and `project_name` are display-only:
# project_type is 1-to-1 with scheme_name, so feeding both would count the same
# information twice and distort distance calculations.
# ---------------------------------------------------------------------------
FEATURES = schema.FEATURES
NUMERIC_FEATURES = schema.NUMERIC_FEATURES
CATEGORICAL_FEATURES = schema.CATEGORICAL_FEATURES
TARGET = schema.TARGET
CONTRACTOR_TYPES = schema.CONTRACTOR_TYPES

ALL_COLUMNS = schema.CSV_COLUMNS

ROW_COUNT = 700
SEED = 42

# ---------------------------------------------------------------------------
# Geography: district lists are authoritative, not decoration
# ---------------------------------------------------------------------------

STATE_DISTRICTS = {
    "Maharashtra": [
        "Pune", "Mumbai City", "Nashik", "Nagpur", "Thane",
        "Chhatrapati Sambhajinagar", "Amravati",
    ],
    "Karnataka": [
        "Bengaluru Urban", "Mysuru", "Belagavi", "Dakshina Kannada", "Kalaburagi",
    ],
    "Delhi": ["New Delhi", "Central Delhi", "East Delhi", "North-East Delhi"],
    "Uttar Pradesh": [
        "Lucknow", "Kanpur Nagar", "Gautam Buddh Nagar", "Agra",
        "Prayagraj", "Varanasi", "Ghaziabad",
    ],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Jabalpur", "Gwalior"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Tirupati"],
    "West Bengal": ["Kolkata", "Howrah", "Darjeeling", "Paschim Medinipur"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar"],
    "Kerala": ["Ernakulam", "Thiruvananthapuram", "Kozhikode"],
}

# Roughly proportional to each state's share of central-scheme project volume.
STATE_WEIGHTS = {
    "Maharashtra": 0.16, "Uttar Pradesh": 0.15, "Karnataka": 0.11,
    "Madhya Pradesh": 0.09, "Gujarat": 0.08, "Rajasthan": 0.08,
    "Tamil Nadu": 0.07, "Bihar": 0.07, "Telangana": 0.06,
    "Andhra Pradesh": 0.05, "West Bengal": 0.05, "Kerala": 0.04,
    "Punjab": 0.035, "Delhi": 0.03,
}

# Short codes used to build official-looking work numbers, e.g. MH-PMGSY-2526-014.
# Scheme registers in India carry a work number per project, and it makes every row in
# our table uniquely identifiable -- bare names collide (48 of them did).
STATE_CODE = {
    "Maharashtra": "MH", "Uttar Pradesh": "UP", "Karnataka": "KA",
    "Madhya Pradesh": "MP", "Gujarat": "GJ", "Rajasthan": "RJ",
    "Tamil Nadu": "TN", "Bihar": "BR", "Telangana": "TS",
    "Andhra Pradesh": "AP", "West Bengal": "WB", "Kerala": "KL",
    "Punjab": "PB", "Delhi": "DL",
}

# Execution-capacity offset per state, in risk-score units. Deliberately coarse (14
# categories, ~50 rows each) so a classifier can actually learn it, unlike district,
# which is too fine-grained to carry signal at this sample size.
STATE_RISK = {
    "Bihar": 0.12, "Uttar Pradesh": 0.10, "West Bengal": 0.05, "Madhya Pradesh": 0.06,
    "Rajasthan": 0.05, "Andhra Pradesh": 0.03, "Maharashtra": 0.02, "Punjab": 0.02,
    "Karnataka": 0.01, "Telangana": 0.0, "Gujarat": 0.0, "Tamil Nadu": -0.01,
    "Delhi": -0.02, "Kerala": -0.03,
}
SCHEME_CODE = {
    "PMGSY (Rural Roads)": "PMGSY", "PMAY (Housing)": "PMAY",
    "Jal Jeevan Mission": "JJM", "Smart Cities Mission": "SC",
}
FISCAL_YEAR_CODE = "2526"

# ---------------------------------------------------------------------------
# Schemes
# ---------------------------------------------------------------------------

SCHEME_COUNTS = {
    "PMGSY (Rural Roads)": 250,
    "PMAY (Housing)": 210,
    "Jal Jeevan Mission": 145,
    "Smart Cities Mission": 95,
}

SCHEME_PROJECT_TYPE = {
    "PMGSY (Rural Roads)": "Road Construction",
    "PMAY (Housing)": "Housing",
    "Jal Jeevan Mission": "Water Supply",
    "Smart Cities Mission": "Smart City Infrastructure",
}

# How many projects of each scheme finish over sanction. Sums to 294 = 42.0 % of 700.
OVER_BUDGET_BY_SCHEME = {
    "PMGSY (Rural Roads)": 110,
    "PMAY (Housing)": 84,
    "Jal Jeevan Mission": 66,
    "Smart Cities Mission": 34,
}

# Latent-risk inputs. BASE_RISK orders the schemes by how often their works slip;
# OVERRUN_MEAN_PCT is the typical size of the slip once it happens.
SCHEME_PROFILE = {
    "PMGSY (Rural Roads)": {
        "base_risk": 0.18, "overrun_mean_pct": 14.0, "overrun_sd_pct": 7.0,
        "budget_meanlog": 5.2, "budget_sigma": 0.75, "budget_min": 30.0, "budget_max": 2000.0,
        "duration_low": 6, "duration_high": 36, "govt_share": 0.55,
    },
    "PMAY (Housing)": {
        "base_risk": 0.12, "overrun_mean_pct": 11.0, "overrun_sd_pct": 6.0,
        "budget_meanlog": 6.9, "budget_sigma": 0.80, "budget_min": 100.0, "budget_max": 9000.0,
        "duration_low": 12, "duration_high": 48, "govt_share": 0.45,
    },
    "Jal Jeevan Mission": {
        "base_risk": 0.22, "overrun_mean_pct": 18.0, "overrun_sd_pct": 8.0,
        "budget_meanlog": 6.2, "budget_sigma": 0.85, "budget_min": 50.0, "budget_max": 6000.0,
        "duration_low": 12, "duration_high": 60, "govt_share": 0.70,
    },
    "Smart Cities Mission": {
        "base_risk": 0.05, "overrun_mean_pct": 9.0, "overrun_sd_pct": 5.0,
        "budget_meanlog": 8.3, "budget_sigma": 0.95, "budget_min": 500.0, "budget_max": 90000.0,
        "duration_low": 18, "duration_high": 72, "govt_share": 0.30,
    },
}

SMART_CITY_COMPONENTS = [
    "Integrated Command and Control Centre", "Smart Parking System",
    "Riverfront Development", "LED Street Lighting", "Underground Drainage",
    "Waste Management Facility", "Cycle Track and Footpath Renewal",
    "Solar Microgrid", "E-Governance Citizen Centre",
]

# ---------------------------------------------------------------------------
# Project names
# ---------------------------------------------------------------------------


def _pmgsy_name(rng, district, state, other_district):
    works = ["Link Road", "Major District Road", "Rural Road Upgradation", "Bridge Causeway"]
    phase = rng.choice(["", ", Phase II", ", Balance Work", ", Package III"], p=[0.55, 0.15, 0.15, 0.15])
    return "{} \u2013 {} {}, {}".format(district, other_district, rng.choice(works), state)


def _pmay_name(rng, district, state):
    colonies = ["Sector", "Ward", "Nagar Palika", "Township"]
    units = int(rng.integers(48, 1200))
    return "{} {} {} PMAY-Urban Housing, {} ({:,} units)".format(
        district, rng.choice(colonies), int(rng.integers(1, 25)), state, units)


def _jjm_name(rng, district, state):
    panchayats = int(rng.integers(6, 140))
    return "{} Gram Panchayat Jal Jeevan Water Supply Scheme ({} GPs), {}".format(
        district, panchayats, state)


def _smart_city_name(rng, district, state):
    return "{} Smart City \u2013 {}".format(district, rng.choice(SMART_CITY_COMPONENTS))


def make_name(rng, scheme, district, state, other_district):
    if scheme == "PMGSY (Rural Roads)":
        return _pmgsy_name(rng, district, state, other_district)
    if scheme == "PMAY (Housing)":
        return _pmay_name(rng, district, state)
    if scheme == "Jal Jeevan Mission":
        return _jjm_name(rng, district, state)
    return _smart_city_name(rng, district, state)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _state_names_and_weights():
    assert set(STATE_WEIGHTS) == set(STATE_DISTRICTS), "state lists disagree"
    names = sorted(STATE_WEIGHTS)
    weights = np.array([STATE_WEIGHTS[s] for s in names], dtype=float)
    return names, weights / weights.sum()


def generate(row_count=ROW_COUNT, seed=SEED):
    """Build the simulated dataset and return it as a DataFrame."""
    if row_count != ROW_COUNT:
        # Quotas are hand-set for 700 rows; scale them proportionally otherwise.
        raise ValueError("SCHEME_COUNTS and OVER_BUDGET_BY_SCHEME are fixed for %d rows" % ROW_COUNT)

    rng = np.random.default_rng(seed)
    states, state_w = _state_names_and_weights()
    schemes = list(SCHEME_COUNTS)

    rows = []
    for scheme in schemes:
        profile = SCHEME_PROFILE[scheme]
        n = SCHEME_COUNTS[scheme]

        chosen_states = rng.choice(states, size=n, p=state_w)
        for state in chosen_states:
            pool = STATE_DISTRICTS[state]
            district = str(rng.choice(pool))
            other = str(rng.choice([d for d in pool if d != district] or pool))
            budget = float(np.clip(np.exp(rng.normal(profile["budget_meanlog"],
                                                     profile["budget_sigma"])),
                                   profile["budget_min"], profile["budget_max"]))
            duration = int(rng.integers(profile["duration_low"], profile["duration_high"] + 1))
            contractor = "Govt" if rng.random() < profile["govt_share"] else "Private"
            rows.append({
                "project_name": make_name(rng, scheme, district, state, other),
                "scheme_name": scheme,
                "project_type": SCHEME_PROJECT_TYPE[scheme],
                "state": state,
                "district": district,
                "contractor_type": contractor,
                "sanctioned_budget_lakh": round(budget, 1),
                "duration_months": duration,
                "_other_district": other,
                "_index": len(rows),
            })
    df = pd.DataFrame(rows)

    # --- latent risk score: the pattern we want the model to discover ----------
    # Every term below is a real modelling assumption about Indian scheme works:
    #   * some schemes (pipinated water, rural roads) slip more than others
    #   * department-executed work carries land and utility-clearance delay
    #   * small works overrun by a larger PERCENTAGE (mobilisation cost over a small base)
    #   * a longer scheduled duration means more change orders
    # RISK_NOISE is the part no feature explains. It is kept small on purpose: with the
    # original 0.10 the signal drowned and KNN scored below the always-No baseline, which
    # would have made the demo meaningless. See SPEC_INDIA.md section 11, assumption 2.
    RISK_NOISE = 0.055
    risk = (
        df["scheme_name"].map({s: SCHEME_PROFILE[s]["base_risk"] for s in schemes})
        + df["state"].map(STATE_RISK)
        + (df["contractor_type"] == "Govt").map({True: 0.12, False: 0.0})
        - 0.07 * np.log10(df["sanctioned_budget_lakh"] / 1000.0)
        + 0.0025 * (df["duration_months"] - 24.0)
        + rng.normal(0.0, RISK_NOISE, size=len(df))
    )
    df["_risk"] = risk

    # Rank within scheme, so each scheme hits its designed overrun count exactly.
    df[TARGET] = "No"
    for scheme, yes_count in OVER_BUDGET_BY_SCHEME.items():
        group = df.index[df["scheme_name"] == scheme]
        ranked = df.loc[group].sort_values("_risk", ascending=False).index.tolist()
        df.loc[ranked[:yes_count], TARGET] = "Yes"

    # --- actual cost, consistent with the label -------------------------------
    overrun_pct = np.zeros(len(df))
    for scheme in schemes:
        profile = SCHEME_PROFILE[scheme]
        for label, is_yes in (("Yes", True), ("No", False)):
            idx = df.index[(df["scheme_name"] == scheme) & (df[TARGET] == label)]
            if len(idx) == 0:
                continue
            if is_yes:
                values = rng.normal(profile["overrun_mean_pct"], profile["overrun_sd_pct"], size=len(idx))
                values = np.clip(values, 0.8, 65.0)
            else:
                # Finished below sanction: between a hair under and 12 % under.
                values = -np.clip(rng.normal(4.0, 3.0, size=len(idx)), 0.1, 12.0)
            overrun_pct[df.index.get_indexer(idx)] = values

    budget = df["sanctioned_budget_lakh"].to_numpy(dtype=float)
    actual = budget * (1.0 + overrun_pct / 100.0)
    df["actual_cost_lakh"] = np.round(actual, 1)

    # The label must agree with the *rounded* stored numbers, not the pre-round ones.
    df[TARGET] = np.where(df["actual_cost_lakh"] > df["sanctioned_budget_lakh"], "Yes", "No")

    seq = df.groupby(["state", "scheme_name"], sort=True).cumcount() + 1
    work_code = (df["state"].map(STATE_CODE) + "-" + df["scheme_name"].map(SCHEME_CODE)
                 + "-" + FISCAL_YEAR_CODE + "-" + seq.map("{:03d}".format))
    df["project_name"] = df["project_name"] + " [Work " + work_code + "]"

    df = df.drop(columns=["_risk", "_other_district", "_index"])
    df = df[ALL_COLUMNS].reset_index(drop=True)
    return df


def self_check(df):
    """Fail loudly if the dataset broke its own contract."""
    assert len(df) == ROW_COUNT, "expected %d rows, got %d" % (ROW_COUNT, len(df))
    assert list(df.columns) == ALL_COLUMNS, list(df.columns)
    assert df.notna().all().all(), "dataset has empty cells"
    assert df["project_name"].is_unique, "project names must identify exactly one row"

    for _, row in df.iterrows():
        assert row["district"] in STATE_DISTRICTS[row["state"]], (
            "district %r is not in state %r" % (row["district"], row["state"]))
        expected = "Yes" if row["actual_cost_lakh"] > row["sanctioned_budget_lakh"] else "No"
        assert row[TARGET] == expected, "label disagrees with costs on %r" % row["project_name"]

    yes = int((df[TARGET] == "Yes").sum())
    share = 100.0 * yes / len(df)
    assert 39.0 <= share <= 45.0, "overrun share %.2f%% outside the 42%% +/-3pp band" % share

    for scheme, want in SCHEME_COUNTS.items():
        got = int((df["scheme_name"] == scheme).sum())
        assert got == want, "%s: wanted %d rows, got %d" % (scheme, want, got)
    return {"rows": len(df), "yes": yes, "no": len(df) - yes, "yes_share_pct": round(share, 2)}


def write_outputs(directory=None):
    directory = directory or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(directory, "data")
    os.makedirs(data_dir, exist_ok=True)

    df = generate()
    stats = self_check(df)

    main_path = os.path.join(data_dir, "india_projects_2025_26.csv")
    df.to_csv(main_path, index=False, encoding="utf-8")

    # 20-row sample the spec asks for: readable, hand-pickable, used by tests.
    # Take the largest overrun in each scheme plus a spread of underruns so the
    # sample shows both classes and every scheme.
    parts = []
    for scheme in SCHEME_COUNTS:
        subset = df[df["scheme_name"] == scheme].copy()
        subset["slip"] = subset["actual_cost_lakh"] - subset["sanctioned_budget_lakh"]
        yes_rows = subset[subset[TARGET] == "Yes"].sort_values("slip", ascending=False).head(3)
        no_rows = subset[subset[TARGET] == "No"].sort_values("slip").head(2)
        parts.append(pd.concat([yes_rows, no_rows]).drop(columns=["slip"]))
    sample = pd.concat(parts, ignore_index=True)
    sample_path = os.path.join(data_dir, "sample_20_projects.csv")
    sample.to_csv(sample_path, index=False, encoding="utf-8")

    return {"main": main_path, "sample": sample_path, "stats": stats,
            "sample_rows": len(sample), "frame": df, "sample_frame": sample}


if __name__ == "__main__":
    result = write_outputs()
    stats = result["stats"]
    print("Wrote %s" % result["main"])
    print("Wrote %s (%d rows)" % (result["sample"], result["sample_rows"]))
    print("rows=%d  over_budget Yes=%d (%.2f%%)  No=%d" % (
        stats["rows"], stats["yes"], stats["yes_share_pct"], stats["no"]))
    print("states=%d  districts=%d  schemes=%d" % (
        len(STATE_DISTRICTS), sum(len(v) for v in STATE_DISTRICTS.values()), len(SCHEME_COUNTS)))
