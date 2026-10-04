"""Government building cost: a second estimator, separate from the road engine.

    Total = Built-up area x Cost per sq ft x Floor factor x Location factor
            x Quality factor x Building type factor

Nothing here is learned from the 700 project rows -- the classifier was trained on
sanctioned *scheme* projects, not on panchayat buildings, so a per-square-foot rate with
its factors printed is the honest answer at this scope. The road corridor engine in
`route_engine.py` and the planning formula in `planning.py` are untouched by this module;
they share only the location hierarchy and the rupee formatting conventions.
"""

from __future__ import annotations

from model.route_engine import DAYS_PER_MONTH, RouteError, format_inr

# ₹ per square foot of built-up area, by where the plot is. The brief gives rural and
# urban; "remote" is our documented middle case -- it is not just rural, it carries
# mobilisation and haulage that a village on a motorable road does not.
BASE_COST_PER_SQFT = {"rural": 1500.0, "urban": 2500.0, "remote": 1800.0}

LOCATION_FACTORS = {"rural": 1.00, "urban": 1.00, "remote": 1.10}

# `built_up_area_sqft` is the ground footprint. Each additional storeys costs slightly
# less than a full repeat of it, because the foundation, the staircase and the plot
# works are already paid for -- hence 1.90 rather than 2.00 for two floors.
FLOOR_FACTORS = {1: 1.00, 2: 1.90, 3: 2.65}

QUALITY_FACTORS = {"low": 0.85, "medium": 1.00, "high": 1.25}
QUALITY_LABELS = {"low": "Basic / cost-effective", "medium": "Standard PWD",
                  "high": "Premium / institutional"}

# What the building is for changes the fit-out, and a hospital is the clearest case in
# the three types this module knows.
BUILDING_TYPES = {
    "gram_panchayat": {"label": "Gram Panchayat office", "factor": 0.90,
                       "note": "Public hall, offices, records room; no specialist "
                               "services."},
    "school": {"label": "School building", "factor": 1.00,
               "note": "Classrooms, verandahs, toilets, staff room; single-storey "
                       "span-driven structure."},
    "hospital": {"label": "Hospital / urban health centre", "factor": 1.35,
                 "note": "Wing loading, medical gas, power backup, HVAC and OT-grade "
                         "finishes are what makes a health centre cost more per "
                         "square foot than a classroom block."},
}

# Square feet of built-up area one site crew raises per day, structure and finishes
# together. Small numbers on purpose: a 4,000 sq ft school block is a 10-14 month site,
# and a rate an order of magnitude higher would print a completion time no engineer
# would sign for.
CONSTRUCTION_SQFT_PER_DAY = {"low": 55.0, "medium": 45.0, "high": 35.0}
# Foundation, plinth and the finishing tail do not scale with area: a 600 sq ft
# panchayat office still needs them. Charged as fixed days rather than pretending the
# rate covers them.
MOBILISATION_DAYS = 120.0
MIN_MOBILISATION_MONTHS = 3.0

# Work-section split of a government building cost. Percentages of the total, and they
# add to 100 -- asserted by a test, because a pie chart that disagrees with the card is
# the mistake this project has already been caught making once.
WORK_SPLIT_PCT = {"structure": 42.0, "masonry": 12.0, "finishes": 15.0,
                  "electrical_plumbing": 11.0, "labour": 12.0,
                  "fixtures_equipment": 6.0, "site_development": 2.0}
WORK_LABELS = {"structure": "Structure (cement, steel, aggregate)",
               "masonry": "Masonry and brickwork",
               "finishes": "Finishes, flooring, painting",
               "electrical_plumbing": "Electrical and plumbing",
               "labour": "Contract labour",
               "fixtures_equipment": "Fixtures and equipment",
               "site_development": "Site development and mobilisation"}

MIN_AREA, MAX_AREA = 100.0, 500000.0
MIN_FLOORS, MAX_FLOORS = 1, 3


def _factors(location_type, floors, quality):
    return (LOCATION_FACTORS[location_type], FLOOR_FACTORS[floors],
            QUALITY_FACTORS[quality])


def construction_time(built_sqft, quality, floors):
    """Months from the area actually raised, plus the fixed site time around it."""
    rate = CONSTRUCTION_SQFT_PER_DAY[quality]
    days = built_sqft / rate + MOBILISATION_DAYS
    sequence = 1.0 + 0.08 * (floors - 1)
    months = days / DAYS_PER_MONTH * sequence
    storeys = "storey" if floors == 1 else "storeys"
    return {"sqft_per_day": rate,
            "built_up_sqft": round(built_sqft, 1),
            "working_days": round(days, 1),
            "mobilisation_days": MOBILISATION_DAYS,
            "floor_sequence_factor": round(sequence, 2),
            "months": round(max(MIN_MOBILISATION_MONTHS, months), 1),
            "time_months": max(1, int(round(max(MIN_MOBILISATION_MONTHS, months)))),
            "basis": ("%.0f sq ft ÷ %.0f sq ft per day + %.0f days of site set-up = "
                      "%.0f days ÷ %.2f days per month = %.1f months, x %.2f for %d "
                      "%s to cure in sequence"
                      % (built_sqft, rate, MOBILISATION_DAYS, days, DAYS_PER_MONTH,
                         months, sequence, floors, storeys))}


def estimate(state=None, district=None, taluka=None, village=None,
             building_type="gram_panchayat", area_sqft=1200, floors=1,
             quality="medium", location_type="rural"):
    """Cost one government building. Every factor used is returned, and so is the
    sentence that multiplies them."""
    errors = {}
    if building_type not in BUILDING_TYPES:
        errors["building_type"] = "'%s' is not a known building type." % building_type
    if location_type not in BASE_COST_PER_SQFT:
        errors["location_type"] = "Location must be rural, urban or remote."
    try:
        floors = int(floors)
    except (TypeError, ValueError):
        floors = 0
        errors["floors"] = "Floors must be a whole number."
    if not errors and not (MIN_FLOORS <= floors <= MAX_FLOORS):
        errors["floors"] = "Floors must be 1, 2 or 3."
    if quality not in QUALITY_FACTORS:
        errors["quality"] = "Quality must be low, medium or high."
    try:
        area = float(area_sqft)
    except (TypeError, ValueError):
        area = 0.0
        errors["area_sqft"] = "Built-up area must be a number of square feet."
    if not errors and not (MIN_AREA <= area <= MAX_AREA):
        errors["area_sqft"] = ("Built-up area must be between %.0f and %.0f sq ft."
                               % (MIN_AREA, MAX_AREA))
    if errors:
        raise RouteError("The building request is not valid.", errors)

    base = BASE_COST_PER_SQFT[location_type]
    type_factor = BUILDING_TYPES[building_type]["factor"]
    location_factor, floor_factor, quality_factor = _factors(location_type, floors,
                                                             quality)
    cost_rupees = area * base * floor_factor * location_factor * quality_factor \
        * type_factor
    built_sqft = area * floor_factor
    time = construction_time(built_sqft, quality, floors)

    # Split the *rounded* total, so the seven lines can only ever add back to the one
    # number the card prints; any rounding residual goes into the largest line.
    total_rupees = int(round(cost_rupees))
    total_lakh = round(cost_rupees / 100000.0, 2)
    works = {}
    for key, pct in WORK_SPLIT_PCT.items():
        works[key] = {"label": WORK_LABELS[key], "percent": pct,
                      "lakh": round(total_lakh * pct / 100.0, 2),
                      "rupees": int(round(total_rupees * pct / 100.0))}
    lakh_residual = round(total_lakh - sum(row["lakh"] for row in works.values()), 2)
    rupee_residual = total_rupees - sum(row["rupees"] for row in works.values())
    works["structure"]["lakh"] = round(works["structure"]["lakh"] + lakh_residual, 2)
    works["structure"]["rupees"] += rupee_residual

    return {
        "location": {"state": state, "district": district, "taluka": taluka,
                     "village": village},
        "building_type": building_type,
        "building_type_label": BUILDING_TYPES[building_type]["label"],
        "building_type_note": BUILDING_TYPES[building_type]["note"],
        "building_type_factor": type_factor,
        "area_sqft": round(area, 1),
        "floors": floors,
        "quality": quality,
        "quality_label": QUALITY_LABELS[quality],
        "location_type": location_type,
        "base_cost_per_sqft": base,
        "floor_factor": floor_factor,
        "location_factor": location_factor,
        "quality_factor": quality_factor,
        "effective_cost_per_sqft": round(base * floor_factor * location_factor
                                         * quality_factor * type_factor, 2),
        "built_up_sqft": round(built_sqft, 1),
        "cost_rupees": total_rupees,
        # Two decimals, so the seven breakdown lines add up to exactly the figure the
        # card above them prints.
        "cost_lakh": total_lakh,
        "cost_crore": round(total_lakh / 100.0, 2),
        "cost_formatted": format_inr(total_lakh),
        "breakdown": works,
        "work_split_pct": dict(WORK_SPLIT_PCT),
        "time_months": time["time_months"],
        "construction_time": time,
        "brief": {"cost": total_rupees,
                  "time_months": time["time_months"],
                  "breakdown": {key: row["rupees"] for key, row in works.items()}},
        "basis": ("%.0f sq ft x ₹ %.0f per sq ft (%s) x %.2f floors x %.2f location "
                  "x %.2f quality x %.2f building type = %s"
                  % (area, base, location_type, floor_factor, location_factor,
                     quality_factor, type_factor, format_inr(total_lakh))),
        "note": ("Cost-per-square-foot estimate for a public building, from published "
                 "FY 2025-26 assumptions. It is not a trained-model output, and it "
                 "prices one standard structural frame: no land, no boundary wall, no "
                 "furniture, no specialist medical equipment."),
    }


def catalogue():
    """The tables the form and the on-page explanation both read from one source."""
    return {
        "building_types": [{"key": key, "label": spec["label"],
                            "factor": spec["factor"], "note": spec["note"]}
                           for key, spec in BUILDING_TYPES.items()],
        "location_types": [{"key": key, "base_cost_per_sqft": value,
                            "factor": LOCATION_FACTORS[key]}
                           for key, value in BASE_COST_PER_SQFT.items()],
        "floor_factors": [{"floors": key, "factor": value}
                          for key, value in sorted(FLOOR_FACTORS.items())],
        "qualities": [{"key": key, "label": QUALITY_LABELS[key], "factor": value,
                       "sqft_per_day": CONSTRUCTION_SQFT_PER_DAY[key]}
                      for key, value in sorted(QUALITY_FACTORS.items())],
        "work_split": [{"key": key, "label": WORK_LABELS[key], "percent": value}
                       for key, value in WORK_SPLIT_PCT.items()],
        "mobilisation_days": MOBILISATION_DAYS,
        "area_limits": {"min": MIN_AREA, "max": MAX_AREA},
    }
