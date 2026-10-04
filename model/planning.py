"""Planning-grade corridor estimate: the one-line formula, kept beside the take-off.

`model/route_engine.py` prices a road bottom-up (quantities x unit rates). This module
prices the same corridor with the top-down formula a planning officer actually types:

    Total = Distance x Base cost per km x Road width factor x Terrain factor
            x Soil factor

Both numbers are shown. They are not the same measurement and the page says so: the
take-off answers "what do the materials cost", this answers "what does a per-km
assumption say". Nothing here is learned from data, and every figure carries the
arithmetic that produced it.
"""

from __future__ import annotations

from model.route_engine import (DAYS_PER_MONTH, HILLY_TERRAINS, RouteError,
                                TERRAIN_FACTORS, great_circle_km, route_distance)

# The brief's three base rates, ₹ crore per kilometre.
BASE_COST_PER_KM_CRORE = {"rural": 1.0, "highway": 2.0, "expressway": 5.0}

# The cross-sections the map offers, mapped onto those three rate classes.
ROAD_RATE_CLASS = {"rural_two_lane": "rural", "state_highway": "highway",
                   "national_highway_4": "highway", "expressway_6": "expressway"}

# Width is a multiplier on the base rate, but normalised to *that class's own*
# standard cross-section: the rural ₹ 1 crore/km figure already buys a 7 m road and
# the expressway ₹ 5 crore figure already buys 24 m. Dividing by one global 7 m would
# re-price every class twice, once through its rate and once through its width.
CLASS_STANDARD_WIDTH_M = {"rural": 7.0, "highway": 14.0, "expressway": 24.0}
MIN_WIDTH_FACTOR = 0.5
MAX_WIDTH_FACTOR = 3.5

# The brief names three soil premia; the other soils the catalogue knows are stated
# here rather than borrowed from the take-off table, which prices soil through
# earthwork quantities instead.
SOIL_FACTORS = {"rocky": 1.20, "clay": 1.10, "sandy": 1.05,
                "black_cotton": 1.10, "laterite": 1.05, "red": 1.05,
                "alluvial": 1.00}

# cement / steel / bitumen / labour / machinery, as percentages of the total.
MATERIAL_SPLIT_PCT = {"cement": 25.0, "steel": 20.0, "bitumen": 15.0,
                      "labor": 20.0, "machinery": 20.0}

MATERIAL_LABELS = {"cement": "Cement", "steel": "Steel", "bitumen": "Bitumen",
                   "labor": "Labour", "machinery": "Machinery"}

# Single speeds, not bands: the planning formula quotes one figure per class. An
# expressway is laid at highway speed -- the brief names three speeds, not four.
SPEED_KM_PER_DAY = {"highway": 4.0, "rural": 2.0, "ghat": 1.0}
SPEED_LABELS = {"highway": "Highway", "rural": "Rural", "ghat": "Ghat / hilly"}
RATE_CLASS_SPEED = {"rural": "rural", "highway": "highway", "expressway": "highway"}

# The brief's risk rule is a budget threshold, not a weight sum.
RISK_THRESHOLDS_CRORE = {"low_max": 200.0, "medium_max": 500.0}

MIN_DISTANCE_KM, MAX_DISTANCE_KM = 0.1, 4000.0
MIN_WIDTH_M, MAX_WIDTH_M = 3.0, 30.0


def risk_band(total_crore):
    """Low below ₹ 200 crore, Medium up to ₹ 500 crore, High above it."""
    if total_crore > RISK_THRESHOLDS_CRORE["medium_max"]:
        band = "High"
    elif total_crore > RISK_THRESHOLDS_CRORE["low_max"]:
        band = "Medium"
    else:
        band = "Low"
    return {"band": band,
            "total_crore": round(total_crore, 2),
            "thresholds": dict(RISK_THRESHOLDS_CRORE),
            "rule": ("budget > ₹ 500 crore = High, ₹ 200-500 crore = Medium, "
                     "below ₹ 200 crore = Low -- a sanction-size rule, not a "
                     "probability of overrun")}


def planning_time(distance_km, rate_class, terrain):
    """Months at the brief's single km-per-day speed for the corridor's class."""
    band = "ghat" if terrain in HILLY_TERRAINS else RATE_CLASS_SPEED[rate_class]
    speed = SPEED_KM_PER_DAY[band]
    days = distance_km / speed
    months = days / DAYS_PER_MONTH
    return {"band": band,
            "km_per_day": speed,
            "working_days": round(days, 1),
            "months": round(months, 1),
            # A short connector really does work out under a month of laying; the
            # integer the card and the CSV carry is never zero, and `months` beside it
            # is the unrounded figure.
            "time_months": max(1, int(round(months))),
            "under_one_month": months < 1.0,
            "basis": ("%.1f km ÷ %.1f km per day (%s) = %.0f working days ÷ %.2f days "
                      "per month = %.1f months"
                      % (distance_km, speed, SPEED_LABELS[band], days, DAYS_PER_MONTH,
                         months))}


def width_factor(width_m, rate_class):
    """The typed width against the width that class's base rate already buys."""
    standard = CLASS_STANDARD_WIDTH_M[rate_class]
    return max(MIN_WIDTH_FACTOR, min(MAX_WIDTH_FACTOR, width_m / standard))


def plan(distance_km=None, start=None, end=None, road_type="rural_two_lane",
         terrain="plain", soil="alluvial", width_m=None):
    """Price one corridor with the top-down planning formula.

    Takes either a length, or two catalogue cities to measure one from -- the same
    choice the take-off engine offers, so both views can be asked about one corridor.
    """
    if road_type not in ROAD_RATE_CLASS:
        raise RouteError("Unknown road type '%s'." % road_type, {"road_type": "road_type"})
    if terrain not in TERRAIN_FACTORS:
        raise RouteError("Unknown terrain '%s'." % terrain, {"terrain": "terrain"})
    if soil not in SOIL_FACTORS:
        raise RouteError("Unknown soil type '%s'." % soil, {"soil_type": "soil"})

    calibrated = None
    if distance_km is None:
        if not (start and end):
            raise RouteError("Give either a length, or a start and an end city.",
                             {"distance_km": "distance_km"})
        measured = route_distance(start, end)
        calibrated = measured["road_km"]
        distance_km = calibrated
    else:
        try:
            distance_km = round(float(distance_km), 1)
        except (TypeError, ValueError):
            raise RouteError("Distance must be a number of kilometres.",
                             {"distance_km": "distance_km"})
    if not (MIN_DISTANCE_KM <= distance_km <= MAX_DISTANCE_KM):
        raise RouteError("Distance must be between %s and %s km."
                         % (MIN_DISTANCE_KM, MAX_DISTANCE_KM), {"distance_km": "distance_km"})

    rate_class = ROAD_RATE_CLASS[road_type]
    base = BASE_COST_PER_KM_CRORE[rate_class]
    terrain_mult = TERRAIN_FACTORS[terrain]["factor"]
    soil_mult = SOIL_FACTORS[soil]
    default_width = CLASS_STANDARD_WIDTH_M[rate_class]
    effective_width = float(width_m) if width_m else default_width
    if not (MIN_WIDTH_M <= effective_width <= MAX_WIDTH_M):
        raise RouteError("Carriageway width must be between %s and %s m."
                         % (MIN_WIDTH_M, MAX_WIDTH_M), {"width_m": "width_m"})
    factor = width_factor(effective_width, rate_class)

    per_km = base * factor
    total_crore = distance_km * per_km * terrain_mult * soil_mult
    total_lakh = total_crore * 100.0
    time = planning_time(distance_km, rate_class, terrain)
    total_lakh_shown = round(total_lakh, 1)
    total_rupees = int(round(total_lakh * 100000.0))
    # An ordered list, not a dict: Flask sorts JSON keys on the way out, so a mapping
    # would arrive at the browser alphabetically and the five lines would lose the
    # cement-steel-bitumen-labour-machinery order the brief states.
    materials = []
    for key, pct in MATERIAL_SPLIT_PCT.items():
        materials.append({"key": key, "label": MATERIAL_LABELS[key], "percent": pct,
                          "lakh": round(total_lakh_shown * pct / 100.0, 2),
                          "crore": round(total_crore * pct / 100.0, 3),
                          "rupees": int(round(total_lakh * pct * 1000.0))})
    if abs(sum(MATERIAL_SPLIT_PCT.values()) - 100.0) > 1e-9:
        raise RouteError("The material split does not add up to 100 %.", {})
    # Each line is rounded on its own, so the five can be a rupee -- or a paisa -- short
    # of the printed total. Put the difference in the largest line: the split then adds
    # back to exactly the budget the card shows, in rupees and in lakh.
    rupee_residual = total_rupees - sum(row["rupees"] for row in materials)
    lakh_residual = round(total_lakh_shown - sum(row["lakh"] for row in materials), 2)
    cement = [row for row in materials if row["key"] == "cement"][0]
    cement["rupees"] += rupee_residual
    cement["lakh"] = round(cement["lakh"] + lakh_residual, 2)

    return {
        "distance_km": distance_km,
        "calibrated_km": calibrated,
        "road_type": road_type,
        "rate_class": rate_class,
        "base_cost_per_km_crore": base,
        "width_m": effective_width,
        "width_factor": round(factor, 3),
        "standard_width_m": CLASS_STANDARD_WIDTH_M[rate_class],
        "terrain": terrain,
        "terrain_factor": terrain_mult,
        "soil": soil,
        "soil_factor": soil_mult,
        "cost_per_km_crore": round(per_km * terrain_mult * soil_mult, 4),
        "total_crore": round(total_lakh / 100.0, 2),
        "total_lakh": total_lakh_shown,
        "total_rupees": total_rupees,
        "materials": materials,
        "material_split_pct": dict(MATERIAL_SPLIT_PCT),
        # The brief's `breakdown` object: the five material lines in rupees.
        "breakdown": {row["key"]: row["rupees"] for row in materials},
        "time": time,
        "time_months": time["time_months"],
        "risk": risk_band(total_crore),
        "basis": ("%.1f km x ₹ %.1f crore per km (%s base) x %.3f width factor "
                  "(%.1f m ÷ %.0f m for this class) x %.2f terrain x %.2f soil = "
                  "₹ %.2f crore"
                  % (distance_km, base, rate_class, factor, effective_width,
                     CLASS_STANDARD_WIDTH_M[rate_class], terrain_mult, soil_mult,
                     total_crore)),
        "note": ("Top-down planning formula from the brief. It is a different "
                 "measurement from the itemised quantity take-off, not a check on it; "
                 "the take-off prices materials and labour, this prices the corridor "
                 "at one assumed ₹ per km for the cross-section class."),
    }


def straight_line_km(point_a, point_b):
    """Haversine between two (lat, lon) pairs, rounded the way every other distance
    in the project is -- one place, so the map and the card agree."""
    return round(great_circle_km(point_a, point_b), 1)


def road_km(point_a, point_b):
    """Straight line converted to a road length by the same corridor coefficient the
    take-off uses, so the hierarchy path and the city path cannot disagree."""
    from model.route_engine import CORRIDOR_FACTOR
    return round(great_circle_km(point_a, point_b) * CORRIDOR_FACTOR, 1)
