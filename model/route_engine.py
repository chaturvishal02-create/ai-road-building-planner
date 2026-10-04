"""
Route-based construction cost engine for Indian roads (FY 2025-26 rates).

This module answers a different question from model/train.py. The trained classifiers
answer "will this sanctioned project overrun?" from data. This module answers "what
would it cost to build this road, and out of what?" from a civil-engineering quantity
take-off. Nothing here is learned: every rate is a documented assumption about Indian
market prices in 2025-26, and the payload says so in `basis`.

Two things make it defensible rather than a guess:

1. Distance is *measured*, not typed per route. Great-circle distance comes from city
   coordinates; a corridor factor calibrated against published road distances converts
   it into a driving distance (see CALIBRATION_ROUTES and tests/test_routes.py, which
   fails if the calibrated distances drift more than 15 % away from the published ones).
2. The rupee figure is assembled bottom-up from quantities -- cubic metres of aggregate,
   tonnes of bitumen, bags of cement, tonnes of steel, person-days, machine-hours --
   each multiplied by a unit rate. Every component is returnable, so the UI can show
   where the money goes instead of presenting one number.

No external or paid API is used: coordinates are static data and the distance is
spherical geometry.

Run directly to print a worked example:  py model/route_engine.py
"""

import math
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.schema import format_inr  # noqa: E402

# ---------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------
# name -> (latitude, longitude, state, default terrain, default soil).
# terrain is "plain" | "urban" | "ghat" | "hilly" | "rocky_or_hilly" | "coastal"; soil
# is "black_cotton" | "alluvial" | "red" | "sandy" | "rocky" | "laterite" | "clay".
# These are the labels the cost multipliers below are keyed on, and the dropdowns can
# override them.
CITIES = {
    # Maharashtra -------------------------------------------------------------
    "Mumbai City": (19.0760, 72.8777, "Maharashtra", "plain", "clay"),
    "Thane": (19.1972, 72.9720, "Maharashtra", "plain", "laterite"),
    "Nashik": (19.9975, 73.7898, "Maharashtra", "ghat", "black_cotton"),
    "Pune": (18.5204, 73.8567, "Maharashtra", "ghat", "black_cotton"),
    "Chhatrapati Sambhajinagar": (19.8762, 75.3433, "Maharashtra", "plain", "black_cotton"),
    "Nagpur": (21.1458, 79.0882, "Maharashtra", "plain", "black_cotton"),
    "Amravati": (20.9333, 77.7500, "Maharashtra", "hilly", "black_cotton"),
    # Delhi / Uttar Pradesh / Punjab / Bihar ----------------------------------
    "New Delhi": (28.6139, 77.2090, "Delhi", "plain", "alluvial"),
    "Central Delhi": (28.6300, 77.2200, "Delhi", "plain", "alluvial"),
    "Gautam Buddh Nagar": (28.5830, 77.4000, "Uttar Pradesh", "plain", "alluvial"),
    "Agra": (27.1767, 78.0081, "Uttar Pradesh", "plain", "alluvial"),
    "Kanpur Nagar": (26.4499, 80.3319, "Uttar Pradesh", "plain", "alluvial"),
    "Lucknow": (26.8467, 80.9462, "Uttar Pradesh", "plain", "alluvial"),
    "Prayagraj": (25.4358, 81.8463, "Uttar Pradesh", "plain", "alluvial"),
    "Varanasi": (25.3176, 82.9739, "Uttar Pradesh", "plain", "sandy"),
    "Amritsar": (31.6340, 74.8723, "Punjab", "plain", "alluvial"),
    "Jalandhar": (31.3266, 75.5690, "Punjab", "plain", "alluvial"),
    "Ludhiana": (30.9010, 75.8573, "Punjab", "plain", "alluvial"),
    "Patna": (25.5941, 85.1376, "Bihar", "plain", "alluvial"),
    "Muzaffarpur": (26.1176, 85.3968, "Bihar", "plain", "sandy"),
    "Bhagalpur": (25.2942, 86.9910, "Bihar", "plain", "sandy"),
    "Gaya": (24.7914, 85.0002, "Bihar", "hilly", "rocky"),
    # Rajasthan / Gujarat / Madhya Pradesh -----------------------------------
    "Jaipur": (26.9124, 75.7873, "Rajasthan", "plain", "sandy"),
    "Udaipur": (24.5854, 73.7125, "Rajasthan", "hilly", "rocky"),
    "Jodhpur": (26.2389, 73.0243, "Rajasthan", "plain", "sandy"),
    "Kota": (25.2138, 75.8648, "Rajasthan", "plain", "alluvial"),
    "Ahmedabad": (23.0225, 72.5714, "Gujarat", "plain", "black_cotton"),
    "Surat": (21.1675, 72.8311, "Gujarat", "plain", "black_cotton"),
    "Vadodara": (22.3072, 73.1812, "Gujarat", "plain", "black_cotton"),
    "Rajkot": (22.2559, 70.7923, "Gujarat", "plain", "sandy"),
    "Bhopal": (23.2599, 77.4126, "Madhya Pradesh", "plain", "black_cotton"),
    "Indore": (22.7196, 75.8575, "Madhya Pradesh", "plain", "black_cotton"),
    "Gwalior": (26.2183, 78.1828, "Madhya Pradesh", "rocky_or_hilly", "rocky"),
    "Jabalpur": (23.1815, 79.9864, "Madhya Pradesh", "hilly", "rocky"),
    # West Bengal -------------------------------------------------------------
    "Kolkata": (22.5726, 88.3639, "West Bengal", "plain", "clay"),
    "Howrah": (22.5958, 88.3151, "West Bengal", "plain", "clay"),
    "Paschim Medinipur": (22.4352, 87.2720, "West Bengal", "plain", "laterite"),
    "Darjeeling": (27.0410, 88.2663, "West Bengal", "hilly", "rocky"),
    # Karnataka / Telangana / Andhra Pradesh ---------------------------------
    "Bengaluru Urban": (12.9716, 77.5946, "Karnataka", "plain", "rocky"),
    "Mysuru": (12.2958, 76.6394, "Karnataka", "plain", "clay"),
    "Belagavi": (15.8497, 74.4975, "Karnataka", "plain", "black_cotton"),
    "Kalaburagi": (17.3297, 76.8343, "Karnataka", "plain", "black_cotton"),
    "Dakshina Kannada": (12.9141, 74.8560, "Karnataka", "ghat", "laterite"),
    "Hyderabad": (17.3850, 78.4867, "Telangana", "plain", "rocky"),
    "Warangal": (17.9689, 79.5941, "Telangana", "plain", "red"),
    "Nizamabad": (18.6722, 78.0941, "Telangana", "plain", "red"),
    "Vijayawada": (16.5062, 80.6489, "Andhra Pradesh", "plain", "clay"),
    "Visakhapatnam": (17.6868, 83.2185, "Andhra Pradesh", "ghat", "laterite"),
    "Tirupati": (13.6288, 79.4192, "Andhra Pradesh", "hilly", "red"),
    # Tamil Nadu / Kerala ----------------------------------------------------
    "Chennai": (13.0827, 80.2707, "Tamil Nadu", "plain", "sandy"),
    "Salem": (11.6643, 78.1464, "Tamil Nadu", "hilly", "rocky"),
    "Coimbatore": (11.0168, 76.9558, "Tamil Nadu", "ghat", "red"),
    "Madurai": (9.9252, 78.1198, "Tamil Nadu", "plain", "red"),
    "Kozhikode": (11.2588, 75.7804, "Kerala", "ghat", "laterite"),
    "Ernakulam": (9.9312, 76.2673, "Kerala", "plain", "clay"),
    "Thiruvananthapuram": (8.5241, 76.9366, "Kerala", "plain", "laterite"),
}

# Routes whose published road distance is quoted in the same order of magnitude that
# this engine needs to reproduce. The tests assert the calibrated estimate stays
# within 15 % of each published figure -- this is what keeps a "200 km" claim honest
# instead of hard-coding 200 for one pair and letting the rest of the map drift.
CALIBRATION_ROUTES = [
    ("Nashik", "Pune", 199),
    ("Mumbai City", "Pune", 149),
    ("Nashik", "Mumbai City", 165),
    ("New Delhi", "Jaipur", 274),
    ("Ahmedabad", "Vadodara", 101),
    ("Bengaluru Urban", "Mysuru", 146),
    ("Hyderabad", "Warangal", 150),
    ("Chennai", "Salem", 335),
    ("Indore", "Surat", 400),
    ("Agra", "Jaipur", 240),
    ("Lucknow", "Kanpur Nagar", 80),
    ("Vijayawada", "Visakhapatnam", 350),
    ("Coimbatore", "Ernakulam", 170),
    ("Bhopal", "Indore", 195),
    ("Surat", "Mumbai City", 275),
    ("Pune", "Nagpur", 620),
    ("Kolkata", "Patna", 540),
]

# Routes the home screen offers as one-click examples.
FEATURED_ROUTES = [("Nashik", "Pune"), ("Mumbai City", "Pune"), ("New Delhi", "Jaipur")]

# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0


def great_circle_km(a, b):
    """Straight-line distance between two (lat, lon) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (math.sin(dlat / 2.0) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def corridor_factor(pairs=None):
    """Median road-distance / great-circle distance over the calibration routes.

    Indian national highways do not run straight, but they are also not as crooked as
    the old metalled trunk roads: the published distances above sit between 1.0x and
    1.25x of the straight line. The median is used rather than the mean because two of
    the long corridors (Ahmedabad-Vadodara, Pune-Nagpur) really are near-straight, and
    averaging would let them drag the short, crooked hops down with them.
    """
    ratios = []
    for start, end, published in (pairs if pairs is not None else CALIBRATION_ROUTES):
        straight = great_circle_km(CITIES[start][:2], CITIES[end][:2])
        if straight > 0:
            ratios.append(published / straight)
    if not ratios:
        return 1.16
    ratios.sort()
    middle = len(ratios) // 2
    if len(ratios) % 2:
        return ratios[middle]
    return (ratios[middle - 1] + ratios[middle]) / 2.0


CORRIDOR_FACTOR = round(corridor_factor(), 4)


def route_distance(start, end):
    """Return {"start", "end", "straight_km", "road_km", "coefficient"} for a pair.

    The road distance is the great-circle distance multiplied by the corridor
    coefficient, rounded to one decimal. This is an estimate, not a routing result:
    a real alignment would come from a surveyed DPR or an OSRM self-hosted service.
    """
    a, b = CITIES[start][:2], CITIES[end][:2]
    straight = great_circle_km(a, b)
    return {"start": start, "end": end,
            "straight_km": round(straight, 1),
            "road_km": round(straight * CORRIDOR_FACTOR, 1),
            "coefficient": CORRIDOR_FACTOR,
            "basis": ("great-circle distance x %.4f corridor coefficient "
                      "(calibrated on %d published Indian road distances)"
                      % (CORRIDOR_FACTOR, len(CALIBRATION_ROUTES)))}


def geodesic_path(start, end, steps=24):
    """Sample the great circle between two catalogue cities for the map polyline."""
    return geodesic_between(CITIES[start][:2], CITIES[end][:2], steps=steps)


def geodesic_between(point_a, point_b, steps=24):
    """The same sampling between two raw (lat, lon) points.

    A straight line in lat/lon is wrong on a Mercator tile; interpolating along the
    great circle draws the curve a driver would broadly follow. It is still a
    corridor, not an alignment, and the UI labels it as one. Village-level points
    come from the planning dataset, so this takes coordinates rather than city names.
    """
    lat1, lon1 = math.radians(point_a[0]), math.radians(point_a[1])
    lat2, lon2 = math.radians(point_b[0]), math.radians(point_b[1])
    delta = great_circle_km(point_a, point_b) / EARTH_RADIUS_KM
    if delta < 1e-9:
        return [[round(math.degrees(lat1), 5), round(math.degrees(lon1), 5)]]
    points = []
    for i in range(steps + 1):
        fraction = i / float(steps)
        a_s = math.sin((1.0 - fraction) * delta) / math.sin(delta)
        b_s = math.sin(fraction * delta) / math.sin(delta)
        x = a_s * math.cos(lat1) * math.cos(lon1) + b_s * math.cos(lat2) * math.cos(lon2)
        y = a_s * math.cos(lat1) * math.sin(lon1) + b_s * math.cos(lat2) * math.sin(lon2)
        z = a_s * math.sin(lat1) + b_s * math.sin(lat2)
        points.append([round(math.degrees(math.atan2(z, math.sqrt(x * x + y * y))), 5),
                       round(math.degrees(math.atan2(y, x)), 5)])
    return points


# ---------------------------------------------------------------------------
# Terrain, soil and cross-section
# ---------------------------------------------------------------------------

TERRAIN_FACTORS = {
    "plain": {"factor": 1.00, "label": "Plain / flat terrain", "note": "Base rate."},
    "urban": {"factor": 1.15, "label": "Urban / city stretch",
              "note": "Utility shifting, traffic diversions, night working and "
                      "restricted access to the carriageway."},
    "ghat": {"factor": 1.30, "label": "Ghat section",
             "note": "Hairpin geometry, rock cutting, retaining walls, diverging "
                     "two-way traffic on a single carriageway."},
    "hilly": {"factor": 1.25, "label": "Hilly / undulating",
              "note": "Extra earthwork, curves and drainage; less rock face than a ghat."},
    "rocky_or_hilly": {"factor": 1.25, "label": "Rocky / hilly",
                       "note": "Blasting and hard-rock excavation in cut sections."},
    "coastal": {"factor": 1.15, "label": "Coastal / saline",
                "note": "Saline exposure demands sulphate-resistant cement and protection."},
}

# How hard each terrain is to build through, used when a corridor has to be priced to
# one of its two ends. Plain ranks lowest, ghat highest; urban and coastal tie.
TERRAIN_RANK = {"plain": 0, "urban": 1, "coastal": 1, "hilly": 2,
                "rocky_or_hilly": 2, "ghat": 3}

SOIL_FACTORS = {
    "black_cotton": {"factor": 1.10, "label": "Black cotton soil",
                     "note": "Expansive clay: needs an extra sub-base layer or lime "
                             "stabilisation."},
    "alluvial": {"factor": 1.00, "label": "Alluvial soil",
                 "note": "Reasonably workable, watch the water table in the plains."},
    "red": {"factor": 1.05, "label": "Red / lateritic soil",
            "note": "Free-draining but low CBR; a thin stabilised layer helps."},
    "sandy": {"factor": 1.12, "label": "Sandy soil",
              "note": "Low bearing capacity and erosion: wider shoulders, more compaction."},
    "rocky": {"factor": 1.08, "label": "Rocky / boulder strata",
              "note": "Good bearing but costly to excavate; less fill required."},
    "clay": {"factor": 1.15, "label": "Soft clay / low CBR",
             "note": "Weakest subgrade of the set: deeper pavement and pre-loading."},
    "laterite": {"factor": 1.05, "label": "Laterite",
                 "note": "Usable as sub-base when quarried dry; disintegrates when wet."},
}

# carriageway width by cross-section, and the pavement build that goes with it.
# labour_days_per_km and machinery_hours_per_km cover the whole stretch crew:
# earthwork, sub-base, surface course, shoulders, drains and cross drains.
ROAD_TYPES = {
    "rural_two_lane": {
        "label": "Rural / PMGSY-style two-lane", "lanes": 2, "width_m": 7.0,
        "pavement_depth_m": 0.45, "gsb_depth_m": 0.20, "bitumen_kg_per_m2": 5.0,
        "cement_bags_per_m2": 0.030, "steel_tonnes_per_km": 2.5,
        "drain_lines": 1, "shoulder_width_m": 5.0,
        "labour_days_per_km": 3200.0, "machinery_hours_per_km": 300.0,
    },
    "state_highway": {
        "label": "State highway, single lane + shoulders", "lanes": 1, "width_m": 3.75,
        "pavement_depth_m": 0.40, "gsb_depth_m": 0.20, "bitumen_kg_per_m2": 4.4,
        "cement_bags_per_m2": 0.026, "steel_tonnes_per_km": 1.5,
        "drain_lines": 1, "shoulder_width_m": 5.0,
        "labour_days_per_km": 2400.0, "machinery_hours_per_km": 200.0,
    },
    "national_highway_4": {
        "label": "National highway, 4-lane", "lanes": 4, "width_m": 14.0,
        "pavement_depth_m": 0.60, "gsb_depth_m": 0.22, "bitumen_kg_per_m2": 6.2,
        "cement_bags_per_m2": 0.048, "steel_tonnes_per_km": 9.0,
        "drain_lines": 2, "shoulder_width_m": 5.0,
        "labour_days_per_km": 7000.0, "machinery_hours_per_km": 900.0,
    },
    "expressway_6": {
        "label": "Expressway / ring road, 6-lane", "lanes": 6, "width_m": 24.0,
        "pavement_depth_m": 0.70, "gsb_depth_m": 0.25, "bitumen_kg_per_m2": 7.4,
        "cement_bags_per_m2": 0.075, "steel_tonnes_per_km": 22.0,
        "drain_lines": 2, "shoulder_width_m": 7.5,
        "labour_days_per_km": 12000.0, "machinery_hours_per_km": 1600.0,
    },
}

# Unit rates, rupees, FY 2025-26 assumptions for a Deccan / north-plain belt project.
# Change them here and every derived figure follows; the UI prints this table so a
# viva examiner can see exactly which number produced which rupee.
UNIT_RATES = {
    "cement_bag": 350.0,            # ₹ per 50 kg bag, OPC 53 grade, delivered
    "steel_tonne": 60000.0,         # ₹ per tonne, TMT Fe500D
    "bitumen_tonne": 50000.0,       # ₹ per tonne, VG-30 / CRMB
    "aggregate_tonne": 1250.0,      # ₹ per tonne, graded stone aggregate 10-20 mm
    "sand_tonne": 900.0,            # ₹ per tonne, M-sand / crushed sand
    "gsb_m3": 640.0,                # ₹ per m³ laid, granular sub-base (moorum)
    "earthwork_m3": 210.0,          # ₹ per m³, combined cutting and embankment
    "labour_day": 650.0,            # ₹ per person-day, skilled/unskilled blend
    "machinery_hour": 1450.0,       # ₹ per machine-hour, paver/roller/plant blend
    "shoulder_lakh_per_m_km": 1.2,  # ₹ lakh per km per metre of total shoulder width
    "drain_lakh_per_line_km": 7.5,  # ₹ lakh per km per drainage line (side or divider)
}

# Mixed-material density of a compacted pavement layer, and the split between the
# coarse aggregate and the fine/sand fraction inside it.
PAVEMENT_DENSITY_T_PER_M3 = 2.2
AGGREGATE_SPLIT = 0.70
EARTHWORK_DEPTH_M = 0.35            # average cut + fill height per km of corridor
OVERHEADS_PCT = 18.0                # GST, supervision, contingency, escalation

DEFAULT_COST_PER_KM_CRORE = 2.0     # the brief's adjustable baseline, ₹ crore per km
GHAT_UPLIFT_RANGE = "20-30 %"

# Programme completion speed, months per km of new carriageway. Used only to turn a
# cost estimate into the "scheduled duration" the trained model expects: ghat work and
# wider cross-sections slow a contractor down. Planning assumptions, not measurements.
MONTHS_PER_KM = {"state_highway": 0.16, "rural_two_lane": 0.18,
                 "national_highway_4": 0.30, "expressway_6": 0.45}
DURATION_MIN, DURATION_MAX = 6, 240


def planned_duration(road_type, distance_km, terrain):
    """Months a contractor would be given for this stretch (integer, 6-240)."""
    slow = 1.35 if terrain == "ghat" else (1.20 if terrain in ("hilly", "rocky_or_hilly") else 1.0)
    months = MONTHS_PER_KM[road_type] * distance_km * slow
    return int(max(DURATION_MIN, min(DURATION_MAX, round(months))))


# Construction speed in km of completed carriageway per day -- the three bands the
# brief names: highway 3-5, rural 1-2, ghat or hilly 0.5-1. A cross-section picks its
# band, and a hilly corridor always drops to the slowest band whatever it is called,
# because on a ghat the geometry, not the number of lanes, governs the job.
SPEED_KM_PER_DAY = {"highway": (3.0, 5.0), "rural": (1.0, 2.0), "ghat": (0.5, 1.0)}
ROAD_SPEED_BAND = {"state_highway": "rural", "rural_two_lane": "rural",
                   "national_highway_4": "highway", "expressway_6": "highway"}
HILLY_TERRAINS = ("ghat", "hilly", "rocky_or_hilly")
# An urban stretch is plain-country concrete that the crew only gets the road back on
# at night: +15 % on the clock, the same uplift the bill carries.
URBAN_TIME_FACTOR = 1.15
DAYS_PER_MONTH = 365.25 / 12.0    # calendar days -- no monsoon shutdown deducted


def construction_time(distance_km, road_type, terrain):
    """Months of physical running, from Time = Distance / Construction Speed.

    The band's fastest speed gives the early finish, its slowest the late finish, and
    its midpoint the headline figure. Days here are calendar days, so this answers "how
    long does the plant take to lay this road". It is deliberately not the same number
    as planned_duration(): that one is the contract programme the trained model consumes,
    and it carries land acquisition, utility shifting, statutory notices and monsoon
    idle time, which on an Indian corridor usually run for the whole length of the job.
    """
    band = "ghat" if terrain in HILLY_TERRAINS else ROAD_SPEED_BAND[road_type]
    slow, fast = SPEED_KM_PER_DAY[band]
    drag = URBAN_TIME_FACTOR if terrain == "urban" else 1.0

    def months_at(speed):
        return distance_km / speed * drag / DAYS_PER_MONTH

    mid = (slow + fast) / 2.0
    return {
        "band": band,
        "km_per_day": [slow, fast],
        "min_months": round(months_at(fast), 1),
        "likely_months": round(months_at(mid), 1),
        "max_months": round(months_at(slow), 1),
        "likely_days": round(distance_km / mid * drag, 1),
        "terrain_drag": drag,
        "basis": ("%.1f km ÷ %.2f km per day (%s band %s-%s km per day%s) = %.0f calendar "
                  "days = %.1f months; the band's own fastest and slowest speeds give "
                  "%.1f-%.1f months" % (
                      distance_km, mid, band, slow, fast,
                      ", x %.2f for urban working" % drag if drag != 1.0 else "",
                      distance_km / mid * drag, months_at(mid),
                      months_at(fast), months_at(slow))),
    }


# Deterministic overrun-risk weights. This is a stated judgement, not a learned
# probability -- /predict-route also asks the trained Random Forest, whose figure is
# labelled as a model probability, and the UI shows both side by side.
RISK_WEIGHTS = {"plain": 6.0, "urban": 14.0, "coastal": 12.0, "hilly": 16.0,
                "rocky_or_hilly": 16.0, "ghat": 22.0}
RISK_COMPLEXITY = {"state_highway": 3.0, "rural_two_lane": 4.0,
                   "national_highway_4": 7.0, "expressway_6": 10.0}
RISK_BASE = 12.0
RISK_MIN, RISK_MAX = 5.0, 75.0


class RouteError(ValueError):
    """A route request that cannot be answered. Becomes a 400 at the API edge."""

    def __init__(self, message, fields=None):
        super().__init__(message)
        self.message = message
        self.fields = fields or {}


# ---------------------------------------------------------------------------
# Quantity take-off
# ---------------------------------------------------------------------------


def quantities(spec, distance_km):
    """Physical quantities for a stretch: the things a DPR actually measures.

    Returns rupee-free numbers -- square metres of carriageway, tonnes of bitumen,
    bags of cement, cubic metres of earthwork -- so the cost table is a transparent
    multiplication of these by UNIT_RATES rather than a set of magic rupee figures.
    """
    area_m2 = spec["width_m"] * 1000.0 * distance_km
    pavement_m3 = area_m2 * spec["pavement_depth_m"]
    gsb_m3 = area_m2 * spec["gsb_depth_m"]
    earthwork_m3 = spec["width_m"] * EARTHWORK_DEPTH_M * 1000.0 * distance_km
    pavement_tonnes = pavement_m3 * PAVEMENT_DENSITY_T_PER_M3
    return {
        "carriageway_m2": round(area_m2, 1),
        "pavement_m3": round(pavement_m3, 1),
        "gsb_m3": round(gsb_m3, 1),
        "earthwork_m3": round(earthwork_m3, 1),
        "aggregate_tonnes": round(pavement_tonnes * AGGREGATE_SPLIT, 1),
        "sand_tonnes": round(pavement_tonnes * (1.0 - AGGREGATE_SPLIT), 1),
        "bitumen_tonnes": round(area_m2 * spec["bitumen_kg_per_m2"] / 1000.0, 2),
        "cement_bags": round(area_m2 * spec["cement_bags_per_m2"], 1),
        "steel_tonnes": round(spec["steel_tonnes_per_km"] * distance_km, 2),
        "labour_person_days": round(spec["labour_days_per_km"] * distance_km, 1),
        "machinery_hours": round(spec["machinery_hours_per_km"] * distance_km, 1),
        "drain_lines": spec["drain_lines"],
    }


def cost_components(spec, q, distance_km, terrain_factor, soil_factor):
    """Multiply the take-off by the unit rates. Everything is ₹ lakh.

    Terrain and soil multipliers apply to every works item: both are geometry and
    ground conditions, and they change what the crew and plant actually do on the
    whole corridor, not just on one layer.
    """
    r = UNIT_RATES
    site = terrain_factor * soil_factor
    items = [
        ("Cement", "₹ %s / bag (50 kg)" % int(r["cement_bag"]),
         q["cement_bags"] * r["cement_bag"]),
        ("Steel", "₹ %s / tonne (TMT Fe500D)" % int(r["steel_tonne"]),
         q["steel_tonnes"] * r["steel_tonne"]),
        ("Bitumen", "₹ %s / tonne (VG-30)" % int(r["bitumen_tonne"]),
         q["bitumen_tonnes"] * r["bitumen_tonne"]),
        ("Aggregate", "₹ %s / tonne (10-20 mm)" % int(r["aggregate_tonne"]),
         q["aggregate_tonnes"] * r["aggregate_tonne"]),
        ("Sand", "₹ %s / tonne (M-sand)" % int(r["sand_tonne"]),
         q["sand_tonnes"] * r["sand_tonne"]),
        ("Sub-base (GSB)", "₹ %s / m³ laid" % int(r["gsb_m3"]),
         q["gsb_m3"] * r["gsb_m3"]),
        ("Earthwork", "₹ %s / m³ cut + fill" % int(r["earthwork_m3"]),
         q["earthwork_m3"] * r["earthwork_m3"]),
        ("Labour", "₹ %s / person-day" % int(r["labour_day"]),
         q["labour_person_days"] * r["labour_day"]),
        ("Machinery", "₹ %s / machine-hour" % int(r["machinery_hour"]),
         q["machinery_hours"] * r["machinery_hour"]),
        ("Shoulders & furniture",
         "₹ %s lakh / km per m of shoulder width" % r["shoulder_lakh_per_m_km"],
         r["shoulder_lakh_per_m_km"] * spec["shoulder_width_m"] * distance_km * 1e5),
        ("Cross drainage",
         "₹ %s lakh / km per drain line" % r["drain_lakh_per_line_km"],
         r["drain_lakh_per_line_km"] * q["drain_lines"] * distance_km * 1e5),
    ]
    return [(label, basis, round(value * site / 1e5, 2)) for label, basis, value in items]


def estimate(start=None, end=None, distance_km=None, road_type="rural_two_lane",
             terrain=None, soil=None, width_m=None, cost_per_km_crore=None):
    """Full route cost estimate. Amounts are ₹ lakh unless a key says crore.

    `start`/`end` must be keys of CITIES; pass `distance_km` instead to price a
    stretch whose length is already known from a DPR.
    """
    measured = None
    if start is not None and end is not None:
        if start not in CITIES:
            raise RouteError("Unknown start city: %s" % start, fields={"start": start})
        if end not in CITIES:
            raise RouteError("Unknown end city: %s" % end, fields={"end": end})
        if start == end:
            raise RouteError("Start and end city are the same.", fields={"end": end})
        measured = route_distance(start, end)
    if distance_km is None:
        if measured is None:
            raise RouteError("Give either a start and end city, or a distance in km.",
                             fields={"distance_km": "required"})
        # A corridor typed in from a DPR wins over the calibrated one; both are
        # returned so the difference stays visible instead of one silently replacing
        # the other.
        distance_km = measured["road_km"]
    distance_km = float(distance_km)
    if distance_km <= 0 or distance_km > 4000:
        raise RouteError("Route length must be between 0.1 and 4000 km.",
                         fields={"distance_km": distance_km})
    # One decimal everywhere: the length on screen, the length the quantities use and
    # the length the ₹/km formula multiplies have to be the same number, or a reader
    # checking 189.2 x 2 against the printed total gets a different answer.
    distance_km = round(distance_km, 1)

    if road_type not in ROAD_TYPES:
        raise RouteError("Unknown road type: %s" % road_type, fields={"road_type": road_type})
    spec = dict(ROAD_TYPES[road_type])
    if width_m is not None:
        width_m = float(width_m)
        if width_m < 3.0 or width_m > 30.0:
            raise RouteError("Carriageway width must be between 3 and 30 metres.",
                             fields={"width_m": width_m})
        spec["width_m"] = width_m

    if terrain is None:
        terrain = _suggest_terrain(start, end)
    if soil is None:
        soil = _suggest_soil(start, end)
    if terrain not in TERRAIN_FACTORS:
        raise RouteError("Unknown terrain: %s" % terrain, fields={"terrain": terrain})
    if soil not in SOIL_FACTORS:
        raise RouteError("Unknown soil type: %s" % soil, fields={"soil": soil})

    terrain_factor = TERRAIN_FACTORS[terrain]["factor"]
    soil_factor = SOIL_FACTORS[soil]["factor"]

    q = quantities(spec, distance_km)
    components = cost_components(spec, q, distance_km, terrain_factor, soil_factor)
    works = round(sum(value for _, _, value in components), 2)
    overheads = round(works * OVERHEADS_PCT / 100.0, 2)
    total_lakh = round(works + overheads, 2)

    derived_per_km = round(total_lakh / distance_km, 2)
    duration = planned_duration(road_type, distance_km, terrain)
    build_time = construction_time(distance_km, road_type, terrain)
    site_factor = terrain_factor * soil_factor
    # Split the ground/site uplift into the two multipliers so the UI can name which
    # one the money went to. works already carries both, so divide them back out.
    base_works = works / site_factor if site_factor else works
    overhead_multiplier = 1.0 + OVERHEADS_PCT / 100.0
    payload = {
        "start": start, "end": end,
        "states": [CITIES[start][2], CITIES[end][2]] if start and end else [],
        "distance_km": round(distance_km, 1),
        "calibrated_km": measured["road_km"] if measured else None,
        "straight_km": measured["straight_km"] if measured else None,
        "distance_basis": (measured["basis"] if measured and distance_km == measured["road_km"]
                           else (measured["basis"] + "; the length typed in the request "
                                 "replaced it" if measured else
                                 "length taken from the request, not calibrated")),
        "road_type": road_type, "road_type_label": spec["label"],
        "lanes": spec["lanes"], "width_m": spec["width_m"],
        "terrain": terrain, "terrain_label": TERRAIN_FACTORS[terrain]["label"],
        "terrain_factor": terrain_factor, "terrain_note": TERRAIN_FACTORS[terrain]["note"],
        "soil": soil, "soil_label": SOIL_FACTORS[soil]["label"],
        "soil_factor": soil_factor, "soil_note": SOIL_FACTORS[soil]["note"],
        "combined_factor": round(site_factor, 4),
        "terrain_impact_lakh": round(base_works * (terrain_factor - 1.0) * overhead_multiplier, 2),
        "soil_impact_lakh": round(base_works * terrain_factor * (soil_factor - 1.0)
                                  * overhead_multiplier, 2),
        "quantities": q,
        "components": [{"label": label, "basis": basis, "lakh": value}
                       for label, basis, value in components],
        "works_lakh": works,
        "overheads_pct": OVERHEADS_PCT,
        "overheads_lakh": overheads,
        "total_lakh": total_lakh,
        "total_crore": round(total_lakh / 100.0, 2),
        "total_inr": format_inr(total_lakh),
        "total_rupees": int(round(total_lakh * 100000.0)),
        "derived_cost_per_km_lakh": derived_per_km,
        "derived_cost_per_km_crore": round(derived_per_km / 100.0, 3),
        "planned_duration_months": duration,
        "time_months": int(round(build_time["likely_months"])),
        "construction_time": build_time,
        "risk_percent": route_risk(terrain, soil, road_type, distance_km),
        # The same stretch, expressed as the six fields the trained classifier takes.
        # app.py feeds this to the model and reports the model's own probability
        # alongside the weight-based grade, labelled as which is which.
        "model_input": {
            "scheme_name": "PMGSY (Rural Roads)",
            "state": CITIES[start][2] if start else "",
            "district": start if start else "",
            "sanctioned_budget_lakh": round(total_lakh, 1),
            "duration_months": duration,
        },
        "basis": ("Bottom-up quantity take-off x FY 2025-26 unit rates, multiplied by "
                  "terrain and soil factors. This is an engineering estimate, not a "
                  "figure measured from the trained model."),
    }

    if cost_per_km_crore is not None:
        # The brief's adjustable headline rate: Budget = Distance x Cost_per_km.
        # When the user types a rate it is authoritative -- the take-off stays visible
        # as the takeoff_* figures and the component split is rescaled to the total,
        # so the page never shows two different budgets for one corridor.
        override = float(cost_per_km_crore)
        if override <= 0 or override > 200:
            raise RouteError("Cost per km must be between 0.01 and 200 ₹ crore.",
                             fields={"cost_per_km_crore": override})
        assumed_lakh = round(override * 100.0 * distance_km, 2)
        factor = assumed_lakh / total_lakh if total_lakh else 0.0

        payload["assumed_cost_per_km_crore"] = override
        payload["takeoff_cost_per_km_crore"] = round(derived_per_km / 100.0, 3)
        payload["takeoff_total_lakh"] = total_lakh
        payload["takeoff_total_crore"] = round(total_lakh / 100.0, 2)
        payload["components"] = [
            {"label": row["label"],
             "basis": row["basis"] + ("  · share rescaled" if abs(factor - 1.0) > 0.001 else ""),
             "lakh": round(row["lakh"] * factor, 2)}
            for row in payload["components"]]
        payload["works_lakh"] = round(works * factor, 2)
        payload["overheads_lakh"] = round(overheads * factor, 2)
        payload["terrain_impact_lakh"] = round(payload["terrain_impact_lakh"] * factor, 2)
        payload["soil_impact_lakh"] = round(payload["soil_impact_lakh"] * factor, 2)
        payload["total_lakh"] = assumed_lakh
        payload["total_crore"] = round(assumed_lakh / 100.0, 2)
        payload["total_inr"] = format_inr(assumed_lakh)
        payload["total_rupees"] = int(round(assumed_lakh * 100000.0))
        payload["derived_cost_per_km_lakh"] = round(assumed_lakh / distance_km, 2)
        payload["derived_cost_per_km_crore"] = override
        payload["model_input"]["sanctioned_budget_lakh"] = round(assumed_lakh, 1)
        payload["basis"] = (
            "Distance x your assumed ₹ %.2f crore/km. The component split is the "
            "bottom-up FY 2025-26 take-off rescaled to that total. This is an "
            "engineering estimate, not a figure measured from the trained model." % override)
        payload["scaling_note"] = (
            "Budget forced to your assumed ₹ %.2f crore/km x %.1f km = ₹ %.2f crore. "
            "The unforced itemised take-off for this cross-section is ₹ %.2f crore/km "
            "(₹ %.2f crore), so the components are its shares rescaled by %.3f -- the "
            "total is your assumption, the proportions are the engineer's." % (
                override, distance_km, assumed_lakh / 100.0,
                derived_per_km / 100.0, total_lakh / 100.0, factor))
    return payload


# ---------------------------------------------------------------------------
# Terrain / soil suggestion and risk band
# ---------------------------------------------------------------------------

def worst_terrain(terrain_a, terrain_b):
    """The more expensive of two terrains. A corridor is built to its hardest section."""
    return (terrain_a if TERRAIN_RANK[terrain_a] >= TERRAIN_RANK[terrain_b]
            else terrain_b)


def costlier_soil(soil_a, soil_b):
    """The weaker of two subgrades, judged by the factor each one carries."""
    if soil_a == soil_b:
        return soil_a
    return (soil_a if SOIL_FACTORS[soil_a]["factor"] >= SOIL_FACTORS[soil_b]["factor"]
            else soil_b)


def _suggest_terrain(start, end):
    """Terrain for a corridor: the worst of the two ends."""
    if not start or not end:
        return "plain"
    return worst_terrain(CITIES[start][3], CITIES[end][3])


def _suggest_soil(start, end):
    """Soil for a corridor: the ends are averaged by taking the costlier one."""
    if not start or not end:
        return "alluvial"
    return costlier_soil(CITIES[start][4], CITIES[end][4])


def route_risk(terrain, soil, road_type, distance_km):
    """Deterministic risk percentage for the corridor.

    Sum of explicit weights, clamped to 5-75 %. Every contributor is returned next to
    it in the API payload, so a reviewer can see why a number is what it is instead of
    trusting a black box. The trained model's own probability is reported separately
    and labelled as a model probability.
    """
    parts = {
        "base": RISK_BASE,
        "terrain": RISK_WEIGHTS[terrain],
        "soil_excess": round((SOIL_FACTORS[soil]["factor"] - 1.0) * 100.0 * 0.6, 2),
        "complexity": RISK_COMPLEXITY[road_type],
        "length": 8.0 if distance_km > 300 else (4.0 if distance_km > 150 else 0.0),
    }
    total = max(RISK_MIN, min(RISK_MAX, round(sum(parts.values()), 1)))
    return {"percent": total, "weights": parts,
            "band": "High" if total >= 45 else ("Medium" if total >= 25 else "Low"),
            "note": ("Sum of stated weights (base %.0f + terrain %.0f + soil %.2f + "
                     "complexity %.0f + length %.0f), clamped to 5-75 %%, and not a "
                     "calibrated model probability."
                     % (parts["base"], parts["terrain"], parts["soil_excess"],
                        parts["complexity"], parts["length"]))}


def city_catalogue():
    """Everything the browser needs to build the map and the dropdowns."""
    return {
        "cities": [{"name": name, "lat": values[0], "lon": values[1], "state": values[2],
                    "terrain": values[3], "soil": values[4]}
                   for name, values in sorted(CITIES.items())],
        "states": sorted({values[2] for values in CITIES.values()}),
        "road_types": [{"key": key, "label": values["label"], "lanes": values["lanes"],
                        "width_m": values["width_m"]} for key, values in ROAD_TYPES.items()],
        "terrains": [{"key": key, "label": values["label"], "factor": values["factor"],
                      "note": values["note"]} for key, values in TERRAIN_FACTORS.items()],
        "soils": [{"key": key, "label": values["label"], "factor": values["factor"],
                   "note": values["note"]} for key, values in SOIL_FACTORS.items()],
        "unit_rates": UNIT_RATES,
        "construction_speeds": [{"band": key, "km_per_day": list(values),
                                 "road_types": [road for road, value in ROAD_SPEED_BAND.items()
                                                if value == key]}
                                for key, values in SPEED_KM_PER_DAY.items()],
        "corridor_coefficient": CORRIDOR_FACTOR,
        "calibration_routes": [{"start": a, "end": b, "published_km": km}
                               for a, b, km in CALIBRATION_ROUTES],
        "featured_routes": [{"start": a, "end": b} for a, b in FEATURED_ROUTES],
        "centre": [19.0, 75.5],
        "default_cost_per_km_crore": DEFAULT_COST_PER_KM_CRORE,
        "ghat_uplift_range": GHAT_UPLIFT_RANGE,
        "overheads_pct": OVERHEADS_PCT,
    }


def sample_routes(road_type="rural_two_lane"):
    """Budget, distance and completion time over every calibration route.

    The chart data is produced by estimate() at request time, so a change to the unit
    rates or to the speed bands moves the bars; the series is never stored or hard-coded.
    """
    rows = []
    for start, end, published in CALIBRATION_ROUTES:
        result = estimate(start=start, end=end, road_type=road_type)
        rows.append({"label": "%s → %s" % (start, end), "start": start, "end": end,
                     "distance_km": result["distance_km"],
                     "published_km": published,
                     "total_crore": result["total_crore"],
                     "cost_per_km_crore": result["derived_cost_per_km_crore"],
                     "time_months": result["time_months"],
                     "min_time_months": result["construction_time"]["min_months"],
                     "max_time_months": result["construction_time"]["max_months"],
                     "km_per_day": result["construction_time"]["km_per_day"]})
    return rows


if __name__ == "__main__":
    # cp1252 consoles cannot encode the arrow or the rupee sign.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    for _s, _e in FEATURED_ROUTES:
        _r = estimate(_s, _e)
        _t = _r["construction_time"]
        print("%-14s → %-22s %6.1f km  %s  %2d mo (%.1f-%.1f)  risk %s%% (%s)" % (
            _s, _e, _r["distance_km"], format_inr(_r["total_lakh"]),
            _r["time_months"], _t["min_months"], _t["max_months"],
            _r["risk_percent"]["percent"], _r["risk_percent"]["band"]))
    print("\ncorridor coefficient %.4f  (median of %d published/straight ratios)"
          % (CORRIDOR_FACTOR, len(CALIBRATION_ROUTES)))


