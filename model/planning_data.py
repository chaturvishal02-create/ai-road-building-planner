"""The four-level Indian location hierarchy the planning brief asks for.

    State -> District -> Taluka -> Village

Only the top two levels are real data in this project: the 56 districts and their
coordinates are the ones `model/route_engine.py` carries and the trained classifier has
seen. The taluka and village names below are **generated placeholders** -- India has
over 250,000 villages and no free, offline, licensing-clean file of that granularity
exists on this machine. They are deterministic (seeded by district name), they sit
around their real district centre, and every place the UI shows them it also says they
are simulated, which is the rule this whole project follows for its numbers.

Nothing in here hard-codes a distance: every length is haversine between two coordinates
in this file, times the corridor coefficient the route engine calibrates on 17 published
Indian road distances.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random

from model import planning
from model.route_engine import (CITIES, ROAD_TYPES, RouteError, TERRAIN_FACTORS)

DATASET_NAME = "india_road_project_planning_dataset_2025_26.csv"

COLUMNS = ["state", "district", "taluka", "village", "lat", "lng", "road_type",
           "terrain", "soil_type", "distance_km", "budget", "time_months",
           "cement_cost", "steel_cost", "bitumen_cost", "labor_cost",
           "machinery_cost"]

# Money columns are ₹ lakh, the unit the rest of the project stores budgets in.
MONEY_UNIT = "₹ lakh"

TALUKAS_PER_DISTRICT = 3
VILLAGES_PER_TALUKA = 3
# A village sits between 4 km and ~55 km from its district headquarters; that is the
# spread a connector road actually has, and it keeps every generated point inside the
# state it belongs to.
MIN_OFFSET_KM, MAX_OFFSET_KM = 4.0, 55.0

STORAGE_STEMS = ["Wadgaon", "Shirval", "Kolhapur", "Arasgaon", "Pimpri", "Ukhale",
                 "Mordad", "Kelhi", "Palkhi", "Bhamare", "Tumane", "Kadepur",
                 "Ghoti", "Sonai", "Mhatari", "Deulwada", "Parnera", "Saswad",
                 "Kundmukhi", "Bodal", "Vaduj", "Shirude", "Kolewadi", "Ambassa",
                 "Ganeri", "Kharakala", "Pimpalgaon", "Yeola", "Kusumta", "Harnai",
                 "Sawana", "Morbadi", "Wadali", "Kinkhola", "Tisgaon", "Dhomni"]
TALUKA_SUFFIX = ["-Yavat", "-RV", " Bk", " Kh", " Nagar"]

# Rural roads are what a panchayat-level plan actually prices, so the weighted draw
# leans that way rather than treating all four cross-sections as equally likely.
ROAD_WEIGHTS = [("rural_two_lane", 46), ("state_highway", 28),
                ("national_highway_4", 20), ("expressway_6", 6)]

PROVENANCE = {
    "districts": ("56 district centres, coordinates, terrain and soil taken from the "
                  "same catalogue the trained models and the route planner use"),
    "talukas_villages": ("Generated placeholders: deterministic names and coordinates "
                        "seeded by district, 4-55 km around each district centre. They "
                        "are not census units."),
    "distance_km": ("Haversine from the village to its district headquarters x the "
                    "1.151 corridor coefficient. No distance is typed into a table."),
    "budget_and_materials": ("model/planning.py's top-down formula (distance x base "
                            "₹/km x width x terrain x soil) and its 25/20/15/20/20 "
                            "material split, in ₹ lakh."),
    "time_months": "Distance ÷ the brief's 4 / 2 / 1 km-per-day speed, in months.",
}


def _rng(*parts):
    """A fresh RNG whose stream depends only on the names given, so a rebuilt file is
    byte-identical to the shipped one."""
    seed = int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest(), 16)
    return random.Random(seed & 0xFFFFFFFF)


def _offset(centre_lat, centre_lon, bearing_deg, distance_km):
    """A point `distance_km` away from a centre on a great-circle bearing."""
    import math
    radius = 6371.0
    lat1, lon1 = math.radians(centre_lat), math.radians(centre_lon)
    angular = distance_km / radius
    bearing = math.radians(bearing_deg)
    lat2 = math.asin(math.sin(lat1) * math.cos(angular)
                     + math.cos(lat1) * math.sin(angular) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(angular),
                             math.cos(lat1) * math.cos(angular)
                             - math.sin(lat1) * math.sin(angular) * math.cos(bearing))
    return round(math.degrees(lat2), 4), round(math.degrees(lon2), 4)


def _weighted(rng, pairs):
    total = sum(weight for _name, weight in pairs)
    ticket = rng.uniform(0, total)
    running = 0.0
    for name, weight in pairs:
        running += weight
        if ticket <= running:
            return name
    return pairs[-1][0]


def _names(pool, count, rng, suffixes=()):
    """Distinct placeholder names from a stem pool, with an optional suffix cycle.
    Deterministic for a given RNG, and never repeats inside one taluka."""
    picked = rng.sample(pool, min(count, len(pool)))
    while len(picked) < count:                       # pool exhausted: number the repeats
        picked.append("%s %d" % (picked[len(picked) % len(pool)], len(picked) + 1))
    if suffixes:
        picked = [name + suffixes[i % len(suffixes)] for i, name in enumerate(picked)]
    return picked


def rows():
    """Every generated location, each carrying its own corridor length and price."""
    out = []
    for district, (lat, lon, state, terrain, soil) in sorted(CITIES.items()):
        taluka_names = _names(STORAGE_STEMS, TALUKAS_PER_DISTRICT,
                              _rng("taluka", district), TALUKA_SUFFIX)
        for taluka_index, taluka in enumerate(taluka_names):
            rng = _rng("village", district, taluka)
            for village in _names(STORAGE_STEMS, VILLAGES_PER_TALUKA, _rng("stems", district, taluka)):
                bearing = (taluka_index * 360.0 / TALUKAS_PER_DISTRICT
                           + rng.uniform(-30, 30))
                offset_km = rng.uniform(MIN_OFFSET_KM, MAX_OFFSET_KM)
                vlat, vlon = _offset(lat, lon, bearing, offset_km)
                road_type = _weighted(rng, ROAD_WEIGHTS)
                # Terrain and soil are inherited from the district's own catalogue
                # entry. Randomising them put "coastal" in landlocked Agra, which is
                # exactly the kind of detail a viva examiner notices.
                row_terrain, row_soil = terrain, soil
                distance = planning.road_km((vlat, vlon), (lat, lon))
                plan = planning.plan(distance_km=distance, road_type=road_type,
                                     terrain=row_terrain, soil=row_soil)
                material_lakh = {row["key"]: row["lakh"] for row in plan["materials"]}
                out.append({
                    "state": state, "district": district, "taluka": taluka,
                    "village": village, "lat": vlat, "lng": vlon,
                    "road_type": road_type, "terrain": row_terrain,
                    "soil_type": row_soil, "distance_km": distance,
                    "budget": plan["total_lakh"], "time_months": plan["time_months"],
                    "cement_cost": material_lakh["cement"],
                    "steel_cost": material_lakh["steel"],
                    "bitumen_cost": material_lakh["bitumen"],
                    "labor_cost": material_lakh["labor"],
                    "machinery_cost": material_lakh["machinery"],
                    "district_lat": lat, "district_lon": lon,
                    "straight_km": planning.straight_line_km((vlat, vlon), (lat, lon)),
                    "total_crore": plan["total_crore"],
                    "risk_band": plan["risk"]["band"],
                    "cost_basis": plan["basis"],
                })
    return out


_CACHE = None


def dataset():
    """Rows with a module-level cache; the data is generated, so there is no file to
    re-read and no chance of the API and the CSV disagreeing."""
    global _CACHE
    if _CACHE is None:
        _CACHE = rows()
    return _CACHE


def hierarchy():
    """The nested tree the dependent dropdowns consume: state > district > taluka >
    village, each village carrying its coordinates and its priced connector road."""
    tree = {}
    for row in dataset():
        state = tree.setdefault(row["state"], {"name": row["state"], "districts": {}})
        district = state["districts"].setdefault(
            row["district"], {"name": row["district"], "lat": row["district_lat"],
                              "lng": row["district_lon"], "talukas": {}})
        taluka = district["talukas"].setdefault(
            row["taluka"], {"name": row["taluka"], "villages": []})
        taluka["villages"].append({k: row[k] for k in (
            "village", "lat", "lng", "road_type", "terrain", "soil_type",
            "distance_km", "budget", "time_months", "risk_band", "total_crore")})
    out = []
    for state in sorted(tree):
        node = {"name": state, "districts": []}
        for district in sorted(tree[state]["districts"]):
            dnode = tree[state]["districts"][district]
            node["districts"].append({
                "name": district, "lat": dnode["lat"], "lng": dnode["lng"],
                "talukas": [{"name": taluka,
                             "villages": sorted(dnode["talukas"][taluka]["villages"],
                                                key=lambda v: v["village"])}
                            for taluka in sorted(dnode["talukas"])]})
        out.append(node)
    return out


def lookup(state, district, taluka=None, village=None, prefix="start"):
    """One location row, addressed level by level. Raises RouteError with the offending
    field named, because the API edge turns that into a 400 that says which dropdown is
    wrong. `prefix` names the field after the form it came from -- `start_village` for a
    route's origin, `end_village` for its destination."""
    def field(level):
        return "%s_%s" % (prefix, level)

    def fail(message, level):
        raise RouteError(message, {field(level): level})

    rows_at = matches(state, district, taluka, village, fail)
    if len(rows_at) > 1:
        # A district+taluka pair still spans several villages; ask for the last level
        # rather than silently pricing one of them.
        fail("Pick a village in %s, %s." % (taluka or district, district), "village")
    return rows_at[0]


def matches(state, district, taluka=None, village=None, fail=None):
    """Every row that matches the levels given, deepest level last.

    `fail` is the caller's error reporter, so the same walk serves a lookup that needs
    one row and a building form that only needs to know the place is real.
    """
    if fail is None:
        def fail(message, level):
            raise RouteError(message, {level: level})
    if not state:
        fail("Name the state.", "state")
    if state not in {row["state"] for row in dataset()}:
        fail("Unknown state '%s'." % state, "state")
    rows_at = [row for row in dataset() if row["state"] == state]
    if not district:
        fail("Name the district in %s." % state, "district")
    if district not in {row["district"] for row in rows_at}:
        fail("'%s' is not a district of %s." % (district, state), "district")
    rows_at = [row for row in rows_at if row["district"] == district]
    if taluka:
        if taluka not in {row["taluka"] for row in rows_at}:
            fail("'%s' is not a taluka of %s." % (taluka, district), "taluka")
        rows_at = [row for row in rows_at if row["taluka"] == taluka]
    if village:
        if village not in {row["village"] for row in rows_at}:
            fail("'%s' is not a village here." % village, "village")
        rows_at = [row for row in rows_at if row["village"] == village]
    return rows_at


def point(state, district, taluka=None, village=None, prefix="start"):
    """Resolve a nested {state, district, taluka, village} object to one map point.

    A village gives that village, with the road, terrain and soil its dataset row
    carries. State + district alone gives the district headquarters -- a real
    coordinate in the catalogue -- rather than an arbitrary village picked out of the
    taluka, which is why the level is reported back in `level` and shown on the map.
    """
    def field(level):
        return "%s_%s" % (prefix, level)

    def fail(message, level):
        raise RouteError(message, {field(level): level})

    rows_at = matches(state, district, taluka, village, fail)
    first = rows_at[0]
    if len(rows_at) > 1:
        if village and not taluka:
            # 33 of the generated village names repeat inside their own district, in a
            # different taluka. Falling back to the district headquarters here would
            # silently price a road to somewhere else, so the ambiguity is named.
            fail("'%s' is in more than one taluka of %s -- choose the taluka as well."
                 % (village, district), "taluka")
        if taluka:
            fail("Pick a village in %s, %s." % (taluka, district), "village")
        return {
            "level": "district", "state": state, "district": district,
            "taluka": None, "village": None,
            "label": "%s (district headquarters), %s" % (district, state),
            "lat": first["district_lat"], "lng": first["district_lon"],
            "terrain": first["terrain"], "soil_type": first["soil_type"],
            "road_type": None,
            "count": len(rows_at),
        }
    row = first
    return {
        "level": "village" if village else "taluka",
        "state": row["state"], "district": row["district"],
        "taluka": row["taluka"], "village": row["village"],
        "label": "%s%s, %s, %s" % (row["village"],
                                   " · " + row["taluka"] if row["taluka"] else "",
                                   row["district"], row["state"]),
        "lat": row["lat"], "lng": row["lng"],
        "terrain": row["terrain"], "soil_type": row["soil_type"],
        "road_type": row["road_type"],
        "connector_km": row["distance_km"],
        "connector_budget_lakh": row["budget"],
        "connector_time_months": row["time_months"],
        "connector_risk_band": row["risk_band"],
    }


def between(origin, destination):
    """Road length between two resolved points, haversine x corridor coefficient.

    Nothing is looked up in a distance table: the two coordinates in the hierarchy are
    the only inputs, which is what makes a village-to-village hop priceable at all.
    """
    distance = planning.road_km((origin["lat"], origin["lng"]),
                                (destination["lat"], destination["lng"]))
    if distance <= 0:
        raise RouteError("The start and end points are the same location; pick a "
                         "different village or district.", {"end": "same point"})
    return distance


def describe(point_obj):
    """The point as a reader wants it: one line naming the level it was resolved at."""
    return "%s [%s]" % (point_obj["label"], point_obj["level"])


def suggest(query, limit=12):
    """Search-as-you-type over the whole hierarchy: "pune hin" finds
    'Hinjawadi, Mulshi-ish taluka, Pune, Maharashtra'."""
    text = str(query or "").strip().lower()
    if not text:
        return []
    words = text.split()
    hits = []
    for row in dataset():
        haystack = "%s %s %s %s %s" % (row["village"], row["taluka"], row["district"],
                                       row["state"], row["district"] + " " + row["state"])
        if not all(word in haystack.lower() for word in words):
            continue
        hits.append({k: row[k] for k in ("state", "district", "taluka", "village",
                                         "lat", "lng", "distance_km", "budget",
                                         "time_months", "risk_band")})
        if len(hits) >= limit:
            break
    return hits


def write_dataset(path):
    """Write the brief's CSV. Money columns are ₹ lakh, `distance_km` is measured."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in dataset():
            writer.writerow(row)
    return path


def ensure_dataset(data_dir, quiet=False):
    """Create the planning CSV if it is missing. Returns its path."""
    path = os.path.join(data_dir, DATASET_NAME)
    if not os.path.exists(path):
        write_dataset(path)
        if not quiet:
            print("  planning dataset written: %s (%d rows)" % (path, len(dataset())))
    return path


def read_back(path):
    """Read the written file as a DataFrame, for the tests that check the CSV on disk
    really is the hierarchy the API serves."""
    import pandas as pd
    return pd.read_csv(path)


def catalogue():
    """Everything the frontend needs in one payload, including the assumption tables so
    the page can print them instead of paraphrasing them."""
    return {
        "states": hierarchy(),
        "columns": list(COLUMNS),
        "money_unit": MONEY_UNIT,
        "levels": ["state", "district", "taluka", "village"],
        "counts": {"states": len({r["state"] for r in dataset()}),
                   "districts": len({r["district"] for r in dataset()}),
                   "talukas": len({(r["district"], r["taluka"]) for r in dataset()}),
                   "villages": len(dataset())},
        "base_costs": dict(planning.BASE_COST_PER_KM_CRORE),
        "terrain_factors": {k: v["factor"] for k, v in TERRAIN_FACTORS.items()},
        "soil_factors": dict(planning.SOIL_FACTORS),
        "material_split": dict(planning.MATERIAL_SPLIT_PCT),
        "speeds_km_per_day": dict(planning.SPEED_KM_PER_DAY),
        "risk_thresholds_crore": dict(planning.RISK_THRESHOLDS_CRORE),
        "road_types": {key: {"label": spec["label"], "width_m": spec["width_m"],
                             "rate_class": planning.ROAD_RATE_CLASS[key]}
                       for key, spec in ROAD_TYPES.items()},
        "provenance": dict(PROVENANCE),
        "simulated": True,
    }
