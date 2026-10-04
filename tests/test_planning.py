"""Tests for the planning layer: the brief's one-line formula, the four-level
hierarchy it is addressed by, and the two HTTP routes that expose them.

These are arithmetic and contract tests, not accuracy tests. The planning formula is a
stated assumption, not a measurement, so "is ₹ 2,897 crore right?" has no answer -- what
can be checked is that the number on screen is the number the formula produces, that the
five material lines add back to it to the rupee, that the CSV on disk is the same data the
API serves, and that a bad location names the dropdown that is wrong instead of
returning a 500.

Nothing here is mocked and no number is typed in from outside: the expected values are
recomputed from the same constants the engine documents.
"""

import csv
import os
import statistics

import pytest

import model.planning as planning
import model.planning_data as planning_data
import model.route_engine as routes

RUPEE = "₹"


# ---------------------------------------------------------------------------
# The formula itself
# ---------------------------------------------------------------------------

def test_plan_reproduces_the_briefs_multiplication():
    result = planning.plan(distance_km=100.0, road_type="national_highway_4",
                           terrain="plain", soil="alluvial")
    base = planning.BASE_COST_PER_KM_CRORE["highway"]
    width = planning.CLASS_STANDARD_WIDTH_M["highway"]
    factor = planning.width_factor(width, "highway")
    expected = (100.0 * base * factor
                * planning.TERRAIN_FACTORS["plain"]["factor"]
                * planning.SOIL_FACTORS["alluvial"])
    assert result["total_crore"] == pytest.approx(round(expected, 2))
    assert result["total_lakh"] == pytest.approx(round(expected * 100.0, 1))
    assert result["width_factor"] == 1.0
    assert RUPEE in result["basis"] and "100.0 km" in result["basis"]


def test_ghat_and_rocky_cost_more_than_plain_and_alluvial():
    plain = planning.plan(distance_km=50.0, road_type="state_highway", terrain="plain",
                          soil="alluvial")
    hard = planning.plan(distance_km=50.0, road_type="state_highway", terrain="ghat",
                         soil="rocky")
    assert hard["total_crore"] > plain["total_crore"]
    assert hard["terrain_factor"] == pytest.approx(1.30)
    assert hard["soil_factor"] == pytest.approx(planning.SOIL_FACTORS["rocky"])
    # and it takes longer: the ghat speed band is 1 km/day against highway 4 km/day
    assert hard["time"]["months"] > plain["time"]["months"]


def test_width_is_normalised_to_each_class_own_standard():
    # A 14 m national highway is the standard cross-section its ₹ 2 crore/km rate
    # already buys; a 14 m rural road is double the 7 m the ₹ 1 crore figure assumes.
    highway = planning.plan(distance_km=10.0, road_type="national_highway_4",
                            width_m=14.0)
    rural = planning.plan(distance_km=10.0, road_type="rural_two_lane", width_m=14.0)
    assert highway["width_factor"] == pytest.approx(1.0)
    assert rural["width_factor"] == pytest.approx(2.0)
    assert rural["standard_width_m"] == 7.0


def test_width_factor_is_clamped_at_both_ends():
    assert planning.width_factor(3.0, "expressway") == planning.MIN_WIDTH_FACTOR
    assert planning.width_factor(30.0, "rural") == planning.MAX_WIDTH_FACTOR
    assert planning.width_factor(24.0, "expressway") == pytest.approx(1.0)


@pytest.mark.parametrize("road_type, terrain, band", [
    ("national_highway_4", "plain", "highway"),
    ("expressway_6", "plain", "highway"),      # the brief names three speeds, not four
    ("rural_two_lane", "plain", "rural"),
    ("state_highway", "ghat", "ghat"),         # ground outranks cross-section
    ("national_highway_4", "hilly", "ghat"),
])
def test_time_uses_the_briefs_three_speeds(road_type, terrain, band):
    result = planning.plan(distance_km=40.0, road_type=road_type, terrain=terrain)
    assert result["time"]["band"] == band
    assert result["time"]["km_per_day"] == planning.SPEED_KM_PER_DAY[band]
    assert result["time"]["working_days"] == pytest.approx(40.0 / result["time"]["km_per_day"])


def test_a_short_connector_never_reports_zero_months():
    result = planning.plan(distance_km=0.5, road_type="rural_two_lane")
    assert result["time"]["months"] < 1.0
    assert result["time"]["under_one_month"] is True
    assert result["time"]["time_months"] == 1


# ---------------------------------------------------------------------------
# The material split
# ---------------------------------------------------------------------------

def test_material_split_is_the_briefs_five_percentages():
    assert planning.MATERIAL_SPLIT_PCT == {"cement": 25.0, "steel": 20.0,
                                           "bitumen": 15.0, "labor": 20.0,
                                           "machinery": 20.0}
    assert sum(planning.MATERIAL_SPLIT_PCT.values()) == 100.0


def test_the_five_material_lines_add_back_to_the_printed_total():
    result = planning.plan(distance_km=137.4, road_type="state_highway",
                           terrain="urban", soil="black_cotton")
    rows = result["materials"]
    assert sum(row["rupees"] for row in rows) == result["total_rupees"]
    assert sum(row["lakh"] for row in rows) == pytest.approx(result["total_lakh"])
    assert sum(result["breakdown"].values()) == result["total_rupees"]


def test_materials_carry_the_brief_order_not_alphabetical():
    # Flask sorts JSON keys on the way out, so a dict would reach the browser as
    # bitumen, cement, labor, machinery, steel. The list keeps the stated order.
    result = planning.plan(distance_km=20.0)
    assert [row["key"] for row in result["materials"]] == list(planning.MATERIAL_SPLIT_PCT)
    assert [row["label"] for row in result["materials"]] == ["Cement", "Steel", "Bitumen",
                                                             "Labour", "Machinery"]


# ---------------------------------------------------------------------------
# Risk by sanction size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("crore, band", [
    (1.0, "Low"), (200.0, "Low"), (200.01, "Medium"), (350.0, "Medium"),
    (500.0, "Medium"), (500.01, "High"), (4200.0, "High"),
])
def test_risk_band_follows_the_threshold_rule(crore, band):
    assert planning.risk_band(crore)["band"] == band


def test_risk_band_says_it_is_a_size_rule_not_a_probability():
    rule = planning.risk_band(600.0)["rule"]
    assert "sanction-size" in rule and "probability" in rule


# ---------------------------------------------------------------------------
# Validation: a bad input names the field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, field", [
    ({"road_type": "motorway"}, "road_type"),
    ({"terrain": "tundra"}, "terrain"),
    ({"soil": "volcanic"}, "soil_type"),
    ({"distance_km": 0.0}, "distance_km"),
    ({"distance_km": 9000.0}, "distance_km"),
    ({"distance_km": "twelve"}, "distance_km"),
    ({"distance_km": 10.0, "width_m": 1.0}, "width_m"),
    ({"distance_km": 10.0, "width_m": 80.0}, "width_m"),
    ({}, "distance_km"),
])
def test_bad_planning_inputs_raise_a_field_level_error(kwargs, field):
    with pytest.raises(routes.RouteError) as excinfo:
        planning.plan(**kwargs)
    assert field in excinfo.value.fields


def test_a_city_pair_is_measured_not_typed():
    result = planning.plan(start="Pune", end="Nashik", road_type="national_highway_4")
    measured = routes.route_distance("Pune", "Nashik")["road_km"]
    assert result["distance_km"] == pytest.approx(measured)
    assert result["calibrated_km"] == pytest.approx(measured)


# ---------------------------------------------------------------------------
# The two soil and terrain tables are allowed to differ, but not in their keys
# ---------------------------------------------------------------------------

def test_planning_and_take_off_agree_on_terrain_and_soil_names():
    assert set(planning.SOIL_FACTORS) == set(routes.SOIL_FACTORS)
    assert set(planning.TERRAIN_FACTORS) == set(routes.TERRAIN_FACTORS)
    # ...and they deliberately disagree on the numbers: the take-off prices soil through
    # earthwork quantities, this multiplies a per-km rate. Documented, not a drift.
    assert planning.SOIL_FACTORS["sandy"] != routes.SOIL_FACTORS["sandy"]


def test_terrain_multipliers_match_the_brief_premia():
    assert planning.TERRAIN_FACTORS["plain"]["factor"] == pytest.approx(1.00)
    assert planning.TERRAIN_FACTORS["ghat"]["factor"] == pytest.approx(1.30)
    assert planning.TERRAIN_FACTORS["urban"]["factor"] == pytest.approx(1.15)
    assert planning.TERRAIN_FACTORS["rocky_or_hilly"]["factor"] == pytest.approx(1.25)
    assert planning.SOIL_FACTORS["rocky"] == pytest.approx(1.20)
    assert planning.SOIL_FACTORS["clay"] == pytest.approx(1.10)
    assert planning.SOIL_FACTORS["sandy"] == pytest.approx(1.05)


# ---------------------------------------------------------------------------
# The hierarchy dataset
# ---------------------------------------------------------------------------

def test_dataset_has_the_briefs_columns_and_row_count():
    rows = planning_data.dataset()
    # The in-memory row is a superset: the CSV writer emits COLUMNS and ignores the
    # rest (district centre, straight-line km, band), which is what the brief asked for.
    assert list(rows[0])[:len(planning_data.COLUMNS)] == planning_data.COLUMNS
    assert planning_data.COLUMNS == [
        "state", "district", "taluka", "village", "lat", "lng", "road_type", "terrain",
        "soil_type", "distance_km", "budget", "time_months", "cement_cost",
        "steel_cost", "bitumen_cost", "labor_cost", "machinery_cost"]
    expected = (len(routes.CITIES) * planning_data.TALUKAS_PER_DISTRICT
                * planning_data.VILLAGES_PER_TALUKA)
    assert len(rows) == expected == 504


def test_generation_is_deterministic():
    first = planning_data.rows()
    second = planning_data.rows()
    assert first == second
    assert len({(r["state"], r["district"], r["taluka"], r["village"])
                for r in first}) == len(first)


def test_every_row_carries_its_own_priced_materials():
    for row in planning_data.dataset()[:60]:
        parts = (row["cement_cost"] + row["steel_cost"] + row["bitumen_cost"]
                 + row["labor_cost"] + row["machinery_cost"])
        assert parts == pytest.approx(row["budget"], abs=0.11)
        assert row["cement_cost"] == pytest.approx(row["budget"] * 0.25, abs=0.06)


def test_distances_are_measured_from_coordinates_not_written_into_a_table():
    for row in planning_data.dataset()[:60]:
        centre = (row["district_lat"], row["district_lon"])
        village = (row["lat"], row["lng"])
        straight = routes.great_circle_km(centre, village)
        assert row["straight_km"] == pytest.approx(straight, abs=0.06)
        assert row["distance_km"] == pytest.approx(
            straight * routes.corridor_factor(), abs=0.06)
        # the generator's own promise: a village is 4-55 km from its headquarters
        assert planning_data.MIN_OFFSET_KM - 0.6 <= straight \
            <= planning_data.MAX_OFFSET_KM + 0.6


def test_budget_column_is_the_planning_formula_for_the_same_row():
    for row in planning_data.dataset()[:40]:
        plan = planning.plan(distance_km=row["distance_km"], road_type=row["road_type"],
                             terrain=row["terrain"], soil=row["soil_type"])
        assert row["budget"] == pytest.approx(plan["total_lakh"])
        assert row["time_months"] == plan["time_months"]
        assert row["risk_band"] == plan["risk"]["band"]


def test_hierarchy_is_sorted_and_complete():
    tree = planning_data.hierarchy()
    assert [node["name"] for node in tree] == sorted(node["name"] for node in tree)
    states = {node["name"] for node in tree}
    assert states == {row["state"] for row in planning_data.dataset()}
    for node in tree:
        districts = node["districts"]
        assert [d["name"] for d in districts] == sorted(d["name"] for d in districts)
        for district in districts:
            assert district["lat"] and district["lng"]
            for taluka in district["talukas"]:
                names = [v["village"] for v in taluka["villages"]]
                assert names == sorted(names)
                assert len(names) == planning_data.VILLAGES_PER_TALUKA


def test_catalogue_publishes_counts_and_the_assumption_tables():
    catalogue = planning_data.catalogue()
    assert catalogue["levels"] == ["state", "district", "taluka", "village"]
    assert catalogue["counts"]["villages"] == len(planning_data.dataset())
    assert catalogue["counts"]["districts"] == len(routes.CITIES)
    assert catalogue["base_costs"] == planning.BASE_COST_PER_KM_CRORE
    assert catalogue["material_split"] == planning.MATERIAL_SPLIT_PCT
    assert catalogue["risk_thresholds_crore"] == planning.RISK_THRESHOLDS_CRORE
    assert catalogue["money_unit"] == "₹ lakh"
    assert catalogue["simulated"] is True
    assert "placeholder" in catalogue["provenance"]["talukas_villages"].lower()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def test_suggest_matches_across_levels_and_respects_the_limit():
    hits = planning_data.suggest("Pune", limit=5)
    assert 0 < len(hits) <= 5
    assert all("Pune" in (h["district"] + h["state"] + h["taluka"] + h["village"])
               for h in hits)
    assert planning_data.suggest("", limit=5) == []
    assert planning_data.suggest("zzzz-no-such-place") == []
    words = planning_data.suggest(hits[0]["village"] + " " + hits[0]["taluka"])
    assert any(h["village"] == hits[0]["village"] for h in words)


# ---------------------------------------------------------------------------
# Resolving a location
# ---------------------------------------------------------------------------

def test_a_village_resolves_to_its_own_coordinates_and_connector():
    row = planning_data.dataset()[0]
    point = planning_data.point(row["state"], row["district"], row["taluka"],
                                row["village"])
    assert point["level"] == "village"
    assert (point["lat"], point["lng"]) == pytest.approx((row["lat"], row["lng"]), abs=1e-6)
    assert point["connector_km"] == pytest.approx(row["distance_km"])
    assert row["village"] in point["label"] and row["state"] in point["label"]


def test_a_district_alone_gives_the_headquarters_and_says_so():
    row = planning_data.dataset()[0]
    point = planning_data.point(row["state"], row["district"])
    assert point["level"] == "district"
    assert "headquarters" in point["label"]
    assert point["count"] == planning_data.TALUKAS_PER_DISTRICT * planning_data.VILLAGES_PER_TALUKA
    assert point["lat"] == pytest.approx(row["district_lat"])


def test_an_ambiguous_village_asks_for_the_taluka_instead_of_moving_the_site():
    # 33 of the generated village names repeat inside their own district. Silently
    # pricing the district headquarters for one of them would be a wrong answer that
    # looks right, so the resolver refuses.
    from collections import Counter
    counts = Counter((r["state"], r["district"], r["village"])
                     for r in planning_data.dataset())
    target = next(key for key, n in counts.items() if n > 1)
    with pytest.raises(routes.RouteError) as excinfo:
        planning_data.point(target[0], target[1], None, target[2])
    assert excinfo.value.fields == {"start_taluka": "taluka"}
    assert "more than one taluka" in excinfo.value.message


@pytest.mark.parametrize("args, field", [
    ({"state": "", "district": "Pune"}, "start_state"),
    ({"state": "Atlantis", "district": "Pune"}, "start_state"),
    ({"state": "Maharashtra", "district": "Nowhere"}, "start_district"),
])
def test_a_bad_location_names_the_level_that_is_wrong(args, field):
    # `prefix` is what lets the same resolver serve both ends of a corridor: the 400
    # that comes back names start_district or end_district, so the right dropdown goes
    # red on the page.
    with pytest.raises(routes.RouteError) as excinfo:
        planning_data.point(**args)
    assert field in excinfo.value.fields
    with pytest.raises(routes.RouteError) as excinfo:
        planning_data.point(prefix="end", **args)
    assert field.replace("start_", "end_") in excinfo.value.fields


def test_between_is_symmetric_and_refuses_a_zero_length_corridor():
    row = planning_data.dataset()[0]
    origin = planning_data.point(row["state"], row["district"], row["taluka"],
                                 row["village"])
    other = planning_data.point(row["state"], row["district"])
    forward = planning_data.between(origin, other)
    back = planning_data.between(other, origin)
    assert forward == pytest.approx(back)
    assert forward > 0.0
    with pytest.raises(routes.RouteError) as excinfo:
        planning_data.between(origin, origin)
    assert excinfo.value.fields == {"end": "same point"}


def test_lookup_only_returns_a_row_when_the_address_is_unique():
    row = planning_data.dataset()[0]
    found = planning_data.lookup(row["state"], row["district"], row["taluka"],
                                 row["village"])
    assert found["village"] == row["village"]
    with pytest.raises(routes.RouteError):
        planning_data.lookup(row["state"], row["district"])


def test_describe_shows_the_level_the_point_resolved_at():
    row = planning_data.dataset()[0]
    point = planning_data.point(row["state"], row["district"])
    assert planning_data.describe(point).endswith("[district]")


# ---------------------------------------------------------------------------
# The CSV on disk
# ---------------------------------------------------------------------------

def test_ensure_dataset_writes_the_file_the_api_serves(tmp_path):
    path = planning_data.ensure_dataset(str(tmp_path), quiet=True)
    assert os.path.basename(path) == planning_data.DATASET_NAME
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == planning_data.COLUMNS
        written = list(reader)
    assert len(written) == len(planning_data.dataset())
    assert written[0]["village"] == planning_data.dataset()[0]["village"]
    # calling it again must not touch the file that is already there
    stamp = os.path.getmtime(path)
    assert planning_data.ensure_dataset(str(tmp_path), quiet=True) == path
    assert os.path.getmtime(path) == stamp


def test_the_shipped_file_is_the_hierarchy_the_api_serves():
    frame = planning_data.read_back(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
        planning_data.DATASET_NAME))
    rows = planning_data.dataset()
    assert len(frame) == len(rows)
    assert float(frame["budget"].min()) > 0
    assert float(frame["budget"].max()) < 100000  # ₹ lakh, so under ₹ 1,000 crore
    assert {str(s) for s in frame["state"]} == {r["state"] for r in rows}


# ---------------------------------------------------------------------------
# GET /locations
# ---------------------------------------------------------------------------

def test_locations_serves_the_whole_catalogue(client):
    payload = client.get("/locations").get_json()
    assert payload["catalogue"]["counts"]["villages"] == 504
    assert payload["credit_line"]
    assert "haversine" in payload["note"]
    assert len(payload["catalogue"]["states"]) == planning_data.catalogue()["counts"]["states"]


def test_locations_search_and_limit_validation(client):
    results = client.get("/locations?q=nashik&limit=3").get_json()["results"]
    assert 0 < len(results) <= 3
    assert client.get("/locations?q=nashik&limit=0").status_code == 400
    assert client.get("/locations?q=nashik&limit=99").status_code == 400
    assert client.get("/locations?q=nashik&limit=many").status_code == 400
    bad = client.get("/locations?q=nashik&limit=0").get_json()
    assert "limit" in bad["fields"]


# ---------------------------------------------------------------------------
# POST /predict-route in hierarchy mode
# ---------------------------------------------------------------------------

def _village_pair():
    rows = planning_data.dataset()
    first = rows[0]
    second = next(r for r in rows if r["district"] != first["district"])
    return first, second


def _nested(row):
    return {"state": row["state"], "district": row["district"],
            "taluka": row["taluka"], "village": row["village"]}


def test_a_village_to_village_corridor_is_priced_end_to_end(client):
    origin, destination = _village_pair()
    body = {"start": _nested(origin), "end": _nested(destination),
            "road_type": "national_highway_4"}
    response = client.post("/predict-route", json=body)
    assert response.status_code == 200
    result = response.get_json()

    assert result["location_levels"] == ["village", "village"]
    assert origin["village"] in result["start"] and destination["village"] in result["end"]
    expected = planning_data.between(
        planning_data.point(**_nested(origin)), planning_data.point(**_nested(destination)))
    assert result["distance_km"] == pytest.approx(expected, abs=0.05)
    assert result["calibrated_km"] == pytest.approx(result["distance_km"])
    assert result["straight_km"] < result["distance_km"]
    assert "haversine" in result["distance_basis"]

    # the brief's contract, unchanged in shape from city mode
    assert set(result["brief"]) == {"distance", "budget", "time_months", "risk",
                                    "risk_percent"}
    assert result["brief"]["distance"] == pytest.approx(result["distance_km"])
    assert result["brief"]["budget"] == result["total_rupees"]

    # the same corridor through the planning formula, beside it and different
    plan = result["planning"]
    assert plan["brief"]["distance"] == pytest.approx(result["distance_km"])
    assert sum(plan["brief"]["breakdown"].values()) == plan["total_rupees"]
    assert plan["total_rupees"] != result["total_rupees"]


def test_the_model_cross_check_still_gets_a_state_the_classifier_has_seen(client):
    origin, destination = _village_pair()
    result = client.post("/predict-route", json={"start": _nested(origin),
                                                 "end": _nested(destination)}).get_json()
    check = result["model_check"]
    assert check["available"] is True, check.get("reason")
    assert result["model_input"]["state"] == origin["state"]
    assert result["model_input"]["district"] == origin["district"]


def test_a_district_headquarters_is_an_acceptable_end(client):
    origin, _ = _village_pair()
    result = client.post("/predict-route", json={
        "start": _nested(origin),
        "end": {"state": "Maharashtra", "district": "Nashik"}}).get_json()
    assert result["location_levels"] == ["village", "district"]
    assert "headquarters" in result["end"]


def test_mixing_a_city_name_with_a_hierarchy_object_is_refused(client):
    origin, _ = _village_pair()
    response = client.post("/predict-route", json={"start": "Pune",
                                                   "end": _nested(origin)})
    assert response.status_code == 400
    fields = response.get_json()["fields"]
    assert fields["start"] == "mixed location forms"
    assert fields["end"] == "mixed location forms"


def test_the_same_place_at_both_ends_is_a_400_not_a_500(client):
    origin, _ = _village_pair()
    response = client.post("/predict-route", json={"start": _nested(origin),
                                                   "end": _nested(origin)})
    assert response.status_code == 400
    assert response.get_json()["fields"]["end"] == "same point"


def test_an_unknown_village_names_which_end_it_belongs_to(client):
    origin, destination = _village_pair()
    bad = dict(_nested(origin))
    bad["village"] = "No Such Gaon"
    response = client.post("/predict-route", json={"start": bad,
                                                   "end": _nested(destination)})
    assert response.status_code == 400
    assert "start_village" in response.get_json()["fields"]


def test_city_mode_still_answers_exactly_as_before(client):
    result = client.post("/predict-route", json={"start": "Pune",
                                                 "end": "Nashik"}).get_json()
    assert set(result["brief"]) == {"distance", "budget", "time_months", "risk",
                                    "risk_percent"}
    assert result["start"] == "Pune" and result["end"] == "Nashik"
    assert "start_point" not in result
    assert result["planning"]["brief"]["distance"] == pytest.approx(result["distance_km"])
    assert [row["key"] for row in result["planning"]["materials"]] == \
        list(planning.MATERIAL_SPLIT_PCT)


def test_the_take_off_and_the_formula_are_two_measurements_of_one_road(client):
    # They are shown side by side, so a test should catch them ever being merged into
    # one number -- which would mean one of the two engines stopped being used.
    origin, destination = _village_pair()
    result = client.post("/predict-route", json={"start": _nested(origin),
                                                 "end": _nested(destination),
                                                 "road_type": "rural_two_lane"}).get_json()
    assert result["total_crore"] != result["planning"]["total_crore"]
    assert result["time_months"] != result["planned_duration_months"]
    assert result["risk_percent"]["band"] != result["planning"]["risk"]["band"] or \
        result["risk_percent"]["percent"] != result["planning"]["total_crore"]
    assert "not a trained-model output" in result["planning"]["note"] or \
        "top-down" in result["planning"]["note"].lower()


def test_the_briefs_three_base_rates_are_what_the_page_claims(client):
    cities = client.get("/route/cities").get_json()["catalogue"]
    per_km = planning.BASE_COST_PER_KM_CRORE
    assert per_km["rural"] == 1.0 and per_km["highway"] == 2.0 and per_km["expressway"] == 5.0
    assert cities["default_cost_per_km_crore"] > 0
    assert statistics.median(per_km.values()) == 2.0
