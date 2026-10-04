"""Tests for the route planner: the engineering engine and its three HTTP routes.

The engine has no training data behind it, so these tests do not ask "is this number
right?" -- they ask whether the number is the one the stated arithmetic produces:

  * the corridor coefficient really reproduces the published Indian road distances;
  * the components sum to the works cost, works + overheads sum to the budget;
  * a forced Rupees-per-km rate wins, and the unforced take-off survives beside it;
  * terrain, soil, width and cross-section move the money in the documented direction;
  * completion months are the distance divided by the km-per-day speed band the brief
    names -- and are kept separate from the contract programme the model is shown;
  * every city is a district the trained model has already seen, so the cross-check
    can never hit an unknown category;
  * bad requests become field-level 400s, never a 500.

Nothing here is mocked.
"""

import json
import math
import statistics

import pytest

import model.route_engine as routes

RUPEE = "₹"


# ---------------------------------------------------------------------------
# Great-circle distance and the calibration
# ---------------------------------------------------------------------------

def test_great_circle_is_symmetric_and_zero_on_itself():
    mumbai = routes.CITIES["Mumbai City"][:2]
    pune = routes.CITIES["Pune"][:2]
    assert routes.great_circle_km(mumbai, mumbai) == 0.0
    assert routes.great_circle_km(mumbai, pune) == pytest.approx(
        routes.great_circle_km(pune, mumbai))


def test_great_circle_matches_a_known_pair():
    # Delhi to Mumbai is about 1,160 km straight-line; anything near that shows the
    # formula has the right radius and the right argument order.
    km = routes.great_circle_km(routes.CITIES["New Delhi"][:2],
                                routes.CITIES["Mumbai City"][:2])
    assert 1000 <= km <= 1250, km


def test_corridor_coefficient_is_the_median_of_the_published_ratios():
    ratios = [published / routes.great_circle_km(routes.CITIES[start][:2],
                                                 routes.CITIES[end][:2])
              for start, end, published in routes.CALIBRATION_ROUTES]
    assert routes.CORRIDOR_FACTOR == pytest.approx(statistics.median(ratios), abs=0.0002)
    assert routes.CORRIDOR_FACTOR == pytest.approx(routes.corridor_factor(), abs=0.0002)
    assert 1.0 < routes.CORRIDOR_FACTOR < 1.6


def test_every_calibration_route_lands_within_16_percent_of_the_published_distance():
    worst, worst_pair = 0.0, None
    for start, end, published in routes.CALIBRATION_ROUTES:
        estimated = routes.route_distance(start, end)["road_km"]
        error = abs(estimated - published) / published * 100.0
        if error > worst:
            worst, worst_pair = error, (start, end, published, estimated)
    # The two worst are Ahmedabad-Vadodara and Pune-Nagpur: corridors that really are
    # near-straight, which one shared coefficient cannot fit at both ends.
    assert worst <= 16.0, worst_pair


def test_road_distance_is_never_shorter_than_the_straight_line():
    for start, end, _published in routes.CALIBRATION_ROUTES:
        measured = routes.route_distance(start, end)
        assert measured["road_km"] >= measured["straight_km"]
        assert "corridor coefficient" in measured["basis"]


def test_geodesic_path_runs_between_the_two_cities():
    path = routes.geodesic_path("Nashik", "Pune")
    assert len(path) == 25
    assert path[0] == pytest.approx(list(routes.CITIES["Nashik"][:2]), abs=1e-4)
    assert path[-1] == pytest.approx(list(routes.CITIES["Pune"][:2]), abs=1e-4)


def test_geodesic_path_stays_inside_india():
    for lat, lon in routes.geodesic_path("New Delhi", "Chennai"):
        assert 8.0 <= lat <= 37.5 and 68.0 <= lon <= 97.5


# ---------------------------------------------------------------------------
# The city catalogue and its link to the trained vocabulary
# ---------------------------------------------------------------------------

def test_catalogue_cities_are_districts_the_model_has_seen(meta_json):
    districts = set(meta_json["vocabularies"]["district"])
    states = set(meta_json["vocabularies"]["state"])
    unseen = [name for name in routes.CITIES if name not in districts]
    assert unseen == [], f"cities the one-hot encoder has never seen: {unseen}"
    wrong_state = [name for name, row in routes.CITIES.items() if row[2] not in states]
    assert wrong_state == []


def test_catalogue_payload_describes_the_engine(client):
    catalogue = client.get("/route/cities").get_json()["catalogue"]
    assert len(catalogue["cities"]) == len(routes.CITIES)
    assert {row["terrain"] for row in catalogue["cities"]} <= set(routes.TERRAIN_FACTORS)
    assert {row["soil"] for row in catalogue["cities"]} <= set(routes.SOIL_FACTORS)
    assert catalogue["corridor_coefficient"] == routes.CORRIDOR_FACTOR
    assert {row["key"] for row in catalogue["road_types"]} == set(routes.ROAD_TYPES)
    assert {row["key"] for row in catalogue["terrains"]} == set(routes.TERRAIN_FACTORS)
    assert {row["key"] for row in catalogue["soils"]} == set(routes.SOIL_FACTORS)
    for pair in catalogue["featured_routes"]:
        assert pair["start"] in routes.CITIES and pair["end"] in routes.CITIES
    assert catalogue["default_cost_per_km_crore"] == routes.DEFAULT_COST_PER_KM_CRORE
    assert catalogue["overheads_pct"] == routes.OVERHEADS_PCT
    assert {row["band"] for row in catalogue["construction_speeds"]} == set(routes.SPEED_KM_PER_DAY)
    assert "urban" in {row["key"] for row in catalogue["terrains"]}
    assert catalogue["calibration_routes"] and catalogue["centre"]


def test_city_positions_are_inscribed_lat_lon():
    for name, (lat, lon, state, terrain, soil) in routes.CITIES.items():
        assert 8.0 < lat < 37.5 and 68.0 < lon < 97.5, name
        assert state
        assert terrain in routes.TERRAIN_FACTORS and soil in routes.SOIL_FACTORS


# ---------------------------------------------------------------------------
# Factors and rates the brief fixed
# ---------------------------------------------------------------------------

def test_ghat_uplift_is_the_briefs_twenty_to_thirty_percent():
    assert routes.TERRAIN_FACTORS["plain"]["factor"] == 1.0
    assert 1.20 <= routes.TERRAIN_FACTORS["ghat"]["factor"] <= 1.30
    assert routes.GHAT_UPLIFT_RANGE == "20-30 %"
    assert routes.DEFAULT_COST_PER_KM_CRORE == 2.0


def test_urban_terrain_is_the_briefs_fifteen_percent():
    assert routes.TERRAIN_FACTORS["urban"]["factor"] == 1.15
    plain = routes.estimate("New Delhi", "Jaipur", terrain="plain")
    urban = routes.estimate("New Delhi", "Jaipur", terrain="urban")
    assert urban["total_lakh"] == pytest.approx(
        plain["total_lakh"] * 1.15, rel=1e-3)
    assert urban["terrain_impact_lakh"] > 0
    # Urban sits between plain and hilly on the risk scale, not below both.
    assert (routes.RISK_WEIGHTS["plain"] < routes.RISK_WEIGHTS["urban"]
            < routes.RISK_WEIGHTS["hilly"])


def test_unit_rates_match_the_2025_26_numbers_quoted_in_the_brief():
    assert routes.UNIT_RATES["cement_bag"] == 350
    assert routes.UNIT_RATES["steel_tonne"] == 60000
    assert routes.UNIT_RATES["bitumen_tonne"] == 50000
    assert 500 <= routes.UNIT_RATES["labour_day"] <= 800
    assert all(value > 0 for value in routes.UNIT_RATES.values())


def test_road_types_widen_with_lanes():
    widths = {key: spec["width_m"] for key, spec in routes.ROAD_TYPES.items()}
    assert widths["state_highway"] < widths["rural_two_lane"]
    assert widths["rural_two_lane"] < widths["national_highway_4"]
    assert widths["national_highway_4"] < widths["expressway_6"]


# ---------------------------------------------------------------------------
# The take-off arithmetic
# ---------------------------------------------------------------------------

def test_components_add_up_to_works_and_works_plus_overheads_to_the_budget():
    result = routes.estimate("Nashik", "Pune")
    assert sum(row["lakh"] for row in result["components"]) == pytest.approx(
        result["works_lakh"], abs=0.05)
    assert result["overheads_lakh"] == pytest.approx(
        result["works_lakh"] * routes.OVERHEADS_PCT / 100.0, abs=0.05)
    assert result["works_lakh"] + result["overheads_lakh"] == pytest.approx(
        result["total_lakh"], abs=0.05)
    assert result["total_crore"] == pytest.approx(result["total_lakh"] / 100.0, abs=0.01)
    assert result["total_rupees"] == int(round(result["total_lakh"] * 100000.0))


def test_per_km_figure_is_the_budget_divided_by_the_distance_it_prints():
    result = routes.estimate("Mumbai City", "Pune")
    assert result["derived_cost_per_km_lakh"] == pytest.approx(
        result["total_lakh"] / result["distance_km"], abs=0.02)
    assert result["derived_cost_per_km_crore"] == pytest.approx(
        result["total_crore"] / result["distance_km"], abs=0.005)


def test_budget_scales_linearly_with_distance_when_nothing_else_moves():
    args = dict(road_type="rural_two_lane", terrain="plain", soil="alluvial")
    short = routes.estimate(distance_km=100, **args)
    double = routes.estimate(distance_km=200, **args)
    # Every line is per km or per m2 of a longer carriageway, so doubling the length
    # doubles the money -- the formula the brief asked for, minus the hard-coded rate.
    assert double["total_lakh"] == pytest.approx(short["total_lakh"] * 2.0, rel=1e-4)
    assert double["quantities"]["carriageway_m2"] == pytest.approx(
        short["quantities"]["carriageway_m2"] * 2.0, rel=1e-9)


def test_a_wider_carriageway_costs_more_than_the_road_type_default():
    default = routes.estimate("Nashik", "Pune", road_type="rural_two_lane")
    widened = routes.estimate("Nashik", "Pune", road_type="rural_two_lane", width_m=14.0)
    assert widened["width_m"] == 14.0
    assert widened["total_lakh"] > default["total_lakh"]


def test_heavier_cross_sections_cost_more_in_order():
    totals = [routes.estimate("Nashik", "Pune", road_type=key)["total_lakh"]
              for key in ("state_highway", "rural_two_lane", "national_highway_4",
                          "expressway_6")]
    assert totals == sorted(totals)
    assert totals[0] < totals[-1]


def test_ghat_multiplies_the_unmultiplied_works():
    plain = routes.estimate("New Delhi", "Jaipur")
    ghat = routes.estimate("New Delhi", "Jaipur", terrain="ghat")
    assert plain["terrain_factor"] == 1.0 and plain["terrain_impact_lakh"] == 0.0
    assert ghat["total_lakh"] == pytest.approx(
        plain["total_lakh"] * routes.TERRAIN_FACTORS["ghat"]["factor"], rel=1e-3)
    assert ghat["terrain_impact_lakh"] > 0


def test_site_uplift_is_split_between_terrain_and_soil():
    result = routes.estimate("Nashik", "Pune")
    assert result["combined_factor"] == pytest.approx(
        result["terrain_factor"] * result["soil_factor"], abs=0.001)
    total = result["total_lakh"]
    # Closed form of the engine's split: the terrain uplift is charged before the soil
    # multiplier is applied on top of it, so its share of the printed budget is
    # total x (terrainFactor-1) / (terrainFactor x soilFactor), and soil's is
    # total x (soilFactor-1) / soilFactor.
    assert result["terrain_impact_lakh"] == pytest.approx(
        total * (result["terrain_factor"] - 1.0) / result["combined_factor"], rel=0.01)
    assert result["soil_impact_lakh"] == pytest.approx(
        total * (result["soil_factor"] - 1.0) / result["soil_factor"], rel=0.01)
    assert total - result["terrain_impact_lakh"] - result["soil_impact_lakh"] > 0
    # Applying them together costs more than either alone, which is why the two
    # impacts do not simply add up to the percentage sum of the factors.
    combined = routes.estimate("Nashik", "Pune")
    only_terrain = routes.estimate("Nashik", "Pune", soil="alluvial")
    assert combined["total_lakh"] > only_terrain["total_lakh"]


def test_quantities_are_physical_and_in_the_right_proportions():
    result = routes.estimate("Nashik", "Pune")
    quantities = result["quantities"]
    assert all(value > 0 for value in quantities.values()), quantities
    assert quantities["carriageway_m2"] == pytest.approx(
        result["width_m"] * result["distance_km"] * 1000.0, rel=1e-6)
    assert quantities["bitumen_tonnes"] < quantities["aggregate_tonnes"]
    assert quantities["cement_bags"] == pytest.approx(
        quantities["carriageway_m2"]
        * routes.ROAD_TYPES[result["road_type"]]["cement_bags_per_m2"], rel=1e-4)


def test_forced_cost_per_km_wins_and_keeps_the_take_off_visible():
    forced = routes.estimate("Nashik", "Pune", cost_per_km_crore=2.0)
    unforced = routes.estimate("Nashik", "Pune")
    assert forced["total_crore"] == pytest.approx(forced["distance_km"] * 2.0, abs=0.01)
    assert forced["derived_cost_per_km_crore"] == 2.0
    assert forced["takeoff_total_crore"] == pytest.approx(unforced["total_crore"], abs=0.01)
    assert forced["takeoff_total_crore"] > forced["total_crore"]
    # The rescaled split still adds up, so the pie chart cannot disagree with the card.
    assert sum(row["lakh"] for row in forced["components"]) == pytest.approx(
        forced["works_lakh"], abs=0.05)
    assert forced["works_lakh"] + forced["overheads_lakh"] == pytest.approx(
        forced["total_lakh"], abs=0.05)
    assert "rescaled" in forced["scaling_note"]
    assert "your assumed" in forced["basis"]
    assert forced["model_input"]["sanctioned_budget_lakh"] == pytest.approx(
        forced["total_lakh"], abs=0.1)


def test_unforced_estimate_does_not_pretend_a_rate_was_assumed():
    assert "assumed_cost_per_km_crore" not in routes.estimate("Nashik", "Pune")
    assert "scaling_note" not in routes.estimate("Nashik", "Pune")


def test_typed_distance_beats_the_calibrated_one_and_both_are_returned():
    result = routes.estimate("Nashik", "Pune", distance_km=210.0)
    assert result["distance_km"] == 210.0
    assert result["calibrated_km"] == routes.route_distance("Nashik", "Pune")["road_km"]
    assert "request" in result["distance_basis"]
    assert result["total_lakh"] > routes.estimate("Nashik", "Pune")["total_lakh"]


def test_distance_is_rounded_once_so_the_printed_arithmetic_closes():
    result = routes.estimate("Nashik", "Pune", distance_km=189.249)
    assert result["distance_km"] == 189.2
    # The per-km rate is the budget divided by the same rounded length the card
    # prints, so a reader with a calculator gets the budget back.
    assert result["derived_cost_per_km_lakh"] * result["distance_km"] == pytest.approx(
        result["total_lakh"], abs=1.0)
    assert routes.estimate("Nashik", "Pune", distance_km=189.249)["total_lakh"] == \
        result["total_lakh"]


def test_planned_duration_stays_inside_the_range_the_schema_accepts():
    # schema.normalise_frame rejects duration_months <= 0 or > 240; a route that
    # handed one of those to the model would break its own cross-check.
    for key in routes.ROAD_TYPES:
        for distance in (5, 120, 900, 3800):
            months = routes.planned_duration(key, distance, "ghat")
            assert 1 <= months <= 240, (key, distance, months)


def test_the_model_input_is_a_record_the_classifier_can_read(meta_json):
    result = routes.estimate("Nashik", "Pune")
    model_input = result["model_input"]
    vocab = meta_json["vocabularies"]
    assert model_input["scheme_name"] in vocab["scheme_name"]
    assert model_input["state"] in vocab["state"]
    assert model_input["district"] in vocab["district"]
    assert 0 < model_input["sanctioned_budget_lakh"] < 1e6
    assert 1 <= model_input["duration_months"] <= 240


# ---------------------------------------------------------------------------
# Completion time: Time = Distance / Construction Speed
# ---------------------------------------------------------------------------

def test_speed_bands_are_the_three_the_brief_names():
    assert routes.SPEED_KM_PER_DAY["highway"] == (3.0, 5.0)
    assert routes.SPEED_KM_PER_DAY["rural"] == (1.0, 2.0)
    assert routes.SPEED_KM_PER_DAY["ghat"] == (0.5, 1.0)
    assert set(routes.ROAD_SPEED_BAND) == set(routes.ROAD_TYPES)
    # Every terrain has to fall in one of the two groups, or it has no time rule.
    assert set(routes.TERRAIN_FACTORS) <= ({"plain", "urban", "coastal"}
                                           | set(routes.HILLY_TERRAINS))


def test_months_are_the_distance_divided_by_the_speed_it_prints():
    time = routes.construction_time(200.0, "rural_two_lane", "plain")
    assert time["band"] == "rural" and time["km_per_day"] == [1.0, 2.0]
    assert time["min_months"] == pytest.approx(200.0 / 2.0 / routes.DAYS_PER_MONTH, abs=0.05)
    assert time["likely_months"] == pytest.approx(200.0 / 1.5 / routes.DAYS_PER_MONTH, abs=0.05)
    assert time["max_months"] == pytest.approx(200.0 / 1.0 / routes.DAYS_PER_MONTH, abs=0.05)
    assert time["likely_days"] == pytest.approx(200.0 / 1.5, abs=0.2)


def test_a_hilly_corridor_drops_to_the_slowest_band_whatever_the_cross_section():
    for road_type in routes.ROAD_TYPES:
        for terrain in routes.HILLY_TERRAINS:
            assert routes.construction_time(120, road_type, terrain)["band"] == "ghat"
    assert routes.construction_time(120, "national_highway_4", "plain")["band"] == "highway"


def test_the_briefs_nashik_pune_ghat_example_comes_out_between_six_and_ten_months():
    result = routes.estimate("Nashik", "Pune", terrain="ghat")
    time = result["construction_time"]
    assert time["min_months"] >= 6.0
    assert 6.0 <= time["likely_months"] <= 10.0
    assert result["time_months"] == int(round(time["likely_months"]))
    assert time["basis"].startswith("%.1f km ÷" % result["distance_km"])


def test_time_band_is_ordered_and_grows_with_distance():
    for road_type in routes.ROAD_TYPES:
        for terrain in routes.TERRAIN_FACTORS:
            time = routes.construction_time(240, road_type, terrain)
            assert time["band"] in routes.SPEED_KM_PER_DAY
            assert 0 < time["min_months"] <= time["likely_months"] <= time["max_months"]
    half = routes.construction_time(100, "rural_two_lane", "plain")
    double = routes.construction_time(200, "rural_two_lane", "plain")
    assert double["likely_months"] == pytest.approx(half["likely_months"] * 2.0, abs=0.2)


def test_urban_working_is_fifteen_percent_slower_on_the_clock_too():
    plain = routes.construction_time(150, "state_highway", "plain")
    urban = routes.construction_time(150, "state_highway", "urban")
    assert urban["km_per_day"] == plain["km_per_day"]
    assert urban["likely_months"] == pytest.approx(plain["likely_months"] * 1.15, abs=0.1)


def test_construction_runtime_is_not_the_contract_programme():
    # Two different questions, both answered: how long the plant takes to lay the
    # road, and the programme a contractor is contracted for. The model is shown the
    # second one, because that is the field the training data carries.
    result = routes.estimate("Nashik", "Pune")
    assert result["time_months"] < result["planned_duration_months"]
    assert result["model_input"]["duration_months"] == result["planned_duration_months"]
    assert "calendar days" in result["construction_time"]["basis"]


def test_the_forced_rupee_rate_does_not_move_the_clock():
    forced = routes.estimate("Nashik", "Pune", cost_per_km_crore=5.0)
    assert forced["time_months"] == routes.estimate("Nashik", "Pune")["time_months"]


# ---------------------------------------------------------------------------
# Risk grade
# ---------------------------------------------------------------------------

def test_risk_grade_is_the_sum_of_the_weights_it_prints():
    risk = routes.estimate("Nashik", "Pune")["risk_percent"]
    assert sum(risk["weights"].values()) == pytest.approx(risk["percent"], abs=0.01)
    assert 5.0 <= risk["percent"] <= 75.0
    assert risk["band"] in ("Low", "Medium", "High")
    assert "not a calibrated model probability" in risk["note"]


def test_risk_band_follows_the_printed_percentage():
    for terrain, soil, road_type in [("plain", "alluvial", "state_highway"),
                                     ("urban", "sandy", "national_highway_4"),
                                     ("coastal", "red", "rural_two_lane"),
                                     ("ghat", "clay", "expressway_6")]:
        risk = routes.route_risk(terrain, soil, road_type, 189.2)
        percent, band = risk["percent"], risk["band"]
        expected = "High" if percent >= 45 else ("Medium" if percent >= 25 else "Low")
        assert band == expected, (terrain, percent, band)


def test_every_terrain_the_cost_model_knows_also_has_a_risk_weight():
    assert set(routes.RISK_WEIGHTS) == set(routes.TERRAIN_FACTORS)


def test_risk_rises_with_terrain_and_with_length():
    ghat = routes.route_risk("ghat", "black_cotton", "rural_two_lane", 189.2)["percent"]
    plain = routes.route_risk("plain", "black_cotton", "rural_two_lane", 189.2)["percent"]
    assert ghat > plain
    short = routes.route_risk("plain", "alluvial", "rural_two_lane", 40)["percent"]
    long_ = routes.route_risk("plain", "alluvial", "rural_two_lane", 400)["percent"]
    assert long_ > short


def test_risk_is_clamped_at_both_ends():
    floor = routes.route_risk("plain", "alluvial", "state_highway", 1)
    assert floor["percent"] >= routes.RISK_MIN
    ceiling = routes.route_risk("ghat", "clay", "expressway_6", 4000)
    assert ceiling["percent"] <= routes.RISK_MAX


# ---------------------------------------------------------------------------
# Engine input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,message_part", [
    ({"start": "Atlantis", "end": "Pune"}, "Unknown start city"),
    ({"start": "Nashik", "end": "Atlantis"}, "Unknown end city"),
    ({"start": "Pune", "end": "Pune"}, "same"),
    ({"start": None, "end": None}, "either a start and end city"),
    ({"start": "Nashik", "end": "Pune", "road_type": "moon_highway"}, "Unknown road type"),
    ({"start": "Nashik", "end": "Pune", "terrain": "lava"}, "Unknown terrain"),
    ({"start": "Nashik", "end": "Pune", "soil": "moon dust"}, "Unknown soil"),
    ({"distance_km": 0}, "between 0.1 and 4000"),
    ({"distance_km": 5000}, "between 0.1 and 4000"),
    ({"start": "Nashik", "end": "Pune", "width_m": 2.0}, "between 3 and 30"),
    ({"start": "Nashik", "end": "Pune", "cost_per_km_crore": 0}, "Cost per km"),
    ({"start": "Nashik", "end": "Pune", "cost_per_km_crore": 500}, "Cost per km"),
])
def test_bad_engine_input_raises_a_field_carrying_route_error(kwargs, message_part):
    with pytest.raises(routes.RouteError) as caught:
        routes.estimate(**kwargs)
    assert message_part.lower() in str(caught.value).lower()
    assert caught.value.fields


def test_route_error_is_a_value_error_so_the_api_edge_can_catch_it():
    assert issubclass(routes.RouteError, ValueError)


# ---------------------------------------------------------------------------
# The comparison series
# ---------------------------------------------------------------------------

def test_sample_routes_prices_every_calibration_corridor():
    rows = routes.sample_routes("national_highway_4")
    assert len(rows) == len(routes.CALIBRATION_ROUTES)
    for row, (start, end, published) in zip(rows, routes.CALIBRATION_ROUTES):
        assert row["label"] == "%s → %s" % (start, end)
        assert row["published_km"] == published
        assert row["distance_km"] > 0 and row["total_crore"] > 0
        assert row["cost_per_km_crore"] == pytest.approx(
            row["total_crore"] / row["distance_km"], abs=0.01)


def test_sample_routes_carry_the_months_the_time_chart_needs():
    rows = routes.sample_routes("rural_two_lane")
    for row in rows:
        engine = routes.estimate(row["start"], row["end"], road_type="rural_two_lane")
        assert row["time_months"] == engine["time_months"]
        assert row["min_time_months"] <= row["time_months"] <= row["max_time_months"]
        assert len(row["km_per_day"]) == 2


def test_sample_routes_follow_the_road_type_they_are_given():
    rural = {row["label"]: row["total_crore"] for row in routes.sample_routes("rural_two_lane")}
    six_lane = {row["label"]: row["total_crore"]
                for row in routes.sample_routes("expressway_6")}
    assert all(six_lane[label] > value for label, value in rural.items())


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

def test_route_cities_endpoint_serves_the_catalogue_and_credits(client):
    response = client.get("/route/cities")
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["catalogue"]["cities"]) == len(routes.CITIES)
    assert payload["credit_line"]
    assert "$" not in json.dumps(payload)


def test_route_cities_is_get_only(client):
    assert client.post("/route/cities").status_code == 405
    assert client.get("/predict-route").status_code == 405


def test_route_samples_endpoint_computes_the_series_live(client):
    response = client.get("/route/samples")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["road_type"] == "rural_two_lane"
    assert len(payload["rows"]) == len(routes.CALIBRATION_ROUTES)
    assert payload["corridor_coefficient"] == routes.CORRIDOR_FACTOR

    wider = client.get("/route/samples?road_type=expressway_6")
    assert wider.status_code == 200
    assert wider.get_json()["rows"][0]["total_crore"] > payload["rows"][0]["total_crore"]

    bad = client.get("/route/samples?road_type=moon_highway")
    assert bad.status_code == 400
    assert bad.get_json()["fields"]["road_type"] == "moon_highway"


def test_predict_route_matches_the_briefs_contract(client):
    response = client.post("/predict-route", json={"start": "Nashik", "end": "Pune"})
    assert response.status_code == 200
    payload = response.get_json()

    # The exact shape the brief asked for: distance, budget, time_months, risk.
    brief = payload["brief"]
    assert set(brief) == {"distance", "budget", "time_months", "risk", "risk_percent"}
    assert brief["distance"] == pytest.approx(payload["distance_km"], abs=0.01)
    assert brief["budget"] == payload["total_rupees"]
    assert isinstance(brief["budget"], int) and brief["budget"] > 1e9
    assert isinstance(brief["time_months"], int) and brief["time_months"] > 0
    assert brief["time_months"] == payload["time_months"]
    assert brief["risk"] in ("Low", "Medium", "High")

    assert len(payload["route_path"]) == 25
    assert payload["start"] == "Nashik" and payload["end"] == "Pune"
    assert payload["states"] == ["Maharashtra", "Maharashtra"]
    assert RUPEE in payload["total_inr"] and "crore" in payload["total_inr"]
    assert "engineering estimate" in payload["basis"]
    assert len(payload["components"]) >= 10


def test_predict_route_payload_has_no_dollar_sign_anywhere(client):
    payload = client.post("/predict-route",
                          json={"start": "Mumbai City", "end": "Pune"}).get_json()
    assert "$" not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.parametrize("body,field", [
    ({}, "start"),
    ({"start": "Nashik"}, "end"),
    ({"start": "Atlantis", "end": "Pune"}, "start"),
    ({"start": "Pune", "end": "Pune"}, "end"),
    ({"start": "Nashik", "end": "Pune", "width_m": 99}, "width_m"),
    ({"start": "Nashik", "end": "Pune", "distance_km": 9000}, "distance_km"),
    ({"start": "Nashik", "end": "Pune", "cost_per_km_crore": -1}, "cost_per_km_crore"),
    ({"start": "Nashik", "end": "Pune", "road_type": "moon_highway"}, "road_type"),
    ({"start": "Nashik", "end": "Pune", "terrain": "lava"}, "terrain"),
    ({"start": "Nashik", "end": "Pune", "soil": "moon dust"}, "soil"),
    ({"start": "Nashik", "end": "Pune", "contractor_type": "Overseas"}, "contractor_type"),
    ({"start": "Nashik", "end": "Pune", "model": "xgboost"}, "model"),
    ({"start": "Nashik", "end": "Pune", "width_m": "wide"}, "width_m"),
])
def test_predict_route_answers_400_with_the_offending_field(client, body, field):
    response = client.post("/predict-route", json=body)
    assert response.status_code == 400
    assert field in response.get_json()["fields"]


def test_predict_route_cross_checks_with_the_real_trained_model(client, meta_json):
    payload = client.post("/predict-route",
                          json={"start": "Nashik", "end": "Pune"}).get_json()
    check = payload["model_check"]
    assert check["available"] is True
    assert check["model"] in ("Random Forest", "K-Nearest Neighbours")
    assert check["verdict"] in ("OVER BUDGET", "WITHIN BUDGET")
    assert 0.0 <= check["risk_pct"] <= 100.0
    assert check["votes"]["yes"] + check["votes"]["no"] == check["votes"]["total"]
    assert check["expected_actual_lakh"] > 0
    assert RUPEE in check["expected_actual_inr"]

    # A ₹484 crore road is larger than any project in the training data, and the
    # payload must say so instead of quietly extrapolating.
    trained_max = float(meta_json["vocabularies"]["budget_max_lakh"])
    assert check["budget_max_lakh_in_training"] == pytest.approx(trained_max, abs=0.1)
    assert payload["total_lakh"] > trained_max
    assert check["within_training_range"] is False
    assert "extrapolat" in check["note"].lower()


def test_predict_route_model_choice_changes_the_cross_check(client):
    rf = client.post("/predict-route", json={"start": "Nashik", "end": "Pune",
                                             "model": "rf"}).get_json()["model_check"]
    knn = client.post("/predict-route", json={"start": "Nashik", "end": "Pune",
                                              "model": "knn"}).get_json()["model_check"]
    assert rf["model"] == "Random Forest"
    assert knn["model"] == "K-Nearest Neighbours"
    assert knn["votes"]["total"] == 5


def test_predict_route_honours_a_forced_rate_and_a_dpr_length(client):
    payload = client.post("/predict-route", json={
        "start": "Nashik", "end": "Pune", "distance_km": 199,
        "cost_per_km_crore": 2.0}).get_json()
    assert payload["distance_km"] == 199.0
    assert payload["brief"]["budget"] == int(round(199.0 * 2.0 * 1e7))
    assert payload["calibrated_km"] == routes.route_distance("Nashik", "Pune")["road_km"]
    assert "forced to your assumed" in payload["scaling_note"]


def test_route_prediction_is_counted_as_a_served_prediction(client):
    """The cross-check is a real model call, so it increments the served counter.
    This test pins that down so the dashboard cannot silently over- or under-count."""
    before = client.get("/dashboard").get_json()["predictions_served"]
    client.post("/predict-route", json={"start": "Nashik", "end": "Pune"})
    after = client.get("/dashboard").get_json()["predictions_served"]
    assert after == before + 1


@pytest.mark.parametrize("start,end", [("Nashik", "Pune"),
                                       ("Mumbai City", "Pune"),
                                       ("New Delhi", "Jaipur")])
def test_every_example_route_in_the_brief_estimates(client, start, end):
    payload = client.post("/predict-route", json={"start": start, "end": end}).get_json()
    assert payload["distance_km"] > 0 and payload["total_crore"] > 0
    assert payload["brief"]["risk"] in ("Low", "Medium", "High")


def test_index_page_ships_the_map_and_the_route_section(client):
    html = client.get("/").get_data(as_text=True)
    assert 'id="routes"' in html
    assert "leaflet@1.9.4" in html
    assert "openstreetmap" in html.lower()
    assert "js/routes.js" in html
    assert "chart-route-split" in html and "chart-route-budget" in html
    assert "chart-route-time" in html and 'id="rt-out-time"' in html
