"""Tests for the building estimator: the per-square-foot engine, its HTTP route and
the page that hosts it.

A building is not a road and not a sanctioned scheme, so this module deliberately makes
no model call. That shapes what the tests can assert: not accuracy but consistency --
that the printed multiplication really produces the printed rupees, that the seven work
sections add back to the total to the rupee, that the months move the way a site
programme moves, and that the two systems stay separate (this route never touches the
trained classifier, and the road routes never touch this one).

Nothing here is mocked.
"""

import pytest

import model.building as building
import model.route_engine as routes

RUPEE = "₹"


# ---------------------------------------------------------------------------
# The rate tables
# ---------------------------------------------------------------------------

def test_the_briefs_two_named_rates_are_the_ones_in_the_code():
    assert building.BASE_COST_PER_SQFT["rural"] == 1500.0
    assert building.BASE_COST_PER_SQFT["urban"] == 2500.0
    assert building.FLOOR_FACTORS == {1: 1.00, 2: 1.90, 3: 2.65}
    assert building.QUALITY_FACTORS == {"low": 0.85, "medium": 1.00, "high": 1.25}


def test_the_work_split_adds_to_exactly_one_hundred():
    # A pie that disagrees with the card above it is the mistake this project has
    # already been caught making once, so the table itself is pinned.
    assert sum(building.WORK_SPLIT_PCT.values()) == 100.0
    assert set(building.WORK_SPLIT_PCT) == set(building.WORK_LABELS)


def test_a_second_storey_costs_less_than_a_repeat_of_the_first():
    assert 1.0 < building.FLOOR_FACTORS[2] < 2.0
    assert building.FLOOR_FACTORS[3] > building.FLOOR_FACTORS[2]


def test_catalogue_publishes_every_table_the_page_prints():
    catalogue = building.catalogue()
    assert [row["key"] for row in catalogue["building_types"]] == \
        list(building.BUILDING_TYPES)
    assert {row["key"] for row in catalogue["location_types"]} == \
        set(building.BASE_COST_PER_SQFT)
    assert [row["floors"] for row in catalogue["floor_factors"]] == [1, 2, 3]
    assert sum(row["percent"] for row in catalogue["work_split"]) == 100.0
    assert catalogue["area_limits"] == {"min": building.MIN_AREA, "max": building.MAX_AREA}
    assert catalogue["mobilisation_days"] == building.MOBILISATION_DAYS


# ---------------------------------------------------------------------------
# The multiplication
# ---------------------------------------------------------------------------

def test_estimate_is_the_formula_with_no_hidden_terms():
    result = building.estimate(building_type="gram_panchayat", area_sqft=1200, floors=1,
                               quality="medium", location_type="rural")
    rupees = 1200 * 1500.0 * 1.00 * 1.00 * 1.00 * 0.90
    assert result["cost_rupees"] == int(round(rupees))
    assert result["cost_lakh"] == pytest.approx(rupees / 100000.0)
    assert result["cost_crore"] == pytest.approx(rupees / 10000000.0, abs=0.005)
    assert result["effective_cost_per_sqft"] == pytest.approx(1350.0)
    assert result["basis"].startswith("1200 sq ft x " + RUPEE + " 1500 per sq ft (rural)")
    assert RUPEE in result["cost_formatted"]


def test_every_factor_moves_the_money_in_the_documented_direction():
    base = building.estimate(area_sqft=2000, location_type="rural")
    assert building.estimate(area_sqft=2000, location_type="urban")["cost_rupees"] > \
        base["cost_rupees"]
    assert building.estimate(area_sqft=2000, location_type="remote")["cost_rupees"] > \
        base["cost_rupees"]
    for floors in (2, 3):
        assert building.estimate(area_sqft=2000, floors=floors)["cost_rupees"] > \
            building.estimate(area_sqft=2000, floors=floors - 1)["cost_rupees"]
    assert building.estimate(area_sqft=2000, quality="high")["cost_rupees"] > \
        building.estimate(area_sqft=2000, quality="medium")["cost_rupees"] > \
        building.estimate(area_sqft=2000, quality="low")["cost_rupees"]
    for kind in ("hospital", "school"):
        assert building.estimate(area_sqft=2000, building_type=kind)["cost_rupees"] > \
            building.estimate(area_sqft=2000, building_type="gram_panchayat")["cost_rupees"]


def test_the_built_up_area_is_the_footprint_times_the_floor_factor():
    result = building.estimate(area_sqft=1000, floors=3)
    assert result["area_sqft"] == 1000.0
    assert result["built_up_sqft"] == pytest.approx(1000 * building.FLOOR_FACTORS[3])


def test_the_seven_work_lines_add_back_to_the_total_to_the_rupee():
    for kwargs in [{"area_sqft": 600}, {"area_sqft": 4000, "location_type": "urban"},
                   {"area_sqft": 6000, "floors": 2, "quality": "high",
                    "building_type": "hospital"},
                   {"area_sqft": 12345, "floors": 3, "quality": "low",
                    "location_type": "remote", "building_type": "school"}]:
        result = building.estimate(**kwargs)
        lines = result["breakdown"]
        assert len(lines) == len(building.WORK_SPLIT_PCT)
        assert sum(row["rupees"] for row in lines.values()) == result["cost_rupees"]
        assert sum(row["lakh"] for row in lines.values()) == pytest.approx(
            result["cost_lakh"], abs=0.011)
        assert sum(row["percent"] for row in lines.values()) == 100.0


def test_the_briefs_three_key_contract_holds():
    result = building.estimate(area_sqft=2400, floors=2, building_type="school")
    assert set(result["brief"]) == {"cost", "time_months", "breakdown"}
    assert result["brief"]["cost"] == result["cost_rupees"]
    assert result["brief"]["time_months"] == result["time_months"]
    assert sum(result["brief"]["breakdown"].values()) == result["brief"]["cost"]


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def test_time_is_area_over_a_crew_rate_plus_fixed_site_days():
    time = building.construction_time(4000, "medium", 1)
    expected_days = 4000 / building.CONSTRUCTION_SQFT_PER_DAY["medium"] \
        + building.MOBILISATION_DAYS
    assert time["working_days"] == pytest.approx(expected_days, abs=0.05)
    assert time["months"] == pytest.approx(expected_days / routes.DAYS_PER_MONTH, abs=0.05)
    assert "÷" in time["basis"] and "x 1.00 for 1 storey" in time["basis"]


def test_the_fixed_site_days_dominate_a_tiny_building():
    # 100 sq ft is two days of laying; the 120 fixed days of site set-up around it are
    # what makes the programme, so the model must not print a one-month handover.
    time = building.construction_time(100, "medium", 1)
    assert time["working_days"] > building.MOBILISATION_DAYS
    assert time["months"] >= building.MIN_MOBILISATION_MONTHS
    assert building.estimate(area_sqft=100)["time_months"] >= 3


def test_bigger_and_better_both_take_longer():
    small = building.estimate(area_sqft=1500)
    big = building.estimate(area_sqft=15000)
    assert big["time_months"] > small["time_months"]
    assert building.estimate(area_sqft=5000, quality="high")["time_months"] > \
        building.estimate(area_sqft=5000, quality="low")["time_months"]
    assert building.estimate(area_sqft=5000, floors=3)["time_months"] > \
        building.estimate(area_sqft=5000, floors=1)["time_months"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, field", [
    ({"building_type": "stadium"}, "building_type"),
    ({"location_type": "suburban"}, "location_type"),
    ({"quality": "luxury"}, "quality"),
    ({"floors": 7}, "floors"),
    ({"floors": 0}, "floors"),
    ({"floors": "two"}, "floors"),
    ({"area_sqft": 40}, "area_sqft"),
    ({"area_sqft": 9000000}, "area_sqft"),
    ({"area_sqft": "large"}, "area_sqft"),
])
def test_a_bad_building_request_names_its_field(kwargs, field):
    with pytest.raises(routes.RouteError) as excinfo:
        building.estimate(**kwargs)
    assert field in excinfo.value.fields


def test_the_location_is_context_and_is_still_checked():
    # The site does not change the rupees, but a typo must not silently become a
    # building in a district that does not exist.
    result = building.estimate(state="Maharashtra", district="Pune", area_sqft=1200)
    assert result["location"] == {"state": "Maharashtra", "district": "Pune",
                                  "taluka": None, "village": None}
    # The engine itself does not police the names -- it has no copy of the hierarchy.
    # /predict-building checks them against /locations before calling it, which the
    # 400 tests below cover; here the location must simply survive unchanged.
    echoed = building.estimate(state="Atlantis", district="Nowhere", area_sqft=1200)
    assert echoed["location"]["district"] == "Nowhere"
    assert echoed["cost_rupees"] == building.estimate(area_sqft=1200)["cost_rupees"]


# ---------------------------------------------------------------------------
# Honesty: this page makes no model call
# ---------------------------------------------------------------------------

def test_the_answer_says_it_is_a_rate_estimate_not_a_model_output():
    result = building.estimate(area_sqft=1200)
    assert "not a trained-model output" in result["note"]
    assert "no land" in result["note"]
    # no model vocabulary leaks into the response, so nothing here can be mistaken for
    # a probability the classifier produced
    assert "probability" not in result and "risk_percent" not in result
    assert "model" not in result["basis"].lower()


# ---------------------------------------------------------------------------
# POST /predict-building
# ---------------------------------------------------------------------------

def test_the_route_returns_the_engines_answer_verbatim(client):
    body = {"building_type": "hospital", "area_sqft": 6000, "floors": 2,
            "quality": "high", "location_type": "urban",
            "state": "Maharashtra", "district": "Pune"}
    response = client.post("/predict-building", json=body)
    assert response.status_code == 200
    result = response.get_json()
    direct = building.estimate(building_type="hospital", area_sqft=6000, floors=2,
                               quality="high", location_type="urban")
    for key in ("cost_rupees", "cost_lakh", "cost_crore", "time_months", "basis",
                "effective_cost_per_sqft"):
        assert result[key] == direct[key]
    assert result["credit_line"]
    assert sum(row["rupees"] for row in result["breakdown"].values()) == result["cost_rupees"]


def test_the_form_defaults_match_the_engine_defaults(client):
    from_api = client.post("/predict-building", json={"area_sqft": 1200}).get_json()
    from_engine = building.estimate(area_sqft=1200)
    assert from_api["building_type"] == from_engine["building_type"] == "gram_panchayat"
    assert from_api["quality"] == "medium" and from_api["location_type"] == "rural"
    assert from_api["floors"] == 1
    assert from_api["cost_rupees"] == from_engine["cost_rupees"]


@pytest.mark.parametrize("body, field", [
    ({"building_type": "school"}, "area_sqft"),                       # blank area
    ({"area_sqft": "abc"}, "area_sqft"),                              # not a number
    ({"area_sqft": 5}, "area_sqft"),                                  # below the floor
    ({"area_sqft": 1200, "building_type": "mall"}, "building_type"),
    ({"area_sqft": 1200, "quality": "premium"}, "quality"),
    ({"area_sqft": 1200, "floors": 9}, "floors"),
    ({"area_sqft": 1200, "location_type": "metro"}, "location_type"),
    ({"area_sqft": 1200, "state": "Maharashtra", "district": "Nowhere"}, "district"),
    ({"area_sqft": 1200, "state": "Atlantis", "district": "Pune"}, "state"),
    ({"area_sqft": 1200, "state": "", "district": "Pune"}, "state"),
])
def test_a_bad_building_request_is_a_400_naming_the_field(client, body, field):
    response = client.post("/predict-building", json=body)
    assert response.status_code == 400
    assert field in response.get_json()["fields"]


def test_a_taluka_that_is_not_in_the_district_is_refused(client):
    response = client.post("/predict-building", json={
        "area_sqft": 1200, "state": "Maharashtra", "district": "Pune",
        "taluka": "Such A Taluka"})
    assert response.status_code == 400
    assert "taluka" in response.get_json()["fields"]


def test_the_building_route_accepts_form_data_as_well_as_json(client):
    response = client.post("/predict-building", data={"area_sqft": "2400",
                                                      "floors": "2",
                                                      "location_type": "urban"})
    assert response.status_code == 200
    assert response.get_json()["built_up_sqft"] == pytest.approx(
        2400 * building.FLOOR_FACTORS[2])


def test_get_on_the_building_route_is_not_allowed(client):
    assert client.get("/predict-building").status_code == 405


# ---------------------------------------------------------------------------
# The two systems stay separate
# ---------------------------------------------------------------------------

def test_the_building_page_renders_and_links_back_to_the_dashboard(client):
    response = client.get("/building")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'id="building-form"' in html
    assert "js/building.js" in html
    assert "js/app.js" not in html                  # its boot() is index-only
    assert "js/charts.js" in html
    assert "Vishal Raghunath Chatur" in html
    assert "Tapase Miss" in html and "DBATU" in html
    assert "/#routes" in html                       # the way back


def test_the_dashboard_links_to_the_building_page_without_losing_its_own(client):
    html = client.get("/").get_data(as_text=True)
    assert "/building" in html
    for anchor in ("#dataset", "#predictor", "#performance", "#dashboard", "#routes",
                   "#knn", "#about", "#team"):
        assert anchor in html


def test_the_building_module_does_not_touch_the_road_engines():
    # Importing and running the estimator must leave the shared catalogues alone: a
    # regression here would silently re-price every corridor on the dashboard.
    before_soil = dict(routes.SOIL_FACTORS)
    before_cities = len(routes.CITIES)
    building.estimate(area_sqft=4000, floors=2, quality="high", location_type="urban",
                      building_type="hospital")
    assert routes.SOIL_FACTORS == before_soil
    assert len(routes.CITIES) == before_cities
    assert set(building.WORK_SPLIT_PCT).isdisjoint(set(routes.SOIL_FACTORS))


def test_a_road_estimate_still_carries_no_building_fields(client):
    result = client.post("/predict-route", json={"start": "Pune",
                                                 "end": "Nashik"}).get_json()
    for key in ("area_sqft", "floors", "quality", "cost_lakh", "breakdown"):
        assert key not in result, key
