"""Shared pytest fixtures.

Adds the project root to ``sys.path`` so ``import app`` works from any directory,
and exposes a Flask test client that talks to the real application — no mocks and
no fake metrics anywhere in this suite.

The models are trained once per session by ``app.bootstrap()`` (which also
generates ``data/india_projects_2025_26.csv`` if it is missing), so every number a
test asserts is a number the shipped app actually produces.
"""

import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config                                    # noqa: E402
from data_layer import schema                     # noqa: E402
import app as app_module                          # noqa: E402
from app import app as flask_app                  # noqa: E402
import model.train as trainer                     # noqa: E402


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def app():
    flask_app.config.update(TESTING=True)
    return flask_app


@pytest.fixture(scope="session")
def service(app):
    """The loaded ModelService. bootstrap() is idempotent, so this is safe to call
    even when the tests are run after the real server has already trained."""
    app_module.bootstrap()
    return app_module.SERVICE


@pytest.fixture(scope="session")
def client(service):
    return flask_app.test_client()


@pytest.fixture(scope="session")
def meta_json(client):
    """The /meta payload, fetched once for the whole session."""
    response = client.get("/meta")
    assert response.status_code == 200
    return response.get_json()


@pytest.fixture(scope="session")
def metrics_json(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    return response.get_json()


@pytest.fixture(scope="session")
def dashboard_json(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    return response.get_json()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dataset_path():
    return config.LOCAL_DATASET


@pytest.fixture(scope="session")
def dataset_frame(dataset_path):
    """The real CSV, read through the same normalisation the app uses."""
    raw = pd.read_csv(dataset_path)
    return schema.add_derived_columns(schema.normalise_frame(raw))


@pytest.fixture(scope="session")
def sample_frame():
    raw = pd.read_csv(config.SAMPLE_DATASET)
    return schema.add_derived_columns(schema.normalise_frame(raw))


@pytest.fixture
def valid_payload(meta_json):
    """A /predict body assembled from the real vocabularies, so it cannot drift from
    the dataset the way a hard-coded example would."""
    vocab = meta_json["vocabularies"]
    assert "Maharashtra" in vocab["state"]
    assert "Pune" in vocab["district_states"], "Pune must be a district the model has seen"
    assert "Maharashtra" in vocab["district_states"]["Pune"]
    return {
        "scheme_name": vocab["scheme_name"][0],
        "state": "Maharashtra",
        "district": "Pune",
        "contractor_type": "Govt",
        "sanctioned_budget_lakh": 250,
        "duration_months": 18,
        "k": meta_json["default_k"],
        "model": "auto",
    }


@pytest.fixture
def training_row(service):
    """Build a /predict body from a row of the training frame the service holds right
    now, with every value cast to a plain JSON type and the true label alongside it.

    A factory rather than a dict, because a test that retrains needs a row of the
    *new* training frame, and only the service knows which one that is."""
    def build(position=0):
        row = service.train_frame.iloc[position]
        return {
            "scheme_name": str(row["scheme_name"]),
            "state": str(row["state"]),
            "district": str(row["district"]),
            "contractor_type": str(row["contractor_type"]),
            "sanctioned_budget_lakh": float(row["sanctioned_budget_lakh"]),
            "duration_months": int(row["duration_months"]),
            "true_label": str(row[schema.TARGET]),
        }
    return build
