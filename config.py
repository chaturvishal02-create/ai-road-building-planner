"""
Settings, read from environment variables only.

Nothing secret lives in this repository. To point the app at a real bucket you export
credentials in the shell before starting it; if the variables are absent the S3 adapter
simply refuses to build and the app stays on local files.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
FEATURE_STORE_DIR = os.path.join(DATA_DIR, "feature_store")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloads")

# local | http | s3  -- which StorageAdapter to build.
DATA_STORE = os.environ.get("DATA_STORE", "local").strip().lower()

# What to load for each backend.
LOCAL_DATASET = os.environ.get("LOCAL_DATASET", os.path.join(DATA_DIR, "india_projects_2025_26.csv"))
SAMPLE_DATASET = os.environ.get("SAMPLE_DATASET", os.path.join(DATA_DIR, "sample_20_projects.csv"))
HTTP_DATASET_URL = os.environ.get("HTTP_DATASET_URL", "")

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_KEY = os.environ.get("S3_KEY", "india_projects_2025_26.csv")
S3_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", ""))
# Credentials are never read from this file; boto3 takes them from AWS_ACCESS_KEY_ID /
# AWS_SECRET_ACCESS_KEY (or an instance role) when it is actually constructed.

# Uploading a dataset back to cloud storage is opt-in, so a typo in DATA_STORE can
# never overwrite somebody's bucket.
ALLOW_WRITES = os.environ.get("ALLOW_WRITES", "0").strip() in ("1", "true", "yes", "on")

# Rows per partition for the chunked pipeline (see data_layer/pipeline.py).
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "250"))

PORT = int(os.environ.get("PORT", "5000"))

# Tricolour accents and currency, exposed to the template so the values live once.
THEME = {
    "saffron": "#FF9933",
    "saffron_text": "#B25A00",
    "india_green": "#138808",
    "green_text": "#0B6B0B",
    "chakra_navy": "#0A1F44",
}
CURRENCY = {"symbol": "\u20B9", "unit": "lakh", "lakh_per_crore": 100}


def describe():
    """Safe summary of the data-source configuration for the /meta endpoint.
    Deliberately excludes anything that could look like a credential."""
    return {
        "data_store": DATA_STORE,
        "local_dataset": LOCAL_DATASET,
        "sample_dataset": SAMPLE_DATASET,
        "http_url_configured": bool(HTTP_DATASET_URL),
        "s3_bucket_configured": bool(S3_BUCKET),
        "s3_credentials_present": bool(os.environ.get("AWS_ACCESS_KEY_ID")
                                       and os.environ.get("AWS_SECRET_ACCESS_KEY")),
        "allow_writes": ALLOW_WRITES,
        "chunk_size": CHUNK_SIZE,
        "currency": dict(CURRENCY),
        "theme": dict(THEME),
    }
