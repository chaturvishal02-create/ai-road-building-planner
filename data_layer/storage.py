"""
Storage adapters: one ``load``/``put`` interface, three backends.

Why an interface at all for a 700-row student project? Because it is the part of a
production design that can be shown honestly. ``LocalStore`` really runs here.
``HttpStore`` really fetches a CSV over HTTP. ``S3Store`` is complete code that only
activates when ``DATA_STORE=s3`` and real credentials are in the environment, so the
application never needs a second data-loading path once it moves to a bucket.

No adapter validates. They hand back whatever the source contained and the caller runs
:data:`data_layer.schema` over it -- one place to decide what a good row looks like.
"""

import io
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

import config
from data_layer import schema


class StorageError(Exception):
    """Any failure talking to a storage backend, with a message fit to show a user."""


class StorageAdapter:
    """Base class. Subclasses set ``name`` and implement ``load``/``put``."""

    name = "base"
    writable = True

    def load(self, ref=None):
        raise NotImplementedError

    def put(self, frame, ref):
        raise StorageError("%s storage cannot write" % self.name)

    def iter_chunks(self, ref=None, chunksize=None):
        """Default: load everything, then slice. LocalStore overrides this with a true
        streaming reader."""
        frame = self.load(ref)
        chunksize = chunksize or config.CHUNK_SIZE
        for start in range(0, len(frame), chunksize):
            yield frame.iloc[start:start + chunksize]

    def describe(self):
        return {"name": self.name, "writable": self.writable, "configured": True}

    def missing_reason(self):
        """Why this backend is not usable yet, for the Data Source panel. None means it
        is ready."""
        return None

    # -----------------------------------------------------------------------
    @staticmethod
    def _read_csv_bytes(payload, source_label):
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise StorageError("%s is not valid UTF-8. Re-save the CSV as UTF-8 "
                               "(the rupee sign and en-dashes need it)." % source_label)
        try:
            return pd.read_csv(io.StringIO(text))
        except Exception as exc:  # pandas raises many shapes of parse error
            raise StorageError("%s could not be parsed as CSV: %s" % (source_label, exc))


class LocalStore(StorageAdapter):
    """Reads and writes CSV files on this machine. The default backend."""

    name = "local"

    def __init__(self, root=None):
        self.root = os.path.abspath(root or config.PROJECT_ROOT)

    def resolve(self, ref):
        """Absolute paths pass through; relative ones are anchored to the project. Only
        files under data/ may be read, so a dataset ref can never name app.py or a
        future .env file."""
        if not ref:
            return config.LOCAL_DATASET
        candidate = ref if os.path.isabs(ref) else os.path.join(self.root, ref)
        candidate = os.path.normpath(candidate)
        if not _is_under(candidate, os.path.normpath(config.DATA_DIR)):
            raise StorageError("dataset files must live under the data/ folder: %s" % ref)
        return candidate

    def load(self, ref=None):
        path = self.resolve(ref)
        if not os.path.exists(path):
            raise StorageError("no such file: %s (run py model/generate_dataset.py)" % path)
        with open(path, "rb") as handle:
            return self._read_csv_bytes(handle.read(), path)

    def iter_chunks(self, ref=None, chunksize=None):
        """pandas' own chunked reader: constant memory, one partition at a time."""
        path = self.resolve(ref)
        if not os.path.exists(path):
            raise StorageError("no such file: %s" % path)
        return pd.read_csv(path, chunksize=chunksize or config.CHUNK_SIZE, encoding="utf-8-sig")

    def put(self, frame, ref):
        path = self.resolve(ref)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8")
        return path


class HttpStore(StorageAdapter):
    """API-based data loading: pull a published CSV straight from a government or
    project URL, with no bucket involved."""

    name = "http"
    MAX_BYTES = 25 * 1024 * 1024

    def __init__(self, url=None, cache_dir=None):
        self.url = url if url is not None else config.HTTP_DATASET_URL
        self.cache_dir = cache_dir or config.DOWNLOAD_DIR

    def load(self, ref=None):
        url = ref or self.url
        if not url:
            raise StorageError("HTTP source is not configured. Set HTTP_DATASET_URL "
                               "to a public .csv URL and restart.")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https"):
            raise StorageError("only http:// and https:// data sources are allowed")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "budget-prediction/2.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > self.MAX_BYTES:
                    raise StorageError("data source is larger than %d MB" % (self.MAX_BYTES // 2**20))
                payload = response.read(self.MAX_BYTES + 1)
        except StorageError:
            raise
        except urllib.error.HTTPError as exc:
            raise StorageError("data source returned HTTP %s" % exc.code)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise StorageError("could not reach the data source: %s" % exc)

        if len(payload) > self.MAX_BYTES:
            raise StorageError("data source exceeded the %d MB limit" % (self.MAX_BYTES // 2**20))

        frame = self._read_csv_bytes(payload, url)
        self._cache(payload)
        return frame

    def _cache(self, payload):
        """Keep the downloaded bytes so the dashboard still works offline afterwards."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(os.path.join(self.cache_dir, "last_download.csv"), "wb") as handle:
                handle.write(payload)
        except OSError:
            pass  # caching is a convenience, never a reason to fail a load

    def put(self, frame, ref):
        raise StorageError("HTTP downloads are read-only; use local or s3 storage to write")

    def missing_reason(self):
        if self.url:
            return None
        return ("no data URL set -- export HTTP_DATASET_URL pointing at a public .csv "
                "and restart with DATA_STORE=http")

    def describe(self):
        return {"name": self.name, "writable": False, "configured": bool(self.url),
                "url": self.url or "", "reason": self.missing_reason()}


class S3Store(StorageAdapter):
    """Bucket-backed storage. Constructed only when DATA_STORE=s3 and credentials
    exist in the environment; boto3 is imported lazily so it is not a hard dependency
    for the 99 % of runs that never touch a bucket."""

    name = "s3"
    writable = False  # flipped to True only when config.ALLOW_WRITES

    def __init__(self, bucket=None, key=None, region=None):
        self.bucket = bucket if bucket is not None else config.S3_BUCKET
        self.key = key if key is not None else config.S3_KEY
        self.region = region if region is not None else config.S3_REGION
        self.writable = config.ALLOW_WRITES

        if not self.bucket:
            raise StorageError("S3 source is not configured. Set S3_BUCKET (and S3_KEY) "
                               "in the environment, then restart with DATA_STORE=s3.")
        if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
            raise StorageError("no AWS credentials in the environment. Export "
                               "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY first -- "
                               "they must never be written into this repository.")
        try:
            import boto3  # imported here, not at module top, on purpose
        except ImportError:
            raise StorageError("boto3 is not installed. Run: py -m pip install boto3")
        self._client = boto3.client("s3", region_name=self.region or None)

    def load(self, ref=None):
        key = ref or self.key
        try:
            obj = self._client.get_object(Bucket=self.bucket, Key=key)
            payload = obj["Body"].read()
        except Exception as exc:
            raise StorageError("could not read s3://%s/%s -- %s" % (self.bucket, key, exc))
        return self._read_csv_bytes(payload, "s3://%s/%s" % (self.bucket, key))

    def put(self, frame, ref):
        if not config.ALLOW_WRITES:
            raise StorageError("writing to a bucket is disabled. Set ALLOW_WRITES=1 if "
                               "you really intend to overwrite s3://%s/%s." % (self.bucket, ref or self.key))
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        key = ref or self.key
        try:
            self._client.put_object(Bucket=self.bucket, Key=key,
                                    Body=buffer.getvalue().encode("utf-8"))
        except Exception as exc:
            raise StorageError("could not write s3://%s/%s -- %s" % (self.bucket, key, exc))
        return "s3://%s/%s" % (self.bucket, key)

    def describe(self):
        return {"name": self.name, "writable": self.writable, "configured": bool(self.bucket),
                "bucket": self.bucket, "key": self.key, "region": self.region or "default"}


def _is_under(path, base):
    path, base = os.path.abspath(path), os.path.abspath(base)
    return os.path.commonpath([path, base]) == base


_BUILDERS = {
    "local": lambda: LocalStore(),
    "http": lambda: HttpStore(),
    "s3": lambda: S3Store(),
}


def build_adapter(name=None):
    """Return the adapter for ``name`` (default: config.DATA_STORE)."""
    chosen = (name or config.DATA_STORE or "local").strip().lower()
    if chosen not in _BUILDERS:
        raise StorageError("unknown data store %r -- choose local, http or s3" % chosen)
    return _BUILDERS[chosen]()


def adapter_status(name=None):
    """Describe each backend for the UI without ever raising, so the Data Source panel
    can show why http/s3 are inactive instead of hiding them."""
    active = (name or config.DATA_STORE or "local").strip().lower()
    report = []
    for key in ("local", "http", "s3"):
        entry = {"name": key, "configured": False, "writable": False,
                 "detail": "", "reason": None, "active": key == active}
        try:
            adapter = _BUILDERS[key]()
            description = adapter.describe()
            entry["configured"] = bool(description.get("configured", True))
            entry["writable"] = bool(description.get("writable", False))
            entry["detail"] = (description.get("bucket") or description.get("url")
                               or (config.LOCAL_DATASET if key == "local" else ""))
            entry["reason"] = description.get("reason") or adapter.missing_reason()
        except StorageError as exc:
            # Unconfigured cloud backends refuse to construct; that message is the
            # reason we show, not an error to hide.
            entry["reason"] = str(exc)
            entry["detail"] = str(exc)
        report.append(entry)
    return report


def reset_feature_store(directory=None):
    """Remove a previous partitioned output so a rebuild cannot leave stale files."""
    target = directory or config.FEATURE_STORE_DIR
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.makedirs(target, exist_ok=True)
    return target
