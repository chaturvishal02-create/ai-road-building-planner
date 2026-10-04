"""
The "scalable pipeline": batch processing that is shaped like a Spark job but runs on
a laptop with no JVM.

Three ideas borrowed from Spark, deliberately:

1. **Partition-at-a-time processing.** ``map_partitions`` applies one function to each
   chunk and concatenates the results. It is the same programming shape as
   ``DataFrame.mapInPandas`` -- a partition-local transform, no per-row Python loop --
   so porting to Spark means replacing this one function, not rewriting the transform.
2. **A partitioned table on disk.** Output lands as
   ``scheme_name=<scheme>/part-00000.csv`` directories plus a ``_manifest.json``. That is
   how Spark lays out a partitioned parquet table (directory per partition, many part
   files, metadata alongside).
3. **Constant memory.** Local sources stream through ``pandas.read_csv(chunksize=...)``
   instead of materialising the whole file.

Two honest limitations, stated here because a viva will ask:
* We write CSV, not parquet. Parquet needs pyarrow, which is not installed and whose
  compiled binaries would hit the same Windows Smart App Control block that already
  forced us to pin scikit-learn and scipy. Switching ``to_csv`` to ``to_parquet`` is a
  one-line change on any machine where pyarrow loads.
* 700 rows do not need a cluster. This design is about being *ready* for the day the
  data is 7 million rows, not about speeding up today's run.
"""

import json
import os
import re
import time

import pandas as pd

import config
from data_layer import schema
from data_layer.storage import build_adapter

ENGINE = "pandas-chunked (Spark-shaped: map_partitions + partitioned output)"

MANIFEST_NAME = "_manifest.json"


def sanitise_partition_value(value):
    """Turn 'PMGSY (Rural Roads)' into a filesystem-safe directory name, the way a
    Spark partition path would."""
    text = re.sub(r"[^0-9A-Za-z\u0900-\u097F]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def read_chunks(source=None, chunksize=None, adapter=None):
    """Yield DataFrames of at most ``chunksize`` rows from any storage backend."""
    store = adapter or build_adapter()
    size = chunksize or config.CHUNK_SIZE
    if size < 1:
        raise ValueError("chunksize must be positive")
    return store.iter_chunks(source, size)


def map_partitions(transform, chunks):
    """Apply ``transform`` to every partition and concatenate. Spark equivalent:
    ``df.mapInPandas`` / ``df.transform``."""
    parts = [transform(chunk) for chunk in chunks]
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def enrich_partition(chunk):
    """The business transform: normalise types, then add the derived overrun columns
    the dashboard and the risk heuristic need."""
    frame = schema.normalise_frame(chunk)
    return schema.add_derived_columns(frame)


def transform_source(source=None, chunksize=None, adapter=None):
    """Full pass: stream partitions through ``enrich_partition``."""
    chunks = read_chunks(source=source, chunksize=chunksize, adapter=adapter)
    return map_partitions(enrich_partition, chunks)


def _part_filename(index):
    return "part-%05d.csv" % index


def build_feature_store(source=None, chunksize=None, adapter=None, out_dir=None,
                        partition_column="scheme_name", quiet=True):
    """Stream the dataset through the pipeline and write a partitioned feature store.

    Returns the manifest dictionary (also written as ``_manifest.json``).
    """
    store = adapter or build_adapter()
    target = out_dir or config.FEATURE_STORE_DIR
    size = chunksize or config.CHUNK_SIZE
    started = time.time()

    schema_data = {"columns": None, "partitions": {}, "rows_total": 0, "files": 0}
    counters = {}
    source_label = source or getattr(store, "url", None) or config.LOCAL_DATASET
    source_hash = None
    resolver = getattr(store, "resolve", None)  # only LocalStore has one
    if resolver is not None:
        try:
            resolved = resolver(source)
            if os.path.isfile(resolved):
                source_hash = schema.sha256_of_file(resolved)
        except Exception:
            source_hash = None

    os.makedirs(target, exist_ok=True)
    first = True
    for chunk in store.iter_chunks(source, size):
        frame = enrich_partition(chunk)
        if first:
            schema_data["columns"] = list(frame.columns)
            first = False
        if not quiet:
            print("partition of %d rows" % len(frame))

        for value, group in frame.groupby(partition_column, sort=True):
            partition_dir = os.path.join(target, "%s=%s" % (partition_column,
                                                            sanitise_partition_value(value)))
            os.makedirs(partition_dir, exist_ok=True)
            index = counters.get(partition_dir, 0)
            counters[partition_dir] = index + 1
            group.to_csv(os.path.join(partition_dir, _part_filename(index)),
                         index=False, encoding="utf-8")

            key = os.path.basename(partition_dir)
            entry = schema_data["partitions"].setdefault(
                key, {"partition_value": None, "rows": 0, "files": 0, "states": []})
            entry["partition_value"] = str(value)
            entry["rows"] += len(group)
            entry["files"] += 1
            seen_states = sorted(set(group["state"].unique().tolist()))
            entry["states"] = sorted(set(entry["states"]) | set(seen_states))
            schema_data["rows_total"] += len(group)
            schema_data["files"] += 1

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "engine": ENGINE,
        "source": str(source_label),
        "source_sha256": source_hash,
        "chunk_size": size,
        "partition_column": partition_column,
        "row_count": schema_data["rows_total"],
        "file_count": schema_data["files"],
        "columns": schema_data["columns"],
        "partitions": schema_data["partitions"],
        "format": "csv",
        "format_note": "to_parquet() replaces to_csv() here once pyarrow is available",
    }
    with open(os.path.join(target, MANIFEST_NAME), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def load_feature_store(out_dir=None):
    """Read every partition back into one frame, restoring the true partition labels
    from the manifest (directory names are sanitised, so they are not authoritative).

    Returns the frame alone; the manifest is available from build_feature_store() or by
    reading _manifest.json directly, which is what the /pipeline route does."""
    target = out_dir or config.FEATURE_STORE_DIR
    manifest_path = os.path.join(target, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("feature store has no %s -- run the pipeline first" % MANIFEST_NAME)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    frames = []
    for entry in manifest["partitions"].values():
        directory = os.path.join(target, "%s=%s" % (manifest["partition_column"],
                                                    sanitise_partition_value(entry["partition_value"])))
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".csv"):
                continue
            frame = pd.read_csv(os.path.join(directory, name), encoding="utf-8")
            frame[manifest["partition_column"]] = entry["partition_value"]
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(combined) != manifest["row_count"]:
        raise ValueError("feature store holds %d rows but its manifest claims %d"
                         % (len(combined), manifest["row_count"]))
    return combined
