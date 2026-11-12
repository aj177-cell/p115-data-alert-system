"""Load the pipeline table the batch checks read.

``alert_system.data.load_data`` turns that table into one event dict per row.
Those dicts rename book columns, drop ``flags``, and store a quote midpoint in
``price``. The batch checks need the pipeline table itself: ``ts_recv`` as the
index, ``bid_px_00`` / ``ask_px_00``, ``flags``, and ``sequence``. This module
reads the same bundled or local file the loader reads, and it does not call
Databento.
"""

import json
from pathlib import Path

import pandas as pd

from alert_system.config import PROJECT_ROOT
from alert_system.data import _read_file, local_path, validate_request


def pipeline_frame(payload, root=PROJECT_ROOT):
    """Return the pipeline DataFrame for a bundled sample or a local file."""
    root = Path(root).resolve()
    request = validate_request(payload)
    source = request["source"]
    if source == "databento":
        raise ValueError(
            "Batch checks read a Parquet file the pipeline already wrote. "
            "Use source 'bundled' or 'local'. This function does not download data."
        )
    if source == "bundled":
        folder = root / "sample_data"
        name = "quotes_provenance.json" if request["sample"] == "quotes" else "trades_provenance.json"
        provenance = json.loads((folder / name).read_text())
        path = folder / provenance["file"]
        if not path.is_file():
            raise ValueError(f"Bundled sample file is missing: {provenance['file']}")
    else:
        path = local_path(request["path"], root)
    return _read_file(path, {})


def pipeline_parquets(path):
    """One Parquet file, or every ``data.parquet`` under a pipeline download directory."""
    path = Path(path)
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError("Batch checks read a Parquet file, or a directory of data.parquet files")
        return [path]
    if not path.is_dir():
        raise ValueError(f"No such path: {path}")
    found = sorted(item for item in path.rglob("data.parquet") if item.is_file())
    if not found:
        raise ValueError(
            "No data.parquet under that directory. Pass a Parquet file directly, "
            "or a folder written by databento_pipeline.download_and_show."
        )
    return found


def read_pipeline_parquet(path):
    """Read one pipeline Parquet file. ``ts_recv`` stays the index when the file has it."""
    path = Path(path)
    if not path.is_file() or path.suffix != ".parquet":
        raise ValueError("Expected an existing Parquet file")
    return pd.read_parquet(path)
