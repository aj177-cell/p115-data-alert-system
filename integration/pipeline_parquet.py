"""Load pipeline Parquet and run the v3 batch checks, without relocating that package.

Adapted from Aryan Jhaveri's alert_system/quality/inputs.py and the batch helpers
in his pull-request run_checks.py. The detector code is called through
integration.combine.load_v3, so p115_detectors/ stays the only copy.

These functions do not call Databento and do not download data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alert_system.config import PROJECT_ROOT
from alert_system.data import _read_file, local_path, validate_request
from integration.combine import load_v3

PRESET_NAMES = ("equities", "cme", "base")


def presets(v3=None):
    """Threshold factories from p115_detectors/detectors/config.py."""
    v3 = v3 or load_v3()
    return {
        "equities": v3.config.equities_config,
        "cme": v3.config.cme_futures_config,
        "base": v3.config.DetectorConfig,
    }


def pipeline_frame(payload, root=PROJECT_ROOT):
    """Return the pipeline DataFrame for a bundled sample or a local file under data/."""
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
    """One Parquet file, or every data.parquet under a pipeline download directory."""
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
    """Read one pipeline Parquet file. ts_recv stays the index when the file has it."""
    path = Path(path)
    if not path.is_file() or path.suffix != ".parquet":
        raise ValueError("Expected an existing Parquet file")
    return pd.read_parquet(path)


def run_batch_frame(frame, preset="equities"):
    """Run the seven batch checks and the vendor-flag check on one pipeline table."""
    if preset not in PRESET_NAMES:
        raise ValueError("preset must be equities, cme, or base")
    v3 = load_v3()
    return v3.detectors.run_all(frame, presets(v3)[preset]())


def run_batch_sample(sample, preset="equities"):
    """Batch checks on the same bundled file the loader reads."""
    frame = pipeline_frame({"source": "bundled", "sample": sample})
    flags, summary = run_batch_frame(frame, preset)
    return frame, flags, summary
