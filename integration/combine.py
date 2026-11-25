"""Run both detector sets on the same Databento frame and merge their output.

engine_m1     alert_system.detectors.AlertEngine (Milestone 1): online, one record at a
              time in arrival order, past-only state, cooldown, explained alerts.
detectors_v3  p115_detectors/detectors/detectors.py::run_all (v3): vectorized over the
              whole frame, seven checks plus Databento vendor flags.

Both are called only through their own public functions. The adapter adds three things:
chunking for files above the engine's MAX_EVENTS, one shared table (UNIFIED_COLUMNS), and an
overlap report for the two checks that both systems implement.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from alert_system.config import MAX_EVENTS, PROJECT_ROOT, SUPPORTED_DATASETS, SUPPORTED_SCHEMAS
from alert_system.data import local_path, normalize_frame, utc_time
from alert_system.detectors import AlertEngine

V3_DIR = PROJECT_ROOT / "p115_detectors" / "detectors"
ENGINE, BATCH = "engine_m1", "detectors_v3"
F_LAST = 128
LEVEL = {"info": 0, "low": 1, "medium": 2, "warning": 2, "high": 3, "critical": 3}
UNIFIED_COLUMNS = ["source", "mode", "rule", "subtype", "severity", "level", "ts_event", "ts_recv", "symbol",
                   "publisher_id", "instrument_id", "value", "threshold", "detail", "f_last"]
# Checks that both systems implement: (name, engine rule_id, (v3 check, v3 subtype))
OVERLAPS = [
    ("crossed quote", "crossed_quote", ("crossed_books", "crossed")),
    ("event time moved backwards", "time_order", ("timestamp_inversions", "ts_event_inversion")),
]


# ---------------------------------------------------------------------------- v3 package
def load_v3(path: Path = V3_DIR, names: tuple[str, ...] = ("config", "detectors", "synthetic")) -> SimpleNamespace:
    """Import the v3 modules, which use the top-level names config/detectors/synthetic.

    Raises if a module of the same name was already imported from somewhere else, so the
    v3 code can never silently run against the wrong ``config``.
    """
    path = Path(path).resolve()
    if not (path / "detectors.py").is_file():
        raise FileNotFoundError(f"{path / 'detectors.py'} not found. Unzip p115_detectors_v3_final.zip "
                                "into p115_detectors/ at the repository root.")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    modules = {}
    for name in names:
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve().parent if getattr(module, "__file__", None) else None
        if origin != path:
            raise ImportError(f"Top-level module {name!r} was imported from {origin}, not {path}")
        modules[name] = module
    return SimpleNamespace(**modules)


def v3_config(dataset: str, v3: SimpleNamespace | None = None):
    """The v3 preset the pilot used for this dataset (see p115_detectors/analyze_v2.py::cfg_for)."""
    v3 = v3 or load_v3()
    if dataset in SUPPORTED_DATASETS:
        return v3.config.equities_config()
    if dataset == "GLBX.MDP3":
        config = v3.config.cme_futures_config()
        config.sequence.gap_mode = "skip"  # symbol-filtered pulls: per-channel sequence shows false gaps
        return config
    raise ValueError(f"No v3 preset for dataset {dataset!r}")


def run_v3(frame: pd.DataFrame, dataset: str, config=None, v3: SimpleNamespace | None = None, **kwargs) -> pd.DataFrame:
    v3 = v3 or load_v3()
    flags, _ = v3.detectors.run_all(frame, config or v3_config(dataset, v3), **kwargs)
    return flags


# ---------------------------------------------------------------------------- Milestone 1 engine
def engine_supports(metadata: dict) -> bool:
    return metadata.get("dataset") in SUPPORTED_DATASETS and metadata.get("schema") in SUPPORTED_SCHEMAS


def run_engine(frame: pd.DataFrame, metadata: dict, config: dict | None = None,
               chunk_rows: int = MAX_EVENTS) -> tuple[list[dict], dict]:
    """Feed every record to one AlertEngine in file (arrival) order.

    Chunks of at most ``chunk_rows`` go through alert_system.data.normalize_frame, which
    accepts up to MAX_EVENTS rows. Event ids are renumbered to the 1-based row of the whole
    frame, so a frame that fits in one chunk gives exactly the alerts of run_checks.py.
    Returns ``(alerts, context)``; ``context[alert id]`` holds the source row, instrument_id
    and raw ``flags`` of the record that raised the alert.
    """
    if not 0 < chunk_rows <= MAX_EVENTS:
        raise ValueError(f"chunk_rows must be between 1 and {MAX_EVENTS:,}")
    engine = AlertEngine(config)
    alerts, context = [], {}
    raw_flags = frame["flags"].to_numpy() if "flags" in frame else None
    for offset in range(0, len(frame), chunk_rows):
        for event in normalize_frame(frame.iloc[offset:offset + chunk_rows], metadata).events:
            event["id"] = event["source_record_id"] = offset + event["id"]
            for alert in engine.process(event):
                row = event["id"] - 1
                context[alert["id"]] = {"row": row, "instrument_id": event.get("instrument_id"),
                                        "flags": None if raw_flags is None else int(raw_flags[row])}
                alerts.append(alert)
    return alerts, context


# ---------------------------------------------------------------------------- unified table
def _empty() -> pd.DataFrame:
    frame = pd.DataFrame({name: pd.Series(dtype=object) for name in UNIFIED_COLUMNS})
    for name in ("ts_event", "ts_recv"):
        frame[name] = pd.Series(dtype="datetime64[ns, UTC]")
    for name in ("level", "value", "threshold"):
        frame[name] = pd.Series(dtype="float64")
    return frame


def _headline(alert: dict) -> tuple[float, float]:
    observed, thresholds = alert["observed"], alert["thresholds"]
    if alert["rule_id"] == "crossed_quote":
        return observed["cross_amount"], thresholds["maximum_cross"]
    if alert["rule_id"] == "time_order":
        return observed["backward_seconds"], thresholds["backward_seconds_allowed"]
    if alert["rule_id"] == "integrity":
        return float(len(observed["issues"])), thresholds["violations_allowed"]
    return np.nan, np.nan


def engine_table(alerts: list[dict], context: dict | None = None) -> pd.DataFrame:
    """Milestone 1 alerts as unified rows (the full alert dicts stay available unchanged)."""
    context = context or {}
    rows = []
    for alert in alerts:
        ctx = context.get(alert["id"], {})
        f_last = None if ctx.get("flags") is None else bool(ctx["flags"] & F_LAST)
        value, threshold = _headline(alert)
        detail = "; ".join(alert["observed"]["issues"]) if alert["rule_id"] == "integrity" else alert["explanation"]
        if alert["rule_id"] == "crossed_quote" and f_last is False:
            detail += " [record without F_LAST: intermediate book state]"
        rows.append({"source": ENGINE, "mode": "online", "rule": alert["rule_id"], "subtype": None,
                     "severity": alert["severity"], "level": LEVEL[alert["severity"]],
                     "ts_event": alert["timestamp"], "ts_recv": alert["detected_at"], "symbol": alert["symbol"],
                     "publisher_id": alert["publisher_id"], "instrument_id": ctx.get("instrument_id"),
                     "value": value, "threshold": threshold, "detail": detail, "f_last": f_last})
    if not rows:
        return _empty()
    table = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    for name in ("ts_event", "ts_recv"):
        table[name] = pd.to_datetime(table[name], utc=True, format="ISO8601")
    return table


def v3_table(flags: pd.DataFrame) -> pd.DataFrame:
    """v3 flag rows as unified rows (columns renamed, values untouched)."""
    if flags.empty:
        return _empty()
    table = pd.DataFrame({"source": BATCH, "mode": "batch", "rule": flags["check"], "subtype": flags["subtype"],
                          "severity": flags["severity"], "level": flags["severity"].map(LEVEL).astype("float64"),
                          "ts_event": flags["ts_event"], "ts_recv": flags["ts_recv"], "symbol": flags["symbol"],
                          "publisher_id": flags["publisher_id"], "instrument_id": flags["instrument_id"],
                          "value": flags["value"], "threshold": flags["threshold"], "detail": flags["detail"],
                          "f_last": None})
    return table[UNIFIED_COLUMNS].reset_index(drop=True)


def unify(*tables: pd.DataFrame) -> pd.DataFrame:
    tables = [table for table in tables if not table.empty]
    if not tables:
        return _empty()
    table = pd.concat(tables, ignore_index=True)
    return table.sort_values(["ts_recv", "source"], kind="stable", na_position="last").reset_index(drop=True)


def _seen(rows: pd.DataFrame, others: pd.DataFrame, tolerance_s: float) -> int:
    """How many of ``rows`` have a row of ``others`` for the same symbol within the tolerance."""
    if rows.empty or others.empty:
        return 0
    def ns(values: pd.Series) -> np.ndarray:
        return pd.to_datetime(values, utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    tol = int(tolerance_s * 1_000_000_000)
    count = 0
    for symbol, group in rows.groupby("symbol"):
        times = np.sort(ns(others.loc[others["symbol"] == symbol, "ts_recv"]))
        if not len(times):
            continue
        own = ns(group["ts_recv"])
        count += int((np.searchsorted(times, own + tol, side="right") > np.searchsorted(times, own - tol, side="left")).sum())
    return count


def overlap_report(unified: pd.DataFrame, tolerance_s: float = 1.0) -> pd.DataFrame:
    """Counts for the checks both systems implement, and how often each side's finding has a
    counterpart from the other side (same symbol, ts_recv within the tolerance)."""
    rows = []
    for name, rule, (check, subtype) in OVERLAPS:
        ours = unified[(unified["source"] == ENGINE) & (unified["rule"] == rule)]
        theirs = unified[(unified["source"] == BATCH) & (unified["rule"] == check) & (unified["subtype"] == subtype)]
        rows.append({"check": name, "engine_m1_rule": rule, "detectors_v3_rule": f"{check}/{subtype}",
                     "engine_m1": len(ours), "detectors_v3": len(theirs),
                     "engine_m1_with_v3_match": _seen(ours, theirs, tolerance_s),
                     "detectors_v3_with_engine_match": _seen(theirs, ours, tolerance_s)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- inputs
def bundled(sample: str = "quotes", root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, dict]:
    """The same file and context that run_checks.py (load_data) uses for a bundled sample."""
    if sample not in ("quotes", "trades"):
        raise ValueError("sample must be quotes or trades")
    folder = Path(root) / "sample_data"
    provenance = json.loads((folder / f"{sample}_provenance.json").read_text())
    frame = pd.read_parquet(folder / provenance["file"])
    label = f"{', '.join(provenance['symbols'])} · {provenance['dataset']} {provenance['schema']} · bundled sample"
    return frame, {"source": "bundled", "dataset": provenance["dataset"], "schema": provenance["schema"], "label": label}


def local(relative: str, symbol: str | None = None, start=None, end=None,
          root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, dict]:
    """A Parquet file under data/ (for example written by download_and_show), with dataset and
    schema taken from the manifest.json beside it or from the Parquet header. Optional
    ``symbol`` / ``start`` / ``end`` filters (naive times are New York time, as in the pipeline)."""
    path = local_path(relative, root)
    if path.suffix != ".parquet":
        raise ValueError("Combined runs read Parquet files (download_and_show writes data.parquet)")
    metadata = {"source": "local", "label": str(path.relative_to(Path(root) / "data"))}
    manifest = path.parent / "manifest.json"
    if manifest.is_file():
        values = json.loads(manifest.read_text())
        for name in ("dataset", "schema"):
            value = values.get(name) or values.get("request", {}).get(name)
            if value:
                metadata[name] = str(value)
    headers = pq.ParquetFile(path).metadata.metadata or {}
    for name in ("dataset", "schema"):
        if name not in metadata and name.encode() in headers:
            metadata[name] = headers[name.encode()].decode()
    expression = None
    if symbol is not None:
        expression = pads.field("symbol") == symbol
    for value, compare in ((start, "__ge__"), (end, "__lt__")):
        if value is not None:
            term = getattr(pads.field("ts_recv"), compare)(utc_time(value))
            expression = term if expression is None else expression & term
    frame = pads.dataset(path).to_table(filter=expression).to_pandas()
    if "ts_recv" in frame.columns:
        frame = frame.set_index("ts_recv")
    for name in ("dataset", "schema"):
        if name not in metadata and name in frame.columns and frame[name].nunique() == 1:
            metadata[name] = str(frame[name].iloc[0])
    return frame, metadata


# ---------------------------------------------------------------------------- one call
def run_combined(frame: pd.DataFrame, metadata: dict, engine_config: dict | None = None, v3_cfg=None,
                 chunk_rows: int = MAX_EVENTS, **v3_kwargs) -> SimpleNamespace:
    """Run both systems on ``frame``. The native outputs are returned unchanged next to the
    unified table: ``alerts`` (Milestone 1 dicts) and ``flags`` (v3 flag table)."""
    if not metadata.get("dataset") or not metadata.get("schema"):
        raise ValueError("Dataset and schema are required: keep the Databento manifest.json beside the Parquet file")
    v3 = load_v3()
    if engine_supports(metadata):
        alerts, context = run_engine(frame, metadata, engine_config, chunk_rows)
        engine_note = None
    else:
        alerts, context = [], {}
        engine_note = (f"engine_m1 skipped: Milestone 1 supports {', '.join(SUPPORTED_SCHEMAS)} on equity datasets, "
                       f"not {metadata.get('dataset')} / {metadata.get('schema')}")
    flags = run_v3(frame, metadata["dataset"], v3_cfg, v3, **v3_kwargs)
    unified = unify(engine_table(alerts, context), v3_table(flags))
    return SimpleNamespace(alerts=alerts, flags=flags, unified=unified, overlap=overlap_report(unified),
                           records=len(frame), metadata=metadata, engine_note=engine_note)
