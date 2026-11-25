"""Item 5: regenerate the v3 synthetic metrics into a sidecar folder and diff them with the stored file.

The stored p115_detectors/detectors/results/metrics.csv is checksummed by integration.verify and
is never overwritten. evaluate.main() is run with its OUT pointed at the sidecar folder.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from integration.combine import V3_DIR, load_v3

SIDE = Path(__file__).resolve().parent / "results" / "v3_metrics_regenerated"
STORED = Path(V3_DIR) / "results"


def regenerate(out: Path = SIDE, sweep: bool = True) -> pd.DataFrame:
    v3 = load_v3(names=("config", "detectors", "synthetic", "evaluate"))
    ev = v3.evaluate
    out.mkdir(parents=True, exist_ok=True)
    old_out = ev.OUT
    try:
        ev.OUT = out
        if not sweep:
            ev.sweep = lambda *a, **k: pd.DataFrame(columns=["check", "parameter", "value", "precision", "recall", "f1", "flags"])
            ev.plot_sweep = lambda *a, **k: None
        metrics = ev.main()
    finally:
        ev.OUT = old_out
    return metrics


def compare(out: Path = SIDE, stored: Path = STORED) -> pd.DataFrame:
    new = pd.read_csv(out / "metrics.csv").set_index("check")
    old = pd.read_csv(stored / "metrics.csv").set_index("check")
    cols = ["labels", "detected", "flags", "tp_flags", "fp_flags", "collateral_flags", "clean_stream_flags", "precision", "recall"]
    rows = []
    for check in old.index:
        for c in cols:
            a, b = old.loc[check, c], new.loc[check, c]
            if not (pd.isna(a) and pd.isna(b)) and abs(float(a) - float(b)) > 1e-9:
                rows.append(dict(check=check, column=c, stored=a, regenerated=b))
    return pd.DataFrame(rows, columns=["check", "column", "stored", "regenerated"])


def flag_diff(check: str = "extreme_prices", out: Path = SIDE, stored: Path = STORED) -> pd.DataFrame:
    """Flags of one check present in only one of the two flags.csv files (by ts_recv, instrument, subtype)."""
    key = ["ts_recv", "instrument_id", "subtype"]
    a = pd.read_csv(stored / "flags.csv"); b = pd.read_csv(out / "flags.csv")
    a = a[(a["check"] == check) & (a["severity"] != "info")]; b = b[(b["check"] == check) & (b["severity"] != "info")]
    m = a[key + ["severity", "detail"]].merge(b[key + ["severity", "detail"]], on=key, how="outer",
                                              suffixes=("_stored", "_regen"), indicator=True)
    return m[m["_merge"] != "both"]
