"""SPC charts on the 95 symbol-day pull and on injected faults.

    python -m m2.spc.run_spc --mode series   # one-minute bucket series per symbol-day (cached)
    python -m m2.spc.run_spc --mode charts   # Phase I baselines, calibration, signals, overlap
    python -m m2.spc.run_spc --mode inject   # spread widening and partial thinning vs the point checks
    python -m m2.spc.run_spc --mode plots

One symbol-day in memory at a time. Writes m2/results/spc/ only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from integration.combine import load_v3
from m2.extract import DEFAULT_DATA_ROOT, NORMAL_WEEK, day_type_of, load_symbol_slice, load_ticks, tasks
from m2.spc.series import bucket_index, build_series

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "m2" / "results" / "spc"
SERIES = OUT / "series"
V2 = ROOT / "p115_detectors" / "results_real_v2" / "flags"
V3 = ROOT / "p115_detectors" / "results_real_v3" / "flags"
BAYES = ROOT / "m2" / "results" / "bayes" / "real_gaps.parquet"
COLS = ["ts_recv", "flags", "bid_px_00", "ask_px_00"]


def to_ns(col: pd.Series) -> np.ndarray:
    return col.astype("datetime64[ns, UTC]").astype("int64").to_numpy()


def load_arrays(piece, symbol, t0, t1, data_root, columns=COLS):
    frame = load_symbol_slice(piece, symbol, t0, t1, data_root=data_root, columns=columns)
    if "ts_recv" not in frame.columns:
        frame = frame.reset_index()
    return frame


def candidate_buckets(venue, symbol, day, series: pd.DataFrame, bayes: pd.DataFrame | None):
    """Existing point-check candidates per bucket (stored v2 spread flags, v3 Poisson gaps, Bayesian gaps)."""
    out = pd.DataFrame({"bucket": series["bucket"]})
    def count(ts_ns, name):
        b = pd.Series(bucket_index(ts_ns, venue)).value_counts() if len(ts_ns) else pd.Series(dtype=int)
        out[name] = out["bucket"].map(b).fillna(0).astype(int)
    f2 = V2 / f"{venue}_{symbol}_{day}.parquet"
    sp = pd.read_parquet(f2, columns=["ts_recv", "check", "subtype"]) if f2.exists() else pd.DataFrame(columns=["ts_recv", "check", "subtype"])
    sp = sp[(sp["check"] == "unusual_spreads") & (sp["subtype"] == "wide")]
    count(to_ns(sp["ts_recv"]) if len(sp) else np.array([], np.int64), "cand_spread")
    f3 = V3 / f"{venue}_{symbol}_{day}.parquet"
    ms = pd.read_parquet(f3, columns=["ts_recv"]) if f3.exists() else pd.DataFrame(columns=["ts_recv"])
    count(to_ns(ms["ts_recv"]) if len(ms) else np.array([], np.int64), "cand_gap_poisson")
    if bayes is not None:
        bz = bayes[(bayes["symbol"] == symbol) & (bayes["day"] == day) & bayes["bayes"]]
        count(bz["ts_recv_ns"].to_numpy(), "cand_gap_bayes")
    return out


def run_series(data_root):
    v3 = load_v3()
    ticks = load_ticks(data_root)
    bayes = pd.read_parquet(BAYES, columns=["symbol", "day", "ts_recv_ns", "bayes"]) if BAYES.exists() else None
    SERIES.mkdir(parents=True, exist_ok=True)
    for venue, piece, symbol, day, t0, t1 in tasks():
        path = SERIES / f"{venue}_{symbol}_{day}.parquet"
        if path.exists():
            continue
        t_ = time.time()
        fr = load_arrays(piece, symbol, t0, t1, data_root)
        tick = float(ticks.get(symbol, 0.01) or 0.01)
        s = build_series(to_ns(fr["ts_recv"]), fr["flags"].to_numpy().astype(np.int64),
                         pd.to_numeric(fr["bid_px_00"], errors="coerce").to_numpy(),
                         pd.to_numeric(fr["ask_px_00"], errors="coerce").to_numpy(), tick, venue, day, v3)
        n = len(fr); del fr
        s = s.merge(candidate_buckets(venue, symbol, day, s, bayes), on="bucket", how="left")
        s = s.assign(venue=venue, symbol=symbol, day=day, day_type=day_type_of(day), tick=tick)
        s.to_parquet(path, index=False)
        print(dict(symbol=symbol, day=day, records=n, buckets=len(s), secs=round(time.time() - t_, 1)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["series", "charts", "inject", "plots"], required=True)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    a = ap.parse_args()
    if a.mode == "series":
        run_series(a.data_root)
    elif a.mode == "charts":
        from m2.spc.evaluate import run_charts_real
        run_charts_real()
    elif a.mode == "inject":
        from m2.spc.inject import run_inject
        run_inject(a.data_root)
    else:
        from m2.spc.report import main as rep
        rep()


if __name__ == "__main__":
    main()
