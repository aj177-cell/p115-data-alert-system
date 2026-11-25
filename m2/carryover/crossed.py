"""Item 1: engine crossed_quote alerts on records without F_LAST.

Sidecar, opt-in. ``flast_aware_crossed`` post-processes the unified table (integration.combine
UNIFIED_COLUMNS) and never touches the engine. Modes:

  'mark'   (current behaviour)  keep every alert; f_last column already tells them apart
  'demote'                     crossed_quote alerts on records without F_LAST become severity 'info', level 0
  'drop'                       those alerts are removed
  'persist'                    additionally require the cross to last >= min_duration_s on F_LAST records
                               (the v3 crossed_books definition), judged from the frame

``crossed_census`` counts crossed records by F_LAST on raw frames, vectorised, with the
engine's own predicate (bid > ask, both > 0, both sizes non-zero).
"""
from __future__ import annotations

from m2.carryover._util import ns

import numpy as np
import pandas as pd

F_LAST = 128
NS = 1_000_000_000


def crossed_mask(frame: pd.DataFrame) -> np.ndarray:
    bid = pd.to_numeric(frame["bid_px_00"], errors="coerce").to_numpy(dtype="float64")
    ask = pd.to_numeric(frame["ask_px_00"], errors="coerce").to_numpy(dtype="float64")
    bs = pd.to_numeric(frame.get("bid_sz_00", pd.Series(1, index=frame.index)), errors="coerce").to_numpy()
    as_ = pd.to_numeric(frame.get("ask_sz_00", pd.Series(1, index=frame.index)), errors="coerce").to_numpy()
    with np.errstate(invalid="ignore"):
        return np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0) & (bs != 0) & (as_ != 0) & (bid > ask)


def crossed_census(frame: pd.DataFrame) -> dict:
    cr = crossed_mask(frame)
    fl = (frame["flags"].to_numpy().astype(np.int64) & F_LAST) != 0
    return {"records": int(len(frame)), "f_last_records": int(fl.sum()), "crossed": int(cr.sum()),
            "crossed_f_last": int((cr & fl).sum()), "crossed_no_f_last": int((cr & ~fl).sum())}


def _cross_runs(frame: pd.DataFrame, min_duration_s: float) -> np.ndarray:
    """[start, end] ns of crossed runs on F_LAST records lasting >= min_duration_s (per instrument)."""
    work = frame.reset_index() if "ts_recv" not in frame.columns else frame
    t = ns(work["ts_recv"]).to_numpy()
    fl = (work["flags"].to_numpy().astype(np.int64) & F_LAST) != 0
    cr = crossed_mask(work)
    iid = work["instrument_id"].to_numpy() if "instrument_id" in work else np.zeros(len(work))
    out = []
    for k in np.unique(iid):
        m = (iid == k) & fl
        tt, cc = t[m], cr[m]
        o = np.argsort(tt, kind="stable"); tt, cc = tt[o], cc[o]
        if not len(tt):
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, cc.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            end = tt[e] if e < len(tt) else tt[e - 1]  # run ends when the next F_LAST record is uncrossed
            if end - tt[s] >= min_duration_s * NS:
                out.append((tt[s], end))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def flast_aware_crossed(unified: pd.DataFrame, mode: str = "demote", frame: pd.DataFrame | None = None,
                        min_duration_s: float = 0.001, tolerance_s: float = 1.0) -> pd.DataFrame:
    """Return a copy of the unified table with engine crossed_quote alerts handled by ``mode``."""
    if mode not in ("mark", "demote", "drop", "persist"):
        raise ValueError(mode)
    out = unified.copy()
    is_x = (out["source"] == "engine_m1") & (out["rule"] == "crossed_quote")
    no_fl = is_x & out["f_last"].eq(False)
    if mode == "mark":
        return out
    if mode == "demote":
        out.loc[no_fl, "severity"] = "info"; out.loc[no_fl, "level"] = 0
        return out
    out = out[~no_fl].reset_index(drop=True)
    if mode == "drop":
        return out
    if frame is None:
        raise ValueError("mode='persist' needs the frame")
    runs = _cross_runs(frame, min_duration_s)
    is_x = ((out["source"] == "engine_m1") & (out["rule"] == "crossed_quote")).to_numpy()
    ts = ns(out["ts_event"]).to_numpy()
    tol = int(tolerance_s * NS)
    keep = np.ones(len(out), bool)
    for i in np.flatnonzero(is_x):
        keep[i] = bool(((runs[:, 0] - tol <= ts[i]) & (ts[i] <= runs[:, 1] + tol)).any()) if len(runs) else False
    return out[keep].reset_index(drop=True)


def census_all(data_root=None) -> pd.DataFrame:
    """Crossed-record census on every symbol-day of the real pull (cheap columns only)."""
    from m2.extract import DEFAULT_DATA_ROOT, load_symbol_slice, tasks, day_type_of
    root = data_root or DEFAULT_DATA_ROOT
    cols = ["ts_recv", "instrument_id", "flags", "action", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    rows = []
    for venue, piece, symbol, day, start, end in tasks():
        fr = load_symbol_slice(piece, symbol, start, end, root, columns=cols)
        c = crossed_census(fr)
        cr = crossed_mask(fr); fl = (fr["flags"].to_numpy().astype(np.int64) & F_LAST) != 0
        # longest crossed run on F_LAST records (seconds)
        runs = _cross_runs(fr, 0.0)
        c.update(venue=venue, symbol=symbol, day=day, day_type=day_type_of(day),
                 crossed_runs_f_last=int(len(runs)),
                 crossed_runs_f_last_ge_1ms=int(((runs[:, 1] - runs[:, 0]) >= 1_000_000).sum()) if len(runs) else 0,
                 max_run_f_last_s=float((runs[:, 1] - runs[:, 0]).max() / NS) if len(runs) else 0.0)
        rows.append(c)
        del fr, cr, fl
    return pd.DataFrame(rows)
