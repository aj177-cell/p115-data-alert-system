"""Item 4: the ~5.2 s pauses v3 flags on the bundled trades sample.

Evidence (``census_all``): on the real XNAS mbp-1 pull, trade-to-trade pauses longer than 5 s
happen hundreds of times per regular session for AAPL and SPY, and almost every one has quote
records inside it. A pause in prints is not a pause in the feed.

Opt-in policy (``reclassify``), applied to a unified table from a trades-schema run:
  * with a quotes/mbp-1 frame for the same window: a v3 missing_intervals candidate whose window
    contains quote records becomes subtype 'trade_pause', severity 'info', level 0;
  * without quotes: a candidate shorter than a trades-only floor (per symbol, default from the
    real pull's 0.9999 quantile of RTH trade gaps) becomes 'trade_pause_below_floor' / info.
Rows are never removed; ``level`` and ``severity`` change only for the rows the policy matches.
"""
from __future__ import annotations

from m2.carryover._util import ns

import re

import numpy as np
import pandas as pd

NS = 1_000_000_000
RTH_TZ = "America/New_York"
EARLY_CLOSE = {"2025-11-28": "13:00"}
# 0.9999 quantile of RTH trade-to-trade gaps, maximum over the normal week 2026-10-05..09 (seconds,
# rounded up), from m2/carryover/results/trade_pause_census.csv. Used only when no quotes are given.
DEFAULT_FLOOR_S = 15.0


def _rth(day: str):
    close = EARLY_CLOSE.get(day, "16:00")
    a = pd.Timestamp(f"{day} 09:30", tz=RTH_TZ).tz_convert("UTC").value
    b = pd.Timestamp(f"{day} {close}", tz=RTH_TZ).tz_convert("UTC").value
    return a, b


def pause_stats(frame: pd.DataFrame, day: str, threshold_s: float = 5.0) -> dict:
    """RTH trade-gap statistics for one symbol-day of mbp-1 (trades = action 'T')."""
    t = np.asarray(ns(frame["ts_recv"] if "ts_recv" in frame else pd.Series(frame.index)))
    o = np.argsort(t, kind="stable"); t = t[o]
    act = frame["action"].to_numpy()[o]
    a, b = _rth(day)
    tr = t[(act == "T") & (t >= a) & (t < b)]
    if len(tr) < 2:
        return dict(trades=int(len(tr)))
    g = np.diff(tr) / NS
    big = np.flatnonzero(g > threshold_s)
    nonT = np.sort(t[act != "T"])
    inside = np.searchsorted(nonT, tr[big + 1], side="left") - np.searchsorted(nonT, tr[big], side="right")
    return dict(trades=int(len(tr)), trades_per_s=round(len(tr) / ((b - a) / NS), 3),
                pauses_gt=int(len(big)), pauses_with_quotes=int((inside > 0).sum()),
                median_records_inside=float(np.median(inside)) if len(big) else np.nan,
                max_pause_s=round(float(g.max()), 3), q9999_s=round(float(np.quantile(g, 0.9999)), 3),
                q999_s=round(float(np.quantile(g, 0.999)), 3))


def census_all(data_root=None, threshold_s: float = 5.0) -> pd.DataFrame:
    from m2.extract import DEFAULT_DATA_ROOT, day_type_of, load_symbol_slice, tasks
    rows = []
    for venue, piece, symbol, day, start, end in tasks():
        if venue != "XNAS":
            continue
        fr = load_symbol_slice(piece, symbol, start, end, data_root or DEFAULT_DATA_ROOT, columns=["ts_recv", "action"])
        rows.append(dict(symbol=symbol, day=day, day_type=day_type_of(day), **pause_stats(fr, day, threshold_s)))
    return pd.DataFrame(rows)


def floors_from_census(census: pd.DataFrame, day_type: str = "normal_week") -> dict[str, float]:
    c = census[census["day_type"] == day_type]
    return {s: float(np.ceil(v)) for s, v in c.groupby("symbol")["q9999_s"].max().items()}


_GAP_START = re.compile(r"gap_start=(\S+)")


def reclassify(unified: pd.DataFrame, schema: str, quotes: pd.DataFrame | None = None,
               floor_s: float | dict[str, float] | None = DEFAULT_FLOOR_S) -> pd.DataFrame:
    out = unified.copy()
    if schema != "trades":
        return out
    sel = np.flatnonzero(((out["source"] == "detectors_v3") & (out["rule"] == "missing_intervals")).to_numpy())
    if not len(sel):
        return out
    qt = qs = None
    if quotes is not None and len(quotes):
        qt = np.asarray(ns(quotes["ts_recv"] if "ts_recv" in quotes else pd.Series(quotes.index, index=quotes.index)))
        qs = quotes["symbol"].to_numpy() if "symbol" in quotes else None
    for i in sel:
        row = out.iloc[i]
        m = _GAP_START.search(str(row["detail"]))
        if not m:
            continue
        g0 = pd.Timestamp(m.group(1)).value
        g1 = pd.Timestamp(row["ts_recv"]).value
        if qt is not None:
            mask = (qt > g0) & (qt < g1)
            if qs is not None:
                mask &= qs == row["symbol"]
            n = int(mask.sum())
            if n > 0:
                out.iat[i, out.columns.get_loc("subtype")] = "trade_pause"
                out.iat[i, out.columns.get_loc("severity")] = "info"
                out.iat[i, out.columns.get_loc("level")] = 0.0
                out.iat[i, out.columns.get_loc("detail")] = f"{row['detail']} [trade pause: {n} quote records inside]"
            continue
        floor = floor_s.get(row["symbol"], DEFAULT_FLOOR_S) if isinstance(floor_s, dict) else floor_s
        if floor is not None and float(row["value"]) < floor:
            out.iat[i, out.columns.get_loc("subtype")] = "trade_pause_below_floor"
            out.iat[i, out.columns.get_loc("severity")] = "info"
            out.iat[i, out.columns.get_loc("level")] = 0.0
            out.iat[i, out.columns.get_loc("detail")] = f"{row['detail']} [below trades-only floor {floor:g}s]"
    return out
