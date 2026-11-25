"""Per-symbol, per-session one-minute bucket series: record counts and quoted spread.

Buckets are in-session local minutes (v3 SessionClock / calendar presets):
equities 09:30-16:00 ET (bucket 0 = 09:30), CME Globex 17:00 CT to 16:00 CT next day
(bucket 0 = 17:00 CT; the 16:00-17:00 CT break has no buckets). The bucket index is the
time-of-day key used for the baselines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from integration.combine import load_v3

F_LAST = 128
NS = 1_000_000_000
SESSION_OPEN_MIN = {"XNAS": 9 * 60 + 30, "GLBX": 17 * 60}
TZ = {"XNAS": "America/New_York", "GLBX": "America/Chicago"}


def calendar(v3, venue):
    return v3.config.equities_config().missing.calendar if venue == "XNAS" else v3.config.cme_calendar()


def utc_offset_ns(t_ns: np.ndarray, tz: str) -> np.ndarray:
    """Local-time offset per record (vectorised over the distinct UTC hours)."""
    hours = t_ns // (3600 * NS)
    uniq, inv = np.unique(hours, return_inverse=True)
    stamps = pd.to_datetime(uniq * 3600 * NS, unit="ns", utc=True).tz_convert(tz)
    off = np.array([s.utcoffset().total_seconds() for s in stamps]) * NS
    return off.astype(np.int64)[inv]


def bucket_index(t_ns: np.ndarray, venue: str) -> np.ndarray:
    local_min = ((t_ns + utc_offset_ns(t_ns, TZ[venue])) // (60 * NS)) % 1440
    return ((local_min - SESSION_OPEN_MIN[venue]) % 1440).astype(np.int64)


def session_buckets(venue: str, day: str, v3=None) -> np.ndarray:
    """All in-session bucket indices of a session (so empty minutes appear with count 0)."""
    v3 = v3 or load_v3()
    if venue == "XNAS":
        start = pd.Timestamp(f"{day} 09:30", tz=TZ[venue])
    else:
        start = pd.Timestamp(day[:10], tz=TZ[venue]) - pd.Timedelta(hours=7)  # 17:00 CT the evening before
    grid = start.tz_convert("UTC").value + np.arange(1440) * 60 * NS
    clk = v3.detectors.SessionClock.build(calendar(v3, venue), int(grid[0]), int(grid[-1]))
    mid = grid + 30 * NS
    inside = clk.in_session(grid) & clk.in_session(mid)
    return bucket_index(grid[inside], venue)


def build_series(t_ns, flags, bid, ask, tick: float, venue: str, day: str, v3=None) -> pd.DataFrame:
    """Bucket series from raw arrays of one symbol-day.

    n_records: all in-session records; n_quotes: F_LAST two-sided records;
    spread_ticks: median positive F_LAST spread in the bucket, carried forward over empty buckets
    (the last quote is still standing); spread_obs: whether the bucket had its own quote.
    """
    v3 = v3 or load_v3()
    allb = session_buckets(venue, day, v3)
    out = pd.DataFrame({"bucket": allb})
    if len(t_ns) == 0:
        out["n_records"] = 0; out["n_quotes"] = 0; out["spread_ticks"] = np.nan; out["spread_obs"] = False
        return out
    clk = v3.detectors.SessionClock.build(calendar(v3, venue), int(t_ns.min()), int(t_ns.max()))
    ins = clk.in_session(t_ns)
    b = bucket_index(t_ns, venue)
    ok = ins & np.isin(b, allb)
    n = pd.Series(b[ok]).value_counts()
    out["n_records"] = out["bucket"].map(n).fillna(0).astype(np.int64)
    sp = (ask - bid) / tick
    q = ok & ((flags & F_LAST) != 0) & np.isfinite(sp) & (bid > 0) & (ask > 0)
    pos = q & (sp > 1e-9)
    nq = pd.Series(b[q]).value_counts()
    med = pd.Series(sp[pos]).groupby(b[pos]).median()
    out["n_quotes"] = out["bucket"].map(nq).fillna(0).astype(np.int64)
    out["spread_obs"] = out["bucket"].isin(med.index)
    out["spread_ticks"] = out["bucket"].map(med).ffill()
    return out
