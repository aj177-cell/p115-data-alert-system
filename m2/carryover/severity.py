"""Item 3: one severity scale across engine_m1, detectors_v3, the Bayesian gap check and SPC charts.

Additive: ``apply`` adds ``severity_m2``, ``level_m2`` and ``map_key`` next to the native
``severity``/``level`` columns, which are never changed. ``SEVERITY_MAP`` is also written as
m2/carryover/results/severity_map.csv for review.

Scale (same numbers as integration.combine.LEVEL):
  3 high    the record or interval is wrong as delivered; exclude it from analysis until reviewed
  2 medium  probably a data problem; review before use
  1 low     unusual but often legitimate market or venue behaviour; context only
  0 info    expected behaviour, kept for audit
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = {"info": 0, "low": 1, "medium": 2, "high": 3}
MAX_GAP_S = 60.0  # v3 missing.max_gap_s, reused so 'high' means the same for v3 and Bayes

# (source, rule, subtype, condition, severity_m2, native, rationale); '*' matches anything.
# The first matching row wins, so specific rows come before generic ones.
SEVERITY_MAP: list[tuple] = [
    ("engine_m1", "integrity", "*", "", "high", "critical", "malformed record (zero/undefined side, bad timestamp)"),
    ("engine_m1", "crossed_quote", "*", "f_last_false", "info", "warning", "intermediate book state; v3 skips these records"),
    ("engine_m1", "crossed_quote", "*", "", "high", "warning", "same evidence as v3 crossed_books/crossed (high)"),
    ("engine_m1", "time_order", "*", "", "low", "warning", "same evidence as v3 ts_event_inversion (low); ts_event is not guaranteed monotonic"),
    ("detectors_v3", "crossed_books", "crossed", "", "high", "high", ""),
    ("detectors_v3", "crossed_books", "locked", "", "low", "medium", "locked books occur in fast markets; not wrong data"),
    ("detectors_v3", "extreme_prices", "*", "", "high", "high", ""),
    ("detectors_v3", "timestamp_inversions", "ts_recv_inversion", "", "high", "high/info", "info kept when F_BAD_TS_RECV"),
    ("detectors_v3", "timestamp_inversions", "negative_latency", "", "high", "high", ""),
    ("detectors_v3", "timestamp_inversions", "latency_outlier", "", "medium", "medium", ""),
    ("detectors_v3", "timestamp_inversions", "ts_event_inversion", "", "low", "low", ""),
    ("detectors_v3", "timestamp_inversions", "ts_in_delta_saturated", "", "low", "low", ""),
    ("detectors_v3", "missing_intervals", "trade_pause", "", "info", "medium", "trades schema, quotes active inside the pause (item 4)"),
    ("detectors_v3", "missing_intervals", "trade_pause_below_floor", "", "info", "medium", "trades schema, below the trades-only floor (item 4)"),
    ("detectors_v3", "missing_intervals", "gap", "value_gt_max_gap", "high", "high", "longer than max_gap_s = 60 s"),
    ("detectors_v3", "missing_intervals", "gap", "", "medium", "medium", ""),
    ("detectors_v3", "missing_intervals", "*", "", "low", "low", "incomplete_bar_*: OHLCV omits minutes with no trades"),
    ("detectors_v3", "stale_quotes", "quiet", "", "info", "info", ""),
    ("detectors_v3", "stale_quotes", "*", "", "medium", "high", "a stale book is usually a quiet name; review, not exclude"),
    ("detectors_v3", "unusual_spreads", "zero_or_negative", "", "high", "high", ""),
    ("detectors_v3", "unusual_spreads", "*", "", "low", "medium", "wide spreads at open/close and on news are market behaviour"),
    ("detectors_v3", "sequence_discontinuities", "regression", "", "high", "high", ""),
    ("detectors_v3", "sequence_discontinuities", "*", "", "medium", "medium/info", "info kept when gap_mode=report_only"),
    ("detectors_v3", "vendor_flags", "*", "", "low", "native", "Databento flagged the record itself"),
    ("m2_bayes", "missing_intervals", "*", "p_ge_099_gap_gt_max_gap", "high", "n/a", "P(outage) >= 0.99 and longer than 60 s"),
    ("m2_bayes", "missing_intervals", "*", "", "medium", "n/a", "P(outage) >= 0.5 (the candidate threshold)"),
    ("m2_spc", "*", "shewhart", "", "low", "n/a", "single-minute excursion"),
    ("m2_spc", "*", "*", "episode_ge_15", "medium", "n/a", "persistent shift (>= 15 one-minute buckets)"),
    ("m2_spc", "*", "*", "", "low", "n/a", "regime change in rate or spread; context for point checks"),
]
COLUMNS = ["source", "rule", "subtype", "condition", "severity_m2", "native_severity", "rationale"]


def _cond(name: str, row) -> bool:
    if not name:
        return True
    if name == "f_last_false":
        return row.get("f_last") is False or (isinstance(row.get("f_last"), (bool, np.bool_)) and not row.get("f_last"))
    if name == "value_gt_max_gap":
        return float(row.get("value", np.nan)) > MAX_GAP_S
    if name == "p_ge_099_gap_gt_max_gap":
        return float(row.get("p_outage", 0.0)) >= 0.99 and float(row.get("value", 0.0)) > MAX_GAP_S
    if name == "episode_ge_15":
        return float(row.get("value", 0.0)) >= 15
    raise ValueError(name)


def lookup(row) -> tuple[str, str]:
    for src, rule, sub, cond, sev, _native, _why in SEVERITY_MAP:
        if src not in ("*", row.get("source")) or rule not in ("*", row.get("rule")):
            continue
        if sub != "*" and sub != row.get("subtype"):
            continue
        if _cond(cond, row):
            return sev, f"{src}/{rule}/{sub}/{cond or '-'}"
    native = str(row.get("severity", "info"))
    return ("medium" if native == "warning" else "high" if native == "critical" else native), "native"


def apply(unified: pd.DataFrame) -> pd.DataFrame:
    out = unified.copy()
    if out.empty:
        out["severity_m2"] = pd.Series(dtype=object); out["level_m2"] = pd.Series(dtype=float); out["map_key"] = pd.Series(dtype=object)
        return out
    res = [lookup(r) for r in out.to_dict("records")]
    out["severity_m2"] = [s for s, _ in res]
    out["level_m2"] = [float(LEVELS[s]) for s, _ in res]
    out["map_key"] = [k for _, k in res]
    return out


def bayes_rows(gaps: pd.DataFrame) -> pd.DataFrame:
    """Bayesian candidates (m2/results/bayes/real_gaps.parquet layout) as unified-style rows."""
    g = gaps[gaps["bayes"]].copy()
    ts = pd.to_datetime(g["ts_recv_ns"], utc=True)
    return pd.DataFrame({"source": "m2_bayes", "mode": "batch", "rule": "missing_intervals", "subtype": "bayes_outage",
                         "severity": None, "level": np.nan, "ts_event": ts, "ts_recv": ts, "symbol": g["symbol"].to_numpy(),
                         "value": g["gap_s"].to_numpy(), "threshold": 0.5, "p_outage": g["p_outage"].to_numpy(),
                         "detail": [f"p_outage={p:.3g} gap={s:.3f}s" for p, s in zip(g["p_outage"], g["gap_s"])]})


def spc_rows(signals: pd.DataFrame, gap: int = 15) -> pd.DataFrame:
    """SPC signal list (symbol, day, kind, chart, bucket) grouped into episodes: signals of one chart
    no more than ``gap`` buckets apart. value = episode length in buckets (minutes)."""
    rows = []
    for (sym, day, kind, chart), g in signals.groupby(["symbol", "day", "kind", "chart"], sort=False):
        b = np.sort(g["bucket"].to_numpy())
        cut = np.flatnonzero(np.diff(b) > gap)
        for s, e in zip(np.r_[0, cut + 1], np.r_[cut, len(b) - 1]):
            rows.append({"source": "m2_spc", "mode": "batch", "rule": f"spc_{kind}", "subtype": chart, "severity": None,
                         "level": np.nan, "symbol": sym, "day": day, "value": float(b[e] - b[s] + 1),
                         "threshold": np.nan, "detail": f"buckets {b[s]}-{b[e]}, {e - s + 1} signals"})
    return pd.DataFrame(rows)


def write_map(path: Path) -> pd.DataFrame:
    table = pd.DataFrame(SEVERITY_MAP, columns=COLUMNS)
    table["level_m2"] = table["severity_m2"].map(LEVELS)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table
