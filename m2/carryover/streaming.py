"""Item 2: real-time (record-at-a-time) versions of the v3 batch checks.

Nothing here changes the batch checks. ``CAUSALITY`` records, check by check, whether the batch
code uses only past data and when a live system could raise the same candidate. Three streaming
classes cover the checks where a live version is useful and the batch rule is causal or nearly so:

  StreamingGapMonitor     v3 missing_intervals (Poisson, adaptive): the same candidates at the
                          record that closes the gap (tested equal to batch), plus an
                          *open-gap* alarm from ``poll(now)`` as soon as the gap passes its threshold.
  StreamingCrossed        v3 crossed_books/crossed on F_LAST records: the same run starts as batch
                          (tested), raised as soon as the cross has lasted min_duration_s.
  TrailingLatencyOutlier  replaces the whole-day median/MAD of latency_outlier (look-ahead) with
                          a trailing window: the only change needed to make that rule causal.

The Bayesian gap check (m2/bayes) is already causal: the discounted Gamma state at gap i uses only
gaps before i. SPC charts are causal per minute; only their Phase I baseline must be built from
*earlier* days in a live system (the batch study used other days of the same week).
"""
from __future__ import annotations

from m2.carryover._util import ns

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NS = 1_000_000_000
F_LAST = 128
MAD_SCALE = 1.4826

# check, subtype -> (causal in batch?, earliest live detection, what a live version needs)
CAUSALITY = [
    ("crossed_books", "crossed/locked", "yes", "run start + min_duration_s (1 ms)", "StreamingCrossed"),
    ("extreme_prices", "robust_z_*", "yes (window shifted by one)", "the record itself", "none: rolling median/MAD of prior W returns is already streaming"),
    ("extreme_prices", "static_band_*", "yes", "the record itself", "none"),
    ("timestamp_inversions", "ts_recv/ts_event inversion, negative_latency", "yes", "the record itself", "none"),
    ("timestamp_inversions", "latency_outlier", "NO: whole-day median/MAD per key", "the record itself", "TrailingLatencyOutlier"),
    ("missing_intervals", "gap", "yes (rate from prior n gaps)", "gap start + threshold (open-gap alarm), batch raises at gap end", "StreamingGapMonitor"),
    ("stale_quotes", "stale/quiet", "NO: adaptive theta is a whole-day quantile of book ages", "book change + theta, once min_trades prints arrive", "trailing quantile of ages (not built)"),
    ("unusual_spreads", "wide", "yes (rolling quantile shifted by one)", "the record itself", "none"),
    ("unusual_spreads", "zero_or_negative", "yes", "the record itself", "none"),
    ("sequence_discontinuities", "gap/regression", "yes", "the next record", "none"),
    ("m2_bayes", "bayes_outage", "yes (discounted state from earlier gaps)", "gap end; survival form P(outage | gap > elapsed) could alarm earlier", "none for gap-end candidates"),
    ("m2_spc", "cusum/ewma/shewhart", "per minute yes; Phase I baseline used other days of the same week", "bucket close (1 min)", "baseline from earlier days only"),
]


@dataclass
class StreamingGapMonitor:
    """Live version of v3 missing_intervals (model='poisson', adaptive=True).

    update(t_ns, key) takes records in ts_recv order per key and returns a candidate dict when the
    gap that the record closes exceeds the threshold the batch rule would use. poll(now_ns) returns
    open-gap alarms for keys whose current silence already exceeds their threshold.
    ``clock`` maps ns to session-clock ns (v3 SessionClock.clock); identity by default.
    """
    alpha: float = 1e-6
    rate_window: int = 500
    min_gap_s: float = 5.0
    max_gap_s: float = 60.0
    clock: object = None
    _hist: dict = field(default_factory=dict)
    _open_alarmed: dict = field(default_factory=dict)

    def _c(self, t):
        if self.clock is None:
            return int(t)
        return int(self.clock(np.array([t], dtype=np.int64))[0])

    def threshold(self, key) -> float:
        h = self._hist.get(key)
        n = self.rate_window
        if h is None or len(h) < n + 1:
            return self.max_gap_s
        span = (h[-1] - h[0]) / NS  # prev - prev_n over the last n gaps
        if span <= 0:
            return self.max_gap_s
        lam = n / span
        return float(np.clip(-np.log(self.alpha) / lam, self.min_gap_s, self.max_gap_s))

    def update(self, t_ns: int, key=0) -> dict | None:
        s = self._c(t_ns)
        h = self._hist.setdefault(key, deque(maxlen=self.rate_window + 1))
        out = None
        if h:
            gap = (s - h[-1]) / NS
            thr = self.threshold(key)
            if gap > thr:
                out = {"key": key, "ts_recv_ns": int(t_ns), "gap_s": gap, "threshold_s": thr}
        h.append(s)
        self._open_alarmed.pop(key, None)
        return out

    def poll(self, now_ns: int) -> list[dict]:
        s = self._c(now_ns)
        alarms = []
        for key, h in self._hist.items():
            if not h or self._open_alarmed.get(key):
                continue
            thr = self.threshold(key)
            if (s - h[-1]) / NS > thr:
                alarms.append({"key": key, "now_ns": int(now_ns), "silence_s": (s - h[-1]) / NS, "threshold_s": thr})
                self._open_alarmed[key] = True
        return alarms


@dataclass
class StreamingCrossed:
    """Live version of v3 crossed_books 'crossed' on F_LAST records (key = publisher, instrument).

    Batch flags a run if (time of the first record after the run - run start) >= min_duration_s.
    Live: the candidate is raised at the first moment that is known — a later record of the same key
    (crossed or not) arriving at least min_duration_s after the start, or ``poll(now)``.
    """
    min_duration_s: float = 0.001
    _start: dict = field(default_factory=dict)
    _done: dict = field(default_factory=dict)

    def update(self, t_ns: int, bid: float, ask: float, key=0, flags: int = F_LAST) -> dict | None:
        if not (int(flags) & F_LAST) or not (np.isfinite(bid) and np.isfinite(ask)):
            return None
        out = None
        start = self._start.get(key)
        if start is not None and not self._done.get(key) and t_ns - start[0] >= self.min_duration_s * NS:
            out = {"key": key, "start_ns": start[0], "detected_ns": int(t_ns), "cross": start[1]}
            self._done[key] = True
        if bid - ask > 1e-9:
            if start is None:
                self._start[key] = (int(t_ns), bid - ask); self._done[key] = False
        else:
            self._start.pop(key, None); self._done.pop(key, None)
        return out

    def poll(self, now_ns: int) -> list[dict]:
        out = []
        for key, (t0, x) in self._start.items():
            if not self._done.get(key) and now_ns - t0 >= self.min_duration_s * NS:
                out.append({"key": key, "start_ns": t0, "detected_ns": int(now_ns), "cross": x}); self._done[key] = True
        return out


@dataclass
class TrailingLatencyOutlier:
    """Causal latency_outlier: latency > max(med + k*1.4826*MAD, med + min_excess) over the last
    ``window`` latencies of the key (v3 uses the whole day of the key)."""
    k: float = 10.0
    min_excess_ns: int = 1_000_000
    window: int = 5000
    min_periods: int = 500
    _buf: dict = field(default_factory=dict)

    def update(self, latency_ns: int, key=0) -> tuple[bool, float]:
        b = self._buf.setdefault(key, deque(maxlen=self.window))
        hit, thr = False, np.nan
        if len(b) >= self.min_periods:
            arr = np.fromiter(b, dtype=np.float64)
            med = float(np.median(arr)); mad = float(np.median(np.abs(arr - med)))
            thr = max(med + self.k * MAD_SCALE * mad, med + self.min_excess_ns)
            hit = latency_ns > thr
        b.append(float(latency_ns))
        return hit, thr


def replay_gaps(frame: pd.DataFrame, monitor: StreamingGapMonitor) -> pd.DataFrame:
    """Feed a frame (ts_recv index or column) through the monitor in (key, ts_recv) order, as batch does."""
    w = frame.reset_index() if "ts_recv" not in frame.columns else frame.copy()
    t = ns(w["ts_recv"]).to_numpy()
    keys = list(zip(w["publisher_id"].to_numpy(), w["instrument_id"].to_numpy()))
    order = sorted(range(len(t)), key=lambda i: (keys[i], t[i]))
    rows = [r for i in order if (r := monitor.update(int(t[i]), keys[i])) is not None]
    return pd.DataFrame(rows, columns=["key", "ts_recv_ns", "gap_s", "threshold_s"])


def replay_crossed(frame: pd.DataFrame, det: StreamingCrossed) -> pd.DataFrame:
    w = frame.reset_index() if "ts_recv" not in frame.columns else frame.copy()
    w = w[(w["flags"].astype(int) & F_LAST) != 0]
    if "action" in w:
        w = w[w["action"] != "R"]
    t = ns(w["ts_recv"]).to_numpy()
    b = pd.to_numeric(w["bid_px_00"], errors="coerce").to_numpy(); a = pd.to_numeric(w["ask_px_00"], errors="coerce").to_numpy()
    keys = list(zip(w["publisher_id"].to_numpy(), w["instrument_id"].to_numpy()))
    order = sorted(range(len(t)), key=lambda i: (keys[i], t[i]))
    rows = []
    last_key = None
    for i in order:
        if last_key is not None and keys[i] != last_key:
            rows += det.poll(np.iinfo(np.int64).max)  # end of key: an open-ended run counts as batch does
        r = det.update(int(t[i]), b[i], a[i], keys[i]); last_key = keys[i]
        if r:
            rows.append(r)
    return pd.DataFrame(rows, columns=["key", "start_ns", "detected_ns", "cross"])


def detected_at(unified: pd.DataFrame, min_duration_s: float = 0.001) -> pd.DataFrame:
    """Annotate unified rows with the earliest time a live system could raise them (additive columns
    ``detected_at_live`` and ``live_lead_s`` = batch ts_recv - live time; positive = live is earlier)."""
    out = unified.copy()
    ts = pd.to_datetime(out["ts_recv"], utc=True)
    live = ts.copy()
    is_gap = (out["rule"] == "missing_intervals") & out["subtype"].astype(str).str.startswith(("gap", "trade_pause"))
    if is_gap.any():
        start = pd.to_datetime(out.loc[is_gap, "detail"].astype(str).str.extract(r"gap_start=(\S+)")[0], utc=True, format="ISO8601")
        live.loc[is_gap] = start + pd.to_timedelta(out.loc[is_gap, "threshold"].astype(float), unit="s")
    is_x = out["rule"] == "crossed_books"
    live.loc[is_x] = ts[is_x] + pd.Timedelta(seconds=min_duration_s)
    out["detected_at_live"] = live
    out["live_lead_s"] = (ts - live).dt.total_seconds()
    return out
