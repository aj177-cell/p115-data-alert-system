"""Seven market-data quality detectors for Databento mbp-1 (and trades/mbo).

Call ``run_all`` from ``alert_system.quality``. Streaming checks stay in
``alert_system.detectors`` and are a separate implementation.

Input: a DataFrame as produced by the team pipeline (``download_and_show`` ->
``data.parquet``; ``pd.read_parquet`` restores the ``ts_recv`` index), i.e. the
output of ``DBNStore.to_df()`` with prices as float and timestamps as UTC
datetimes. Raw frames (int64 fixed-point prices with the ``UNDEF_PRICE``
sentinel, uint64 nanosecond timestamps, ``ts_recv`` as a column) are also
accepted; :func:`normalize` converts both into one internal layout.

Every detector returns a tidy flag table with the columns in ``FLAG_COLUMNS``:
``ts_recv, ts_event, publisher_id, instrument_id, symbol, check, subtype,
severity, value, threshold, detail``.

Conventions
-----------
* Records are grouped by ``(publisher_id, instrument_id)``; within a key the
  original file order is kept (so ts_recv inversions remain visible).
* Book-state checks (crossed, spreads, stale, mid-based extreme prices) use
  only records with ``F_LAST`` (128) set and ``F_SNAPSHOT`` (32) clear.
* ``F_BAD_TS_RECV`` (8) and ``F_MAYBE_BAD_BOOK`` (4) are kept as boolean
  features (``f_bad_ts_recv``, ``f_maybe_bad_book``), used to annotate or
  downgrade flags, and surfaced on their own by :func:`vendor_flags`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import (
    F_BAD_TS_RECV, F_LAST, F_MAYBE_BAD_BOOK, F_SNAPSHOT, FIXED_PRICE_SCALE,
    UNDEF_PRICE, UNDEF_TIMESTAMP, CrossedBooksConfig, DetectorConfig,
    ExtremePricesConfig, MissingIntervalsConfig, SequenceConfig, SessionCalendar,
    StaleQuotesConfig, TimestampConfig, UnusualSpreadsConfig,
)

KEY = ["publisher_id", "instrument_id"]
FLAG_COLUMNS = ["ts_recv", "ts_event", "publisher_id", "instrument_id", "symbol", "check",
                "subtype", "severity", "value", "threshold", "detail"]
PRICE_COLUMNS = ["price", "bid_px_00", "ask_px_00"]
NS = 1_000_000_000
MAD_SCALE = 1.4826


# ===========================================================================
# Normalization
# ===========================================================================
def _to_ns(values: pd.Series) -> np.ndarray:
    """Datetime (any unit/tz) or raw uint64/int64 ns -> int64 ns, NaT/UNDEF -> INT64_MIN."""
    if isinstance(values.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(values):
        ts = pd.to_datetime(values, utc=True).dt.as_unit("ns")
        out = ts.astype("int64").to_numpy().copy()
        out[ts.isna().to_numpy()] = np.iinfo(np.int64).min
        return out
    raw = values.to_numpy()
    if raw.dtype == np.uint64:
        bad = raw == np.uint64(UNDEF_TIMESTAMP)
        out = raw.astype(np.int64)
    else:
        num = pd.to_numeric(values, errors="coerce")
        bad = num.isna().to_numpy() | (num.to_numpy(dtype="float64") >= 2**63 - 1)
        out = num.fillna(0).to_numpy(dtype="float64").astype(np.int64) if num.dtype.kind == "f" else num.to_numpy(dtype=np.int64)
    out = out.copy()
    out[bad] = np.iinfo(np.int64).min
    return out


def _to_price(values: pd.Series) -> np.ndarray:
    """Raw int64 fixed-point (UNDEF_PRICE sentinel) or float -> float64 with NaN."""
    raw = values.to_numpy()
    if np.issubdtype(raw.dtype, np.integer):
        out = raw.astype("float64") * FIXED_PRICE_SCALE
        out[raw == UNDEF_PRICE] = np.nan
        return out
    out = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64").copy()
    out[np.abs(out) >= 9.2e18] = np.nan  # raw sentinel stored as float
    return out


def _to_char(values: pd.Series) -> np.ndarray:
    """c_char fields (action, side) as 1-char str; accepts int codes, bytes or str."""
    if values.dtype.kind in "iu":
        return np.array([chr(int(v)) if v else "" for v in values.to_numpy()], dtype=object)
    return np.array([v.decode("ascii", "ignore") if isinstance(v, bytes) else ("" if pd.isna(v) else str(v))
                     for v in values.to_numpy(dtype=object)], dtype=object)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return an internal frame with int64-ns timestamps and float prices.

    Accepts ``ts_recv`` as the index (team pipeline Parquet / ``to_df``) or as
    a column (raw arrays). Adds ``_row`` (original position), ``ts_recv_ns``,
    ``ts_event_ns``, flag booleans and ``is_book`` (F_LAST set, not snapshot).
    Output is stably sorted by key, then original order.
    """
    frame = df.reset_index() if "ts_recv" not in df.columns else df.reset_index(drop=True)
    if "ts_recv" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "ts_recv"})
    out = pd.DataFrame({"_row": np.arange(len(frame))})
    out["ts_recv_ns"] = _to_ns(frame["ts_recv"])
    out["ts_event_ns"] = _to_ns(frame["ts_event"]) if "ts_event" in frame else out["ts_recv_ns"]
    for col in KEY:
        out[col] = frame[col].to_numpy(dtype=np.int64) if col in frame else 0
    out["symbol"] = frame["symbol"].astype("string").to_numpy(dtype=object) if "symbol" in frame else None
    for col in PRICE_COLUMNS:
        out[col] = _to_price(frame[col]) if col in frame else np.nan
    for col in ["size", "bid_sz_00", "ask_sz_00", "bid_ct_00", "ask_ct_00", "sequence", "ts_in_delta", "rtype", "depth"]:
        out[col] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype="float64") if col in frame else np.nan
    for col in ["action", "side"]:
        out[col] = _to_char(frame[col]) if col in frame else ""
    flags = frame["flags"].to_numpy(dtype=np.int64) if "flags" in frame else np.full(len(frame), F_LAST)
    out["flags"] = flags
    out["f_last"] = (flags & F_LAST) != 0
    out["f_snapshot"] = (flags & F_SNAPSHOT) != 0
    out["f_bad_ts_recv"] = (flags & F_BAD_TS_RECV) != 0
    out["f_maybe_bad_book"] = (flags & F_MAYBE_BAD_BOOK) != 0
    out["is_book"] = out["f_last"] & ~out["f_snapshot"]
    out["both_sides"] = np.isfinite(out["bid_px_00"]) & np.isfinite(out["ask_px_00"])
    return out.sort_values(KEY + ["_row"], kind="stable").reset_index(drop=True)


def _ensure_norm(df: pd.DataFrame) -> pd.DataFrame:
    return df if "ts_recv_ns" in df.columns and "_row" in df.columns else normalize(df)


def _ts(ns: np.ndarray | pd.Series) -> pd.Series:
    arr = np.asarray(ns, dtype=np.int64)
    s = pd.Series(pd.to_datetime(arr, unit="ns", utc=True))
    s[arr == np.iinfo(np.int64).min] = pd.NaT
    return s


def make_flags(rows: pd.DataFrame, check: str, subtype: Any, severity: Any, value: Any,
               threshold: Any, detail: Any) -> pd.DataFrame:
    """Build the tidy flag table from normalized ``rows`` (scalars broadcast)."""
    if rows.empty:
        return empty_flags()
    n = len(rows)
    out = pd.DataFrame({
        "ts_recv": _ts(rows["ts_recv_ns"].to_numpy()).to_numpy(),
        "ts_event": _ts(rows["ts_event_ns"].to_numpy()).to_numpy(),
        "publisher_id": rows["publisher_id"].to_numpy(),
        "instrument_id": rows["instrument_id"].to_numpy(),
        "symbol": rows["symbol"].to_numpy() if "symbol" in rows else None,
    })
    out["check"] = check
    for name, val in (("subtype", subtype), ("severity", severity), ("value", value),
                      ("threshold", threshold), ("detail", detail)):
        out[name] = np.asarray(val) if np.ndim(val) else np.repeat(val, n)
    out["value"] = out["value"].astype("float64")
    out["threshold"] = out["threshold"].astype("float64")
    return out[FLAG_COLUMNS]


def empty_flags() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="float64" if c in ("value", "threshold") else object) for c in FLAG_COLUMNS})
    frame["ts_recv"] = pd.Series(dtype="datetime64[ns, UTC]")
    frame["ts_event"] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return empty_flags()
    return pd.concat(frames, ignore_index=True).sort_values(["ts_recv", "check"], kind="stable").reset_index(drop=True)


# ===========================================================================
# Shared helpers
# ===========================================================================
def _book(norm: pd.DataFrame, sort_time: bool = True) -> pd.DataFrame:
    """F_LAST, non-snapshot records; optionally sorted by ts_recv within key."""
    book = norm[norm["is_book"]]
    if sort_time:
        book = book.sort_values(KEY + ["ts_recv_ns", "_row"], kind="stable")
    return book.reset_index(drop=True)


def _key_id(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby(KEY, sort=False).ngroup().to_numpy()


def _runs(mask: np.ndarray, key: np.ndarray, ts: np.ndarray) -> pd.DataFrame:
    """Contiguous True runs of ``mask`` within each key.

    Duration = ts of the first record after the run (same key) minus ts of the
    run's first record; a run reaching the end of a key lasts to its last record
    and is marked ``open_ended``.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return pd.DataFrame(columns=["start", "end", "duration_ns", "open_ended"])
    n = len(mask)
    new_key = np.r_[True, key[1:] != key[:-1]]
    starts = np.flatnonzero(mask & (new_key | ~np.r_[False, mask[:-1]]))
    last_key = np.r_[key[1:] != key[:-1], True]
    ends = np.flatnonzero(mask & (last_key | ~np.r_[mask[1:], False]))
    nxt = np.minimum(ends + 1, n - 1)
    open_ended = last_key[ends]
    end_ts = np.where(open_ended, ts[ends], ts[nxt])
    return pd.DataFrame({"start": starts, "end": ends, "duration_ns": end_ts - ts[starts], "open_ended": open_ended})


def _lookup(mapping: dict[Any, Any], symbol: Any, instrument_id: int, default: Any = None) -> Any:
    def ok(v: Any) -> bool:
        return v is not None and not (isinstance(v, float) and np.isnan(v))
    if symbol is not None and symbol in mapping and ok(mapping[symbol]):
        return mapping[symbol]
    v = mapping.get(instrument_id, mapping.get(str(instrument_id)))
    return v if ok(v) else default  # v2: NaN counts as missing


def infer_tick(bids: np.ndarray) -> float:
    """Smallest positive difference among observed price levels (rounded)."""
    levels = np.unique(np.round(bids[np.isfinite(bids)], 9))
    if len(levels) < 2:
        return np.nan
    diffs = np.diff(levels)
    diffs = diffs[diffs > 1e-9]
    return float(np.round(diffs.min(), 9)) if len(diffs) else np.nan


def tick_sizes(norm: pd.DataFrame, mapping: dict[Any, float], default: float | None = None) -> pd.Series:
    """Per-record tick size from config (symbol or instrument_id), then
    ``default``, else inferred."""
    out = pd.Series(np.nan, index=norm.index)
    for (_, iid), grp in norm.groupby(KEY, sort=False):
        sym = grp["symbol"].iloc[0] if "symbol" in grp else None
        tick = _lookup(mapping, sym, iid, default)
        if tick is None:
            tick = infer_tick(np.r_[grp["bid_px_00"].to_numpy(), grp["ask_px_00"].to_numpy()])
        out.loc[grp.index] = tick
    return out


def _local_windows_mask(ts_ns: np.ndarray, windows: list[tuple[str, str, str]]) -> np.ndarray:
    """True where ts falls in any daily local window (tz, 'HH:MM', 'HH:MM')."""
    mask = np.zeros(len(ts_ns), bool)
    if not len(ts_ns):
        return mask
    for tz, start, end in windows:
        local = pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert(tz)
        minutes = local.hour.to_numpy() * 60 + local.minute.to_numpy() + local.second.to_numpy() / 60
        s = int(start[:2]) * 60 + int(start[3:])
        e = int(end[:2]) * 60 + int(end[3:])
        mask |= (minutes >= s) & (minutes < e) if s <= e else (minutes >= s) | (minutes < e)
    return mask


def status_windows(status: pd.DataFrame) -> pd.DataFrame:
    """Turn Databento *status* records into not-trading windows.

    Uses the documented ``is_trading`` field ('Y', 'N', '~' unknown). Returns
    columns instrument_id, start_ns, end_ns for spans where is_trading != 'Y'.
    Feed the result to :func:`crossed_books` via ``mask_windows``.
    """
    st = normalize(status)
    st["is_trading"] = _to_char(status.reset_index()["is_trading"]) if "is_trading" in status else "~"
    rows = []
    for iid, grp in st.sort_values("ts_recv_ns").groupby("instrument_id"):
        t = grp["ts_recv_ns"].to_numpy()
        trading = grp["is_trading"].to_numpy() == "Y"
        nxt = np.r_[t[1:], np.iinfo(np.int64).max]
        for a, b, tr in zip(t, nxt, trading):
            if not tr:
                rows.append((iid, a, b))
    return pd.DataFrame(rows, columns=["instrument_id", "start_ns", "end_ns"])


def _in_windows(ts_ns: np.ndarray, iid: np.ndarray, windows: pd.DataFrame | None) -> np.ndarray:
    mask = np.zeros(len(ts_ns), bool)
    if windows is None or windows.empty:
        return mask
    for row in windows.itertuples(index=False):
        sel = (ts_ns >= row.start_ns) & (ts_ns < row.end_ns)
        if "instrument_id" in windows.columns and not pd.isna(row.instrument_id):
            sel &= iid == row.instrument_id
        mask |= sel
    return mask


# ---------------------------------------------------------------------------
# Session calendar -> "session clock" (cumulative open seconds)
# ---------------------------------------------------------------------------
@dataclass
class SessionClock:
    """Maps a timestamp to cumulative open-session nanoseconds, so a gap's
    in-session length is ``clock(t1) - clock(t0)`` (closed time excluded)."""

    starts: np.ndarray
    ends: np.ndarray
    cum: np.ndarray

    @classmethod
    def build(cls, cal: SessionCalendar, t_min: int, t_max: int) -> "SessionClock":
        days = pd.date_range(pd.Timestamp(t_min, unit="ns", tz="UTC").tz_convert(cal.tz).normalize() - pd.Timedelta(days=1),
                             pd.Timestamp(t_max, unit="ns", tz="UTC").tz_convert(cal.tz).normalize() + pd.Timedelta(days=1),
                             freq="D").tz_localize(None)
        holidays = {pd.Timestamp(h).date() for h in cal.holidays}
        early = {pd.Timestamp(d).date(): v for d, v in cal.early_closes.items()}
        spans: list[tuple[int, int]] = []
        for day in days:
            if day.date() in holidays:
                continue
            for w in cal.windows:
                if day.weekday() not in w.weekdays:
                    continue
                start = day + pd.Timedelta(hours=int(w.start[:2]), minutes=int(w.start[3:]))
                end = day + pd.Timedelta(hours=int(w.end[:2]), minutes=int(w.end[3:]))
                if day.date() in early:
                    close = day + pd.Timedelta(hours=int(early[day.date()][:2]), minutes=int(early[day.date()][3:]))
                    end = min(end, close)
                if end <= start:
                    continue
                loc = [pd.Timestamp(x).tz_localize(cal.tz, nonexistent="shift_forward", ambiguous=False).tz_convert("UTC") for x in (start, end)]
                spans.append((loc[0].value, loc[1].value))
        spans.sort()
        merged: list[list[int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        arr = np.array(merged, dtype=np.int64).reshape(-1, 2)
        lengths = arr[:, 1] - arr[:, 0]
        return cls(arr[:, 0], arr[:, 1], np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64))

    def clock(self, ts: np.ndarray) -> np.ndarray:
        if not len(self.starts):
            return np.zeros(len(ts), np.int64)
        idx = np.searchsorted(self.starts, ts, side="right") - 1
        safe = np.clip(idx, 0, None)
        inside = np.clip(ts - self.starts[safe], 0, self.ends[safe] - self.starts[safe])
        return np.where(idx < 0, 0, self.cum[safe] + inside)

    def in_session(self, ts: np.ndarray) -> np.ndarray:
        if not len(self.starts):
            return np.zeros(len(ts), bool)
        idx = np.searchsorted(self.starts, ts, side="right") - 1
        safe = np.clip(idx, 0, None)
        return (idx >= 0) & (ts < self.ends[safe])


# ===========================================================================
# (1) Crossed / locked books
# ===========================================================================
def crossed_books(df: pd.DataFrame, config: CrossedBooksConfig | None = None,
                  mask_windows: pd.DataFrame | None = None) -> pd.DataFrame:
    r"""Crossed ($b_t > a_t$) and locked ($b_t = a_t$) top of book.

    Evaluated on F_LAST, non-snapshot mbp-1 records with both sides defined
    (UNDEF_PRICE / NaN sides ignored). Consecutive crossed (or locked) records
    form a run; a run is flagged once (at its first record) if it lasts at
    least ``min_duration_s`` until the next book update. Runs starting inside
    ``expected_windows_local`` / ``expected_windows_utc`` or ``mask_windows``
    (e.g. from :func:`status_windows`) are downgraded to 'info' (or dropped).
    ``value`` = bid - ask at the run start; ``threshold`` = tau_c in seconds.
    """
    cfg = config or CrossedBooksConfig()
    book = _book(_ensure_norm(df))
    book = book[book["both_sides"]].reset_index(drop=True)
    if book.empty:
        return empty_flags()
    key = _key_id(book)
    ts = book["ts_recv_ns"].to_numpy()
    diff = book["bid_px_00"].to_numpy() - book["ask_px_00"].to_numpy()
    eps = 1e-9
    out = []
    for subtype, mask, sev in (("crossed", diff > eps, "high"), ("locked", np.abs(diff) <= eps, "medium")):
        if subtype == "locked" and not cfg.report_locked:
            continue
        runs = _runs(mask, key, ts)
        if runs.empty:
            continue
        runs = runs[runs["duration_ns"] >= cfg.min_duration_s * NS]
        rows = book.iloc[runs["start"].to_numpy()]
        t0 = rows["ts_recv_ns"].to_numpy()
        expected = _local_windows_mask(t0, cfg.expected_windows_local)
        utc = pd.DataFrame([(pd.Timestamp(a).value, pd.Timestamp(b).value) for a, b in cfg.expected_windows_utc],
                           columns=["start_ns", "end_ns"])
        expected |= _in_windows(t0, rows["instrument_id"].to_numpy(), utc)
        expected |= _in_windows(t0, rows["instrument_id"].to_numpy(), mask_windows)
        severity = np.where(expected, "info", sev)
        detail = [f"duration={d / NS:.6f}s records={e - s + 1}{' open_ended' if o else ''}{' expected_window' if x else ''}"
                  f"{' F_MAYBE_BAD_BOOK' if b else ''}"
                  for d, s, e, o, x, b in zip(runs["duration_ns"], runs["start"], runs["end"], runs["open_ended"],
                                              expected, rows["f_maybe_bad_book"])]
        flags = make_flags(rows, "crossed_books", subtype, severity, diff[runs["start"].to_numpy()], cfg.min_duration_s, detail)
        if cfg.drop_expected:
            flags = flags[~expected]
        out.append(flags)
    return _concat(out)


# ===========================================================================
# (2) Extreme prices
# ===========================================================================
def _robust_z(values: pd.Series, groups: np.ndarray, window: int, min_periods: int, floor: np.ndarray) -> np.ndarray:
    """z_i = (r_i - med_W) / (1.4826 * max(MAD_W, floor)) over the *prior* W obs.

    MAD_W is the rolling median of |r - rolling median| (a standard streaming
    approximation of the windowed MAD).
    """
    grouped = values.groupby(groups, sort=False)
    med = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).median())
    dev = (values - med).abs()
    mad = dev.groupby(groups, sort=False).transform(lambda s: s.rolling(window, min_periods=min_periods).median())
    med_prev = med.groupby(groups, sort=False).shift(1)
    mad_prev = mad.groupby(groups, sort=False).shift(1)
    scale = MAD_SCALE * np.maximum(mad_prev.to_numpy(), floor)
    return ((values - med_prev).to_numpy()) / scale


def extreme_prices(df: pd.DataFrame, config: ExtremePricesConfig | None = None,
                   tick_size: dict[Any, float] | None = None, default_tick: float | None = None) -> pd.DataFrame:
    r"""Rolling robust z-score of log returns, plus optional static bands.

    $r_i = \ln p_i - \ln p_{i-1}$ on trade prints (``action == 'T'``, side
    'N' excluded) and/or the F_LAST book mid $m=(b+a)/2$;
    $z_i = (r_i - \mathrm{med}_W r)/(1.4826\max(\mathrm{MAD}_W r, \epsilon))$
    with $\epsilon=\max(\delta/p,\ 10^{-4})$ (v2). Flag $|z_i| > k$. With
    ``allow_negative`` the score uses $p_i - p_{i-1}$ (and an absolute
    tick floor) so negative spread-instrument prices are handled.
    Works on mbp-1 and trades schemas (mbo: use trade/fill prints only).
    """
    cfg = config or ExtremePricesConfig()
    norm = _ensure_norm(df)
    ticks = tick_size or {}
    series = []
    if cfg.price_source in ("trade", "both"):
        tr = norm[(norm["action"] == "T") & np.isfinite(norm["price"]) & ~norm["side"].isin(cfg.exclude_sides)]
        series.append(("trade", tr.sort_values(KEY + ["ts_recv_ns", "_row"], kind="stable").assign(p=lambda x: x["price"])))
    if cfg.price_source in ("mid", "both"):
        bk = _book(norm)
        bk = bk[bk["both_sides"]]
        series.append(("mid", bk.assign(p=(bk["bid_px_00"] + bk["ask_px_00"]) / 2)))
    out = []
    clock = None
    if cfg.calendar is not None and len(norm):
        tv = norm["ts_recv_ns"].to_numpy()
        tv = tv[tv != np.iinfo(np.int64).min]
        if len(tv):
            clock = SessionClock.build(cfg.calendar, int(tv.min()), int(tv.max()))
    for subtype, frame in series:
        frame = frame.reset_index(drop=True)
        if frame.empty:
            continue
        p = frame["p"].to_numpy()
        groups = _key_id(frame)
        tick = tick_sizes(frame, ticks, default_tick).to_numpy()
        if cfg.allow_negative:
            level = p
            floor = np.where(np.isfinite(tick), cfg.mad_floor_ticks * tick, np.nanmedian(np.abs(p)) * cfg.mad_floor_log)
        else:
            valid = p > 0
            frame, p, groups, tick = frame[valid].reset_index(drop=True), p[valid], groups[valid], tick[valid]
            level = np.log(p)
            floor = np.where(np.isfinite(tick), np.maximum(cfg.mad_floor_ticks * tick / p, cfg.mad_floor_log),
                             cfg.mad_floor_log)
        r = pd.Series(level).groupby(groups, sort=False).diff()
        z = _robust_z(r, groups, cfg.window, cfg.min_periods, floor)
        hit = np.abs(np.nan_to_num(z)) > cfg.k
        if clock is not None:
            hit &= clock.in_session(frame["ts_recv_ns"].to_numpy())
        rows = frame[hit]
        out.append(make_flags(rows, "extreme_prices", f"robust_z_{subtype}", "high", z[hit], cfg.k,
                              [f"price={x:.9g} return={y:.6g}" for x, y in zip(p[hit], r.to_numpy()[hit])]))
        if cfg.static_bands:
            lo = np.full(len(frame), -np.inf)
            hi = np.full(len(frame), np.inf)
            for i, (sym, iid) in enumerate(zip(frame["symbol"], frame["instrument_id"])):
                band = _lookup(cfg.static_bands, sym, iid)
                if band is not None:
                    lo[i], hi[i] = band
            bad = (p < lo) | (p > hi)
            out.append(make_flags(frame[bad], "extreme_prices", f"static_band_{subtype}", "high", p[bad],
                                  np.where(p[bad] < lo[bad], lo[bad], hi[bad]),
                                  [f"band=[{a:.9g},{b:.9g}]" for a, b in zip(lo[bad], hi[bad])]))
    return _concat(out)


# ===========================================================================
# (3) Timestamp inversions
# ===========================================================================
def timestamp_inversions(df: pd.DataFrame, config: TimestampConfig | None = None) -> pd.DataFrame:
    r"""Timestamp ordering and causality per key, in file order.

    * ``ts_recv_inversion`` (high): $ts\_recv_i < ts\_recv_{i-1}$. Databento
      documents ts_recv as monotonic per symbol. Downgraded to 'info' when the
      record carries F_BAD_TS_RECV (vendor already labelled it).
    * ``ts_event_inversion`` (low): $ts\_event_i < ts\_event_{i-1} - \epsilon_e$;
      reported only, since some venues stamp ts_event at unsynchronized gateways.
    * ``negative_latency`` (high): $L_i = ts\_recv_i - ts\_event_i < -\epsilon_{clk}$.
    * ``latency_outlier`` (medium): $L_i > \mathrm{med}(L) + k_L\,1.4826\,\mathrm{MAD}(L)$
      and $L_i - \mathrm{med}(L) >$ ``latency_min_excess_ns``.
    * ``ts_in_delta_saturated`` (low): |ts_in_delta| == 2^31-1.
    """
    cfg = config or TimestampConfig()
    norm = _ensure_norm(df)
    valid = (norm["ts_recv_ns"] != np.iinfo(np.int64).min) & (norm["ts_event_ns"] != np.iinfo(np.int64).min)
    norm = norm[valid].reset_index(drop=True)
    if norm.empty:
        return empty_flags()
    g = _key_id(norm)
    recv = norm["ts_recv_ns"].to_numpy()
    event = norm["ts_event_ns"].to_numpy()
    same = np.r_[False, g[1:] == g[:-1]]
    d_recv = np.r_[0, np.diff(recv)]
    d_event = np.r_[0, np.diff(event)]
    lat = recv - event
    out = []
    m = same & (d_recv < 0)
    out.append(make_flags(norm[m], "timestamp_inversions", "ts_recv_inversion",
                          np.where(norm["f_bad_ts_recv"][m], "info", "high"), d_recv[m] / NS, 0.0,
                          ["F_BAD_TS_RECV" if b else "ts_recv decreased" for b in norm["f_bad_ts_recv"][m]]))
    m = same & (d_event < -cfg.event_tolerance_ns)
    out.append(make_flags(norm[m], "timestamp_inversions", "ts_event_inversion", "low", d_event[m] / NS,
                          -cfg.event_tolerance_ns / NS, "ts_event decreased (may be venue behaviour)"))
    m = lat < -cfg.clock_tolerance_ns
    out.append(make_flags(norm[m], "timestamp_inversions", "negative_latency", "high", lat[m] / NS,
                          -cfg.clock_tolerance_ns / NS, "ts_recv < ts_event"))
    lat_s = pd.Series(lat.astype("float64"))
    med = lat_s.groupby(g).transform("median").to_numpy()
    mad = (lat_s - med).abs().groupby(g).transform("median").to_numpy()
    thr = np.maximum(med + cfg.latency_k * MAD_SCALE * mad, med + cfg.latency_min_excess_ns)
    m = lat > thr
    if cfg.latency_episode_gap_s is None:
        out.append(make_flags(norm[m], "timestamp_inversions", "latency_outlier", "medium", lat[m] / NS, thr[m] / NS,
                              [f"latency_median={x / NS:.6f}s" for x in med[m]]))
    else:  # v2: one flag per episode
        idx = np.flatnonzero(m)
        if len(idx):
            gi, ti = g[idx], recv[idx]
            new = np.r_[True, (gi[1:] != gi[:-1]) | (np.diff(ti) > cfg.latency_episode_gap_s * NS)]
            ep = np.cumsum(new) - 1
            first = idx[new]
            mx = pd.Series(lat[idx]).groupby(ep).max().to_numpy()
            cnt = np.bincount(ep)
            last_t = pd.Series(ti).groupby(ep).max().to_numpy()
            out.append(make_flags(norm.iloc[first], "timestamp_inversions", "latency_outlier", "medium", mx / NS,
                                  thr[first] / NS,
                                  [f"records={c} duration={(e - s) / NS:.6f}s latency_median={md / NS:.6f}s"
                                   for c, e, s, md in zip(cnt, last_t, recv[first], med[first])]))
    if cfg.flag_saturated_ts_in_delta:
        m = np.abs(norm["ts_in_delta"].to_numpy()) >= 2**31 - 1
        out.append(make_flags(norm[m], "timestamp_inversions", "ts_in_delta_saturated", "low",
                              norm["ts_in_delta"].to_numpy()[m], 2**31 - 1, "int32 ts_in_delta clamped"))
    return _concat(out)


# ===========================================================================
# (4) Missing intervals
# ===========================================================================
def missing_intervals(df: pd.DataFrame, config: MissingIntervalsConfig | None = None) -> pd.DataFrame:
    r"""In-session gaps in ts_recv per key.

    $g_i$ is the *in-session* time between consecutive arrivals (closed time
    removed via the session calendar). Threshold: static ``max_gap_s``; if
    ``adaptive``, $\min(\text{max\_gap},\ \max(\text{min\_gap},\ -\ln\alpha/\hat\lambda))$
    where $\hat\lambda$ = trailing arrivals per in-session second over
    ``rate_window`` messages, i.e. flag when the Poisson tail $e^{-\hat\lambda g}<\alpha$.
    The flag is placed on the record that *ends* the gap; detail has the start.
    Gaps before the first / after the last record are not seen (no boundary
    information in a symbol pull).
    """
    cfg = config or MissingIntervalsConfig()
    norm = _ensure_norm(df)
    norm = norm[norm["ts_recv_ns"] != np.iinfo(np.int64).min]
    if norm.empty:
        return empty_flags()
    norm = norm.sort_values(KEY + ["ts_recv_ns"], kind="stable").reset_index(drop=True)
    t = norm["ts_recv_ns"].to_numpy()
    clock = SessionClock.build(cfg.calendar, int(t.min()), int(t.max()))
    s = clock.clock(t)
    g = _key_id(norm)
    same = np.r_[False, g[1:] == g[:-1]]
    gap = np.r_[0, np.diff(s)].astype("float64") / NS
    thr = np.full(len(t), cfg.max_gap_s)
    if cfg.adaptive:
        n = cfg.rate_window
        s_series = pd.Series(s.astype("float64"))
        prev = s_series.groupby(g).shift(1)
        prev_n = s_series.groupby(g).shift(n + 1)
        span = (prev - prev_n).to_numpy() / NS
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(span > 0, n / span, np.nan)
        L = -np.log(cfg.alpha)
        raw_thr = L / lam
        if cfg.model == "overdispersed":
            # window moments of the previous n in-session gaps (gap zeroed at key starts)
            gz = np.where(same, gap, 0.0)
            c1 = np.r_[0.0, np.cumsum(gz)]; c2 = np.r_[0.0, np.cumsum(gz * gz)]
            i = np.arange(len(t)); lo = i - n
            ok = np.isfinite(lam) & (lo >= 0)
            lo_c = np.clip(lo, 0, None)
            m1 = (c1[i] - c1[lo_c]) / n; m2 = (c2[i] - c2[lo_c]) / n
            with np.errstate(divide="ignore", invalid="ignore"):
                cv2 = (m2 - m1 * m1) / (m1 * m1)
                kappa = np.maximum(2 * cv2 / (cv2 - 1), cfg.min_kappa)
                # Lomax: S(g) = (1 + g/sigma)^-kappa, sigma = mean*(kappa-1)
                lomax = m1 * (kappa - 1) * (np.power(cfg.alpha, -1.0 / kappa) - 1)
            od = ok & (cv2 > 1) & np.isfinite(lomax)
            raw_thr = np.where(od, lomax, raw_thr)
            lam = np.where(od, np.nan, lam)  # p-value in detail is Poisson-only
        elif cfg.model == "tod":
            b = np.floor(s / (cfg.tod_bucket_s * NS)).astype("int64")
            kb = pd.DataFrame({"g": g, "b": b})
            cnt = kb.groupby(["g", "b"])["b"].transform("size").to_numpy()
            lam = cnt / cfg.tod_bucket_s
            raw_thr = L / lam
        elif cfg.model != "poisson":
            raise ValueError(f"unknown missing model {cfg.model!r}")
        adaptive = np.clip(raw_thr, cfg.min_gap_s, cfg.max_gap_s)
        thr = np.where(np.isfinite(adaptive), adaptive, thr)
    else:
        lam = np.full(len(t), np.nan)
    m = same & (gap > thr)
    idx = np.flatnonzero(m)
    detail = [f"gap_start={pd.Timestamp(t[i - 1], unit='ns', tz='UTC').isoformat()} wall={(t[i] - t[i - 1]) / NS:.3f}s"
              + (f" lambda={lam[i]:.3g}/s p={np.exp(-lam[i] * gap[i]):.3g}" if np.isfinite(lam[i]) else "")
              for i in idx]
    sev = np.where(gap[m] > cfg.max_gap_s, "high", "medium")
    return make_flags(norm.iloc[idx], "missing_intervals", "gap", sev, gap[m], thr[m], detail)


def bar_completeness(bars_1m: pd.DataFrame, interval: str = "5m",
                     calendar: SessionCalendar | None = None) -> pd.DataFrame:
    """Count the ohlcv-1m bars underlying each 5m/15m/30m bucket.

    ``bars_1m`` is ohlcv-1m output (index or column ``ts_event`` = bar start).
    The expected count is the number of in-session minutes in the bucket
    (all minutes if no calendar). Buckets with fewer bars are flagged
    'incomplete_bar' (severity low): OHLCV prints no bar for a minute with no
    trades, so on illiquid names this is often legitimate, not missing data.
    """
    frame = bars_1m.reset_index() if "ts_event" not in bars_1m.columns else bars_1m.reset_index(drop=True)
    minutes = int(interval.rstrip("m"))
    ts = pd.to_datetime(frame["ts_event"], utc=True)
    if not pd.api.types.is_datetime64_any_dtype(frame["ts_event"]):
        ts = pd.to_datetime(_to_ns(frame["ts_event"]), unit="ns", utc=True)
    frame = frame.assign(bucket=ts.dt.floor(f"{minutes}min"))
    keys = [c for c in KEY if c in frame.columns]
    counts = frame.groupby(keys + ["bucket"], sort=True).size().rename("n_bars").reset_index()
    offsets = np.arange(minutes) * 60 * NS
    starts = counts["bucket"].dt.as_unit("ns").astype("int64").to_numpy()
    if calendar is not None:
        allm = starts[:, None] + offsets[None, :]
        clock = SessionClock.build(calendar, int(allm.min()), int(allm.max()))
        expected = clock.in_session(allm.ravel()).reshape(allm.shape).sum(axis=1)
    else:
        expected = np.full(len(counts), minutes)
    bad = counts["n_bars"].to_numpy() < expected
    rows = counts[bad].copy()
    rows["ts_recv_ns"] = np.iinfo(np.int64).min
    rows["ts_event_ns"] = starts[bad]
    rows["symbol"] = frame.groupby(keys + ["bucket"])["symbol"].first().to_numpy()[bad] if "symbol" in frame else None
    for k in KEY:
        if k not in rows:
            rows[k] = 0
    return make_flags(rows, "missing_intervals", f"incomplete_bar_{interval}", "low",
                      rows["n_bars"].to_numpy(), expected[bad],
                      [f"{n}/{e} one-minute bars" for n, e in zip(rows["n_bars"], expected[bad])])


# ===========================================================================
# (5) Stale quotes
# ===========================================================================
def _mid_asof(ref: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    rt = ref["ts_recv_ns"].to_numpy()
    idx = np.searchsorted(rt, t, side="right") - 1
    mid = ((ref["bid_px_00"] + ref["ask_px_00"]) / 2).to_numpy()
    return np.where(idx >= 0, mid[np.clip(idx, 0, None)], np.nan)


def stale_quotes(df: pd.DataFrame, config: StaleQuotesConfig | None = None,
                 reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    r"""BBO unchanged for longer than $\theta$ while the market is active.

    Quote age intervals are spans between consecutive changes of
    $(b, a, \text{bid\_sz}, \text{ask\_sz})$ on F_LAST records. An interval of
    length $A > \theta$ starting in session is **stale** if
    $N^{trades}_{(\tau, \tau+A]} \ge n_0$ (own prints while the quote froze) or,
    with a reference instrument, $|\ln m^{ref}_{\tau+A} - \ln m^{ref}_\tau| \ge \kappa$
    (bps). Otherwise it is merely *quiet* (reported only if ``report_quiet``).
    The reference (e.g. SPY) is looked up in ``reference_df`` or in ``df``.
    """
    cfg = config or StaleQuotesConfig()
    norm = _ensure_norm(df)
    book = _book(norm)
    book = book[book["both_sides"]].reset_index(drop=True)
    if book.empty:
        return empty_flags()
    ref_book = None
    if cfg.reference:
        rnorm = _ensure_norm(reference_df) if reference_df is not None else norm
        sel = np.ones(len(rnorm), bool)
        for k, v in cfg.reference.items():
            sel &= (rnorm[k] == v).to_numpy()
        ref_book = _book(rnorm[sel])
        ref_book = ref_book[ref_book["both_sides"]].sort_values("ts_recv_ns").reset_index(drop=True)
    trades = norm[norm["action"] == "T"]
    out = []
    t_all = book["ts_recv_ns"].to_numpy()
    clock = SessionClock.build(cfg.calendar, int(t_all.min()), int(t_all.max()))
    for (pid, iid), grp in book.groupby(KEY, sort=False):
        if ref_book is not None and len(ref_book) and (ref_book["instrument_id"].iloc[0] == iid and ref_book["publisher_id"].iloc[0] == pid):
            continue
        q = grp[["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]].to_numpy()
        changed = np.r_[True, np.any(q[1:] != q[:-1], axis=1)]
        t = grp["ts_recv_ns"].to_numpy()
        ch = np.flatnonzero(changed)
        t0 = t[ch]
        t1 = np.r_[t0[1:], t[-1]]
        age = (t1 - t0) / NS
        theta = cfg.min_age_s
        if cfg.adaptive_quantile is not None and len(age) > 10:
            theta = max(theta, cfg.adaptive_multiplier * float(np.quantile(age, cfg.adaptive_quantile)))
        cand = (age > theta) & clock.in_session(t0)
        if not cand.any():
            continue
        tt = np.sort(trades.loc[(trades["publisher_id"] == pid) & (trades["instrument_id"] == iid), "ts_recv_ns"].to_numpy())
        n_tr = np.searchsorted(tt, t1, side="right") - np.searchsorted(tt, t0, side="right")
        move = np.full(len(t0), np.nan)
        if ref_book is not None and len(ref_book):
            move = np.abs(np.log(_mid_asof(ref_book, t1)) - np.log(_mid_asof(ref_book, t0))) * 1e4
        stale = cand & ((n_tr >= cfg.min_trades) | (np.nan_to_num(move) >= cfg.ref_move_bps))
        quiet = cand & ~stale
        rows = grp.iloc[ch]
        for mask, sub, sev in ((stale, "stale", "high"), (quiet, "quiet", "info")):
            if sub == "quiet" and not cfg.report_quiet:
                continue
            out.append(make_flags(rows[mask], "stale_quotes", sub, sev, age[mask], theta,
                                  [f"until={pd.Timestamp(e, unit='ns', tz='UTC').isoformat()} trades={n} ref_move_bps={m:.2f}"
                                   for e, n, m in zip(t1[mask], n_tr[mask], move[mask])]))
    return _concat(out)


# ===========================================================================
# (6) Unusual spreads
# ===========================================================================
def unusual_spreads(df: pd.DataFrame, config: UnusualSpreadsConfig | None = None) -> pd.DataFrame:
    r"""Spread in ticks $S_t=(a_t-b_t)/\Delta$ and relative $s_t=(a_t-b_t)/m_t$.

    * ``zero_or_negative`` (high): $a_t - b_t \le 0$ (overlaps crossed_books,
      reported separately on purpose).
    * ``wide`` (medium), method ``rolling_quantile``: $S_t > \max(c\,Q_q, Q_q + e)$
      where $Q_q$ is the trailing-``window`` q-quantile of $S$ (prior records).
    * method ``time_of_day``: $z^s_t = (\ln s_t - \mathrm{med}_h)/(1.4826\,\mathrm{MAD}_h) > k_s$
      per local time-of-day bucket $h$ (needs several days for a good baseline).
    Wide runs must persist ``min_duration_s``; one flag per run.
    """
    cfg = config or UnusualSpreadsConfig()
    book = _book(_ensure_norm(df))
    book = book[book["both_sides"]].reset_index(drop=True)
    if book.empty:
        return empty_flags()
    book["tick"] = tick_sizes(book, cfg.tick_size, cfg.default_tick)
    t_book = book["ts_recv_ns"].to_numpy()
    in_sess = (SessionClock.build(cfg.calendar, int(t_book.min()), int(t_book.max())).in_session(t_book)
               if cfg.calendar is not None else np.ones(len(book), bool))
    spread = (book["ask_px_00"] - book["bid_px_00"]).to_numpy()
    ticks = spread / book["tick"].to_numpy()
    mid = ((book["ask_px_00"] + book["bid_px_00"]) / 2).to_numpy()
    rel = spread / np.abs(mid)
    g = _key_id(book)
    out = []
    m = spread <= 1e-12
    zr = _runs(m, g, book["ts_recv_ns"].to_numpy())
    if not zr.empty:
        zi = zr["start"].to_numpy()
        keep = in_sess[zi]
        zr, zi = zr[keep], zi[keep]
        out.append(make_flags(book.iloc[zi], "unusual_spreads", "zero_or_negative", "high", ticks[zi], 0.0,
                              [f"spread={spread[i]:.9g} records={e - i + 1} duration={d / NS:.6f}s"
                               for i, e, d in zip(zi, zr["end"], zr["duration_ns"])]))
    pos = ~m
    if cfg.method == "rolling_quantile":
        idx = pd.DatetimeIndex(pd.to_datetime(book["ts_recv_ns"].to_numpy(), unit="ns", utc=True))
        st = pd.Series(np.where(pos, ticks, np.nan), index=idx)
        base = st.groupby(g, sort=False).transform(
            lambda s: s.rolling(cfg.window, min_periods=cfg.min_periods).quantile(cfg.quantile).shift(1)).to_numpy()
        thr = np.maximum(cfg.multiplier * base, base + cfg.min_excess_ticks)
        score, unit = ticks, "ticks"
    elif cfg.method == "time_of_day":
        local = pd.to_datetime(book["ts_recv_ns"].to_numpy(), unit="ns", utc=True).tz_convert(cfg.tz)
        bucket = (local - local.normalize()) // pd.Timedelta(cfg.tod_bucket)
        ls = pd.Series(np.log(np.where(pos, rel, np.nan)))
        keys = [g, np.asarray(bucket)]
        med = ls.groupby(keys).transform("median")
        mad = (ls - med).abs().groupby(keys).transform("median")
        score = ((ls - med) / (MAD_SCALE * mad.clip(lower=1e-6))).to_numpy()
        thr, unit = np.full(len(book), cfg.k), "z"
    else:
        raise ValueError(f"unknown method {cfg.method!r}")
    wide = pos & (np.nan_to_num(score, nan=-np.inf) > np.nan_to_num(thr, nan=np.inf))
    runs = _runs(wide, g, book["ts_recv_ns"].to_numpy())
    if not runs.empty:
        runs = runs[runs["duration_ns"] >= cfg.min_duration_s * NS]
        runs = runs[in_sess[runs["start"].to_numpy().astype(int)]]
        st_i = runs["start"].to_numpy().astype(int)
        out.append(make_flags(book.iloc[st_i], "unusual_spreads", "wide", "medium", score[st_i], thr[st_i],
                              [f"{unit}; spread_ticks={ticks[i]:.2f} rel={rel[i] * 1e4:.2f}bps duration={d / NS:.3f}s"
                               for i, d in zip(st_i, runs["duration_ns"])]))
    return _concat(out)


# ===========================================================================
# (7) Sequence discontinuities
# ===========================================================================
def sequence_discontinuities(df: pd.DataFrame, config: SequenceConfig | None = None) -> pd.DataFrame:
    r"""Gaps and regressions in ``sequence``, $\delta_i = seq_i - seq_{i-1}$.

    Domain = publisher_id (default) or (publisher_id, instrument_id). In file
    order: $\delta \in \{0, 1\}$ OK (records from one venue message share a
    sequence); $\delta > 1$ = ``gap`` of $\delta - 1$ messages (value = gap
    size); $\delta < 0$ = ``regression``. Records with sequence == 0 are skipped
    (EQUS.MINI and other datasets that do not populate it).

    .. warning:: Databento's ``sequence`` is believed to be a venue/channel
       level counter (unconfirmed, open question to the sponsor). A
       symbol-filtered pull then contains only a subset of a channel's
       messages and shows *false gaps* between consecutive records of the
       pulled symbols. Trust gap flags only on full-channel pulls; on
       symbol pulls rely on regressions and on F_MAYBE_BAD_BOOK.
    """
    cfg = config or SequenceConfig()
    norm = _ensure_norm(df)
    norm = norm[np.isfinite(norm["sequence"]) & (norm["sequence"] != 0)]
    if norm.empty:
        return empty_flags()
    dom = ["publisher_id", "instrument_id"] if cfg.per_instrument else ["publisher_id"]
    norm = norm.sort_values(dom + ["_row"], kind="stable").reset_index(drop=True)
    g = norm.groupby(dom, sort=False).ngroup().to_numpy()
    seq = norm["sequence"].to_numpy().astype(np.int64)
    same = np.r_[False, g[1:] == g[:-1]]
    d = np.r_[0, np.diff(seq)]
    lo = 0 if cfg.allow_equal else 1
    gap = same & (d - 1 >= cfg.min_gap) & (d > 1)
    reg = same & (d < lo)
    vendor = norm["f_maybe_bad_book"].to_numpy()
    if cfg.gap_mode not in ("flag", "report_only", "skip"):
        raise ValueError(f"unknown gap_mode {cfg.gap_mode!r}")
    if cfg.gap_mode == "skip":
        gap = np.zeros(len(norm), bool)
    out = [
        make_flags(norm[gap], "sequence_discontinuities", "gap", "info" if cfg.gap_mode == "report_only" else "medium", d[gap] - 1, cfg.min_gap,
                   [f"prev={a - b} cur={a}{' F_MAYBE_BAD_BOOK' if v else ''}" for a, b, v in zip(seq[gap], d[gap], vendor[gap])]),
        make_flags(norm[reg], "sequence_discontinuities", "regression", "high", d[reg], lo,
                   [f"prev={a - b} cur={a}{' F_MAYBE_BAD_BOOK' if v else ''}" for a, b, v in zip(seq[reg], d[reg], vendor[reg])]),
    ]
    return _concat(out)


# ===========================================================================
# Vendor flags and orchestration
# ===========================================================================
def vendor_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Surface records Databento itself marked F_BAD_TS_RECV or F_MAYBE_BAD_BOOK."""
    norm = _ensure_norm(df)
    return _concat([
        make_flags(norm[norm["f_bad_ts_recv"]], "vendor_flags", "F_BAD_TS_RECV", "info", F_BAD_TS_RECV, np.nan, "ts_recv flagged inaccurate by Databento"),
        make_flags(norm[norm["f_maybe_bad_book"]], "vendor_flags", "F_MAYBE_BAD_BOOK", "medium", F_MAYBE_BAD_BOOK, np.nan, "unrecoverable channel gap per Databento"),
    ])


CHECKS = ["crossed_books", "extreme_prices", "timestamp_inversions", "missing_intervals",
          "stale_quotes", "unusual_spreads", "sequence_discontinuities"]


def run_all(df: pd.DataFrame, config: DetectorConfig | None = None, *, status: pd.DataFrame | None = None,
            reference_df: pd.DataFrame | None = None, checks: Iterable[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every check and return ``(flags, summary)``.

    ``status``: optional Databento status records; not-trading spans mask
    expected crosses. ``reference_df``: optional frame holding the stale-quote
    reference instrument if it is not in ``df``. ``summary`` counts flags by
    check, subtype, severity and instrument.
    """
    cfg = config or DetectorConfig()
    norm = _ensure_norm(df)
    wanted = set(checks or CHECKS)
    mask = status_windows(status) if status is not None else None
    runners = {
        "crossed_books": lambda: crossed_books(norm, cfg.crossed, mask_windows=mask),
        "extreme_prices": lambda: extreme_prices(norm, cfg.extreme, tick_size=cfg.spreads.tick_size,
                                                  default_tick=cfg.spreads.default_tick),
        "timestamp_inversions": lambda: timestamp_inversions(norm, cfg.timestamps),
        "missing_intervals": lambda: missing_intervals(norm, cfg.missing),
        "stale_quotes": lambda: stale_quotes(norm, cfg.stale, reference_df=reference_df),
        "unusual_spreads": lambda: unusual_spreads(norm, cfg.spreads),
        "sequence_discontinuities": lambda: sequence_discontinuities(norm, cfg.sequence),
    }
    parts = [fn() for name, fn in runners.items() if name in wanted]
    if cfg.report_vendor_flags:
        parts.append(vendor_flags(norm))
    flags = _concat(parts)
    return flags, summarize(flags)


def summarize(flags: pd.DataFrame) -> pd.DataFrame:
    """Counts by check, subtype, publisher, instrument and severity (wide)."""
    if flags.empty:
        return pd.DataFrame(columns=["check", "subtype", "publisher_id", "instrument_id", "symbol", "n_flags"])
    f = flags.assign(symbol=flags["symbol"].fillna(""))
    keys = ["check", "subtype", "publisher_id", "instrument_id", "symbol"]
    table = f.pivot_table(index=keys, columns="severity", values="value", aggfunc="size", fill_value=0)
    table["n_flags"] = table.sum(axis=1)
    return table.reset_index().rename_axis(columns=None)
