"""Extract in-session gaps, spreads and log returns from team-pipeline Parquet.

Large piece files are filtered with pyarrow (same layout as
integration.pipeline_parquet.read_pipeline_parquet / analyze.py::load).
Bundled samples used in tests go through read_pipeline_parquet.
Session membership uses p115_detectors SessionClock / SessionCalendar presets
via integration.combine.load_v3, so the masks match the v3 detectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from alert_system.config import PROJECT_ROOT
from integration.combine import load_v3
from integration.pipeline_parquet import read_pipeline_parquet

# Default location of the 95 symbol-day pipeline pull on this box.
DEFAULT_DATA_ROOT = Path("/workspace/p115/data_pipeline")

EQ_SYMBOLS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "CRWV", "NBIS", "LWAY", "PLAB", "ADTN"]
COLS = ["ts_recv", "ts_event", "publisher_id", "instrument_id", "symbol", "flags",
        "action", "side", "price", "bid_px_00", "ask_px_00"]

# 08:30-16:00 America/Chicago is the CME day session for equity-index futures;
# the rest of the Globex window (after the 16:00-17:00 CT break) is overnight.
CME_DAY_START = "08:30"
CME_DAY_END = "16:00"
CME_DAY_TZ = "America/Chicago"

NORMAL_WEEK = {f"2026-10-0{d}" for d in range(5, 10)}
DAY_TYPE = {
    "2025-11-28": "half_day",
    "2026-09-16": "vix_high",
    "2026-10-14": "cpi",
    "2026-08-29_weekend": "degraded_saturday",
    "2026-08-28": "around_degraded",
    "2026-08-31": "around_degraded",
}
for _d in NORMAL_WEEK:
    DAY_TYPE[_d] = "normal_week"

F_LAST = 128


@dataclass
class SeriesBundle:
    venue: str
    symbol: str
    day: str
    day_type: str
    regime: str
    tick: float
    n_records: int
    n_flast: int
    gaps_s: np.ndarray
    spreads_ticks: np.ndarray
    mid_returns: np.ndarray
    trade_returns: np.ndarray
    share_zero_or_neg_spread: float
    share_zero_mid_return: float
    mid_unique_abs: int


def day_type_of(day: str) -> str:
    return DAY_TYPE.get(day, "other")


def tasks():
    """Symbol-day list matching p115_detectors/analyze.py::tasks()."""
    T = []
    for piece, days in [
        ("A_mbp-1_2025-11-28", ["2025-11-28"]),
        ("A_mbp-1_2026-09-16", ["2026-09-16"]),
        ("A_mbp-1_2026-10-05", ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"]),
        ("A_mbp-1_2026-10-14", ["2026-10-14"]),
    ]:
        for day in days:
            for symbol in EQ_SYMBOLS:
                start = pd.Timestamp(day, tz="UTC")
                T.append(("XNAS", piece, symbol, day, start, start + pd.Timedelta(days=1)))
    for piece, days, symbols in [
        ("B_mbp-1_wk", ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"], ["ES.c.0", "CL.c.0"]),
        ("B_mbp-1_vol", ["2026-10-14"], ["ES.c.0", "CL.c.0"]),
        ("B_mbp-1_deg", ["2026-08-28", "2026-08-31"], ["ES.c.0"]),
    ]:
        for day in days:
            for symbol in symbols:
                start = pd.Timestamp(day, tz="UTC") - pd.Timedelta(hours=2)
                T.append(("GLBX", piece, symbol, day, start, start + pd.Timedelta(days=1)))
    T.append(("GLBX", "B_mbp-1_deg", "ES.c.0", "2026-08-29_weekend",
              pd.Timestamp("2026-08-28T22:00Z"), pd.Timestamp("2026-08-30T22:00Z")))
    return T


def load_ticks(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, float]:
    """min_price_increment from definition pieces; XNAS NaNs become 0.01."""
    ticks: dict[str, float] = {}
    for folder in sorted(data_root.glob("*definition*")):
        path = folder / "data.parquet"
        if not path.is_file():
            continue
        table = ds.dataset(path).to_table(columns=["symbol", "min_price_increment"])
        frame = table.to_pandas()
        for symbol, value in zip(frame["symbol"], frame["min_price_increment"]):
            if symbol and np.isfinite(value) and value > 0:
                ticks[str(symbol)] = float(value)
    for symbol in EQ_SYMBOLS:
        ticks.setdefault(symbol, 0.01)
    return ticks


def load_symbol_slice(piece: str, symbol: str, start, end, data_root: Path = DEFAULT_DATA_ROOT,
                      columns=None) -> pd.DataFrame:
    """One symbol and time window from a pipeline piece (pyarrow filter)."""
    path = Path(data_root) / piece / "data.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    filt = (
        (pc.field("symbol") == symbol)
        & (pc.field("ts_recv") >= pd.Timestamp(start))
        & (pc.field("ts_recv") < pd.Timestamp(end))
    )
    table = ds.dataset(path).to_table(filter=filt, columns=list(columns or COLS))
    frame = table.to_pandas()
    if "ts_recv" in frame.columns and frame.index.name != "ts_recv":
        # keep ts_recv as a column for numpy extraction; also set index when present
        pass
    return frame


def load_bundled_sample(sample: str = "quotes", root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Tests and demos: the same bundled file the pipeline reader loads."""
    return read_pipeline_parquet(
        # resolve through the provenance used by pipeline_frame
        _bundled_path(sample, root)
    )


def _bundled_path(sample: str, root: Path) -> Path:
    import json
    folder = Path(root) / "sample_data"
    name = "quotes_provenance.json" if sample == "quotes" else "trades_provenance.json"
    provenance = json.loads((folder / name).read_text())
    return folder / provenance["file"]


def _session_clock(v3, venue: str, ts_ns: np.ndarray):
    cal = v3.config.equities_config().missing.calendar if venue == "XNAS" else v3.config.cme_calendar()
    if not len(ts_ns):
        return None, cal
    return v3.detectors.SessionClock.build(cal, int(ts_ns.min()), int(ts_ns.max())), cal


def _in_cme_day(ts_ns: np.ndarray) -> np.ndarray:
    local = pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert(CME_DAY_TZ)
    minutes = local.hour.to_numpy() * 60 + local.minute.to_numpy() + local.second.to_numpy() / 60.0
    start = 8 * 60 + 30
    end = 16 * 60
    return (minutes >= start) & (minutes < end)


def _ns(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series):
        s = series
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize("UTC")
        else:
            s = s.dt.tz_convert("UTC")
        return s.astype("int64").to_numpy()
    if np.issubdtype(np.asarray(series.dropna().iloc[:1]).dtype, np.integer) if len(series) else False:
        return series.to_numpy(dtype="int64")
    return pd.to_datetime(series, utc=True).astype("int64").to_numpy()


def extract_series(frame: pd.DataFrame, venue: str, symbol: str, day: str, tick: float,
                   v3=None, regime: str | None = None) -> list[SeriesBundle]:
    """Build in-session gap, spread and return arrays. Equities: one RTH bundle.

    Futures: two bundles, CME day (08:30-16:00 CT) and overnight (rest of the
    Globex session), unless ``regime`` is set to one of those names.
    """
    v3 = v3 or load_v3()
    if frame.empty:
        return []
    work = frame.reset_index() if "ts_recv" not in frame.columns else frame
    if "ts_recv" not in work.columns:
        raise ValueError("expected a ts_recv column or index")
    ts = _ns(work["ts_recv"])
    flags = work["flags"].to_numpy().astype(np.int64) if "flags" in work.columns else np.zeros(len(work), np.int64)
    flast = (flags & F_LAST) != 0
    bid = pd.to_numeric(work["bid_px_00"], errors="coerce").to_numpy() if "bid_px_00" in work.columns else np.full(len(work), np.nan)
    ask = pd.to_numeric(work["ask_px_00"], errors="coerce").to_numpy() if "ask_px_00" in work.columns else np.full(len(work), np.nan)
    both = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0)
    action = work["action"].astype(str).to_numpy() if "action" in work.columns else np.array([""] * len(work))
    side = work["side"].astype(str).to_numpy() if "side" in work.columns else np.array([""] * len(work))
    price = pd.to_numeric(work["price"], errors="coerce").to_numpy() if "price" in work.columns else np.full(len(work), np.nan)
    pub = work["publisher_id"].to_numpy() if "publisher_id" in work.columns else np.zeros(len(work), np.int64)
    iid = work["instrument_id"].to_numpy() if "instrument_id" in work.columns else np.zeros(len(work), np.int64)

    clock, _cal = _session_clock(v3, venue, ts)
    in_session = clock.in_session(ts) if clock is not None else np.ones(len(work), bool)
    sess_time = clock.clock(ts).astype("float64") if clock is not None else ts.astype("float64")

    if venue == "XNAS":
        regimes = [("equity_rth", in_session)]
    else:
        day_mask = in_session & _in_cme_day(ts)
        night_mask = in_session & ~_in_cme_day(ts)
        regimes = [("cme_day", day_mask), ("cme_overnight", night_mask)]
    if regime is not None:
        regimes = [(name, mask) for name, mask in regimes if name == regime]

    bundles = []
    for name, mask in regimes:
        gaps = _gaps(sess_time, pub, iid, flast & mask)
        spreads, share_bad = _spreads(bid, ask, tick, flast & both & mask)
        mid_r, share0, nunique = _mid_returns(bid, ask, pub, iid, ts, flast & both & mask)
        tr_r = _trade_returns(price, action, side, pub, iid, ts, mask)
        bundles.append(SeriesBundle(
            venue=venue, symbol=symbol, day=day, day_type=day_type_of(day), regime=name,
            tick=float(tick), n_records=int(mask.sum()), n_flast=int((flast & mask).sum()),
            gaps_s=gaps, spreads_ticks=spreads, mid_returns=mid_r, trade_returns=tr_r,
            share_zero_or_neg_spread=share_bad, share_zero_mid_return=share0, mid_unique_abs=nunique,
        ))
    return bundles


def _gaps(sess_time: np.ndarray, pub, iid, mask: np.ndarray) -> np.ndarray:
    """In-session gaps (seconds) between consecutive F_LAST records of one key."""
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return np.array([], dtype="float64")
    order = np.lexsort((sess_time[idx], iid[idx], pub[idx]))
    i = idx[order]
    same = (pub[i][1:] == pub[i][:-1]) & (iid[i][1:] == iid[i][:-1])
    gap = np.diff(sess_time[i]) / 1_000_000_000.0
    return gap[same & (gap > 0)]


def _spreads(bid, ask, tick: float, mask: np.ndarray) -> tuple[np.ndarray, float]:
    if tick <= 0 or not mask.any():
        return np.array([], dtype="float64"), float("nan")
    spread = ask[mask] - bid[mask]
    n = len(spread)
    bad = float(np.mean(spread <= 0)) if n else float("nan")
    ticks = spread[spread > 0] / tick
    return ticks.astype("float64"), bad


def _mid_returns(bid, ask, pub, iid, ts, mask: np.ndarray) -> tuple[np.ndarray, float, int]:
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return np.array([], dtype="float64"), float("nan"), 0
    mid = (bid + ask) / 2.0
    order = np.lexsort((ts[idx], iid[idx], pub[idx]))
    i = idx[order]
    same = (pub[i][1:] == pub[i][:-1]) & (iid[i][1:] == iid[i][:-1])
    p0, p1 = mid[i][:-1], mid[i][1:]
    ok = same & (p0 > 0) & (p1 > 0)
    r = np.log(p1[ok] / p0[ok])
    r = r[np.isfinite(r)]
    share0 = float(np.mean(r == 0)) if len(r) else float("nan")
    uniq = int(np.unique(np.round(np.abs(r[r != 0]), 12)).size) if np.any(r != 0) else 0
    return r, share0, uniq


def _trade_returns(price, action, side, pub, iid, ts, mask: np.ndarray) -> np.ndarray:
    trade = mask & (action == "T") & (side != "N") & np.isfinite(price) & (price > 0)
    idx = np.flatnonzero(trade)
    if len(idx) < 2:
        return np.array([], dtype="float64")
    order = np.lexsort((ts[idx], iid[idx], pub[idx]))
    i = idx[order]
    same = (pub[i][1:] == pub[i][:-1]) & (iid[i][1:] == iid[i][:-1])
    p0, p1 = price[i][:-1], price[i][1:]
    ok = same & (p0 > 0) & (p1 > 0)
    r = np.log(p1[ok] / p0[ok])
    return r[np.isfinite(r)]


def pool(bundles: list[SeriesBundle]) -> SeriesBundle:
    """Concatenate arrays; metadata from the first bundle, day='pooled'."""
    first = bundles[0]
    def cat(name):
        parts = [getattr(b, name) for b in bundles if len(getattr(b, name))]
        return np.concatenate(parts) if parts else np.array([], dtype="float64")
    n = sum(b.n_records for b in bundles)
    nf = sum(b.n_flast for b in bundles)
    mid = cat("mid_returns")
    share0 = float(np.mean(mid == 0)) if len(mid) else float("nan")
    if len(mid) and np.any(mid != 0):
        uniq = int(np.unique(np.round(np.abs(mid[mid != 0]), 12)).size)
    else:
        uniq = 0
    return SeriesBundle(
        venue=first.venue, symbol=first.symbol, day="pooled", day_type=first.day_type,
        regime=first.regime, tick=first.tick, n_records=n, n_flast=nf,
        gaps_s=cat("gaps_s"), spreads_ticks=cat("spreads_ticks"), mid_returns=mid,
        trade_returns=cat("trade_returns"),
        share_zero_or_neg_spread=float(np.nanmean([b.share_zero_or_neg_spread for b in bundles])),
        share_zero_mid_return=share0, mid_unique_abs=uniq,
    )
