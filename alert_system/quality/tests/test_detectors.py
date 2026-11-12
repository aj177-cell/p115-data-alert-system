"""Positive and negative cases for each detector (synthetic data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alert_system.quality.config import (UNDEF_PRICE, DetectorConfig, ExtremePricesConfig, MissingIntervalsConfig,
                                          SequenceConfig, StaleQuotesConfig, TimestampConfig, UnusualSpreadsConfig,
                                          cme_futures_config)
from alert_system.quality.detectors import (FLAG_COLUMNS, bar_completeness, crossed_books, extreme_prices,
                                            missing_intervals, normalize, run_all, sequence_discontinuities,
                                            stale_quotes, status_windows, timestamp_inversions, unusual_spreads,
                                            vendor_flags)
from alert_system.quality.synthetic import PX, generate_mbp1, inject_faults, to_databento_df

NS = 1_000_000_000
TICK = int(0.01 * PX)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return generate_mbp1(duration_s=600, seed=3)


@pytest.fixture()
def clean(raw) -> pd.DataFrame:
    return raw.copy()


def window(df: pd.DataFrame, iid: int, a_s: float, b_s: float, f_last_only: bool = False) -> np.ndarray:
    t0 = df["ts_recv"].min()
    t = df["ts_recv"].to_numpy()
    m = (df["instrument_id"].to_numpy() == iid) & (t >= t0 + a_s * NS) & (t <= t0 + b_s * NS)
    if f_last_only:
        m &= (df["flags"].to_numpy() & 128) != 0
    return np.flatnonzero(m)


def nonblank(flags: pd.DataFrame, check: str, subtype: str | None = None) -> pd.DataFrame:
    f = flags[(flags["check"] == check) & (flags["severity"] != "info")]
    return f if subtype is None else f[f["subtype"] == subtype]


# ---------------------------------------------------------------- general
def test_raw_and_converted_inputs_agree(clean):
    a, b = normalize(clean), normalize(to_databento_df(clean))
    for col in ["ts_recv_ns", "ts_event_ns", "bid_px_00", "ask_px_00", "price", "sequence"]:
        np.testing.assert_allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float))


def test_flag_schema(clean):
    flags, summary = run_all(to_databento_df(clean))
    assert list(flags.columns) == FLAG_COLUMNS
    assert {"check", "instrument_id", "n_flags"} <= set(summary.columns) or flags.empty


def test_parquet_roundtrip(tmp_path, clean):
    df = to_databento_df(clean)
    df.to_parquet(tmp_path / "data.parquet")
    back = pd.read_parquet(tmp_path / "data.parquet")
    assert back.index.name == "ts_recv"
    flags, _ = run_all(back)
    assert nonblank(flags, "crossed_books").empty


# ---------------------------------------------------------------- (1) crossed
def test_crossed_positive_and_locked(clean):
    r = window(clean, 101, 100, 102)
    clean.loc[r, "bid_px_00"] = clean.loc[r, "ask_px_00"] + TICK
    r2 = window(clean, 102, 200, 201)
    clean.loc[r2, "bid_px_00"] = clean.loc[r2, "ask_px_00"]
    f = crossed_books(to_databento_df(clean))
    assert (f["subtype"] == "crossed").sum() == 1 and (f["subtype"] == "locked").sum() == 1


def test_crossed_negative_ignores_non_f_last_and_undef(clean):
    # synthetic intermediates (flags=0) are crossed by construction; must not fire
    assert ((clean["flags"] & 128) == 0).any()
    clean.loc[window(clean, 101, 50, 60), "ask_px_00"] = UNDEF_PRICE  # one-sided book
    assert crossed_books(clean).empty


def test_crossed_expected_window_masked(clean):
    r = window(clean, 101, 100, 102)
    clean.loc[r, "bid_px_00"] = clean.loc[r, "ask_px_00"] + TICK
    cfg = cme_futures_config().crossed
    cfg.expected_windows_local = [("UTC", "13:30", "14:00")]
    assert (crossed_books(clean, cfg)["severity"] == "info").all()
    t0 = pd.Timestamp(int(clean["ts_recv"].min()), unit="ns", tz="UTC")
    status = pd.DataFrame({"ts_recv": [t0, t0 + pd.Timedelta(minutes=5)], "ts_event": [t0, t0 + pd.Timedelta(minutes=5)],
                           "publisher_id": 2, "instrument_id": 101, "is_trading": ["N", "Y"]}).set_index("ts_recv")
    f = crossed_books(clean, DetectorConfig().crossed, mask_windows=status_windows(status))
    assert (f["severity"] == "info").all() and len(f) == 1


def test_crossed_min_duration(clean):
    r = window(clean, 101, 100, 102)
    clean.loc[r, "bid_px_00"] = clean.loc[r, "ask_px_00"] + TICK
    cfg = DetectorConfig().crossed
    cfg.min_duration_s = 10.0
    assert crossed_books(clean, cfg).empty


# ---------------------------------------------------------------- (2) extreme
def test_extreme_positive_decimal_slip(clean):
    r = window(clean, 101, 300, 330)
    r = r[clean.loc[r, "action"].to_numpy() == "T"][0]
    clean.loc[r, "price"] *= 10
    f = extreme_prices(to_databento_df(clean))
    assert (f["ts_recv"] == pd.Timestamp(int(clean.loc[r, "ts_recv"]), unit="ns", tz="UTC")).any()


def test_extreme_negative_clean(clean):
    f = extreme_prices(clean, ExtremePricesConfig(k=12))
    assert f.empty


def test_extreme_static_band_and_negative_prices(clean):
    df = to_databento_df(clean)
    for c in ("price", "bid_px_00", "ask_px_00"):
        df[c] = df[c] - 600.0  # all prices negative, like a calendar spread
    cfg = ExtremePricesConfig(allow_negative=True, k=12, static_bands={"TEST": (-505.0, -495.0)})
    f = extreme_prices(df, cfg, tick_size={"TEST": 0.01, "REF": 0.01})
    assert not f["subtype"].str.startswith("robust").any()
    assert f["subtype"].str.startswith("static_band").sum() == 0  # TEST lives near -500
    cfg.static_bands = {"TEST": (-499.0, -495.0)}
    assert (extreme_prices(df, cfg)["subtype"] == "static_band_trade").any()


# ---------------------------------------------------------------- (3) timestamps
def test_timestamp_positive(clean):
    r = window(clean, 101, 100, 110)
    clean.loc[r[5], "ts_recv"] = clean.loc[r[4], "ts_recv"] - 5_000_000
    clean.loc[r[20], "ts_event"] = clean.loc[r[20], "ts_recv"] - 80_000_000
    f = timestamp_inversions(clean)
    assert {"ts_recv_inversion", "latency_outlier", "negative_latency"} <= set(f["subtype"])


def test_timestamp_negative_and_bad_ts_recv_downgraded(clean):
    assert timestamp_inversions(clean).empty
    r = window(clean, 101, 100, 110)
    clean.loc[r[5], "ts_recv"] = clean.loc[r[4], "ts_recv"] - 1000
    clean.loc[r[5], "ts_event"] = clean.loc[r[5], "ts_recv"] - 20_000
    clean.loc[r[5], "flags"] = 128 | 8
    f = timestamp_inversions(clean)
    assert (f.loc[f["subtype"] == "ts_recv_inversion", "severity"] == "info").all()
    assert (vendor_flags(clean)["subtype"] == "F_BAD_TS_RECV").sum() == 1


# ---------------------------------------------------------------- (4) missing
def test_missing_positive(clean):
    df = clean.drop(index=window(clean, 101, 200, 215))
    f = missing_intervals(df)
    assert len(f) == 1 and f["value"].iloc[0] > 14


def test_missing_negative_and_out_of_session(clean):
    assert missing_intervals(clean).empty
    # a gap that is entirely after the 16:00 ET close is not flagged
    shifted = clean.copy()
    shifted["ts_recv"] += int(7 * 3600 * NS)  # 20:30Z start = 16:30 ET
    shifted["ts_event"] += int(7 * 3600 * NS)
    assert missing_intervals(shifted.drop(index=window(shifted, 101, 200, 215))).empty


def test_bar_completeness():
    t = pd.date_range("2026-10-08 13:30", periods=15, freq="1min", tz="UTC").delete([7])
    bars = pd.DataFrame({"publisher_id": 1, "instrument_id": 5, "close": 1.0, "symbol": "X"},
                        index=pd.Index(t, name="ts_event"))
    f = bar_completeness(bars, "5m")
    assert len(f) == 1 and f["value"].iloc[0] == 4
    assert bar_completeness(bars.drop(index=bars.index[6:10]).iloc[:5], "5m").empty


# ---------------------------------------------------------------- (5) stale
def test_stale_positive(clean):
    r = window(clean, 101, 200, 230, f_last_only=True)
    cols = ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    clean.loc[r, cols] = clean.loc[r[:1], cols].to_numpy().repeat(len(r), axis=0)
    f = stale_quotes(clean, StaleQuotesConfig(reference={"symbol": "REF"}))
    assert len(f) == 1 and f["value"].iloc[0] > 25


def test_stale_negative_quiet_market(clean):
    assert stale_quotes(clean).empty
    # frozen quote but nobody trades and reference does not move -> quiet, not stale
    df = clean[clean["instrument_id"] == 101].copy()
    r = window(df.reset_index(drop=True), 101, 200, 230, f_last_only=True)
    df = df.reset_index(drop=True)
    cols = ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    df.loc[r, cols] = df.loc[r[:1], cols].to_numpy().repeat(len(r), axis=0)
    df.loc[r, "action"] = "A"
    cfg = StaleQuotesConfig(report_quiet=True)
    f = stale_quotes(df, cfg)
    assert (f["subtype"] == "quiet").sum() == 1 and (f["subtype"] == "stale").sum() == 0


# ---------------------------------------------------------------- (6) spreads
def test_spreads_positive_wide_and_zero(clean):
    r = window(clean, 101, 300, 305)
    clean.loc[r, "ask_px_00"] += 10 * TICK
    r2 = window(clean, 102, 400, 401)
    clean.loc[r2, "ask_px_00"] = clean.loc[r2, "bid_px_00"]
    f = unusual_spreads(clean)
    assert (f["subtype"] == "wide").sum() == 1 and (f["subtype"] == "zero_or_negative").sum() == 1


def test_spreads_negative(clean):
    assert unusual_spreads(clean).empty


# ---------------------------------------------------------------- (7) sequence
def test_sequence_positive(clean):
    clean.loc[5000:, "sequence"] += 17
    clean.loc[[7000, 7001], "sequence"] = clean.loc[[7001, 7000], "sequence"].to_numpy()
    f = sequence_discontinuities(clean)
    gaps = f[f["subtype"] == "gap"]
    assert 17 in gaps["value"].to_numpy() and (f["subtype"] == "regression").sum() >= 1


def test_sequence_negative_and_zero_skipped(clean):
    assert sequence_discontinuities(clean).empty
    zero = clean.copy()
    zero["sequence"] = 0  # EQUS.MINI style
    assert sequence_discontinuities(zero).empty
    assert sequence_discontinuities(clean, SequenceConfig(per_instrument=True)).shape[0] > 0  # documented false gaps


# ---------------------------------------------------------------- end to end
def test_run_all_on_injected_faults(raw):
    faulted, labels = inject_faults(generate_mbp1(duration_s=1800, seed=5), n_per_type=2, seed=1)
    cfg = DetectorConfig()
    cfg.stale.reference = {"symbol": "REF"}
    flags, summary = run_all(to_databento_df(faulted), cfg)
    from alert_system.quality.evaluate import match
    m = match(flags, labels).set_index("check")
    assert (m["recall"] == 1.0).all()
    assert summary["n_flags"].sum() == len(flags)


# ---------------------------------------------------------------- v2 behaviour
def _seq_df(seq, iids):
    t = pd.date_range("2026-10-08 14:00", periods=len(seq), freq="10ms", tz="UTC")
    return pd.DataFrame({"ts_event": t, "publisher_id": 2, "instrument_id": iids, "symbol": "X", "sequence": seq,
                         "flags": 128, "bid_px_00": 10.0, "ask_px_00": 10.01, "price": np.nan, "action": "A",
                         "side": "N", "size": 1}, index=pd.Index(t, name="ts_recv"))


def test_v2_config_values():
    from alert_system.quality.config import equities_config
    e, c = equities_config(), cme_futures_config()
    assert e.missing.min_gap_s == 5.0 and e.stale.min_trades == 10 and e.extreme.k == 12.0
    assert e.timestamps.latency_min_excess_ns == 5_000_000 and c.timestamps.latency_min_excess_ns == 20_000_000
    assert e.sequence.gap_mode == "skip" and c.sequence.gap_mode == "flag" and c.sequence.per_instrument
    assert e.spreads.default_tick == 0.01


def test_v2_sequence_gap_modes():
    df = _seq_df([1, 2, 50, 51, 40], [7] * 5)  # gap of 47 and a regression
    assert set(sequence_discontinuities(df)["subtype"]) == {"gap", "regression"}
    sk = sequence_discontinuities(df, SequenceConfig(gap_mode="skip"))
    assert list(sk["subtype"]) == ["regression"]
    ro = sequence_discontinuities(df, SequenceConfig(gap_mode="report_only"))
    assert (ro.loc[ro.subtype == "gap", "severity"] == "info").all()


def test_v2_nan_tick_uses_default():
    from alert_system.quality.detectors import _lookup, tick_sizes
    assert _lookup({"AAPL": float("nan")}, "AAPL", 1, 0.01) == 0.01
    n = normalize(_seq_df([1, 2, 3], [7] * 3))
    assert (tick_sizes(n, {"X": float("nan")}, 0.01) == 0.01).all()


def test_v2_missing_1s_pause_not_flagged(clean):
    df = clean.drop(index=window(clean, 101, 200, 201.4))  # ~1.4 s silence
    assert missing_intervals(df, MissingIntervalsConfig(min_gap_s=1.0)).shape[0] >= 1
    assert missing_intervals(df).empty


def test_v2_latency_episode_collapse(clean):
    r = window(clean, 101, 100, 101)
    clean.loc[r[:10], "ts_event"] = clean.loc[r[:10], "ts_recv"] - 80_000_000
    f = timestamp_inversions(clean)
    lo = f[f.subtype == "latency_outlier"]
    n_rec = (timestamp_inversions(clean, TimestampConfig(latency_episode_gap_s=None)).subtype == "latency_outlier").sum()
    assert n_rec >= 5 and len(lo) == 1 and f"records={n_rec}" in lo["detail"].iloc[0]
    # 3 ms excess is below the v2 5 ms equities floor
    clean2 = clean.copy()
    clean2["ts_event"] = clean2["ts_recv"] - 20_000
    clean2.loc[r[:10], "ts_event"] = clean2.loc[r[:10], "ts_recv"] - 3_020_000
    assert timestamp_inversions(clean2).empty


def test_v2_session_gating_spreads_and_extreme(clean):
    # synthetic session starts 09:30 ET; shift so the faults land after 16:00 ET
    r = window(clean, 101, 300, 305)
    clean.loc[r, "ask_px_00"] += 10 * TICK
    shifted = clean.copy()
    for c in ("ts_recv", "ts_event"):
        shifted[c] += int(7 * 3600 * NS)
    assert (unusual_spreads(clean)["subtype"] == "wide").sum() == 1
    assert unusual_spreads(shifted).empty
    assert (unusual_spreads(shifted, UnusualSpreadsConfig(calendar=None))["subtype"] == "wide").sum() == 1
    tr = window(clean, 101, 300, 400)
    tr = [i for i in tr if clean.loc[i, "action"] == "T"][5]
    for f in (clean, shifted):
        f.loc[tr, "price"] = f.loc[tr, "price"] * 10
    assert len(extreme_prices(clean)) >= 1
    assert extreme_prices(shifted).empty
    assert len(extreme_prices(shifted, ExtremePricesConfig(calendar=None))) >= 1


# ---------------- v3: missing-intervals gap model ----------------
def _arrivals(times_s, start="2026-10-08 14:00:00+00:00"):
    """Minimal one-key stream with ts_recv at the given seconds after start (in RTH)."""
    t0 = pd.Timestamp(start).value
    ns = (t0 + np.asarray(times_s) * 1e9).astype("int64")
    ts = pd.to_datetime(ns, unit="ns", utc=True)
    return pd.DataFrame({"ts_recv": ts, "ts_event": ts, "publisher_id": 2, "instrument_id": 1, "symbol": "X",
                         "rtype": 1, "sequence": np.arange(len(ns))})


def test_v3_min_gap_5_default(clean):
    df = clean.drop(index=window(clean, 101, 200, 203.5))  # ~3.5 s silence in a liquid name
    assert missing_intervals(df, MissingIntervalsConfig(min_gap_s=2.0)).shape[0] >= 1
    assert missing_intervals(df).empty
    df2 = clean.drop(index=window(clean, 101, 200, 208))  # 8 s outage still caught
    f = missing_intervals(df2)
    assert len(f) == 1 and f["value"].iloc[0] >= 7.9 and f["severity"].iloc[0] == "medium"


@pytest.mark.parametrize("model", ["poisson", "overdispersed", "tod"])
def test_v3_long_gap_is_high_in_every_model(model):
    t = np.r_[np.arange(0, 600, 0.05), np.arange(675, 1200, 0.05)]  # 20/s with a 75 s hole
    f = missing_intervals(normalize(_arrivals(t)), MissingIntervalsConfig(model=model))
    assert len(f) == 1 and f["severity"].iloc[0] == "high" and f["value"].iloc[0] > 60


def test_v3_overdispersed_model():
    rng = np.random.default_rng(0)
    # Poisson stream (CV ~ 1): overdispersed falls back to Poisson -> same flags
    t = np.cumsum(rng.exponential(0.1, 20000)); t = np.r_[t[t < 900], t[t >= 920]]
    p = missing_intervals(normalize(_arrivals(t)), MissingIntervalsConfig(model="poisson"))
    o = missing_intervals(normalize(_arrivals(t)), MissingIntervalsConfig(model="overdispersed"))
    assert len(p) == len(o) == 1
    # bursty stream: 1 s of 100/s then a 6-9 s lull, repeated -> CV >> 1, Poisson flags the lulls,
    # the Lomax threshold (heavier tail) does not
    parts, s = [], 0.0
    for _ in range(100):
        parts.append(s + np.arange(0, 1, 0.01)); s += 1 + rng.uniform(6, 9)
    tb = np.concatenate(parts)
    nb = normalize(_arrivals(tb))
    assert len(missing_intervals(nb, MissingIntervalsConfig(model="poisson"))) > 50
    assert missing_intervals(nb, MissingIntervalsConfig(model="overdispersed")).empty


def test_v3_unknown_model_raises(clean):
    with pytest.raises(ValueError):
        missing_intervals(clean, MissingIntervalsConfig(model="nb2"))
