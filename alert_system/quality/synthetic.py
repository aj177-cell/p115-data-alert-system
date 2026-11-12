"""Synthetic Databento-style mbp-1 streams and labeled fault injection.

``generate_mbp1`` builds a clean stream with the documented mbp-1 fields:
random-walk bid on a tick grid, 1-3 tick spread, Poisson arrivals, sequential
``sequence`` per publisher, ts_event = ts_recv - latency, F_LAST set (plus a
few non-F_LAST intermediate records whose book is deliberately inconsistent,
to prove book checks ignore them).

``inject_faults`` adds faults of all seven types at known locations and
returns ground-truth labels. Both work on *raw* frames (int64 fixed-point
prices, int64 ns timestamps, ts_recv as a column); ``to_databento_df`` turns a
raw frame into what ``DBNStore.to_df()`` / the team Parquet looks like
(float prices with NaN for UNDEF_PRICE, UTC datetimes, ts_recv index).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import F_LAST, UNDEF_PRICE

NS = 1_000_000_000
PX = 1_000_000_000  # fixed-point scale
MBP1_COLUMNS = ["ts_recv", "ts_event", "rtype", "publisher_id", "instrument_id", "action", "side", "depth",
                "price", "size", "flags", "ts_in_delta", "sequence", "bid_px_00", "ask_px_00",
                "bid_sz_00", "ask_sz_00", "bid_ct_00", "ask_ct_00", "symbol"]


@dataclass
class InstrumentSpec:
    instrument_id: int
    symbol: str
    start_price: float
    rate_per_s: float
    tick: float = 0.01


DEFAULT_INSTRUMENTS = (InstrumentSpec(101, "TEST", 100.0, 10.0), InstrumentSpec(102, "REF", 500.0, 20.0))


def generate_mbp1(start: str = "2026-10-08 13:30:00+00:00", duration_s: float = 3600.0,
                  instruments: tuple[InstrumentSpec, ...] = DEFAULT_INSTRUMENTS, publisher_id: int = 2,
                  intermediate_prob: float = 0.03, seed: int = 7) -> pd.DataFrame:
    """Clean raw mbp-1 stream (see module docstring)."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start).value
    parts = []
    for spec in instruments:
        tick = int(round(spec.tick * PX))
        n = rng.poisson(spec.rate_per_s * duration_s)
        recv = np.sort(t0 + rng.uniform(0, duration_s * NS, n).astype(np.int64))
        steps = rng.choice([-1, 0, 1], size=n, p=[0.15, 0.7, 0.15])
        bid = int(round(spec.start_price * PX / tick)) * tick + np.cumsum(steps) * tick
        ask = bid + rng.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1]) * tick
        is_trade = rng.random(n) < 0.1
        side = np.where(rng.random(n) < 0.5, "B", "A")
        action = np.where(is_trade, "T", rng.choice(["A", "C", "M"], size=n))
        price = np.where(side == "B", bid, ask)
        lat = (20_000 * rng.lognormal(0, 0.3, n)).astype(np.int64)
        frame = pd.DataFrame({
            "ts_recv": recv, "ts_event": np.maximum.accumulate(recv - lat), "rtype": 1, "publisher_id": publisher_id,
            "instrument_id": spec.instrument_id, "action": action, "side": side, "depth": 0,
            "price": price.astype(np.int64), "size": rng.integers(1, 300, n), "flags": F_LAST,
            "ts_in_delta": (lat // 2).astype(np.int32), "sequence": 0,
            "bid_px_00": bid.astype(np.int64), "ask_px_00": ask.astype(np.int64),
            "bid_sz_00": rng.integers(1, 500, n), "ask_sz_00": rng.integers(1, 500, n),
            "bid_ct_00": rng.integers(1, 20, n), "ask_ct_00": rng.integers(1, 20, n), "symbol": spec.symbol,
        })
        inter = frame[rng.random(n) < intermediate_prob].copy()
        inter["flags"] = 0
        inter["bid_px_00"] = inter["ask_px_00"] + 2 * tick  # mid-update book: crossed, must be ignored
        inter["action"] = "C"
        inter["_order"] = 0
        frame["_order"] = 1
        parts += [frame, inter]
    raw = pd.concat(parts, ignore_index=True).sort_values(["ts_recv", "instrument_id", "_order"], kind="stable")
    raw = raw.drop(columns="_order").reset_index(drop=True)
    renumber_sequence(raw)
    return raw[MBP1_COLUMNS]


def renumber_sequence(raw: pd.DataFrame) -> None:
    """Sequential per-publisher sequence in file order; an intermediate
    (non-F_LAST) record shares its sequence with the next record, like several
    records normalized from one venue message."""
    for _, idx in raw.groupby("publisher_id").groups.items():
        last = (raw.loc[idx, "flags"].to_numpy() & F_LAST) != 0
        seq = 1000 + np.cumsum(last.astype(np.int64)) + (~last)
        raw.loc[idx, "sequence"] = seq.astype(np.int64)


def to_databento_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw -> ``to_df()`` layout: float prices, UTC datetimes, ts_recv index."""
    out = raw.copy()
    for col in ("price", "bid_px_00", "ask_px_00"):
        vals = out[col].to_numpy()
        out[col] = np.where(vals == UNDEF_PRICE, np.nan, vals / PX)
    for col in ("ts_recv", "ts_event"):
        out[col] = pd.to_datetime(out[col], unit="ns", utc=True)
    out["flags"] = out["flags"].astype(np.uint8)
    out["sequence"] = out["sequence"].astype(np.uint32)
    out["symbol"] = out["symbol"].astype("string")
    return out.set_index("ts_recv")


FAULT_TYPES = ["crossed_books", "extreme_prices", "timestamp_inversions", "missing_intervals",
               "stale_quotes", "unusual_spreads", "sequence_discontinuities"]


def inject_faults(raw: pd.DataFrame, n_per_type: int = 5, seed: int = 11, tick: float = 0.01,
                  reference_instrument: int | None = 102, slot_s: float = 60.0,
                  warmup_s: float = 180.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject labeled faults into a raw clean stream.

    Each fault gets its own ``slot_s`` slot (non-overlapping) after a warm-up.
    Returns ``(faulted_raw, labels)``; labels have columns check, subtype,
    publisher_id, instrument_id (None for publisher-level sequence faults),
    start, end (UTC Timestamps), magnitude.

    Fault recipes: crossed = bid set k ticks above ask for 2 s; extreme = one
    trade print x10 or x0.1 (decimal slip); timestamps = ts_recv moved back
    5 ms / ts_event moved back 50 ms (latency spike) / ts_event after ts_recv;
    missing = all records of the instrument deleted for 20 s (sequence then
    renumbered so the gap shows up only as a time gap); stale = BBO frozen
    for 30 s while trades keep printing; spreads = ask widened by 10 ticks for
    5 s; sequence = +k jump or a swapped pair.
    """
    rng = np.random.default_rng(seed)
    df = raw.copy().reset_index(drop=True)
    tick_px = int(round(tick * PX))
    t_start, t_end = int(df["ts_recv"].min()), int(df["ts_recv"].max())
    n_slots = int((t_end - t_start) / NS - warmup_s) // int(slot_s)
    jobs = [(c, i) for c in FAULT_TYPES for i in range(n_per_type)]
    if len(jobs) > n_slots:
        raise ValueError(f"need {len(jobs)} slots of {slot_s}s, stream has {n_slots}")
    slots = rng.permutation(n_slots)[:len(jobs)]
    plan = list(zip(jobs, slots))
    instruments = sorted(df["instrument_id"].unique())
    labels: list[dict] = []
    pub = int(df["publisher_id"].iloc[0])

    def slot_time(slot: int) -> int:
        return t_start + int((warmup_s + slot * slot_s + 5) * NS)

    def rows(iid: int, a: int, b: int) -> np.ndarray:
        t = df["ts_recv"].to_numpy()
        return np.flatnonzero((df["instrument_id"].to_numpy() == iid) & (t >= a) & (t <= b))

    # 1) missing intervals first (deletes rows), then renumber sequence
    for (check, i), slot in [p for p in plan if p[0][0] == "missing_intervals"]:
        iid = int(rng.choice(instruments)); a = slot_time(slot); b = a + 20 * NS
        df = df.drop(index=rows(iid, a, b)).reset_index(drop=True)
        labels.append(dict(check=check, subtype="gap", publisher_id=pub, instrument_id=iid, start=a, end=b, magnitude=20.0))
    renumber_sequence(df)
    for (check, i), slot in plan:
        a = slot_time(slot)
        iid = int(rng.choice([x for x in instruments if x != reference_instrument] if check == "stale_quotes" else instruments))
        if check == "crossed_books":
            b = a + 2 * NS; r = rows(iid, a, b); k = int(rng.integers(1, 4))
            df.loc[r, "bid_px_00"] = df.loc[r, "ask_px_00"] + k * tick_px
            labels.append(dict(check=check, subtype="crossed", publisher_id=pub, instrument_id=iid, start=a, end=b, magnitude=k))
        elif check == "extreme_prices":
            r = rows(iid, a, a + 30 * NS)
            r = r[df.loc[r, "action"].to_numpy() == "T"][:1]
            factor = 10.0 if i % 2 == 0 else 0.1
            df.loc[r, "price"] = (df.loc[r, "price"] * factor).round().astype(np.int64)
            t = int(df.loc[r[0], "ts_recv"])
            labels.append(dict(check=check, subtype="decimal_slip", publisher_id=pub, instrument_id=iid, start=t, end=t, magnitude=factor))
        elif check == "timestamp_inversions":
            r = rows(iid, a, a + 30 * NS)[5]
            kind = ["ts_recv_back", "latency_spike", "event_after_recv"][i % 3]
            if kind == "ts_recv_back":
                prev = rows(iid, a, a + 30 * NS)[4]
                df.loc[r, "ts_recv"] = int(df.loc[prev, "ts_recv"]) - 5_000_000
            elif kind == "latency_spike":
                df.loc[r, "ts_event"] = int(df.loc[r, "ts_recv"]) - 50_000_000
            else:
                df.loc[r, "ts_event"] = int(df.loc[r, "ts_recv"]) + 1_000_000
            t = int(df.loc[r, "ts_recv"])
            labels.append(dict(check=check, subtype=kind, publisher_id=pub, instrument_id=iid, start=t - 10_000_000, end=t, magnitude=np.nan))
        elif check == "stale_quotes":
            b = a + 30 * NS; r = rows(iid, a, b)
            r = r[(df.loc[r, "flags"].to_numpy() & F_LAST) != 0]
            cols = ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "bid_ct_00", "ask_ct_00"]
            df.loc[r, cols] = df.loc[r[:1], cols].to_numpy().repeat(len(r), axis=0)
            labels.append(dict(check=check, subtype="frozen_bbo", publisher_id=pub, instrument_id=iid, start=a, end=b, magnitude=30.0))
        elif check == "unusual_spreads":
            b = a + 5 * NS; r = rows(iid, a, b)
            df.loc[r, "ask_px_00"] = df.loc[r, "ask_px_00"] + 10 * tick_px
            labels.append(dict(check=check, subtype="wide", publisher_id=pub, instrument_id=iid, start=a, end=b, magnitude=10.0))
    for (check, i), slot in [p for p in plan if p[0][0] == "sequence_discontinuities"]:
        a = slot_time(slot)
        r = int(np.searchsorted(df["ts_recv"].to_numpy(), a))
        if i % 2 == 0:
            k = int(rng.integers(5, 50))
            df.loc[r:, "sequence"] = df.loc[r:, "sequence"] + k
            labels.append(dict(check=check, subtype="gap", publisher_id=pub, instrument_id=None,
                               start=int(df.loc[r, "ts_recv"]), end=int(df.loc[r, "ts_recv"]), magnitude=k))
        else:
            while not (df.loc[r, "flags"] & F_LAST and df.loc[r + 1, "flags"] & F_LAST and df.loc[r + 2, "flags"] & F_LAST):
                r += 1
            s0, s1 = int(df.loc[r, "sequence"]), int(df.loc[r + 1, "sequence"])
            df.loc[r, "sequence"], df.loc[r + 1, "sequence"] = s1, s0
            labels.append(dict(check=check, subtype="swap", publisher_id=pub, instrument_id=None,
                               start=int(df.loc[r, "ts_recv"]), end=int(df.loc[r + 2, "ts_recv"]), magnitude=-1))
    lab = pd.DataFrame(labels)
    lab["start"] = pd.to_datetime(lab["start"], unit="ns", utc=True)
    lab["end"] = pd.to_datetime(lab["end"], unit="ns", utc=True)
    return df, lab.sort_values("start").reset_index(drop=True)
