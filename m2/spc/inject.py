"""Injected faults on real symbol-days: SPC charts vs the existing point checks.

Faults (three 30-minute windows per run, away from the first/last 30 minutes, fixed seeds):
  widen_x1.5 / widen_x2 / widen_x3 : ask raised so the spread is >= f times wider (whole ticks)
  ramp_1to3                       : the factor rises linearly from 1 to 3 across the window
  thin_30 / thin_50 / thin_70     : that share of all records in the window removed at random
                                    (partial feed degradation, not a full outage)
Compared with: v3 unusual_spreads (rolling-quantile rule, pilot preset) for spread faults; the v3
Poisson gap rule and the Bayesian gap check for thinning faults.

A window counts as detected only by a signal / candidate that the unmodified day does not have
(SPC: same chart and bucket; point checks: no clean candidate within 2 s), so background candidates
that happen to fall in a window are not credited. 'false_alarms' on faulted runs counts new
signals/candidates outside the windows (+15 min guard); the clean-day rows hold the background counts.
"""
from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from integration.combine import load_v3
from m2.bayes.model import BayesGapConfig, posterior_from_clock, poisson_rule_from_clock
from m2.bayes.run_bayes_gaps import prior_for, prior_rates
from m2.extract import COLS, DEFAULT_DATA_ROOT, load_symbol_slice, load_ticks, tasks
from m2.spc.evaluate import KINDS, load_all, params_sets, phase1_history, signals, zscores
from m2.spc.series import bucket_index, build_series, calendar

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "m2" / "results" / "spc"
NS = 1_000_000_000
DAY = "2026-10-09"
SYMBOLS = [("XNAS", "AAPL"), ("XNAS", "MSFT"), ("XNAS", "CRWV"), ("XNAS", "NBIS"), ("XNAS", "PLAB"),
           ("XNAS", "ADTN"), ("XNAS", "LWAY"), ("GLBX", "CL.c.0"), ("GLBX", "ES.c.0")]
FAULTS = {"widen_x1.5": ("spread", 1.5), "widen_x2": ("spread", 2.0), "widen_x3": ("spread", 3.0),
          "ramp_1to3": ("spread", "ramp"), "thin_30": ("rate", 0.3), "thin_50": ("rate", 0.5), "thin_70": ("rate", 0.7)}
WIN_MIN, N_WIN, MARGIN, GAP_MIN = 30, 3, 30, 30
SPC_GRACE = 5      # minutes after the window end still counted as a detection
FA_GUARD = 15      # minutes after the window end excluded from false-alarm counts


def bucket_start_ns(venue, day, buckets):
    from m2.spc.series import TZ
    start = (pd.Timestamp(f"{day} 09:30", tz=TZ[venue]) if venue == "XNAS"
             else pd.Timestamp(day, tz=TZ[venue]) - pd.Timedelta(hours=7)).tz_convert("UTC").value
    return start + np.asarray(buckets, dtype=np.int64) * 60 * NS


def choose_windows(allb: np.ndarray, rng) -> list[int]:
    lo, hi = allb.min() + MARGIN, allb.max() - MARGIN - WIN_MIN
    for _ in range(1000):
        s = np.sort(rng.integers(lo, hi, N_WIN))
        if np.all(np.diff(s) >= WIN_MIN + GAP_MIN) and all(np.isin(np.arange(x, x + WIN_MIN), allb).all() for x in s):
            return [int(x) for x in s]
    raise RuntimeError("no window placement")


def apply_fault(fr: pd.DataFrame, t: np.ndarray, wins_ns, fault, tick, rng):
    kind, val = FAULTS[fault]
    if kind == "rate":
        keep = np.ones(len(fr), bool)
        for s, e in wins_ns:
            m = (t >= s) & (t < e)
            keep[m] = rng.random(int(m.sum())) >= val
        return fr[keep].reset_index(drop=True), t[keep]
    fr = fr.copy()
    bid = pd.to_numeric(fr["bid_px_00"], errors="coerce").to_numpy()
    ask = pd.to_numeric(fr["ask_px_00"], errors="coerce").to_numpy()
    sp = (ask - bid) / tick
    new_ask = ask.copy()
    for s, e in wins_ns:
        m = (t >= s) & (t < e) & np.isfinite(sp) & (sp > 0)
        f = np.full(m.sum(), val) if val != "ramp" else 1.0 + 2.0 * (t[m] - s) / (e - s)
        extra = np.ceil((f - 1.0) * np.round(sp[m]) - 1e-9)
        new_ask[m] = ask[m] + extra * tick
    fr["ask_px_00"] = new_ask
    return fr, t


def first_in(times, s, e):
    x = times[(times >= s) & (times <= e)]
    return int(x.min()) if len(x) else None


def gap_checks(t, pub, iid, venue, v3, prior):
    order = np.lexsort((t, iid, pub))
    ts, p, i = t[order], pub[order], iid[order]
    key = pd.DataFrame({"p": p, "i": i}).groupby(["p", "i"], sort=False).ngroup().to_numpy()
    clk = v3.detectors.SessionClock.build(calendar(v3, venue), int(ts.min()), int(ts.max()))
    s = clk.clock(ts)
    pois = poisson_rule_from_clock(s, key)
    bay = posterior_from_clock(s, key, BayesGapConfig(prior_rate=prior)).candidate
    return {"poisson_gap": ts[pois], "bayes_gap": ts[bay]}


def run_inject(data_root=DEFAULT_DATA_ROOT):
    v3 = load_v3()
    ticks = load_ticks(data_root)
    allseries = load_all()
    sets = params_sets(allseries)
    rates = prior_rates()
    T = {(x[0], x[3], x[2]): x for x in tasks()}
    rows = []
    for venue, sym in SYMBOLS:
        t_ = time.time()
        _, piece, _, day, t0, t1 = T[(venue, DAY, sym)]
        fr = load_symbol_slice(piece, sym, t0, t1, data_root=data_root, columns=COLS)
        if "ts_recv" not in fr.columns:
            fr = fr.reset_index()
        t = fr["ts_recv"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        tick = float(ticks.get(sym, 0.01) or 0.01)
        hist = phase1_history(allseries, sym, day)
        allb = allseries[(sym, day)]["bucket"].to_numpy()
        cfg = (v3.config.equities_config() if venue == "XNAS" else v3.config.cme_futures_config()).spreads
        cfg.tick_size = dict(ticks)
        prior = prior_for(rates, sym, day)

        def spc(frame, tt):
            s = build_series(tt, frame["flags"].to_numpy().astype(np.int64),
                             pd.to_numeric(frame["bid_px_00"], errors="coerce").to_numpy(),
                             pd.to_numeric(frame["ask_px_00"], errors="coerce").to_numpy(), tick, venue, day, v3)
            z = zscores(s, hist)
            return s["bucket"].to_numpy(), {p: signals(z, prm) for p, prm in sets.items()}

        def spread_rule(frame):
            f = v3.detectors.unusual_spreads(frame, cfg)
            f = f[f["subtype"] == "wide"]
            return f["ts_recv"].astype("datetime64[ns, UTC]").astype("int64").to_numpy() if len(f) else np.array([], np.int64)

        # clean day: false-alarm reference for every method
        b0, sig0 = spc(fr, t)
        clean_point = {"spread_rule": spread_rule(fr)}
        clean_point.update(gap_checks(t, fr["publisher_id"].to_numpy(), fr["instrument_id"].to_numpy(), venue, v3, prior))
        base = dict(venue=venue, symbol=sym, day=day)
        for p, sg in sig0.items():
            for (k, c), m in sg.items():
                rows.append({**base, "fault": "clean", "method": f"spc_{k}_{c}", "params": p, "false_alarms": int(m.sum())})
        for nm, ts in clean_point.items():
            rows.append({**base, "fault": "clean", "method": nm, "params": "v3", "false_alarms": len(ts)})

        for fault, (kind, _) in FAULTS.items():
            rng = np.random.default_rng(zlib.crc32(f"{sym}|{day}|{fault}".encode()))
            wins_b = choose_windows(allb, rng)
            wins_ns = [(int(bucket_start_ns(venue, day, [w])[0]), int(bucket_start_ns(venue, day, [w + WIN_MIN])[0])) for w in wins_b]
            ff, tt = apply_fault(fr, t, wins_ns, fault, tick, rng)
            bb, sg = spc(ff, tt)
            for p, sgp in sg.items():
                for c in ("shewhart", "cusum", "ewma"):
                    m = sgp[(kind, c)]
                    clean_b = set(b0[sig0[p][(kind, c)]].tolist())
                    sb = bb[m]
                    sb = sb[~np.isin(sb, list(clean_b))]  # credit only signals the clean day does not have
                    det, delays = 0, []
                    guard = np.zeros(len(sb), bool)
                    for w in wins_b:
                        hit = sb[(sb >= w) & (sb <= w + WIN_MIN - 1 + SPC_GRACE)]
                        guard |= (sb >= w) & (sb <= w + WIN_MIN - 1 + FA_GUARD)
                        if len(hit):
                            det += 1; delays.append(int(hit.min() - w + 1))
                    rows.append({**base, "fault": fault, "method": f"spc_{kind}_{c}", "params": p, "windows": len(wins_b),
                                 "detected": det, "median_delay_min": float(np.median(delays)) if delays else np.nan,
                                 "delays": ";".join(map(str, delays)), "false_alarms": int((~guard).sum())})
            if kind == "spread":
                point = {"spread_rule": spread_rule(ff)}
            else:
                point = gap_checks(tt, ff["publisher_id"].to_numpy(), ff["instrument_id"].to_numpy(), venue, v3, prior)
            for nm, ts in point.items():
                cl = np.sort(clean_point[nm])
                if len(cl) and len(ts):
                    j = np.clip(np.searchsorted(cl, ts), 1, len(cl)) - 1
                    d = np.minimum(np.abs(ts - cl[j]), np.abs(cl[np.clip(j + 1, 0, len(cl) - 1)] - ts))
                    ts = ts[d > 2 * NS]  # credit only candidates the clean day does not have (within 2 s)
                det, delays = 0, []
                guard = np.zeros(len(ts), bool)
                for s, e in wins_ns:
                    x = first_in(ts, s, e + 60 * NS)
                    guard |= (ts >= s) & (ts <= e + FA_GUARD * 60 * NS)
                    if x is not None:
                        det += 1; delays.append((x - s) / 60e9)
                rows.append({**base, "fault": fault, "method": nm, "params": "v3", "windows": len(wins_ns), "detected": det,
                             "median_delay_min": float(np.median(delays)) if delays else np.nan,
                             "delays": ";".join(f"{d:.2f}" for d in delays), "false_alarms": int((~guard).sum())})
            del ff, tt
        print(sym, round(time.time() - t_, 1), "s", flush=True)
        del fr, t
        pd.DataFrame(rows).to_csv(OUT / "inject_raw.csv", index=False)
