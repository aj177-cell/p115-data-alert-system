"""All checks on the representative-sample extension, by stratum, against the pilot normal week.

    python -m m2.sample.run_sample --mode strata    # stratified metadata (2026-09-16 = vix_high + fomc)
    python -m m2.sample.run_sample --mode v3        # seven v3 checks, pilot preset (+ holiday-calendar variant)
    python -m m2.sample.run_sample --mode engine    # Milestone 1 engine on XNAS mbp-1 (equities only)
    python -m m2.sample.run_sample --mode bayes     # Bayesian + Poisson gap rule (+ holiday variant)
    python -m m2.sample.run_sample --mode spc       # bucket series + Phase I charts (+ holiday variant)
    python -m m2.sample.run_sample --mode volume    # front vs back month records/trades
    python -m m2.sample.run_sample --mode report    # stratum tables

v3 and engine also run on the 60 pilot normal-week symbol-days with the same code (baseline).
One symbol-day in memory at a time; per-symbol-day cache under m2/sample/results/cache/<mode>/,
file names built with _p() (dotted symbols). Writes m2/sample/results/ only; never touches stored results.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from integration.combine import load_v3
from m2.extract import load_symbol_slice, load_ticks
from m2.sample import holidays as H
from m2.sample.strata import PILOT_ROOT, SAMPLE_ROOT, normal_week_tasks, sample_tasks, stratum_of, strata_table, tags_for

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "m2" / "sample" / "results"
CACHE = OUT / "cache"
CHECKS = ["crossed_books", "extreme_prices", "timestamp_inversions", "missing_intervals", "stale_quotes",
          "unusual_spreads", "sequence_discontinuities"]


def _p(stem: Path, ext: str) -> Path:
    """stem + extension; never Path.with_suffix (ES.c.0 contains dots)."""
    return stem.parent / (stem.name + ext)


def _stem(mode, venue, symbol, day) -> Path:
    d = CACHE / mode
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{venue}_{symbol}_{day}"


def _root(piece: str) -> Path:
    return SAMPLE_ROOT if piece.startswith("S_") else PILOT_ROOT


def _ticks():
    t = load_ticks(PILOT_ROOT)
    t.update({k: v for k, v in load_ticks(SAMPLE_ROOT).items() if k not in t or not np.isfinite(t[k])})
    return t


SOURCE = "all"


def _all_tasks(include_normal=True):
    T = [("sample", *x) for x in sample_tasks()] if SOURCE in ("all", "sample") else []
    if include_normal and SOURCE in ("all", "pilot"):
        T += [("pilot", *x) for x in normal_week_tasks()]
    return T


def cfg_for(v3, venue, symbol, ticks, present):
    """The pilot preset (p115/analyze_v2.cfg_for): v3 presets + tick table + stale reference."""
    c = v3.config.equities_config() if venue == "XNAS" else v3.config.cme_futures_config()
    c.spreads.tick_size = dict(ticks)
    if venue == "XNAS":
        c.stale.reference = {"symbol": "QQQ" if symbol == "SPY" else "SPY"}
    elif symbol == "CL.c.0" and "ES.c.0" in present:
        c.stale.reference = {"symbol": "ES.c.0"}
    if venue == "GLBX":
        c.sequence.gap_mode = "skip"
    return c


def _load(piece, symbol, t0, t1, columns=None):
    fr = load_symbol_slice(piece, symbol, t0, t1, data_root=_root(piece), columns=columns or [
        "ts_recv", "ts_event", "rtype", "publisher_id", "instrument_id", "action", "side", "depth", "price", "size",
        "flags", "ts_in_delta", "sequence", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "bid_ct_00", "ask_ct_00", "symbol"])
    if "ts_recv" in fr.columns:
        fr = fr.set_index("ts_recv")
    return fr


def _status(piece, symbol):
    path = _root(piece) / piece / "data.parquet"
    if not path.is_file():
        return None
    import pyarrow.dataset as ds, pyarrow.compute as pc
    t = ds.dataset(path).to_table(filter=pc.field("symbol") == symbol).to_pandas()
    if "ts_recv" in t.columns:
        t = t.set_index("ts_recv")
    return t


def _present(source, piece, day):
    if source == "sample":
        from m2.sample.strata import GLBX, XNAS_DAYS
        return set(GLBX.get(day, [])) if piece.startswith("S_mbp-1_G") else set()
    return {"ES.c.0", "CL.c.0"}


def _counts(fl: pd.DataFrame) -> dict:
    out = {}
    f = fl[fl["severity"] != "info"] if len(fl) else fl
    for c in CHECKS:
        out[c] = int((f["check"] == c).sum()) if len(f) else 0
    if len(f):
        for (c, s), n in f.groupby(["check", "subtype"]).size().items():
            out[f"{c}/{s}"] = int(n)
    out["info"] = int((fl["severity"] == "info").sum()) if len(fl) else 0
    return out


def mode_v3():
    v3 = load_v3()
    D = v3.detectors
    ticks = _ticks()
    for source, venue, piece, stp, dfp, symbol, day, t0, t1 in _all_tasks():
        stem = _stem("v3", venue, symbol, day)
        if _p(stem, ".json").exists():
            continue
        t_ = time.time()
        fr = _load(piece, symbol, t0, t1)
        row = dict(source=source, venue=venue, symbol=symbol, day=day, records=len(fr))
        if len(fr):
            present = _present(source, piece, day)
            cfg = cfg_for(v3, venue, symbol, ticks, present)
            st = _status(stp, symbol)
            ref = _load(piece, cfg.stale.reference["symbol"], t0, t1) if cfg.stale.reference else None
            fl, _ = D.run_all(fr, cfg, status=st, reference_df=ref)
            fl.assign(source=source, venue=venue, day=day, symbol=symbol).to_parquet(_p(stem, ".parquet"))
            row.update({f"n:{k}": v for k, v in _counts(fl).items()})
            special = H.is_special(venue, day, H.root_of(symbol))
            if special:
                hc = H.holiday_config(cfg, venue, H.root_of(symbol), v3)
                hfl, _ = D.run_all(fr, hc, status=st, reference_df=ref)
                hfl.assign(source=source, venue=venue, day=day, symbol=symbol).to_parquet(_p(stem, ".holiday.parquet"))
                row.update({f"h:{k}": v for k, v in _counts(hfl).items()}, holiday=special)
            del ref
        row["secs"] = round(time.time() - t_, 1)
        _p(stem, ".json").write_text(json.dumps(row))
        print({k: row[k] for k in ("source", "symbol", "day", "records", "secs")}, flush=True)
        del fr


def mode_engine():
    from integration.combine import engine_table, run_engine
    from m2.carryover.crossed import flast_aware_crossed
    for source, venue, piece, stp, dfp, symbol, day, t0, t1 in _all_tasks():
        if venue != "XNAS":
            continue
        stem = _stem("engine", venue, symbol, day)
        if _p(stem, ".json").exists():
            continue
        t_ = time.time()
        fr = _load(piece, symbol, t0, t1)
        al, ctx = run_engine(fr, {"source": "local", "dataset": "XNAS.ITCH", "schema": "mbp-1"})
        tb = engine_table(al, ctx)
        tb.assign(source=source, day=day).to_parquet(_p(stem, ".parquet"))
        dem = flast_aware_crossed(tb, "demote") if len(tb) else tb
        row = dict(source=source, venue=venue, symbol=symbol, day=day, records=len(fr), alerts=len(tb),
                   **{f"rule:{r}": int(n) for r, n in tb.groupby("rule").size().items()} if len(tb) else {},
                   crossed_no_f_last=int(((tb["rule"] == "crossed_quote") & (tb["f_last"] == False)).sum()) if len(tb) else 0,  # noqa: E712
                   alerts_level_gt0_after_demote=int((dem["level"] > 0).sum()) if len(dem) else 0,
                   secs=round(time.time() - t_, 1))
        _p(stem, ".json").write_text(json.dumps(row))
        print({k: row[k] for k in ("source", "symbol", "day", "records", "alerts", "secs")}, flush=True)
        del fr, al, ctx


def mode_bayes():
    from m2.bayes.model import BayesGapConfig, poisson_rule_from_clock, posterior_from_clock
    from m2.bayes.run_bayes_gaps import load_clock, prior_for, prior_rates, session_clock
    v3 = load_v3()
    rates = prior_rates()
    gaps = []
    for source, venue, piece, stp, dfp, symbol, day, t0, t1 in _all_tasks(include_normal=False):
        stem = _stem("bayes", venue, symbol, day)
        if _p(stem, ".json").exists():
            continue
        t_ = time.time()
        loaded = load_clock(v3, venue, piece, symbol, t0, t1, _root(piece))
        row = dict(source=source, venue=venue, symbol=symbol, day=day)
        if loaded is None:
            row["records"] = 0
        else:
            t, key = loaded
            prior = prior_for(rates, symbol, day)
            cfg = BayesGapConfig(prior_rate=prior)
            variants = [("default", session_clock(v3, venue, t))]
            special = H.is_special(venue, day, H.root_of(symbol))
            if special:
                cal = H.holiday_calendar(venue, H.root_of(symbol), v3)
                variants.append(("holiday", v3.detectors.SessionClock.build(cal, int(t.min()), int(t.max()))))
            row.update(records=len(t), prior_rate=prior, holiday=special)
            for name, clk in variants:
                s = clk.clock(t)
                pois = poisson_rule_from_clock(s, key)
                post = posterior_from_clock(s, key, cfg)
                row.update({f"{name}:poisson": int(pois.sum()), f"{name}:bayes": int(post.candidate.sum()),
                            f"{name}:both": int((pois & post.candidate).sum()),
                            f"{name}:bayes_gt60": int((post.candidate & (post.gap_s > 60)).sum()),
                            f"{name}:max_gap_s": float(np.nanmax(post.gap_s)) if len(post.gap_s) else 0.0})
                sel = pois | post.candidate
                pd.DataFrame({"variant": name, "ts_recv_ns": t[sel], "gap_s": post.gap_s[sel],
                              "p_outage": post.p_outage[sel], "bayes": post.candidate[sel], "poisson": pois[sel]}
                             ).assign(venue=venue, symbol=symbol, day=day).to_parquet(_p(stem, f".{name}.parquet"))
        row["secs"] = round(time.time() - t_, 1)
        _p(stem, ".json").write_text(json.dumps(row))
        print({k: row.get(k) for k in ("symbol", "day", "records", "default:poisson", "default:bayes", "secs")}, flush=True)


def _episodes(buckets, gap=15) -> int:
    """Signals no more than ``gap`` buckets apart form one episode (as m2.spc.report.episodes)."""
    b = np.sort(np.asarray(buckets))
    return 0 if not len(b) else 1 + int((np.diff(b) > gap).sum())


def _spc_mask_holiday(series, venue, day, root, v3):
    """Keep only buckets inside the holiday-aware session."""
    from m2.spc.series import SESSION_OPEN_MIN, TZ
    if venue == "XNAS":
        start = pd.Timestamp(f"{day} 09:30", tz=TZ[venue])
    else:
        start = pd.Timestamp(day[:10], tz=TZ[venue]) - pd.Timedelta(hours=7)
    b = series["bucket"].to_numpy()
    t = start.tz_convert("UTC").value + b * 60 * 1_000_000_000
    cal = H.holiday_calendar(venue, root, v3)
    clk = v3.detectors.SessionClock.build(cal, int(t.min()), int(t.max()) + 60 * 1_000_000_000)
    keep = clk.in_session(t) & clk.in_session(t + 30 * 1_000_000_000)
    return series[keep].reset_index(drop=True)


def mode_spc():
    from m2.spc.evaluate import load_all, params_sets, phase1_history, signals, zscores
    from m2.spc.run_spc import to_ns
    from m2.spc.series import build_series
    v3 = load_v3()
    ticks = _ticks()
    allseries = load_all()  # pilot cache (normal-week history)
    sets = params_sets(allseries)
    rows, sig_rows = [], []
    for source, venue, piece, stp, dfp, symbol, day, t0, t1 in _all_tasks(include_normal=False):
        stem = _stem("spc", venue, symbol, day)
        if _p(stem, ".parquet").exists():
            s = pd.read_parquet(_p(stem, ".parquet"))
        else:
            fr = _load(piece, symbol, t0, t1, columns=["ts_recv", "flags", "bid_px_00", "ask_px_00", "symbol"]).reset_index()
            tick = float(ticks.get(symbol, 0.01) or 0.01)
            s = build_series(to_ns(fr["ts_recv"]), fr["flags"].to_numpy().astype(np.int64),
                             pd.to_numeric(fr["bid_px_00"], errors="coerce").to_numpy(),
                             pd.to_numeric(fr["ask_px_00"], errors="coerce").to_numpy(), tick, venue, day, v3)
            s = s.assign(venue=venue, symbol=symbol, day=day, tick=tick)
            s.to_parquet(_p(stem, ".parquet"), index=False)
            del fr
        hist_sym = symbol.replace(".c.1", ".c.0")
        hist = phase1_history(allseries, hist_sym, day)
        variants = [("default", s)]
        special = H.is_special(venue, day, H.root_of(symbol))
        if special:
            variants.append(("holiday", _spc_mask_holiday(s, venue, day, H.root_of(symbol), v3)))
        for name, ser in variants:
            if not len(ser) or not hist:
                continue
            z = zscores(ser, hist)
            sig = signals(z, sets["phase1"])
            pos = ser["bucket"].to_numpy()
            for (k, c), m in sig.items():
                rows.append(dict(source=source, venue=venue, symbol=symbol, day=day, variant=name, holiday=special,
                                 history=hist_sym, buckets=len(ser), kind=k, chart=c, signals=int(m.sum()),
                                 episodes=_episodes(pos[m]),
                                 median_z=float(np.nanmedian(z[k])),
                                 rate_ratio=float(ser["n_records"].sum() / max(np.median([h["n_records"].sum() for h in hist]), 1))))
                for i in np.flatnonzero(m):
                    sig_rows.append(dict(venue=venue, symbol=symbol, day=day, variant=name, kind=k, chart=c, bucket=int(pos[i]), z=float(z[k][i])))
        print(dict(symbol=symbol, day=day, buckets=len(s)), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "spc_by_symbol_day.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(OUT / "spc_signals.csv", index=False)


def mode_volume():
    rows = []
    for source, venue, piece, stp, dfp, symbol, day, t0, t1 in _all_tasks():
        if venue != "GLBX":
            continue
        fr = _load(piece, symbol, t0, t1, columns=["ts_recv", "instrument_id", "action", "size", "flags"])
        tr = fr["action"] == "T"
        rows.append(dict(source=source, symbol=symbol, day=day, records=len(fr), trades=int(tr.sum()),
                         volume=int(fr.loc[tr, "size"].sum()), instrument_ids="|".join(map(str, sorted(fr["instrument_id"].unique())))))
        del fr
    v = pd.DataFrame(rows)
    v.to_csv(OUT / "futures_volume.csv", index=False)
    print(v.to_string(index=False))


def mode_strata():
    t = strata_table()
    t.to_csv(OUT / "strata.csv", index=False)
    print(t.groupby(["source", "stratum"]).size().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["strata", "v3", "engine", "bayes", "spc", "volume", "report"], required=True)
    ap.add_argument("--source", choices=["all", "sample", "pilot"], default="all")
    a = ap.parse_args()
    global SOURCE
    SOURCE = a.source
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "report":
        from m2.sample.report import main as rep
        rep()
    else:
        globals()[f"mode_{a.mode}"]()


if __name__ == "__main__":
    main()
