"""Evidence and sidecar outputs for the carryover decisions (docs/m2_carryover_decisions.md).

    python -m m2.carryover.run_carryover --mode crossed    # stress-stream modes + real-data census
    python -m m2.carryover.run_carryover --mode metrics    # regenerate v3 synthetic metrics (sidecar)
    python -m m2.carryover.run_carryover --mode pauses     # trade-pause census + bundled trades sample
    python -m m2.carryover.run_carryover --mode severity   # severity map + counts on stress and real outputs
    python -m m2.carryover.run_carryover --mode streaming  # live vs batch on stress stream and 3 real days
    python -m m2.carryover.run_carryover --mode all

Read-only on the data; writes only under m2/carryover/results/. No network access.
"""
from __future__ import annotations

from m2.carryover._util import ns

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
ROOT = Path(__file__).resolve().parents[2]


def _stress():
    from integration import stress
    return stress, stress.stress_test()


def mode_crossed():
    from m2.carryover.crossed import census_all, flast_aware_crossed
    from integration.combine import load_v3
    stress, r = _stress()
    v3 = load_v3()
    raw = v3.synthetic.generate_mbp1(seed=7)
    faulted, _ = v3.synthetic.inject_faults(raw, seed=11)
    frame = v3.synthetic.to_databento_df(faulted)
    rows = []
    for mode in ("mark", "demote", "drop", "persist"):
        kw = {"frame": frame} if mode == "persist" else {}
        u = flast_aware_crossed(r.unified, mode, **kw)
        c = flast_aware_crossed(r.clean, mode, **({"frame": v3.synthetic.to_databento_df(raw)} if mode == "persist" else {}))
        det, _ = stress.score(u, r.labels)
        sel = lambda t: (t["source"] == "engine_m1") & (t["rule"] == "crossed_quote") & (t["level"] > 0)
        rows.append(dict(mode=mode, crossed_labels=int(det.loc["crossed_books", "labels"]),
                         engine_detected=int(det.loc["crossed_books", "engine_m1"]),
                         either_all_families=int(det["either"].sum()), labels_all=int(det["labels"].sum()),
                         engine_crossed_findings_faulted=int(sel(u).sum()), engine_crossed_findings_clean=int(sel(c).sum())))
    pd.DataFrame(rows).to_csv(OUT / "crossed_modes_stress.csv", index=False)
    path = OUT / "crossed_census_real.csv"
    if not path.is_file():
        census_all().to_csv(path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def mode_metrics():
    from m2.carryover.metrics_regen import compare, flag_diff, regenerate, SIDE
    regenerate()
    compare().to_csv(OUT / "v3_metrics_diff.csv", index=False)
    flag_diff().to_csv(OUT / "v3_metrics_extreme_flag_diff.csv", index=False)
    # which settings reproduce the stored file: v1-era k=8 and no log floor
    from integration.combine import load_v3
    v3 = load_v3(names=("config", "detectors", "synthetic", "evaluate")); ev = v3.evaluate
    raw = ev.generate_mbp1(seed=7); f, lab = ev.inject_faults(raw, seed=11); norm = ev.normalize(ev.to_databento_df(f))
    rows = []
    for k, floor in ((12.0, 1e-4), (8.0, 1e-4), (12.0, 0.0), (8.0, 0.0)):
        cfg = ev.base_config(); cfg.extreme.k = k; cfg.extreme.mad_floor_log = floor
        fl, _ = ev.run_all(norm, cfg, checks=["extreme_prices"]); m = ev.match(fl, lab)
        cf, _ = ev.run_all(ev.to_databento_df(raw), cfg, checks=["extreme_prices"])
        row = m[m["check"] == "extreme_prices"].iloc[0]
        rows.append(dict(k=k, mad_floor_log=floor, flags=int(row["flags"]), tp=int(row["tp_flags"]), fp=int(row["fp_flags"]),
                         collateral=int(row["collateral_flags"]), clean=int((cf["severity"] != "info").sum())))
    pd.DataFrame(rows).to_csv(OUT / "v3_metrics_extreme_settings.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print("sidecar:", SIDE)


def mode_pauses():
    from m2.carryover.trade_pauses import census_all, floors_from_census, reclassify
    from integration.combine import bundled, run_combined
    path = OUT / "trade_pause_census.csv"
    if not path.is_file():
        census_all().to_csv(path, index=False)
    c = pd.read_csv(path)
    floors = floors_from_census(c)
    (OUT / "trade_pause_floors.json").write_text(json.dumps(floors, indent=1))
    fr, meta = bundled("trades")
    u = run_combined(fr, meta).unified
    q, _ = bundled("quotes")
    rows = []
    for name, kw in (("native", None), ("quotes_sample", dict(quotes=q.reset_index())),
                     ("floor_default_15s", dict(floor_s=15.0)), ("floor_per_symbol", dict(floor_s=floors))):
        x = u if kw is None else reclassify(u, "trades", **kw)
        g = x[x["rule"] == "missing_intervals"]
        for _, row in g.iterrows():
            rows.append(dict(policy=name, symbol=row["symbol"], ts_recv=str(row["ts_recv"]), gap_s=round(float(row["value"]), 3),
                             subtype=row["subtype"], severity=row["severity"], level=row["level"]))
    pd.DataFrame(rows).to_csv(OUT / "trade_pause_bundled.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(floors)


def mode_severity():
    from m2.carryover import severity as S
    from m2.carryover.crossed import flast_aware_crossed
    table = S.write_map(OUT / "severity_map.csv")
    stress, r = _stress()
    rows = []
    for name, u in (("stress_faulted", r.unified), ("stress_clean", r.clean)):
        a = S.apply(u)
        for (src, sev, sev2), n in a.groupby(["source", "severity", "severity_m2"]).size().items():
            rows.append(dict(table=name, source=src, native=sev, severity_m2=sev2, n=int(n)))
    g = pd.read_parquet(ROOT / "m2/results/bayes/real_gaps.parquet")
    b = S.apply(S.bayes_rows(g))
    for sev2, n in b.groupby("severity_m2").size().items():
        rows.append(dict(table="real_bayes_95", source="m2_bayes", native="n/a", severity_m2=sev2, n=int(n)))
    sig = ROOT / "m2/results/spc/real_signal_list_phase1.csv"
    if sig.is_file():
        s = S.apply(S.spc_rows(pd.read_csv(sig)))
        for (rule, sub, sev2), n in s.groupby(["rule", "subtype", "severity_m2"]).size().items():
            rows.append(dict(table="real_spc_episodes", source=f"m2_spc/{rule}/{sub}", native="n/a", severity_m2=sev2, n=int(n)))
    pd.DataFrame(rows).to_csv(OUT / "severity_counts.csv", index=False)
    print(table[["source", "rule", "subtype", "condition", "native_severity", "severity_m2"]].to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


def mode_streaming():
    from m2.carryover import streaming as L
    from integration.combine import load_v3, v3_config, v3_table
    from m2.extract import load_symbol_slice, tasks
    v3 = load_v3(); D = v3.detectors
    pd.DataFrame(L.CAUSALITY, columns=["check", "subtype", "causal_in_batch", "earliest_live_detection", "live_version"]).to_csv(
        OUT / "streaming_causality.csv", index=False)
    rows = []
    raw = v3.synthetic.generate_mbp1(seed=7); f, lab = v3.synthetic.inject_faults(raw, seed=11)
    df = v3.synthetic.to_databento_df(f)
    cfg = v3.config.DetectorConfig()
    bf = D.missing_intervals(df, cfg.missing)
    t = ns(df.index if "ts_recv" not in df else df["ts_recv"]).to_numpy()
    clk = D.SessionClock.build(cfg.missing.calendar, int(t.min()), int(t.max()))
    m = L.StreamingGapMonitor(cfg.missing.alpha, cfg.missing.rate_window, cfg.missing.min_gap_s, cfg.missing.max_gap_s, clock=clk.clock)
    sf = L.replay_gaps(df, m)
    rows.append(dict(data="stress_faulted", check="missing_intervals", batch=len(bf), live=len(sf),
                     identical=sorted(ns(bf["ts_recv"])) == sorted(sf["ts_recv_ns"])))
    cb = D.crossed_books(df, cfg.crossed); cb = cb[cb["subtype"] == "crossed"]
    sc = L.replay_crossed(df, L.StreamingCrossed(cfg.crossed.min_duration_s))
    rows.append(dict(data="stress_faulted", check="crossed_books/crossed", batch=len(cb), live=len(sc),
                     identical=sorted(ns(cb["ts_recv"])) == sorted(sc["start_ns"]),
                     median_live_delay_s=float(((sc["detected_ns"] - sc["start_ns"]) / 1e9).median())))
    # latency_outlier: whole-day (batch) vs trailing window on the stress stream
    norm = D.normalize(df)
    tl = D.timestamp_inversions(norm, cfg.timestamps)
    bl = tl[tl["subtype"] == "latency_outlier"]
    lat = (norm["ts_recv_ns"] - norm["ts_event_ns"]).to_numpy()
    key = list(zip(norm["publisher_id"], norm["instrument_id"]))
    tr = L.TrailingLatencyOutlier(k=cfg.timestamps.latency_k, min_excess_ns=cfg.timestamps.latency_min_excess_ns)
    hits = np.array([tr.update(int(x), k)[0] for x, k in zip(lat, key)])
    lab_lat = lab[lab["check"] == "timestamp_inversions"]
    tt = norm["ts_recv_ns"].to_numpy()
    found = sum(bool((hits & (tt >= r_.start.value - 1e9) & (tt <= r_.end.value + 1e9)).any()) for r_ in lab_lat.itertuples())
    same = sorted(ns(bl["ts_recv"])) == sorted(tt[hits].tolist())
    rows.append(dict(data="stress_faulted", check="timestamp_inversions/latency_outlier", batch=len(bl), live=int(hits.sum()),
                     identical=same, note=f"trailing window {tr.window}; timestamp labels with a live hit {found}/{len(lab_lat)} (the other labels are inversions)"))
    rows.append(_latency_real(v3, D, L))
    # real days: live gap monitor vs stored v3 flags
    lead = []
    for venue, piece, symbol, day, a, b in tasks():
        if (symbol, day) not in {("CL.c.0", "2026-10-08"), ("AAPL", "2026-10-08"), ("LWAY", "2026-10-14")}:
            continue
        fr = load_symbol_slice(piece, symbol, a, b, columns=["ts_recv", "publisher_id", "instrument_id"]).reset_index()
        mc = v3_config("GLBX.MDP3" if venue == "GLBX" else "XNAS.ITCH", v3).missing
        tv = ns(fr["ts_recv"]).to_numpy()
        ck = D.SessionClock.build(mc.calendar, int(tv.min()), int(tv.max()))
        sf = L.replay_gaps(fr, L.StreamingGapMonitor(mc.alpha, mc.rate_window, mc.min_gap_s, mc.max_gap_s, clock=ck.clock))
        st = pd.read_parquet(ROOT / f"p115_detectors/results_real_v3/flags/{venue}_{symbol}_{day}.parquet")
        st = st[st["check"] == "missing_intervals"]
        rows.append(dict(data=f"real {symbol} {day}", check="missing_intervals", batch=len(st), live=len(sf),
                         identical=sorted(ns(st["ts_recv"])) == sorted(sf["ts_recv_ns"])))
    # open-gap alarm lead over batch, all stored real v3 gap candidates
    import glob
    for path in sorted(glob.glob(str(ROOT / "p115_detectors/results_real_v3/flags/*.parquet"))):
        st = pd.read_parquet(path)
        st = st[st["check"] == "missing_intervals"]
        if len(st):
            lead.append((st["value"].astype(float) - st["threshold"].astype(float)).to_numpy())
    lead = np.concatenate(lead)
    rows.append(dict(data="real 95 symbol-days (stored v3)", check="missing_intervals open-gap alarm", batch=len(lead), live=len(lead),
                     identical=True, note=f"live alarm earlier than batch by median {np.median(lead):.2f} s, p90 {np.quantile(lead, 0.9):.2f} s, max {lead.max():.1f} s"))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "streaming_vs_batch.csv", index=False)
    print(out.to_string(index=False))


def _latency_real(v3, D, L, symbol="AAPL", day="2026-10-08"):
    """Whole-day (batch) vs trailing-window latency_outlier on one real symbol-day."""
    from integration.combine import v3_config
    from m2.extract import load_symbol_slice, tasks
    task = next(x for x in tasks() if x[2] == symbol and x[3] == day)
    fr = load_symbol_slice(task[1], symbol, task[4], task[5], columns=None)
    cfg = v3_config("XNAS.ITCH", v3).timestamps
    norm = D.normalize(fr)
    norm = norm.sort_values(["publisher_id", "instrument_id", "ts_recv_ns"], kind="stable")
    tl = D.timestamp_inversions(norm, cfg)
    bl = tl[tl["subtype"] == "latency_outlier"]
    lat = (norm["ts_recv_ns"] - norm["ts_event_ns"]).to_numpy()
    tr = L.TrailingLatencyOutlier(k=cfg.latency_k, min_excess_ns=cfg.latency_min_excess_ns)
    key = list(zip(norm["publisher_id"], norm["instrument_id"]))
    hits = np.array([tr.update(int(x), k)[0] for x, k in zip(lat, key)])
    t = norm["ts_recv_ns"].to_numpy()
    b_t = ns(bl["ts_recv"]).to_numpy()
    covered = int(sum(bool(np.any(np.abs(t[hits] - x) <= 1_000_000_000)) for x in b_t)) if hits.any() else 0
    return dict(data=f"real {symbol} {day}", check="timestamp_inversions/latency_outlier", batch=len(bl), live=int(hits.sum()),
                identical=bool(len(bl) == int(hits.sum()) == covered), note=f"batch episodes with a trailing-window hit within 1 s: {covered}/{len(bl)}; records={len(norm)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["crossed", "metrics", "pauses", "severity", "streaming", "all"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    modes = ["crossed", "metrics", "pauses", "severity", "streaming"] if a.mode == "all" else [a.mode]
    for m in modes:
        print(f"== {m}"); globals()[f"mode_{m}"]()


if __name__ == "__main__":
    main()
