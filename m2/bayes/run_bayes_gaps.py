"""Run the Bayesian missing-interval check on the 95 symbol-day pull and on injected outages.

    python -m m2.bayes.run_bayes_gaps --mode real   # 95 symbol-days: counts vs the v3 Poisson rule
    python -m m2.bayes.run_bayes_gaps --mode inject # the 12 sens_v3 symbol-days, 10 s / 20 s outages
    python -m m2.bayes.run_bayes_gaps --mode plots  # figures from the saved CSVs

One symbol-day in memory at a time, three columns only. Writes m2/results/bayes/ only.
The v3 Poisson rule is re-implemented in model.poisson_rule_from_clock and checked
against p115_detectors/results_real_v3 (flag counts and injected-outage recall).
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from integration.combine import load_v3
from m2.bayes.model import BayesGapConfig, posterior_from_clock, poisson_rule_from_clock
from m2.extract import DEFAULT_DATA_ROOT, day_type_of, load_symbol_slice, tasks

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "m2" / "results" / "bayes"
V3 = ROOT / "p115_detectors" / "results_real_v3"
COLS = ["ts_recv", "publisher_id", "instrument_id"]

# Bayesian-side grid (only the Bayesian parameters move; the Poisson rule is fixed at v3).
GRID = [dict(discount=d, outage_prior=p, min_outage_s=m)
        for d in (0.9, 0.98, 0.99, 0.998, 0.999)
        for p in (1e-4, 1e-6, 1e-8)
        for m in (2.0, 5.0, 10.0)]


def _p(stem: Path, ext: str) -> Path:
    """stem + extension. Path.with_suffix would treat '.0_2026-10-05' in 'GLBX_ES.c.0_2026-10-05' as a
    suffix and map every futures day to the same file (the bug fixed on 2026-11-22)."""
    return stem.parent / (stem.name + ext)


def cfg_name(c: dict) -> str:
    return f"d{c['discount']}_pi{c['outage_prior']:.0e}_dmin{c['min_outage_s']:g}"


def calendar(v3, venue):
    return v3.config.equities_config().missing.calendar if venue == "XNAS" else v3.config.cme_calendar()


def prior_rates() -> dict:
    """Pooled F_LAST rates from the distribution-fit summaries, leave-one-day-out per symbol."""
    s = pd.read_csv(ROOT / "m2" / "results" / "summaries.csv")
    s = s[(s["day"] != "pooled") & s["gaps.lambda_hat"].notna()]
    s = s[s["regime"].isin(["equity_rth", "cme_overnight"])]
    return {(r.symbol, r.day): float(r["gaps.lambda_hat"]) for _, r in s.iterrows()}


def prior_for(rates: dict, symbol: str, day: str) -> float | None:
    vals = [v for (sym, d), v in rates.items() if sym == symbol and d != day]
    return float(np.median(vals)) if vals else None


def load_clock(v3, venue, piece, symbol, t0, t1, data_root=DEFAULT_DATA_ROOT):
    frame = load_symbol_slice(piece, symbol, t0, t1, data_root=data_root, columns=COLS)
    if frame.empty:
        return None
    if "ts_recv" not in frame.columns:
        frame = frame.reset_index()
    t = frame["ts_recv"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    pub = frame["publisher_id"].to_numpy().astype(np.int64)
    iid = frame["instrument_id"].to_numpy().astype(np.int64)
    del frame
    ok = t != np.iinfo(np.int64).min
    t, pub, iid = t[ok], pub[ok], iid[ok]
    order = np.lexsort((t, iid, pub))  # KEY then ts_recv (stable like the detector)
    t, pub, iid = t[order], pub[order], iid[order]
    key = pd.DataFrame({"p": pub, "i": iid}).groupby(["p", "i"], sort=False).ngroup().to_numpy()
    return t, key


def session_clock(v3, venue, t):
    cal = calendar(v3, venue)
    clk = v3.detectors.SessionClock.build(cal, int(t.min()), int(t.max()))
    return clk


class _Lite:
    """Only the candidate mask of a non-default grid config (keeps memory flat on 4M+ record days)."""
    def __init__(self, candidate):
        self.candidate = candidate


def evaluate(s, key, prior, cfgs, keep_full=None):
    pois = poisson_rule_from_clock(s, key)
    out = {}
    base = BayesGapConfig(prior_rate=prior)
    for c in cfgs:
        post = posterior_from_clock(s, key, replace(base, **c))
        name = cfg_name(c)
        out[name] = post if name == keep_full else _Lite(post.candidate)
        del post
    return pois, out


# ---------------------------------------------------------------------------
def run_real(data_root, only=None):
    v3 = load_v3()
    rates = prior_rates()
    default = asdict(BayesGapConfig())
    cfgs = GRID + ([] if any(cfg_name(c) == cfg_name(default) for c in GRID) else [default])
    dname = cfg_name(default)
    v3_idx = pd.read_csv(V3 / "run_index.csv")
    rows, gap_rows = [], []
    cache = OUT / "cache"; cache.mkdir(parents=True, exist_ok=True)
    for venue, piece, symbol, day, t0, t1 in tasks():
        if only and symbol not in only:
            continue
        stem = cache / f"{venue}_{symbol}_{day}"  # names contain dots (ES.c.0): build paths with _p(), never with_suffix
        if _p(stem, ".json").exists():
            rows.append(json.loads(_p(stem, ".json").read_text()))
            if _p(stem, ".parquet").exists():
                gap_rows.append(pd.read_parquet(_p(stem, ".parquet")))
            continue
        t_ = time.time()
        loaded = load_clock(v3, venue, piece, symbol, t0, t1, data_root)
        stored = v3_idx.query("venue==@venue and day==@day and symbol==@symbol")
        row = dict(venue=venue, symbol=symbol, day=day, day_type=day_type_of(day),
                   v3_stored_flags=int(stored["flags"].iloc[0]) if len(stored) else np.nan)
        if loaded is None:
            row["records"] = 0
            rows.append(row); _p(stem, ".json").write_text(json.dumps(row)); continue
        t, key = loaded
        s = session_clock(v3, venue, t).clock(t)
        prior = prior_for(rates, symbol, day)
        pois, posts = evaluate(s, key, prior, cfgs, keep_full=dname)
        gap = posts[dname].gap_s
        row.update(records=len(t), prior_rate=prior, poisson=int(pois.sum()),
                   poisson_high=int((pois & (gap > 60)).sum()))
        for name, post in posts.items():
            c = post.candidate
            row[f"bayes:{name}"] = int(c.sum())
            if name == dname:
                row.update(bayes=int(c.sum()), bayes_high=int((c & (gap > 60)).sum()),
                           both=int((c & pois).sum()), poisson_only=int((pois & ~c).sum()),
                           bayes_only=int((c & ~pois).sum()),
                           median_shape=float(np.nanmedian(post.a)),
                           median_lambda_post=float(np.nanmedian(post.a / post.b)))
        post = posts[dname]
        m = pois | post.candidate | (post.valid & (gap > 2.0))
        idx = np.flatnonzero(m)
        gap_rows.append(pd.DataFrame(dict(
            venue=venue, symbol=symbol, day=day, ts_recv_ns=t[idx], gap_s=gap[idx], a=post.a[idx], b=post.b[idx],
            tail=post.tail[idx], p_outage=post.p_outage[idx], bayes=post.candidate[idx], poisson=pois[idx])))
        row["secs"] = round(time.time() - t_, 1)
        rows.append(row)
        gap_rows[-1].to_parquet(_p(stem, ".parquet"), index=False)
        _p(stem, ".json").write_text(json.dumps({k: (None if isinstance(v, float) and np.isnan(v) else v)
                                                         for k, v in row.items()}, default=float))
        print({k: row[k] for k in ("symbol", "day", "records", "v3_stored_flags", "poisson", "bayes", "both", "secs")}, flush=True)
        del t, key, s, posts, pois
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if not only else "_" + "_".join(only)
    pd.DataFrame(rows).to_csv(OUT / f"real_symbol_days{tag}.csv", index=False)
    if gap_rows:
        pd.concat(gap_rows, ignore_index=True).to_parquet(OUT / f"real_gaps{tag}.parquet", index=False)


# ---------------------------------------------------------------------------
# Injection and recall: same procedure and seeds as p115_detectors/sens_v3.py (copied, not imported,
# because importing analyze.py creates folders relative to the working directory).
SENS_DAYS = [("XNAS", "2026-10-08", "LWAY"), ("XNAS", "2026-10-14", "PLAB"), ("XNAS", "2026-10-07", "ADTN"),
             ("XNAS", "2026-10-08", "NBIS"), ("XNAS", "2026-10-14", "MSFT"), ("XNAS", "2026-10-14", "CRWV"),
             ("XNAS", "2026-10-14", "AAPL"), ("GLBX", "2026-10-08", "CL.c.0"), ("XNAS", "2026-09-16", "NVDA"),
             ("XNAS", "2026-10-14", "SPY"), ("XNAS", "2025-11-28", "SPY"), ("GLBX", "2026-10-14", "ES.c.0")]


def inject(t_sorted_by_key, clk, rng, L, n=10):
    """sens_v3.inject on the timestamp array; returns keep-mask and windows."""
    tr = t_sorted_by_key
    t = np.sort(tr)
    ins = clk.in_session(t) & clk.in_session(t + L * 1_000_000_000)
    cand = np.flatnonzero(ins & (t > t[0] + 120e9) & (t < t[-1] - 120e9))
    starts = np.sort(rng.choice(t[cand], n, replace=False)); wins = []
    keep = np.ones(len(tr), bool)
    for s in starts:
        if wins and s < wins[-1][1] + 60e9:
            continue
        e = s + int(L * 1e9); keep &= ~((tr >= s) & (tr < e)); wins.append((s, e))
    return keep, wins


def matched(ft, v, wins, L):
    """Per window: any flag in [end, end+60 s] with gap >= 0.9 L (sens_v3.recall)."""
    hit = np.zeros(len(wins), bool); used = np.zeros(len(ft), bool)
    for j, (s, e) in enumerate(wins):
        m = (ft >= e) & (ft <= e + 60e9) & (v >= 0.9 * L)
        hit[j] = m.any(); used |= m
    return hit, used


def run_inject(data_root):
    v3 = load_v3()
    rates = prior_rates()
    default = asdict(BayesGapConfig())
    cfgs = GRID + ([] if any(cfg_name(c) == cfg_name(default) for c in GRID) else [default])
    T = {(x[0], x[3], x[2]): x for x in tasks()}
    stored = pd.read_csv(V3 / "sensitivity_missing_raw.csv").query("option=='P5'")
    rows = []
    for k in SENS_DAYS:
        venue, piece, symbol, day, t0, t1 = T[k]
        t, key = load_clock(v3, venue, piece, symbol, t0, t1, data_root)
        clk = session_clock(v3, venue, t)
        prior = prior_for(rates, symbol, day)
        # real stream (no injection)
        pois, posts = evaluate(clk.clock(t), key, prior, cfgs)
        base = dict(venue=venue, day=day, symbol=symbol, records=len(t))
        rows.append({**base, "kind": "real", "method": "poisson_v3", "candidates": int(pois.sum())})
        for name, post in posts.items():
            rows.append({**base, "kind": "real", "method": name, "candidates": int(post.candidate.sum())})
        for L in (10, 20):
            rng = np.random.default_rng(zlib.crc32(f"{symbol}|{day}|{L}".encode()))
            keep, wins = inject(t, clk, rng, L)
            ti, ki = t[keep], key[keep]
            si = clk.clock(ti)
            gap = np.r_[0.0, np.diff(si).astype("float64")] / 1e9
            pois, posts = evaluate(si, ki, prior, cfgs)
            st = stored[(stored["symbol"] == symbol) & (stored["day"] == day) & (stored["kind"] == f"inject{L}s")]
            for name, mask in [("poisson_v3", pois)] + [(n_, p_.candidate) for n_, p_ in posts.items()]:
                hit, used = matched(ti[mask], gap[mask], wins, L)
                rows.append({**base, "kind": f"inject{L}s", "method": name, "injected": len(wins),
                             "detected": int(hit.sum()), "candidates": int(mask.sum()),
                             "unmatched": int((~used).sum()),
                             "stored_v3_injected": int(st["injected"].iloc[0]) if len(st) else np.nan,
                             "stored_v3_detected": int(st["detected"].iloc[0]) if len(st) else np.nan})
        print(k, flush=True)
        del t, key
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "inject_recall_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "inject", "plots", "summary"], required=True)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.mode == "real":
        run_real(a.data_root, a.only)
    elif a.mode == "inject":
        run_inject(a.data_root)
    else:
        from m2.bayes import report
        report.main(a.mode)


if __name__ == "__main__":
    main()
