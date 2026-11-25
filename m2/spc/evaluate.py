"""Phase I baselines, ARL calibration, signals on the 95 symbol-days, overlap with point checks."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from m2.extract import NORMAL_WEEK
from m2.spc.baseline import DIRECTION, ar1_phi, phase1, prewhiten, standardise
from m2.spc.charts import ChartParams, calibrate, run_charts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "m2" / "results" / "spc"
SERIES = OUT / "series"
KINDS = ("spread", "rate")
CHARTS = ("shewhart", "cusum", "ewma")
ARL0 = 370.0
NEAR = 5  # buckets (minutes) either side for overlap


def load_all() -> dict:
    return {(f.stem.split("_")[1], "_".join(f.stem.split("_")[2:])): pd.read_parquet(f)
            for f in sorted(SERIES.glob("*.parquet"))}


def phase1_history(allseries: dict, symbol: str, day: str) -> list[pd.DataFrame]:
    """In-control history: the symbol's normal-week sessions, excluding the charted day."""
    return [s for (sym, d), s in allseries.items() if sym == symbol and d in NORMAL_WEEK and d != day and len(s)]


def zscores(series: pd.DataFrame, history: list[pd.DataFrame], whiten: bool = True, info: dict | None = None) -> dict:
    """Standardised (and AR(1)-prewhitened) series per kind against the Phase I history."""
    out = {}
    for k in KINDS:
        base = phase1(history, k)
        z = standardise(series, base, k)
        if whiten and base is not None:
            phi = ar1_phi([standardise(h, base, k) for h in history])
            z = prewhiten(z, phi)
            if info is not None:
                info[k] = phi
        out[k] = z
    return out


def signals(z: dict, params: dict) -> dict:
    """{(kind, chart): bool array} using per-kind ChartParams."""
    out = {}
    for k in KINDS:
        for c, sig in run_charts(z[k], params[k], DIRECTION[k]).items():
            out[(k, c)] = sig
    return out


def empirical_calibration(zs: list[dict], nominal: ChartParams, target=ARL0) -> dict:
    """Per kind, limits (k and lambda fixed) giving pooled ARL0 = target on the Phase I (normal-week) z."""
    from m2.spc.charts import cusum, ewma, shewhart
    res = {}
    for k in KINDS:
        series = [z[k] for z in zs]
        n = sum(int(np.isfinite(s).sum()) for s in series)
        d = DIRECTION[k]
        def arl(fn):
            cnt = sum(int(fn(s).sum()) for s in series)
            return n / max(cnt, 0.5)
        c = brentq(lambda x: arl(lambda s: shewhart(s, x, d)) - target, 1.0, 60.0, xtol=1e-3)
        h = brentq(lambda x: arl(lambda s: cusum(s, nominal.cusum_k, x, d)) - target, 1.0, 400.0, xtol=1e-3)
        L = brentq(lambda x: arl(lambda s: ewma(s, nominal.ewma_lambda, x, d)) - target, 1.0, 60.0, xtol=1e-3)
        res[k] = ChartParams(c, nominal.cusum_k, h, nominal.ewma_lambda, L)
    return res


def params_sets(allseries: dict):
    nominal = calibrate(ARL0)
    path = OUT / "params.json"
    if path.exists():
        raw = json.loads(path.read_text())
        return {"nominal": {k: ChartParams(**raw["nominal"][k]) for k in KINDS},
                "phase1": {k: ChartParams(**raw["phase1"][k]) for k in KINDS}}
    zs = [zscores(s, phase1_history(allseries, sym, d)) for (sym, d), s in allseries.items() if d in NORMAL_WEEK]
    emp = empirical_calibration(zs, nominal)
    sets = {"nominal": {k: nominal for k in KINDS}, "phase1": emp}
    path.write_text(json.dumps({n: {k: asdict(p) for k, p in v.items()} for n, v in sets.items()}, indent=2))
    return sets


def near(mask_a: np.ndarray, mask_b: np.ndarray, w=NEAR) -> np.ndarray:
    """For each True in mask_a: any True of mask_b within +-w positions."""
    b = np.convolve(mask_b.astype(int), np.ones(2 * w + 1, int), mode="same") > 0
    return mask_a & b


def run_charts_real():
    allseries = load_all()
    sets = params_sets(allseries)
    rows, sig_rows = [], []
    for (sym, day), s in allseries.items():
        if not len(s):
            rows.append(dict(symbol=sym, day=day, buckets=0)); continue
        z = zscores(s, phase1_history(allseries, sym, day))
        n = len(s)
        pos = s["bucket"].to_numpy()
        for pname, params in sets.items():
            sig = signals(z, params)
            for (k, c), m in sig.items():
                row = dict(symbol=sym, day=day, venue=s["venue"].iloc[0], day_type=s["day_type"].iloc[0], buckets=n,
                           params=pname, kind=k, chart=c, signals=int(m.sum()),
                           first30=int(m[pos < 30].sum()), last30=int(m[pos >= pos.max() - 29].sum()))
                if k == "spread":
                    pc = s["cand_spread"].to_numpy() > 0
                    row.update(point_buckets=int(pc.sum()), sig_near_point=int(near(m, pc).sum()),
                               point_near_sig=int(near(pc, m).sum()))
                else:
                    for nm in ("cand_gap_poisson", "cand_gap_bayes"):
                        pc = s[nm].to_numpy() > 0
                        tag = nm.split("_")[-1]
                        row.update({f"point_buckets_{tag}": int(pc.sum()), f"sig_near_{tag}": int(near(m, pc).sum()),
                                    f"{tag}_near_sig": int(near(pc, m).sum())})
                rows.append(row)
                if pname == "phase1":
                    for i in np.flatnonzero(m):
                        sig_rows.append(dict(symbol=sym, day=day, kind=k, chart=c, bucket=int(pos[i]), z=float(z[k][i])))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "real_signals_by_symbol_day.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(OUT / "real_signal_list_phase1.csv", index=False)
    print(json.dumps({n: {k: {kk: round(vv, 3) for kk, vv in asdict(p).items()} for k, p in v.items()} for n, v in sets.items()}, indent=1))
