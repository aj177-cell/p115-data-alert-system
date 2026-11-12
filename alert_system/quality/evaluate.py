"""Evaluate the detectors on synthetic data with injected, labeled faults.

Event matching: a flag of check C is a true positive if it falls (by ts_recv,
or ts_event when ts_recv is missing) within ``[start - tol, end + tol]`` of a
label of check C for the same publisher (and instrument, unless the label is
publisher-level). A flag that matches only a *different* check's label is
counted as ``collateral`` (a real injected defect seen by another detector,
e.g. an injected cross also yields a zero/negative spread) and excluded from
precision; anything else is a false positive. Recall is label-level.
Severity 'info' flags (masked / vendor-labelled) are ignored.

Run: ``python evaluate.py`` -> results/metrics.csv, results/sensitivity.csv,
results/sensitivity.png.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DetectorConfig
from .detectors import CHECKS, normalize, run_all
from .synthetic import generate_mbp1, inject_faults, to_databento_df

OUT = Path(__file__).parent / "results"


def match(flags: pd.DataFrame, labels: pd.DataFrame, tolerance_s: float = 1.0) -> pd.DataFrame:
    """Per-check precision, recall, F1 with event-matching tolerance."""
    f = flags[(flags["severity"] != "info") & flags["check"].isin(CHECKS)].reset_index(drop=True)
    ts = pd.to_datetime(f["ts_recv"], utc=True).fillna(pd.to_datetime(f["ts_event"], utc=True))
    t = ts.dt.as_unit("ns").astype("int64").to_numpy()
    tol = int(tolerance_s * 1e9)
    own = np.zeros(len(f), bool)
    other = np.zeros(len(f), bool)
    detected = np.zeros(len(labels), bool)
    for j, lab in labels.iterrows():
        hit = ((f["publisher_id"].to_numpy() == lab["publisher_id"])
               & (t >= lab["start"].value - tol) & (t <= lab["end"].value + tol))
        if lab["instrument_id"] is not None and not pd.isna(lab["instrument_id"]):
            hit &= f["instrument_id"].to_numpy() == lab["instrument_id"]
        same = hit & (f["check"].to_numpy() == lab["check"])
        own |= same
        other |= hit & ~same
        detected[j] = same.any()
    rows = []
    for check in CHECKS:
        sel = f["check"].to_numpy() == check
        tp = int((own & sel).sum()); coll = int((~own & other & sel).sum()); fp = int((~own & ~other & sel).sum())
        lab_sel = labels["check"].to_numpy() == check
        n_lab = int(lab_sel.sum()); n_det = int(detected[lab_sel].sum())
        p = tp / (tp + fp) if tp + fp else np.nan
        r = n_det / n_lab if n_lab else np.nan
        f1 = 2 * p * r / (p + r) if p and r and p + r else (0.0 if n_lab else np.nan)
        rows.append(dict(check=check, labels=n_lab, detected=n_det, flags=int(sel.sum()), tp_flags=tp,
                         fp_flags=fp, collateral_flags=coll, precision=p, recall=r, f1=f1))
    return pd.DataFrame(rows)


SWEEPS: dict[str, tuple[str, str, list[float]]] = {
    "crossed_books": ("crossed", "min_duration_s", [0.0, 0.001, 0.01, 0.1, 1.0, 2.5, 5.0]),
    "extreme_prices": ("extreme", "k", [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 50.0]),
    "timestamp_inversions": ("timestamps", "latency_k", [1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 200.0]),
    "missing_intervals": ("missing", "alpha", [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-12, 1e-40]),
    "stale_quotes": ("stale", "min_age_s", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]),
    "unusual_spreads": ("spreads", "multiplier", [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]),
    "sequence_discontinuities": ("sequence", "min_gap", [1, 2, 5, 10, 20, 50]),
}


def base_config() -> DetectorConfig:
    cfg = DetectorConfig()
    cfg.stale.reference = {"symbol": "REF"}
    return cfg


def sweep(norm: pd.DataFrame, labels: pd.DataFrame, tolerance_s: float = 1.0) -> pd.DataFrame:
    rows = []
    for check, (section, param, grid) in SWEEPS.items():
        for v in grid:
            cfg = base_config()
            setattr(getattr(cfg, section), param, type(getattr(getattr(cfg, section), param))(v))
            flags, _ = run_all(norm, cfg, checks=[check])
            m = match(flags, labels, tolerance_s)
            rows.append(dict(check=check, parameter=f"{section}.{param}", value=v,
                             **m[m["check"] == check].iloc[0][["precision", "recall", "f1", "flags"]].to_dict()))
    return pd.DataFrame(rows)


def plot_sweep(res: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (check, grp) in zip(axes.flat, res.groupby("check", sort=False)):
        x = np.arange(len(grp))
        for metric, style in (("precision", "o-"), ("recall", "s--"), ("f1", "^-")):
            ax.plot(x, grp[metric].astype(float), style, label=metric)
        ax.set_xticks(x, [f"{v:g}" for v in grp["value"]], rotation=45, fontsize=8)
        ax.set_title(check, fontsize=10); ax.set_xlabel(grp["parameter"].iloc[0], fontsize=8)
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Sensitivity of P/R/F1 to each check's main threshold (synthetic, injected faults)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(seed: int = 7, tolerance_s: float = 1.0) -> pd.DataFrame:
    OUT.mkdir(exist_ok=True)
    raw = generate_mbp1(seed=seed)
    faulted, labels = inject_faults(raw, seed=seed + 4)
    norm = normalize(to_databento_df(faulted))
    flags, summary = run_all(norm, base_config())
    metrics = match(flags, labels, tolerance_s)
    clean_flags, _ = run_all(to_databento_df(raw), base_config())
    metrics["clean_stream_flags"] = metrics["check"].map(
        clean_flags[clean_flags["severity"] != "info"].groupby("check").size()).fillna(0).astype(int)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    labels.to_csv(OUT / "labels.csv", index=False)
    flags.to_csv(OUT / "flags.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    res = sweep(norm, labels, tolerance_s)
    res.to_csv(OUT / "sensitivity.csv", index=False)
    plot_sweep(res, OUT / "sensitivity.png")
    pd.set_option("display.width", 200)
    print(metrics.round(3).to_string(index=False))
    return metrics


if __name__ == "__main__":
    main()
