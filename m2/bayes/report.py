"""Tables and figures for docs/m2_bayesian_gaps.md from m2/results/bayes/*.csv."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m2.bayes.model import BayesGapConfig, outage_posterior

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "m2" / "results" / "bayes"
PLOTS = OUT / "plots"
V3 = ROOT / "p115_detectors" / "results_real_v3"


def default_name() -> str:
    from m2.bayes.run_bayes_gaps import cfg_name
    return cfg_name(asdict(BayesGapConfig()))


def md(df: pd.DataFrame, floatfmt="{:.3g}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else (f"{int(v)}" if v.is_integer() and abs(v) < 1e9 else floatfmt.format(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def recall_table(raw: pd.DataFrame) -> pd.DataFrame:
    inj = raw[raw["kind"].str.startswith("inject")]
    real = raw[raw["kind"] == "real"].groupby("method")["candidates"].sum().rename("real_candidates_12days")
    g = inj.groupby(["method", "kind"])[["injected", "detected", "unmatched"]].sum().unstack("kind")
    t = pd.DataFrame({
        "recall_10s": g[("detected", "inject10s")] / g[("injected", "inject10s")],
        "recall_20s": g[("detected", "inject20s")] / g[("injected", "inject20s")],
        "detected": g[("detected", "inject10s")] + g[("detected", "inject20s")],
        "injected": g[("injected", "inject10s")] + g[("injected", "inject20s")],
        "other_candidates_on_injected_streams": g[("unmatched", "inject10s")] + g[("unmatched", "inject20s")],
    }).join(real)
    return t


def summary():
    name = default_name()
    real = pd.read_csv(OUT / "real_symbol_days.csv")
    real = real[real["records"] > 0]
    lines = [f"Default Bayesian config: {name}", ""]
    chk = (real["poisson"] == real["v3_stored_flags"]).all()
    lines.append(f"Poisson re-implementation equals stored v3 flag count on all {len(real)} loaded symbol-days: {chk}")
    by = real.groupby(["venue", "symbol"])[["records", "poisson", "bayes", "both", "poisson_only", "bayes_only",
                                             "poisson_high", "bayes_high"]].sum().reset_index()
    tot = by.drop(columns=["venue", "symbol"]).sum()
    by.loc[len(by)] = ["all", "total"] + list(tot.values)
    by.to_csv(OUT / "by_symbol.csv", index=False)
    lines += ["", md(by)]
    dt = real.groupby("day_type")[["poisson", "bayes", "both"]].sum().reset_index()
    dt.to_csv(OUT / "by_day_type.csv", index=False)
    lines += ["", md(dt)]
    grid_cols = [c for c in real.columns if c.startswith("bayes:")]
    grid = pd.DataFrame({"config": [c[6:] for c in grid_cols], "real_candidates_95": [int(real[c].sum()) for c in grid_cols]})
    if (OUT / "inject_recall_raw.csv").exists():
        raw = pd.read_csv(OUT / "inject_recall_raw.csv")
        rt = recall_table(raw)
        rt.to_csv(OUT / "inject_recall_summary.csv")
        grid = grid.merge(rt.reset_index().rename(columns={"method": "config"}), on="config", how="left")
        pv = rt.loc["poisson_v3"]
        lines += ["", "Poisson v3 on injected:", pv.to_string()]
        # stored check
        st = raw[raw["method"] == "poisson_v3"].dropna(subset=["stored_v3_detected"])
        lines.append(f"Injected windows and detections equal stored sens_v3 P5: "
                     f"{(st['injected'] == st['stored_v3_injected']).all()} / {(st['detected'] == st['stored_v3_detected']).all()}")
        per = raw[(raw["method"].isin(["poisson_v3", name])) & raw["kind"].str.startswith("inject")]
        per = per.pivot_table(index=["symbol", "day"], columns=["method", "kind"], values="detected", aggfunc="sum")
        per.to_csv(OUT / "inject_recall_by_symbol_day.csv")
        lines += ["", per.to_string()]
    grid.to_csv(OUT / "grid.csv", index=False)
    lines += ["", grid.sort_values("real_candidates_95").to_string()]
    (OUT / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def boundary(lam, cfg: BayesGapConfig, gmax=4000.0):
    """Smallest gap with P(outage|g) >= threshold at steady state (a=1/(1-delta), b=a/lam)."""
    a = 1.0 / (1.0 - cfg.discount)
    g = np.geomspace(cfg.min_outage_s, gmax, 4000)
    out = []
    for l in lam:
        p = outage_posterior(g, a, a / l, cfg)
        i = np.flatnonzero(p >= cfg.posterior_threshold)
        out.append(g[i[0]] if len(i) else np.nan)
    return np.array(out)


def plots():
    PLOTS.mkdir(parents=True, exist_ok=True)
    name = default_name()
    cfg = BayesGapConfig()
    # 1. decision boundary vs rate
    lam = np.geomspace(0.02, 300, 300)
    pois = np.clip(-np.log(1e-6) / lam, 5, 60)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(lam, pois, color="#1f4e79", lw=2, label=r"v3 Poisson: $\min(\max(-\ln\alpha/\hat\lambda,5),60)$")
    ax.plot(lam, boundary(lam, cfg), color="#c62828", lw=2, label=f"Bayes default ({name})")
    for d in (0.98, 0.9):
        ax.plot(lam, boundary(lam, replace(cfg, discount=d)), lw=1.2, ls="--",
                label=f"Bayes, discount {d} (shape ~{1/(1-d):.0f})")
    real = pd.read_csv(OUT / "real_symbol_days.csv")
    real = real[real["records"] > 0]
    for sym, grp in real.groupby("symbol"):
        l = grp["median_lambda_post"].median()
        ax.axvline(l, color="#9aa3ad", lw=0.6)
        ax.text(l, 1.05, sym, rotation=90, fontsize=7, color="#555", va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"arrival rate $\lambda$ (records per in-session second)")
    ax.set_ylabel("smallest gap flagged (s)")
    ax.set_title("Where each rule starts flagging a gap", pad=40)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(PLOTS / "decision_boundary.png", dpi=130); plt.close(fig)

    # 2. candidates per symbol
    by = pd.read_csv(OUT / "by_symbol.csv")
    by = by[by["symbol"] != "total"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(by)); w = 0.38
    ax.bar(x - w / 2, by["poisson"], w, label="v3 Poisson rule", color="#1f4e79")
    ax.bar(x + w / 2, by["bayes"], w, label="Bayesian (default)", color="#c62828")
    ax.set_xticks(x); ax.set_xticklabels(by["symbol"], rotation=45, ha="right")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("missing-interval candidates (95 symbol-days)")
    ax.set_title("Candidates per symbol, all days")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(PLOTS / "candidates_by_symbol.png", dpi=130); plt.close(fig)

    # 3. recall vs candidates trade-off over the Bayesian grid
    grid = pd.read_csv(OUT / "grid.csv")
    rt = pd.read_csv(OUT / "inject_recall_summary.csv", index_col=0)
    sens = pd.read_csv(V3 / "sensitivity_missing.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    g = grid.dropna(subset=["recall_10s"])
    rec = (g["detected"] / g["injected"])
    sc = ax.scatter(g["real_candidates_12days"], rec, c=np.log10(1 / (1 - g["config"].str.extract(r"d([\d.]+)_")[0].astype(float))),
                    cmap="viridis", s=28)
    cb = fig.colorbar(sc, ax=ax); cb.set_label(r"$\log_{10}$ effective window $1/(1-\delta)$")
    pv = rt.loc["poisson_v3"]
    ax.scatter([pv["real_candidates_12days"]], [pv["detected"] / pv["injected"]], marker="*", s=220, color="#1f4e79", label="v3 Poisson (P5)")
    d = grid[grid["config"] == name].iloc[0]
    ax.scatter([d["real_candidates_12days"]], [d["detected"] / d["injected"]], marker="D", s=80, facecolor="none",
               edgecolor="#c62828", lw=2, label="Bayes default")
    od_flags = sens.loc["medium total", "OD5"] + sens.loc["high total", "OD5"]
    od_rec = (sens.loc["recall 10s", "OD5"] + sens.loc["recall 20s", "OD5"]) / 2
    ax.scatter([od_flags], [od_rec], marker="X", s=120, color="black", label="v3 overdispersed/Lomax (OD5, stored)")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("candidates on the 12 unmodified symbol-days")
    ax.set_ylabel("recall of injected 10 s + 20 s outages")
    ax.set_title("Bayesian grid: recall vs candidate volume")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(PLOTS / "recall_vs_candidates.png", dpi=130); plt.close(fig)

    # 4. LWAY / PLAB / ADTN gap scatter: gap vs posterior rate, coloured by rule
    gaps = pd.read_parquet(OUT / "real_gaps.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, sym in zip(axes, ["LWAY", "PLAB", "ADTN"]):
        x = gaps[gaps["symbol"] == sym]
        lam_post = x["a"] / x["b"]
        for m, col, lab in [((~x.poisson) & (~x.bayes), "#cfd4da", "neither"), (x.poisson & x.bayes, "#6a1b9a", "both"),
                            (x.poisson & ~x.bayes, "#1f4e79", "Poisson only"), (~x.poisson & x.bayes, "#c62828", "Bayes only")]:
            ax.scatter(lam_post[m], x["gap_s"][m], s=5, color=col, label=f"{lab} ({int(m.sum())})", alpha=0.7)
        ll = np.geomspace(max(lam_post.min(), 1e-3), lam_post.max(), 200)
        ax.plot(ll, np.clip(-np.log(1e-6) / ll, 5, 60), color="#1f4e79", lw=1)
        ax.plot(ll, boundary(ll, cfg), color="#c62828", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"{sym}, all 8 days (gaps > 2 s)")
        ax.set_xlabel(r"posterior mean rate $a/b$ (1/s)")
        ax.legend(fontsize=7, markerscale=3)
    axes[0].set_ylabel("gap (in-session s)")
    fig.tight_layout(); fig.savefig(PLOTS / "quiet_names_gaps.png", dpi=130); plt.close(fig)
    print("plots written to", PLOTS)


def main(mode="summary"):
    if mode == "summary":
        summary()
    else:
        summary(); plots()
