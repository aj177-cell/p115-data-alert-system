"""Tables and figures for docs/m2_control_charts.md from m2/results/spc/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m2.spc.baseline import DIRECTION, phase1, transform
from m2.spc.charts import ChartParams, cusum
from m2.spc.evaluate import KINDS, load_all, params_sets, phase1_history, zscores

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "m2" / "results" / "spc"
PLOTS = OUT / "plots"
ORDER = ["normal_week", "cpi", "vix_high", "half_day", "around_degraded"]


def episodes(sig: pd.DataFrame, gap=15) -> int:
    n = 0
    for _, g in sig.groupby(["symbol", "day", "kind", "chart"]):
        b = np.sort(g["bucket"].to_numpy())
        n += 1 + int((np.diff(b) > gap).sum())
    return n


def tables():
    lines = []
    r = pd.read_csv(OUT / "real_signals_by_symbol_day.csv")
    r = r[r["buckets"] > 0]
    allseries = load_all()
    # day-type rates
    dt = (r.groupby(["params", "kind", "chart", "day_type"]).agg(signals=("signals", "sum"), buckets=("buckets", "sum"),
                                                               symbol_days=("day", "size")).reset_index())
    dt["per_1000_buckets"] = 1000 * dt["signals"] / dt["buckets"]
    nw = dt[dt["day_type"] == "normal_week"].set_index(["params", "kind", "chart"])["per_1000_buckets"]
    dt["ratio_to_normal_week"] = dt.apply(lambda x: x["per_1000_buckets"] / nw.loc[(x["params"], x["kind"], x["chart"])], axis=1)
    dt.to_csv(OUT / "by_day_type.csv", index=False)
    lines += ["## signals per 1000 buckets by day type (phase1 params)",
              dt[dt["params"] == "phase1"].pivot_table(index="day_type", columns=["kind", "chart"], values="per_1000_buckets").reindex(ORDER).round(2).to_string()]
    lines += ["## same, nominal params",
              dt[dt["params"] == "nominal"].pivot_table(index="day_type", columns=["kind", "chart"], values="per_1000_buckets").reindex(ORDER).round(2).to_string()]
    # median z (not whitened) by day type and symbol
    rows = []
    for (sym, day), s in allseries.items():
        if not len(s):
            continue
        z = zscores(s, phase1_history(allseries, sym, day), whiten=False)
        rows.append(dict(symbol=sym, day=day, day_type=s["day_type"].iloc[0],
                         median_z_rate=float(np.nanmedian(z["rate"])), median_z_spread=float(np.nanmedian(z["spread"]))))
    mz = pd.DataFrame(rows)
    mz.to_csv(OUT / "median_z_by_symbol_day.csv", index=False)
    lines += ["## median z by day type (unwhitened)", mz.groupby("day_type")[["median_z_rate", "median_z_spread"]].median().reindex(ORDER).round(2).to_string()]
    # per symbol x day type cusum counts
    cu = r[(r["params"] == "phase1") & (r["chart"] == "cusum")]
    piv = cu.pivot_table(index="symbol", columns=["kind", "day_type"], values="signals", aggfunc="sum")
    piv.to_csv(OUT / "cusum_by_symbol_day_type.csv")
    lines += ["## CUSUM signals per symbol and day type", piv.to_string()]
    # overlap
    ph = r[r["params"] == "phase1"]
    sp = ph[ph["kind"] == "spread"].groupby("chart")[["signals", "point_buckets", "sig_near_point", "point_near_sig"]].sum()
    rt = ph[ph["kind"] == "rate"].groupby("chart")[["signals", "point_buckets_poisson", "sig_near_poisson", "poisson_near_sig",
                                                     "point_buckets_bayes", "sig_near_bayes", "bayes_near_sig"]].sum()
    sp.to_csv(OUT / "overlap_spread.csv"); rt.to_csv(OUT / "overlap_rate.csv")
    lines += ["## overlap spread", sp.to_string(), "## overlap rate", rt.to_string()]
    eq = ph[(ph["venue"] == "XNAS") & (ph["day_type"] != "half_day")]
    oc = eq.groupby(["kind", "chart"])[["first30", "last30", "signals"]].sum()
    oc["share_open_close"] = (oc["first30"] + oc["last30"]) / oc["signals"]
    oc.to_csv(OUT / "open_close_share.csv")
    lines += ["## equities open/close share (uniform = 60/390 = 0.154)", oc.round(3).to_string()]
    sig = pd.read_csv(OUT / "real_signal_list_phase1.csv")
    ep = pd.Series({key: episodes(g) for key, g in sig.groupby(["kind", "chart"])})
    lines += ["## signals vs episodes (gap > 15 buckets)", pd.DataFrame({"signals": sig.groupby(["kind", "chart"]).size(), "episodes": ep}).to_string()]
    # injection
    inj = pd.read_csv(OUT / "inject_raw.csv")
    f = inj[(inj["fault"] != "clean") & inj["params"].isin(["phase1", "v3"])]
    t = f.groupby(["fault", "method"]).agg(windows=("windows", "sum"), detected=("detected", "sum"),
                                           median_delay_min=("median_delay_min", "median"),
                                           new_false_alarms=("false_alarms", "sum")).reset_index()
    t.to_csv(OUT / "inject_summary.csv", index=False)
    lines += ["## injection summary (phase1 params / v3 point checks)", t.to_string()]
    tn = inj[(inj["fault"] != "clean") & (inj["params"] == "nominal")].groupby(["fault", "method"])[["detected", "false_alarms"]].sum()
    lines += ["## injection, nominal params", tn.to_string()]
    c = inj[inj["fault"] == "clean"].groupby(["method", "params"])["false_alarms"].sum()
    c.to_csv(OUT / "inject_clean_day_counts.csv")
    lines += ["## clean-day counts (9 symbol-days, 2026-10-09)", c.to_string()]
    per = f.pivot_table(index="symbol", columns=["fault", "method"], values="detected")
    per.to_csv(OUT / "inject_by_symbol.csv")
    (OUT / "summary.txt").write_text("\n\n".join(lines))
    print("\n\n".join(lines))


def plots():
    PLOTS.mkdir(parents=True, exist_ok=True)
    allseries = load_all()
    sets = params_sets(allseries)
    p1 = sets["phase1"]

    # 1. time-of-day baseline (AAPL rate and spread) with one Phase I-excluded day
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, kind in zip(axes, KINDS):
        day = "2026-10-14"
        s = allseries[("AAPL", day)]
        base = phase1(phase1_history(allseries, "AAPL", day), kind)
        ax.fill_between(base.index, base["m"] - 3 * base["s"], base["m"] + 3 * base["s"], color="#cfd8e3", label=r"$m_b \pm 3 s_b$ (Phase I)")
        ax.plot(base.index, base["m"], color="#1f4e79", lw=1.5, label=r"$m_b$")
        ax.plot(s["bucket"], transform(s, kind), color="#c62828", lw=0.8, label="CPI day 2026-10-14")
        ax.set_xlabel("minute since 09:30 ET")
        ax.set_ylabel(r"$2\sqrt{N+3/8}$" if kind == "rate" else r"$\ln S$ (ticks)")
        ax.set_title(f"AAPL {kind}: time-of-day baseline")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "tod_baseline_aapl.png", dpi=130); plt.close(fig)

    # 2. ES around the degraded weekend vs a normal day: rate z and CUSUM
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    for ax, day in zip(axes, ["2026-10-07", "2026-08-28", "2026-08-31"]):
        s = allseries[("ES.c.0", day)]
        z = zscores(s, phase1_history(allseries, "ES.c.0", day))["rate"]
        sig, stat = cusum(z, p1["rate"].cusum_k, p1["rate"].cusum_h, "down", return_stat=True)
        ax.plot(s["bucket"] / 60, -z, color="#9aa3ad", lw=0.6, label=r"$-e_t$ (rate, whitened)")
        ax.plot(s["bucket"] / 60, stat, color="#1f4e79", lw=1.2, label="lower CUSUM")
        ax.axhline(p1["rate"].cusum_h, color="#c62828", ls="--", lw=1, label=f"h = {p1['rate'].cusum_h:.2f}")
        ax.scatter(s["bucket"][sig] / 60, np.full(sig.sum(), p1["rate"].cusum_h), color="#c62828", s=14, zorder=3)
        ax.set_title(f"ES.c.0 {day}: {int(sig.sum())} rate-drop signals")
        ax.set_ylim(-6, max(12, np.nanmax(stat) + 1))
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("hours since 17:00 CT (Globex open)")
    fig.tight_layout(); fig.savefig(PLOTS / "es_rate_cusum.png", dpi=130); plt.close(fig)

    # 3. injection detection
    t = pd.read_csv(OUT / "inject_summary.csv")
    faults = ["widen_x1.5", "widen_x2", "widen_x3", "ramp_1to3", "thin_30", "thin_50", "thin_70"]
    methods = [("spread_rule", "v3 spread rule", "#555555"), ("poisson_gap", "Poisson gap", "#1f4e79"),
               ("bayes_gap", "Bayesian gap", "#6a1b9a"), ("cusum", "SPC CUSUM", "#c62828"), ("ewma", "SPC EWMA", "#ef6c00"),
               ("shewhart", "SPC Shewhart", "#f2b134")]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.14
    for j, (m, lab, col) in enumerate(methods):
        vals = []
        for f in faults:
            kind = "spread" if not f.startswith("thin") else "rate"
            name = m if m in ("spread_rule", "poisson_gap", "bayes_gap") else f"spc_{kind}_{m}"
            x = t[(t["fault"] == f) & (t["method"] == name)]
            vals.append(x["detected"].iloc[0] / x["windows"].iloc[0] if len(x) else np.nan)
        ax.bar(np.arange(len(faults)) + (j - 2.5) * w, vals, w, label=lab, color=col)
    ax.set_xticks(np.arange(len(faults))); ax.set_xticklabels(faults)
    ax.set_ylabel("share of 27 injected windows detected")
    ax.set_title("Injected faults on 9 symbols, 2026-10-09 (blank = method not applicable)")
    ax.legend(fontsize=8, ncol=3, loc="upper left"); ax.set_ylim(0, 1.15)
    fig.tight_layout(); fig.savefig(PLOTS / "inject_detection.png", dpi=130); plt.close(fig)

    # 4. delays
    inj = pd.read_csv(OUT / "inject_raw.csv")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    data, labels = [], []
    for f in faults:
        kind = "spread" if not f.startswith("thin") else "rate"
        for name, lab in ([("spc_%s_cusum" % kind, "CUSUM"), ("spc_%s_ewma" % kind, "EWMA")] +
                          ([("poisson_gap", "Poisson"), ("bayes_gap", "Bayes")] if kind == "rate" else [])):
            x = inj[(inj["fault"] == f) & (inj["method"] == name) & inj["params"].isin(["phase1", "v3"])]["delays"].dropna()
            d = [float(v) for s in x.astype(str) for v in s.split(";") if v and v != "nan"]
            data.append(d if d else [np.nan]); labels.append(f"{f}\n{lab}")
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_ylabel("detection delay (minutes from fault start)")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title("Detection delay, detected windows only")
    fig.tight_layout(); fig.savefig(PLOTS / "inject_delays.png", dpi=130); plt.close(fig)
    print("plots written to", PLOTS)


def main():
    tables(); plots()
