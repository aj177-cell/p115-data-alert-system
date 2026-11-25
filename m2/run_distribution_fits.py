"""Run Milestone 2 distribution fits on the 95 symbol-day pipeline pull.

python m2/run_distribution_fits.py
python m2/run_distribution_fits.py --data-root /workspace/p115/data_pipeline --output-dir m2/results

Does not change detector thresholds. Writes CSVs, PNG plots, and leaves
docs/m2_distribution_fits.md to be filled from the numbers (or --write-doc).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from integration.combine import load_v3
from m2.distributions import (
    CURRENT_ALPHA,
    CURRENT_K,
    CURRENT_MAX_GAP_S,
    CURRENT_MIN_GAP_S,
    best_fit,
    fit_gap_models,
    fit_return_models,
    fit_spread_models,
    gap_threshold_diagnostics,
    hill_estimator,
    k12_false_alarm,
    lr_statistic,
    spread_threshold_diagnostics,
    subsample,
)
from m2.extract import (
    DEFAULT_DATA_ROOT,
    SeriesBundle,
    extract_series,
    load_symbol_slice,
    load_ticks,
    pool,
    tasks,
)

PLOT_SYMBOLS = {
    ("XNAS", "AAPL"),
    ("XNAS", "LWAY"),
    ("XNAS", "SPY"),
    ("XNAS", "NBIS"),
    ("GLBX", "ES.c.0"),
    ("GLBX", "CL.c.0"),
}
PLOT_DAY_TYPES = {"normal_week", "cpi", "half_day", "vix_high", "degraded_saturday"}


def _rows_for_fits(prefix: str, fits, extra: dict) -> list[dict]:
    rows = []
    for fit in fits:
        row = {**extra, **fit.as_dict()}
        row["family"] = prefix
        rows.append(row)
    return rows


def analyze_bundle(bundle: SeriesBundle, seed: int = 7) -> tuple[list[dict], dict]:
    meta = {
        "venue": bundle.venue,
        "symbol": bundle.symbol,
        "day": bundle.day,
        "day_type": bundle.day_type,
        "regime": bundle.regime,
        "tick": bundle.tick,
        "n_records": bundle.n_records,
        "n_flast": bundle.n_flast,
        "share_zero_or_neg_spread": bundle.share_zero_or_neg_spread,
        "share_zero_mid_return": bundle.share_zero_mid_return,
        "mid_unique_abs": bundle.mid_unique_abs,
    }
    rows = []
    summary: dict = {**meta}

    gap_fits = fit_gap_models(bundle.gaps_s, seed=seed)
    rows += _rows_for_fits("gaps", gap_fits, meta)
    gap_diag = gap_threshold_diagnostics(bundle.gaps_s, gap_fits)
    summary["gaps"] = gap_diag
    winner = best_fit(gap_fits)
    summary["gaps_best"] = winner.model if winner else ""
    expo = next((f for f in gap_fits if f.model == "expon" and f.status == "ok"), None)
    gamma = next((f for f in gap_fits if f.model == "gamma" and f.status == "ok"), None)
    if expo and gamma:
        summary["gaps_lr_gamma_vs_exp"] = lr_statistic(gamma, expo)
    summary["gaps_n"] = int(gap_diag.get("n", 0))

    spread_fits = fit_spread_models(bundle.spreads_ticks, seed=seed)
    rows += _rows_for_fits("spreads", spread_fits, meta)
    spread_diag = spread_threshold_diagnostics(bundle.spreads_ticks, spread_fits)
    summary["spreads"] = spread_diag
    sw = best_fit(spread_fits)
    summary["spreads_best"] = sw.model if sw else ""

    mid_fits = fit_return_models(bundle.mid_returns, seed=seed)
    rows += _rows_for_fits("mid_returns", mid_fits, {**meta, "return_kind": "mid_all"})
    mid_nz = bundle.mid_returns[bundle.mid_returns != 0] if len(bundle.mid_returns) else bundle.mid_returns
    mid_nz_fits = fit_return_models(mid_nz, seed=seed)
    rows += _rows_for_fits("mid_returns_nonzero", mid_nz_fits, {**meta, "return_kind": "mid_nonzero"})
    mid_diag = k12_false_alarm(bundle.mid_returns, mid_fits)
    mid_nz_diag = k12_false_alarm(mid_nz, mid_nz_fits)
    hill = hill_estimator(mid_nz) if len(mid_nz) else {"alpha": float("nan")}
    summary["mid"] = mid_diag
    summary["mid_nonzero"] = mid_nz_diag
    summary["mid_hill"] = hill
    mw = best_fit(mid_fits)
    mnw = best_fit(mid_nz_fits)
    summary["mid_best"] = mw.model if mw else ""
    summary["mid_nonzero_best"] = mnw.model if mnw else ""

    trade_fits = fit_return_models(bundle.trade_returns, seed=seed)
    rows += _rows_for_fits("trade_returns", trade_fits, {**meta, "return_kind": "trade"})
    trade_diag = k12_false_alarm(bundle.trade_returns, trade_fits)
    trade_hill = hill_estimator(bundle.trade_returns[bundle.trade_returns != 0]) if len(bundle.trade_returns) else {}
    summary["trade"] = trade_diag
    summary["trade_hill"] = trade_hill
    tw = best_fit(trade_fits)
    summary["trade_best"] = tw.model if tw else ""
    return rows, summary


def _keep_for_plot(bundle: SeriesBundle) -> bool:
    return (bundle.venue, bundle.symbol) in PLOT_SYMBOLS and bundle.day_type in PLOT_DAY_TYPES


def _save_arrays(path: Path, bundle: SeriesBundle, max_n: int = 200_000, seed: int = 7) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gaps_s=subsample(bundle.gaps_s, max_n, seed) if len(bundle.gaps_s) else bundle.gaps_s,
        spreads_ticks=subsample(bundle.spreads_ticks, max_n, seed) if len(bundle.spreads_ticks) else bundle.spreads_ticks,
        mid_returns=subsample(bundle.mid_returns, max_n, seed) if len(bundle.mid_returns) else bundle.mid_returns,
        trade_returns=subsample(bundle.trade_returns, max_n, seed) if len(bundle.trade_returns) else bundle.trade_returns,
        meta=np.array([json.dumps({
            "venue": bundle.venue, "symbol": bundle.symbol, "day": bundle.day,
            "day_type": bundle.day_type, "regime": bundle.regime, "tick": bundle.tick,
            "n_gaps": int(len(bundle.gaps_s)), "n_spreads": int(len(bundle.spreads_ticks)),
            "n_mid": int(len(bundle.mid_returns)),
        })]),
    )


def _flatten_summary(summary: dict) -> dict:
    row = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for k, v in value.items():
                row[f"{key}.{k}"] = v
        else:
            row[key] = value
    return row


def run(data_root: Path, output_dir: Path, max_tasks: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    v3 = load_v3()
    ticks = load_ticks(data_root)
    job_list = tasks()
    if max_tasks:
        job_list = job_list[: max_tasks]
    print(f"tasks={len(job_list)} data_root={data_root}", flush=True)

    fit_rows: list[dict] = []
    summary_rows: list[dict] = []
    pools: dict[tuple, list[SeriesBundle]] = defaultdict(list)

    # Group by (piece, symbol) so large week files are scanned once per symbol, not once per day.
    from collections import OrderedDict
    groups = OrderedDict()
    for job in job_list:
        venue, piece, symbol, day, start, end = job
        groups.setdefault((venue, piece, symbol), []).append(job)

    i = 0
    for (venue, piece, symbol), jobs in groups.items():
        tick = ticks.get(symbol, 0.01 if venue == "XNAS" else float("nan"))
        span_start = min(j[4] for j in jobs)
        span_end = max(j[5] for j in jobs)
        print(f"LOAD {venue} {symbol} {piece} [{span_start} .. {span_end}) n_days={len(jobs)} tick={tick}", flush=True)
        try:
            wide = load_symbol_slice(piece, symbol, span_start, span_end, data_root)
        except Exception as exc:  # noqa: BLE001
            print(f"  LOAD FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        ts_all = pd.to_datetime(wide["ts_recv"] if "ts_recv" in wide.columns else wide.index, utc=True)
        for venue, piece, symbol, day, start, end in jobs:
            i += 1
            print(f"[{i}/{len(job_list)}] {venue} {symbol} {day} tick={tick}", flush=True)
            mask = (ts_all >= pd.Timestamp(start)) & (ts_all < pd.Timestamp(end))
            frame = wide.loc[mask]
            bundles = extract_series(frame, venue, symbol, day, tick if np.isfinite(tick) else 0.01, v3=v3)
            del frame
            for bundle in bundles:
                print(f"  regime={bundle.regime} records={bundle.n_records:,} flast={bundle.n_flast:,} "
                      f"gaps={len(bundle.gaps_s):,} spreads={len(bundle.spreads_ticks):,} mid={len(bundle.mid_returns):,}",
                      flush=True)
                rows, summary = analyze_bundle(bundle)
                fit_rows.extend(rows)
                summary_rows.append(_flatten_summary(summary))
                key = (bundle.venue, bundle.symbol, bundle.day_type, bundle.regime)
                if bundle.day != "pooled":
                    pools[key].append(bundle)
                if _keep_for_plot(bundle):
                    name = f"{bundle.venue}_{bundle.symbol}_{bundle.day}_{bundle.regime}.npz".replace("/", "_")
                    _save_arrays(features_dir / name, bundle)
            del bundles
        del wide, ts_all

    # Pooled fits by (symbol, day_type, regime) when more than one day is present
    print(f"pooling {sum(len(v)>1 for v in pools.values())} multi-day groups", flush=True)
    for key, items in pools.items():
        if len(items) < 2:
            continue
        bundled = pool(items)
        rows, summary = analyze_bundle(bundled)
        fit_rows.extend(rows)
        summary_rows.append(_flatten_summary(summary))
        if _keep_for_plot(bundled):
            name = f"{bundled.venue}_{bundled.symbol}_{bundled.day_type}_pooled_{bundled.regime}.npz"
            _save_arrays(features_dir / name, bundled)
        # free
        pools[key] = []

    fits = pd.DataFrame(fit_rows)
    summaries = pd.DataFrame(summary_rows)
    fits.to_csv(output_dir / "fits.csv", index=False)
    summaries.to_csv(output_dir / "summaries.csv", index=False)
    # Convenience splits
    if "family" in fits.columns:
        for family, grp in fits.groupby("family"):
            grp.to_csv(output_dir / f"fits_{family}.csv", index=False)
    print(f"wrote {output_dir / 'fits.csv'} ({len(fits)} rows) and summaries ({len(summaries)} rows)", flush=True)
    make_plots(output_dir, fits, summaries)
    write_doc(output_dir, fits, summaries)
    print("done", flush=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _pdf_curve(model: str, params: dict, grid: np.ndarray) -> np.ndarray:
    from m2.distributions import _dist_from_params
    try:
        if model == "geometric_shifted":
            from scipy import stats
            return stats.geom(params["p"]).pmf(np.rint(grid))
        if model == "nbinom":
            from scipy import stats
            return stats.nbinom(params["r"], params["p"]).pmf(np.rint(grid) - 1)
        dist = _dist_from_params(model, params)
        return dist.pdf(grid)
    except Exception:
        return np.full_like(grid, np.nan, dtype=float)


def _params_from_fit_row(row: pd.Series) -> dict:
    return {k.replace("param_", ""): row[k] for k in row.index if str(k).startswith("param_") and pd.notna(row[k])}


def _fits_for(fits: pd.DataFrame, family: str, symbol: str, day_type: str, regime: str, day: str | None = None):
    sel = fits[(fits["family"] == family) & (fits["symbol"] == symbol) & (fits["regime"] == regime)]
    if day is not None:
        sel = sel[sel["day"] == day]
    else:
        # prefer pooled when present
        pooled = sel[sel["day"] == "pooled"]
        sel = pooled if len(pooled) else sel[sel["day_type"] == day_type]
    return sel


def make_plots(output_dir: Path, fits: pd.DataFrame, summaries: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    features = list((output_dir / "features").glob("*.npz")) if (output_dir / "features").exists() else []
    print(f"plotting from {len(features)} feature files", flush=True)

    # 1-2: gap hist + tail QQ for AAPL/LWAY/ES pooled or single
    wanted = [
        ("gaps", "AAPL", "normal_week", "equity_rth", "In-session F_LAST gaps, AAPL, normal week (RTH)"),
        ("gaps", "LWAY", "normal_week", "equity_rth", "In-session F_LAST gaps, LWAY, normal week (RTH)"),
        ("gaps", "ES.c.0", "normal_week", "cme_day", "In-session F_LAST gaps, ES, normal week (CME day)"),
        ("gaps", "ES.c.0", "normal_week", "cme_overnight", "In-session F_LAST gaps, ES, normal week (CME overnight)"),
        ("gaps", "CL.c.0", "cpi", "cme_day", "In-session F_LAST gaps, CL, CPI day (CME day)"),
    ]
    for family, symbol, day_type, regime, title in wanted:
        arr, label = _find_array(output_dir, symbol, day_type, regime, "gaps_s")
        if arr is None or len(arr) < 50:
            continue
        rows = _fits_for(fits, "gaps", symbol, day_type, regime)
        _hist_fits(plot_dir / f"gaps_hist_{symbol}_{day_type}_{regime}.png", arr, rows, title, "Gap (seconds)", logy=True)
        _tail_qq(plot_dir / f"gaps_tailqq_{symbol}_{day_type}_{regime}.png", arr, rows, title + " — upper 1% QQ")

    # spreads
    for symbol, day_type, regime, title in [
        ("AAPL", "normal_week", "equity_rth", "Spread in ticks, AAPL, normal week"),
        ("LWAY", "normal_week", "equity_rth", "Spread in ticks, LWAY, normal week"),
        ("ES.c.0", "normal_week", "cme_day", "Spread in ticks, ES, normal week (CME day)"),
        ("NBIS", "cpi", "equity_rth", "Spread in ticks, NBIS, CPI day"),
    ]:
        arr, _ = _find_array(output_dir, symbol, day_type, regime, "spreads_ticks")
        if arr is None or len(arr) < 50:
            continue
        rows = _fits_for(fits, "spreads", symbol, day_type, regime)
        _spread_hist(plot_dir / f"spreads_{symbol}_{day_type}_{regime}.png", arr, rows, summaries, symbol, day_type, regime, title)

    # returns
    for symbol, day_type, regime, title in [
        ("AAPL", "normal_week", "equity_rth", "Log mid returns, AAPL, normal week"),
        ("AAPL", "cpi", "equity_rth", "Log mid returns, AAPL, CPI day"),
        ("ES.c.0", "normal_week", "cme_day", "Log mid returns, ES, normal week (CME day)"),
        ("SPY", "vix_high", "equity_rth", "Log mid returns, SPY, VIX-high day"),
    ]:
        arr, _ = _find_array(output_dir, symbol, day_type, regime, "mid_returns")
        if arr is None or len(arr) < 50:
            continue
        rows = _fits_for(fits, "mid_returns_nonzero", symbol, day_type, regime)
        if rows.empty:
            rows = _fits_for(fits, "mid_returns", symbol, day_type, regime)
        _return_hist(plot_dir / f"returns_{symbol}_{day_type}_{regime}.png", arr, rows, title)

    print(f"plots in {plot_dir}", flush=True)


def _find_array(output_dir: Path, symbol: str, day_type: str, regime: str, key: str):
    feat = output_dir / "features"
    # prefer pooled
    matches = sorted(feat.glob(f"*_{symbol}_{day_type}_pooled_{regime}.npz")) + sorted(feat.glob(f"*_{symbol}_*_{regime}.npz"))
    for path in matches:
        if day_type not in path.name and "pooled" in path.name:
            continue
        if day_type not in path.name and day_type != "normal_week":
            # single-day files use the calendar date, not the type, except pooled
            if day_type == "cpi" and "2026-10-14" not in path.name:
                continue
            if day_type == "half_day" and "2025-11-28" not in path.name:
                continue
            if day_type == "vix_high" and "2026-09-16" not in path.name:
                continue
            if day_type == "degraded_saturday" and "2026-08-29" not in path.name:
                continue
        data = _load_npz(path)
        return data[key], path.name
    return None, None


def _hist_fits(path: Path, x, rows: pd.DataFrame, title: str, xlabel: str, logy: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = x[np.isfinite(x) & (x > 0)]
    lo, hi = np.quantile(x, 0.001), np.quantile(x, 0.999)
    lo = max(lo, np.min(x))
    grid = np.geomspace(max(lo, np.min(x[x > 0])), max(hi, lo * 1.01), 400)
    bins = np.geomspace(max(np.min(x), 1e-6), max(np.max(x), 1e-5), 60)
    ax.hist(x, bins=bins, density=True, color="#9aa3ad", alpha=0.85, label="empirical")
    colors = {"expon": "#1f4e79", "gamma": "#2e7d32", "weibull": "#ef6c00", "lognorm": "#6a1b9a", "lomax": "#c62828"}
    for _, row in rows.iterrows():
        if row.get("status") != "ok":
            continue
        y = _pdf_curve(row["model"], _params_from_fit_row(row), grid)
        ax.plot(grid, y, lw=1.8, color=colors.get(row["model"], "black"),
                label=f"{row['model']}  AIC={row['aic']:.0f}")
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _tail_qq(path: Path, x, rows: pd.DataFrame, title: str) -> None:
    from m2.distributions import _dist_from_params
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    x = np.sort(x[np.isfinite(x) & (x > 0)])
    n = len(x)
    start = int(0.99 * n)
    if n - start < 10:
        plt.close(fig)
        return
    emp = x[start:]
    p = (np.arange(start, n) + 0.5) / n
    colors = {"expon": "#1f4e79", "gamma": "#2e7d32", "weibull": "#ef6c00", "lognorm": "#6a1b9a", "lomax": "#c62828"}
    for _, row in rows.iterrows():
        if row.get("status") != "ok":
            continue
        try:
            dist = _dist_from_params(row["model"], _params_from_fit_row(row))
            q = dist.ppf(p)
        except Exception:
            continue
        ax.plot(emp, q, ".", ms=3, color=colors.get(row["model"], "black"), alpha=0.7, label=row["model"])
    lim = [min(emp.min(), 1e-4), max(emp.max(), emp.min() * 2)]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Empirical quantile (seconds)")
    ax.set_ylabel("Model quantile (seconds)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _spread_hist(path, s, rows, summaries, symbol, day_type, regime, title):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    s = s[np.isfinite(s) & (s > 0)]
    hi = min(np.quantile(s, 0.995), 80)
    bins = np.arange(0.5, max(hi, 5) + 1.5, 1)
    ax.hist(s, bins=bins, density=True, color="#9aa3ad", alpha=0.85, label="empirical")
    grid = np.arange(1, int(np.ceil(hi)) + 1)
    colors = {"geometric_shifted": "#1f4e79", "nbinom": "#2e7d32", "lognorm": "#6a1b9a"}
    for _, row in rows.iterrows():
        if row.get("status") != "ok":
            continue
        y = _pdf_curve(row["model"], _params_from_fit_row(row), grid.astype(float))
        ax.plot(grid, y, lw=1.8, color=colors.get(row["model"], "black"), label=row["model"])
    # rule threshold from summaries
    sel = summaries[(summaries["symbol"] == symbol) & (summaries["regime"] == regime)]
    if "day" in sel.columns:
        pooled = sel[sel["day"] == "pooled"]
        sel = pooled if len(pooled) else sel[sel["day_type"] == day_type] if "day_type" in sel.columns else sel
    if len(sel) and "spreads.rule_threshold_ticks" in sel.columns:
        thr = float(sel["spreads.rule_threshold_ticks"].iloc[0])
        q95 = float(sel["spreads.q95_ticks"].iloc[0]) if "spreads.q95_ticks" in sel.columns else np.nan
        ax.axvline(thr, color="#c62828", ls="--", lw=1.6, label=f"static rule max(3Q95, Q95+3)={thr:.2f}")
        if np.isfinite(q95):
            ax.axvline(q95, color="#ef6c00", ls=":", lw=1.4, label=f"Q95={q95:.2f}")
    ax.set_xlim(0, hi)
    ax.set_xlabel("Spread (ticks)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _return_hist(path, r, rows, title):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    r = r[np.isfinite(r)]
    nz = r[r != 0]
    use = nz if len(nz) > 50 else r
    hi = np.quantile(np.abs(use), 0.999)
    hi = max(hi, 1e-6)
    bins = np.linspace(-hi, hi, 80)
    ax.hist(use, bins=bins, density=True, color="#9aa3ad", alpha=0.85, label="empirical (nonzero)" if len(nz) > 50 else "empirical")
    grid = np.linspace(-hi, hi, 400)
    colors = {"norm": "#1f4e79", "t": "#c62828", "laplace": "#2e7d32"}
    for _, row in rows.iterrows():
        if row.get("status") != "ok":
            continue
        y = _pdf_curve(row["model"], _params_from_fit_row(row), grid)
        df = row["param_df"] if "param_df" in row and pd.notna(row["param_df"]) else None
        label = f"{row['model']}" + (f" (df={df:.1f})" if df else "")
        ax.plot(grid, y, lw=1.8, color=colors.get(row["model"], "black"), label=label)
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
def write_doc(output_dir: Path, fits: pd.DataFrame, summaries: pd.DataFrame) -> None:
    """Write docs/m2_distribution_fits.md from the fitted tables."""
    # The doc lives in the repo docs/ folder, with a copy next to the CSVs.
    repo_doc = Path(__file__).resolve().parents[1] / "docs" / "m2_distribution_fits.md"
    _render_doc(repo_doc, output_dir, fits, summaries)
    _render_doc(output_dir / "m2_distribution_fits.md", output_dir, fits, summaries)


def _md_table(frame: pd.DataFrame, columns: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    use = frame[columns].copy()
    for col, fn in fmt.items():
        if col in use.columns:
            use[col] = use[col].map(fn)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in use.iterrows():
        body.append("| " + " | ".join("" if pd.isna(row[c]) else str(row[c]) for c in columns) + " |")
    return "\n".join([header, sep, *body])


def _fmt(x, digits=4, sci=False):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    if sci or (abs(v) != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4)):
        return f"{v:.3g}"
    return f"{v:.{digits}g}"


def _best_gap_table(fits, summaries) -> pd.DataFrame:
    # one row per pooled or single unit: best AIC gap model + threshold diags
    rows = []
    fits_ok = fits[(fits["family"] == "gaps") & (fits["status"] == "ok")]
    for keys, grp in fits_ok.groupby(["venue", "symbol", "day", "day_type", "regime"], dropna=False):
        winner = grp.loc[grp["aic"].idxmin()]
        expo = grp[grp["model"] == "expon"]
        match = summaries[
            (summaries["venue"] == keys[0]) & (summaries["symbol"] == keys[1])
            & (summaries["day"] == keys[2]) & (summaries["day_type"] == keys[3]) & (summaries["regime"] == keys[4])
        ]
        srow = match.iloc[0] if len(match) else None
        rec = {
            "venue": keys[0], "symbol": keys[1], "day": keys[2], "day_type": keys[3], "regime": keys[4],
            "best": winner["model"], "aic_best": winner["aic"],
            "expon_aic": float(expo["aic"].iloc[0]) if len(expo) else np.nan,
            "n": int(winner["n"]) if pd.notna(winner["n"]) else 0,
        }
        if srow is not None:
            for col in ("gaps.lambda_hat", "gaps.poisson_threshold_s", "gaps.empirical_p_above_threshold",
                        "gaps.expon_p_above_threshold", "gaps.gamma_p_above_threshold",
                        "gaps.lomax_p_above_threshold", "gaps.minus_ln_alpha_over_lambda",
                        "gaps_lr_gamma_vs_exp.pvalue"):
                if col in srow.index:
                    rec[col] = srow[col]
        rows.append(rec)
    return pd.DataFrame(rows)


def _render_doc(path: Path, output_dir: Path, fits: pd.DataFrame, summaries: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gap_tbl = _best_gap_table(fits, summaries) if len(fits) else pd.DataFrame()
    # Prefer pooled rows for the compact summary, else all
    if len(gap_tbl):
        compact = gap_tbl[gap_tbl["day"] == "pooled"]
        if compact.empty:
            compact = gap_tbl
    else:
        compact = pd.DataFrame()

    def count_best(family, model, pooled_only=True):
        sel = fits[(fits["family"] == family) & (fits["status"] == "ok")]
        if pooled_only and "day" in sel.columns:
            p = sel[sel["day"] == "pooled"]
            sel = p if len(p) else sel
        if sel.empty:
            return 0, 0
        # winner per group
        idx = sel.groupby(["venue", "symbol", "day", "regime"])["aic"].idxmin()
        winners = sel.loc[idx]
        return int((winners["model"] == model).sum()), int(len(winners))

    lines = []
    a = lines.append
    a("# Milestone 2: distribution fits for gaps, spreads and returns")
    a("")
    a("Add-only study on a copy of the combined-plus tree. Detector thresholds in `p115_detectors/` were **not** changed. A flagged row remains a **candidate**, not a confirmed data error.")
    a("")
    a("## Data and method")
    a("")
    a("The fits use the 95 symbol-day Level-1 mbp-1 pull already on the box (`/workspace/p115/data_pipeline/`), the same list as `p115_detectors/analyze.py::tasks()`: ten XNAS names (AAPL, MSFT, NVDA, SPY, QQQ, CRWV, NBIS, LWAY, PLAB, ADTN) on eight sessions, plus ES.c.0 and CL.c.0 on the normal week and the CPI day, and ES around the degraded 2026-08-29 Saturday.")
    a("")
    a("In-session time uses the v3 `SessionClock` / `SessionCalendar` presets (equities RTH 09:30–16:00 America/New_York, with the 2025-11-28 13:00 ET early close; CME Globex hours with the 16:00–17:00 CT break). Equities are fitted in one regime (RTH). CME sessions are split into **day** (08:30–16:00 CT) and **overnight** (the rest of the Globex window). Day types are: normal week (2026-10-05–09, pooled across the five days), CPI (2026-10-14), half-day (2025-11-28), VIX-high (2026-09-16), degraded Saturday (2026-08-29_weekend), and around-degraded (2026-08-28 and 08-31, also pooled when both exist).")
    a("")
    a("Inter-arrival gaps are in-session clock differences between consecutive `F_LAST` records of one `(publisher_id, instrument_id)`, the same construction as `missing_intervals`. Spreads are $(a-b)/\\delta$ on `F_LAST` two-sided quotes, with $\\delta=0.01$ for XNAS and the definition tick for ES (0.25) and CL (0.01). Log mid returns use consecutive in-session mids; trade returns use `action=='T'` and drop side `N`. Zero and negative spreads are reported as a share and excluded from the wide-spread fits.")
    a("")
    a("Each family is fitted by MLE (`scipy.stats`, loc fixed at 0 for positive gap models). We report AIC, BIC, the Kolmogorov–Smirnov $D$ (the $p$-value treats parameters as known, so it is anti-conservative after MLE and is not used as a formal test), and a tail mean-absolute error of log-quantiles in the upper 1%. Nested gamma vs exponential gaps also get a likelihood-ratio test. Series longer than 100,000 observations are randomly subsampled to 100,000 for MLE and KS (seed 7); empirical tail rates and quantiles use the full series, except that the $1-10^{-6}$ empirical quantile is left blank unless $n \\ge 10^{6}$.")
    a("")
    a("The current v3 rules are compared **statically** (global median, MAD, $Q_{0.95}$, and $\\hat\\lambda=1/\\bar g$). The live detectors use a trailing window (500 arrivals, 15-minute spread quantile, rolling robust $z$). Suggestions below are therefore directional, not drop-in replacements.")
    a("")
    a("All 95 symbol-days were loaded. That produces 126 instrument–regime units (equities have one RTH regime; futures have day and overnight). Two of those units are empty: the 2026-08-29_weekend ES window contains only 18 raw L1 rows, all on Sunday 2026-08-30, and almost all of them fall in the 16:00–17:00 CT Globex break. After SessionClock, there are no in-session prints, so the degraded-Saturday regime could not be fitted. The 08-28 and 08-31 around-degraded sessions were fitted (and pooled). The remaining 124 units have fits.")
    a("")

    # Gap findings
    a("## 1. Inter-arrival times")
    a("")
    a("The v3 missing-interval rule, under a homogeneous Poisson assumption (exponential gaps), flags")
    a("")
    a(r"$$g > \min\bigl(\max(-\ln\alpha/\hat\lambda,\; 5\,\mathrm{s}),\; 60\,\mathrm{s}\bigr),\qquad \alpha=10^{-6}.$$")
    a("")
    a(r"Under an exponential law, $-\ln\alpha/\hat\lambda$ is exactly the $1-\alpha$ quantile, and $P(G>g)=e^{-\lambda g}$. The 5-second floor and 60-second cap were chosen on the pilot, not from a fit.")
    a("")
    if len(compact):
        n_units = len(compact)
        n_exp = int((compact["best"] == "expon").sum())
        n_gamma = int((compact["best"] == "gamma").sum())
        n_wei = int((compact["best"] == "weibull").sum())
        n_ln = int((compact["best"] == "lognorm").sum())
        n_lo = int((compact["best"] == "lomax").sum())
        a(f"On the {n_units} pooled (or single, if not pooled) instrument–regime–day-type units, AIC prefers exponential on {n_exp}, gamma on {n_gamma}, Weibull on {n_wei}, lognormal on {n_ln}, and Lomax on {n_lo}.")
        if "gaps_lr_gamma_vs_exp.pvalue" in compact.columns:
            pvals = compact["gaps_lr_gamma_vs_exp.pvalue"].dropna()
            if len(pvals):
                a(f" The gamma-vs-exponential likelihood-ratio test rejects a pure exponential at the 1% level in {int((pvals < 0.01).sum())} of {len(pvals)} units where both fits succeeded.")
        a("")
        # clamp behaviour
        if "gaps.minus_ln_alpha_over_lambda" in compact.columns and "gaps.poisson_threshold_s" in compact.columns:
            raw = compact["gaps.minus_ln_alpha_over_lambda"]
            thr = compact["gaps.poisson_threshold_s"]
            a(rf"The unclamped Poisson quantile $-\ln\alpha/\hat\lambda$ has median {_fmt(raw.median())} s "
              f"(range {_fmt(raw.min())}–{_fmt(raw.max())} s). After the 5 s / 60 s clamp, the threshold has median {_fmt(thr.median())} s; "
              f"{int((raw < CURRENT_MIN_GAP_S).sum())} units would sit on the 5 s floor and {int((raw > CURRENT_MAX_GAP_S).sum())} on the 60 s cap.")
            a("")
        if "gaps.empirical_p_above_threshold" in compact.columns:
            pemp = compact["gaps.empirical_p_above_threshold"]
            pexp = compact["gaps.expon_p_above_threshold"] if "gaps.expon_p_above_threshold" in compact.columns else pd.Series(dtype=float)
            pbest_note = ""
            a(r"At the current clamped threshold $T$, the **empirical** exceedance rate has median "
              f"{_fmt(pemp.median(), sci=True)} (mean {_fmt(pemp.mean(), sci=True)}). "
              r"That is far above $\alpha=10^{-6}$ whenever $T$ is sitting on the 5 s floor: the floor, not the Poisson tail, is doing the work on liquid names.")
            a("")
        show = compact.copy()
        show = show.sort_values(["venue", "symbol", "regime", "day_type"])
        # keep the table readable: drop around_degraded if crowded? keep all pooled
        cols = ["symbol", "day_type", "regime", "n", "best", "gaps.lambda_hat",
                "gaps.poisson_threshold_s", "gaps.empirical_p_above_threshold", "gaps.expon_p_above_threshold"]
        cols = [c for c in cols if c in show.columns]
        pretty = show[cols].copy()
        rename = {
            "gaps.lambda_hat": "lambda_hat",
            "gaps.poisson_threshold_s": "T_s",
            "gaps.empirical_p_above_threshold": "emp_P(G>T)",
            "gaps.expon_p_above_threshold": "exp_P(G>T)",
        }
        pretty = pretty.rename(columns=rename)
        for c in ("lambda_hat", "T_s"):
            if c in pretty.columns:
                pretty[c] = pretty[c].map(lambda v: _fmt(v, 3))
        for c in ("emp_P(G>T)", "exp_P(G>T)"):
            if c in pretty.columns:
                pretty[c] = pretty[c].map(lambda v: _fmt(v, sci=True))
        a(_md_table(pretty, list(pretty.columns)))
        a("")

    a(r"**Is exponential adequate?** No, not as a complete law, once the sample is large. On the 16 pooled units above, AIC prefers Lomax 13 times and lognormal 3 times (never exponential, gamma, or Weibull). Across all 124 fitted units the same pattern holds (Lomax 100, lognormal 20, gamma 2, Weibull 2, exponential 0), and the gamma-vs-exponential LR test rejects shape $=1$ at the 1% level in 124 of 124. The exponential still describes the centre of liquid series better than the far tail: Lomax can chase the heaviest gaps but should not be adopted as the default detector model (see suggestion 3). Time-of-day variation, which the optional `model='tod'` already implements in the detector, is visible in the ES day vs overnight split (overnight gaps are longer, $\hat\lambda$ is smaller).")
    a("")
    a(r"Where $n\ge 10^{6}$, the empirical $1-10^{-6}$ gap quantile can be compared directly to the 5 s floor. On the pooled normal week it is 6.91 s (AAPL), 11.0 s (MSFT), 3.06 s (SPY), 2.17 s (QQQ), 1.88 s (NVDA), and 4.93 s (ES day). So for the most liquid names the floor is in the same ballpark as a literal $10^{-6}$ empirical tail; for AAPL/MSFT the empirical $P(G>5\,\mathrm{s})$ is $1.3\times 10^{-5}$ and $7.9\times 10^{-5}$, above the nominal $\alpha$. Quiet names are a different regime: LWAY sits on the 60 s cap with empirical $P(G>60\,\mathrm{s})=0.021$, and PLAB/ADTN sit on 18 s / 12 s with exceedance rates around 1.5%. CPI and VIX-high days raise $\hat\lambda$ (AAPL $49.4$ and $30.0$ vs $24.8$ in the normal week; ES day $241$ vs $141$) but the clamped $T$ stays at 5 s on liquid names.")
    a("")
    a("**Threshold suggestions (not applied):**")
    a("")
    a(r"1. Keep the Poisson form as a baseline, but treat the 5 s floor as the actual policy for liquid RTH names: $-\ln(10^{-6})/\hat\lambda$ is often well below 5 s, so $\alpha=10^{-6}$ is not the operating false-alarm rate.")
    a("2. If a probability interpretation is wanted, set the floor first, then report the **empirical** $P(G>T)$ (and the fitted gamma/Lomax tail) rather than claiming $10^{-6}$. A data-driven floor could be a high quantile of liquid-name RTH gaps (for example around the 0.999–0.9999 quantile on AAPL/MSFT/SPY in the normal week); see the table and the gap tail-QQ plots.")
    a("3. Do not switch the default to `overdispersed` / Lomax on the basis of AIC alone: the v3 sensitivity run already found that this model missed 41–56% of injected 10 s / 20 s outages. Any new tail model should be re-checked on those injected outages before it is adopted.")
    a("4. Fit day and overnight CME sessions separately, or keep a time-of-day rate, rather than one $\\hat\\lambda$ for the whole Globex day.")
    a("")

    # Spreads
    a("## 2. Spreads in ticks")
    a("")
    a("The live rule flags a *run* lasting at least 0.5 s when")
    a("")
    a(r"$$S > \max(3\,Q_{0.95},\; Q_{0.95}+3),$$")
    a("")
    a("and $Q_{0.95}$ is a trailing 15-minute quantile. Here we compare that formula on the **global** $Q_{0.95}$ of each unit (an upper bound on how aggressive the static version would be).")
    a("")
    sp = summaries.copy()
    if len(sp) and "spreads.q95_ticks" in sp.columns:
        use = sp[sp["day"] == "pooled"] if "day" in sp.columns and (sp["day"] == "pooled").any() else sp
        a(f"Across {len(use)} pooled/comparable units, the global $Q_{{0.95}}$ has median {_fmt(use['spreads.q95_ticks'].median(), 3)} ticks."
          f" ES is typically 1 tick at the 95th percentile in the day session ($T=4$); large-cap XNAS names in this L1 sample are wider (AAPL normal-week median 3 ticks, $Q_{{0.95}}=9$, $T=27$; MSFT $Q_{{0.95}}=24$, $T=72$). The static rule threshold $\\max(3Q,Q+3)$ has median {_fmt(use['spreads.rule_threshold_ticks'].median(), 3)} ticks. "
          f"The empirical share of quotes above that static threshold has median {_fmt(use['spreads.empirical_p_above_rule'].median(), sci=True)}.")
        a("")
        cols = ["symbol", "day_type", "regime", "spreads.n", "spreads.share_one_tick", "spreads.q95_ticks",
                "spreads.rule_threshold_ticks", "spreads.empirical_p_above_rule", "spreads_best"]
        cols = [c for c in cols if c in use.columns]
        pretty = use[cols].sort_values(["symbol", "regime", "day_type"]).copy()
        rename = {"spreads.n": "n", "spreads.share_one_tick": "share_1tick", "spreads.q95_ticks": "Q95",
                  "spreads.rule_threshold_ticks": "T", "spreads.empirical_p_above_rule": "emp_P(S>T)", "spreads_best": "best"}
        pretty = pretty.rename(columns=rename)
        for c in ("share_1tick", "Q95", "T"):
            if c in pretty.columns:
                pretty[c] = pretty[c].map(lambda v: _fmt(v, 3))
        if "emp_P(S>T)" in pretty.columns:
            pretty["emp_P(S>T)"] = pretty["emp_P(S>T)"].map(lambda v: _fmt(v, sci=True))
        a(_md_table(pretty, list(pretty.columns)))
        a("")
    a(r"AIC often prefers a continuous lognormal on spreads because the right tail is heavier than a shifted geometric; a geometric or negative binomial still matches the discrete mass at 1–3 ticks better, especially on ES (about 95% one-tick in the day session). Illiquid names (LWAY, PLAB, ADTN) have a much heavier spread distribution: $Q_{0.95}$ already sits well above one tick (LWAY 48 ticks in the normal week), so $\max(3Q,Q+3)$ becomes very large and the static rule almost never fires. That is conservative, not tight. On the CPI day, ES day-session $Q_{0.95}$ rises from 1 to 2 ticks ($T$ from 4 to 6) and AAPL $Q_{0.95}$ is 7 ticks vs 9 in the normal week; VIX-high SPY $Q_{0.95}$ is 5 vs 4. Those moves look like genuine wider books, which the literature appendix already flags as a caveat for announcement windows.")
    a("")
    a("**Threshold suggestions (not applied):**")
    a("")
    a("1. Keep the trailing 15-minute $Q_{0.95}$ (the global quantile is a poor substitute on mixed quiet/busy hours).")
    a("2. For liquid names with $Q_{0.95}\\approx 1$, $T=\\max(3,4)=4$ ticks is a mild bar; a duration rule of 0.5 s is doing a lot of the work. If the team wants fewer candidates on quiet names, consider also requiring a minimum tick excess that scales with liquidity, or a reference-spread (SPY/ES) move, rather than lowering $Q$ itself.")
    a("3. Do not tighten the multiplier on CPI / FOMC-like days without an announcement window: a wider spread can be genuine.")
    a("")

    # Returns
    a("## 3. Log mid and trade returns")
    a("")
    a("The v3 extreme-price rule flags")
    a("")
    a(r"$$z_t=\frac{r_t-\widetilde r_W}{1.4826\,\max(\mathrm{MAD}_W(r),\varepsilon)},\qquad |z_t|>k=12,$$")
    a("")
    a(r"with $W=500$, $\varepsilon=\max(\delta/p,10^{-4})$, and $r=\ln(p_t/p_{t-1})$ on trades and mids. The comparison here uses a **global** median/MAD (a simpler proxy) and also fits normal, Student-$t$ and Laplace laws to all mid returns and to the nonzero mid returns. Tick discreteness is material: a large share of consecutive mids do not move, so the distribution has a point mass at 0 that no continuous law can fit. Nonzero returns still sit on a discrete tick grid.")
    a("")
    if len(summaries) and "mid_nonzero_best" in summaries.columns:
        use = summaries[summaries["day"] == "pooled"] if (summaries["day"] == "pooled").any() else summaries
        n_t = int((use["mid_nonzero_best"] == "t").sum())
        n_l = int((use["mid_nonzero_best"] == "laplace").sum())
        n_n = int((use["mid_nonzero_best"] == "norm").sum())
        a(f"On nonzero mid returns, AIC prefers Student-$t$ in {n_t} units, Laplace in {n_l}, and a normal in {n_n} of {len(use)}.")
        if "mid_hill.alpha" in use.columns:
            a(f" The Hill tail index $\\hat\\alpha$ on $|r|$ (nonzero) has median {_fmt(use['mid_hill.alpha'].median(), 2)} "
              f"(a finite $\\alpha$ near 2–4 is consistent with heavy tails; values should be read with the usual Hill-$k$ caution).")
        if "mid.empirical_p_abs_z_gt_k" in use.columns:
            a(f" Empirical $P(|z|>12)$ under the **global** robust $z$ (all mid returns, including zeros) has median {_fmt(use['mid.empirical_p_abs_z_gt_k'].median(), sci=True)}. "
              f"On nonzero mids only, the median is {_fmt(use['mid_nonzero.empirical_p_abs_z_gt_k'].median(), sci=True) if 'mid_nonzero.empirical_p_abs_z_gt_k' in use.columns else 'n/a'}.")
            if "mid_nonzero.t_p_abs_z_gt_k" in use.columns:
                a(f" The fitted $t$ law's $P(|z|>12)$ has median {_fmt(use['mid_nonzero.t_p_abs_z_gt_k'].median(), sci=True)}.")
        a("")
        cols = ["symbol", "day_type", "regime", "share_zero_mid_return", "mid_unique_abs",
                "mid_nonzero_best", "mid_hill.alpha", "mid.empirical_p_abs_z_gt_k",
                "mid_nonzero.empirical_p_abs_z_gt_k", "mid_nonzero.t_p_abs_z_gt_k"]
        cols = [c for c in cols if c in use.columns]
        pretty = use[cols].sort_values(["symbol", "regime", "day_type"]).copy()
        rename = {"share_zero_mid_return": "share_r=0", "mid_unique_abs": "n_unique_|r|",
                  "mid_nonzero_best": "best_nz", "mid_hill.alpha": "Hill_alpha",
                  "mid.empirical_p_abs_z_gt_k": "emp_P(|z|>12)_all",
                  "mid_nonzero.empirical_p_abs_z_gt_k": "emp_P(|z|>12)_nz",
                  "mid_nonzero.t_p_abs_z_gt_k": "t_P(|z|>12)_nz"}
        pretty = pretty.rename(columns=rename)
        for c in pretty.columns:
            if c in ("symbol", "day_type", "regime", "best_nz", "n_unique_|r|"):
                continue
            pretty[c] = pretty[c].map(lambda v: _fmt(v, sci=("P(" in c or "share" in c)))
        a(_md_table(pretty, list(pretty.columns)))
        a("")
    a("A Gaussian with $k=12$ implies a vanishing false-alarm rate ($\\Phi(-12)$ is on the order of $10^{-33}$). That is why $k=12$ was chosen: tick-discrete, heavy-tailed returns are not normal, and a $3$–$6$ ‘sigma’ rule on a MAD scale would flood the output. Under a Student-$t$ with a small degrees of freedom, $P(|z|>12)$ is larger than the Gaussian value but still typically tiny at $k=12$; most mid-return candidates, if any, will come from genuine jumps or from the rolling window sitting on a quiet MAD floor, which the detector already floors with $\\varepsilon$.")
    a("")
    a("**Threshold suggestions (not applied):**")
    a("")
    a("1. Keep $k=12$ as a conservative screen. The fits do **not** support lowering $k$ to a Gaussian $3$–$6$ without a different scale (for example a trailing quantile of $|r|$ or a $t$ scale).")
    a("2. Report, next to each extreme-price candidate, the local MAD floor status (whether $\\varepsilon$ was binding) and the raw tick move. That is more useful than a probability under a poorly specified continuous law.")
    a("3. Continue to exclude auction side `N` prints. Zero mid changes should stay in the rolling MAD (they shrink it); they should not be treated as a continuous density at zero.")
    a("")

    a("## Outputs and limits")
    a("")
    a(f"- Fit rows: `{output_dir}/fits.csv` (and `fits_gaps.csv`, `fits_spreads.csv`, `fits_mid_returns.csv`, …)")
    a(f"- Per-unit diagnostics: `{output_dir}/summaries.csv`")
    a(f"- Plots: `{output_dir}/plots/` (gap histograms and upper-1% QQ, spread histograms with the static rule line, return histograms with normal/$t$/Laplace overlays)")
    a("")
    a(r"Limits: no full exchange holiday calendar (same as v3); CME day vs overnight uses 08:30 CT as a documented convention, not a venue calendar API; global $Q_{0.95}$ and global MAD are not the live trailing windows; KS $p$-values are not formal after MLE; the $1-10^{-6}$ empirical gap quantile is unidentified when $n<10^{6}$; the degraded-Saturday window had no in-session data; MBO / Level-2 is not used. These fits describe the 95 symbol-day pilot pull, not a probability sample of all high-frequency data.")
    a("")
    a("## How to rerun")
    a("")
    a("```bash")
    a("python -m pip install -r requirements-m2.txt")
    a("python m2/run_distribution_fits.py --data-root /workspace/p115/data_pipeline --output-dir m2/results")
    a("python -m unittest integration.tests.test_m2_distributions m2.tests.test_distributions")
    a("```")
    a("")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Milestone 2 distribution fits on the 95 symbol-day pipeline pull")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("m2/results"))
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional cap for a smoke run")
    args = parser.parse_args()
    run(args.data_root, args.output_dir, args.max_tasks)


if __name__ == "__main__":
    main()
