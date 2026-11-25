"""Fit and compare candidate distributions for Milestone 2.

Pure functions: MLE, AIC/BIC, KS, tail QQ error, Hill tail index, and the
implied probabilities of the current v3 thresholds. Detector thresholds are
never written back.

Current rules (v3 defaults, quoted for comparison only):

* Missing intervals, Poisson/exponential: flag if
  :math:`g > \\min(\\max(-\\ln\\alpha/\\hat\\lambda, 5\\mathrm{s}), 60\\mathrm{s})`
  with :math:`\\alpha=10^{-6}`.
* Unusual spreads, rolling quantile: flag if
  :math:`S > \\max(3 Q_{0.95}, Q_{0.95}+3)` (plus a 0.5 s duration rule, not
  applied in these static fits).
* Extreme prices: :math:`|z|>k=12` with
  :math:`z=(r-\\mathrm{med})/(1.4826\\max(\\mathrm{MAD},\\varepsilon))`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import stats

MAD_SCALE = 1.4826
CURRENT_ALPHA = 1e-6
CURRENT_MIN_GAP_S = 5.0
CURRENT_MAX_GAP_S = 60.0
CURRENT_K = 12.0
CURRENT_SPREAD_Q = 0.95
CURRENT_SPREAD_MULT = 3.0
CURRENT_SPREAD_EXCESS = 3.0
SUBSAMPLE_MAX = 100_000
KS_COMPARE_N = 5_000

# Positive continuous families used for in-session gaps (loc fixed at 0).
GAP_MODELS = ("expon", "gamma", "weibull", "lognorm", "lomax")
# Continuous families for log returns (location free, except we also try loc=0).
RETURN_MODELS = ("norm", "t", "laplace")


@dataclass
class FitResult:
    model: str
    n: int
    n_fit: int
    n_params: int
    params: dict[str, float]
    loglik: float
    aic: float
    bic: float
    ks_d: float
    ks_p: float
    ks_note: str
    tail_mae: float
    status: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        row = {
            "model": self.model,
            "n": self.n,
            "n_fit": self.n_fit,
            "n_params": self.n_params,
            "loglik": self.loglik,
            "aic": self.aic,
            "bic": self.bic,
            "ks_d": self.ks_d,
            "ks_p": self.ks_p,
            "ks_note": self.ks_note,
            "tail_mae": self.tail_mae,
            "status": self.status,
            "detail": self.detail,
        }
        for key, value in self.params.items():
            row[f"param_{key}"] = value
        return row


def subsample(x: np.ndarray, max_n: int = SUBSAMPLE_MAX, seed: int = 7) -> np.ndarray:
    """Random subsample without replacement. Documented, reproducible."""
    x = np.asarray(x)
    if len(x) <= max_n:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), size=max_n, replace=False)]


def aic_bic(loglik: float, n_params: int, n: int) -> tuple[float, float]:
    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(n) - 2 * loglik
    return float(aic), float(bic)


def _finite_positive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x) & (x > 0)]
    return x


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    return x[np.isfinite(x)]


def median_mad(x: np.ndarray) -> tuple[float, float]:
    x = _finite(x)
    if not len(x):
        return float("nan"), float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, mad


def hill_estimator(x: np.ndarray, k: int | None = None) -> dict[str, float]:
    """Hill tail-index estimator on the upper order statistics of |x|.

    :math:`\\hat\\gamma = \\frac{1}{k}\\sum_{i=1}^{k}\\log X_{(n-i+1)}/X_{(n-k)}`,
    with tail index :math:`\\hat\\alpha=1/\\hat\\gamma`. ``k`` defaults to
    :math:`\\min(500, \\lfloor n/20\\rfloor)`, at least 20 when possible.
    """
    x = np.sort(_finite_positive(np.abs(np.asarray(x, dtype="float64"))))
    n = len(x)
    if n < 40:
        return {"n": n, "k": 0, "gamma": float("nan"), "alpha": float("nan")}
    if k is None:
        k = int(min(500, max(20, n // 20)))
    k = int(min(k, n - 2))
    thresh = x[n - k - 1]
    if thresh <= 0:
        return {"n": n, "k": k, "gamma": float("nan"), "alpha": float("nan")}
    gamma = float(np.mean(np.log(x[n - k :] / thresh)))
    alpha = (1.0 / gamma) if gamma > 0 else float("nan")
    return {"n": n, "k": k, "gamma": gamma, "alpha": alpha, "threshold": float(thresh)}


def poisson_gap_threshold(lam: float, alpha: float = CURRENT_ALPHA,
                          min_gap: float = CURRENT_MIN_GAP_S,
                          max_gap: float = CURRENT_MAX_GAP_S) -> float:
    """Current adaptive Poisson threshold :math:`\\min(\\max(-\\ln\\alpha/\\lambda, 5), 60)`."""
    if not np.isfinite(lam) or lam <= 0:
        return float(max_gap)
    return float(np.clip(-np.log(alpha) / lam, min_gap, max_gap))


def empirical_survival(x: np.ndarray, t: float) -> float:
    x = _finite(x)
    if not len(x) or not np.isfinite(t):
        return float("nan")
    return float(np.mean(x > t))


def empirical_quantile(x: np.ndarray, q: float) -> float:
    x = _finite(x)
    if not len(x):
        return float("nan")
    return float(np.quantile(x, q, method="linear"))


def spread_rule_threshold(q95: float, multiplier: float = CURRENT_SPREAD_MULT,
                          excess: float = CURRENT_SPREAD_EXCESS) -> float:
    if not np.isfinite(q95):
        return float("nan")
    return float(max(multiplier * q95, q95 + excess))


def _loglik(dist, x: np.ndarray, discrete: bool = False) -> float:
    if discrete:
        ll = dist.logpmf(x)
    else:
        ll = dist.logpdf(x)
    ll = ll[np.isfinite(ll)]
    if not len(ll):
        return float("-inf")
    return float(np.sum(ll))


def _ks(dist, x: np.ndarray) -> tuple[float, float, str]:
    """KS statistic against a fitted continuous CDF.

    The p-value from ``kstest`` treats the parameters as known, so it is
    anti-conservative after MLE. We still report D, and note the caveat.
    """
    if len(x) < 5:
        return float("nan"), float("nan"), "too_few"
    try:
        result = stats.kstest(x, dist.cdf)
        return float(result.statistic), float(result.pvalue), "params_estimated_p_anticonservative"
    except Exception as exc:  # noqa: BLE001 — fit/cdf edge cases
        return float("nan"), float("nan"), f"ks_failed:{type(exc).__name__}"


def _tail_mae(dist, x: np.ndarray, q: float = 0.99, n_grid: int = 20) -> float:
    """Mean absolute error of log-quantiles in the upper tail (default top 1%)."""
    x = _finite(x)
    if len(x) < 50:
        return float("nan")
    p_hi = 1.0 - 1.0 / len(x)
    if p_hi <= q:
        return float("nan")
    probs = np.linspace(q, p_hi, n_grid)
    emp = np.quantile(x, probs, method="linear")
    try:
        mod = dist.ppf(probs)
    except Exception:
        return float("nan")
    ok = np.isfinite(emp) & np.isfinite(mod) & (emp > 0) & (mod > 0)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(np.log(mod[ok]) - np.log(emp[ok]))))


def _fit_success(model: str, n: int, n_fit: int, n_params: int, params: dict[str, float],
                 loglik: float, dist, x_ks: np.ndarray, x_tail: np.ndarray,
                 status: str = "ok", detail: str = "") -> FitResult:
    aic, bic = aic_bic(loglik, n_params, n_fit)
    ks_d, ks_p, ks_note = _ks(dist, x_ks)
    return FitResult(model=model, n=n, n_fit=n_fit, n_params=n_params, params=params,
                     loglik=loglik, aic=aic, bic=bic, ks_d=ks_d, ks_p=ks_p, ks_note=ks_note,
                     tail_mae=_tail_mae(dist, x_tail), status=status, detail=detail)


def _fit_fail(model: str, n: int, n_fit: int, detail: str) -> FitResult:
    return FitResult(model=model, n=n, n_fit=n_fit, n_params=0, params={},
                     loglik=float("-inf"), aic=float("inf"), bic=float("inf"),
                     ks_d=float("nan"), ks_p=float("nan"), ks_note="not_fit",
                     tail_mae=float("nan"), status="failed", detail=detail)


def _frozen_and_params(name: str, x: np.ndarray):
    """MLE with loc=0 for positive gap models. Returns (frozen, params, n_params)."""
    if name == "expon":
        loc, scale = 0.0, float(np.mean(x))
        return stats.expon(loc=loc, scale=scale), {"loc": loc, "scale": scale}, 1
    if name == "gamma":
        a, loc, scale = stats.gamma.fit(x, floc=0)
        return stats.gamma(a, loc=0.0, scale=scale), {"a": float(a), "loc": 0.0, "scale": float(scale)}, 2
    if name == "weibull":
        c, loc, scale = stats.weibull_min.fit(x, floc=0)
        return stats.weibull_min(c, loc=0.0, scale=scale), {"c": float(c), "loc": 0.0, "scale": float(scale)}, 2
    if name == "lognorm":
        s, loc, scale = stats.lognorm.fit(x, floc=0)
        return stats.lognorm(s, loc=0.0, scale=scale), {"s": float(s), "loc": 0.0, "scale": float(scale)}, 2
    if name == "lomax":
        c, loc, scale = stats.lomax.fit(x, floc=0)
        return stats.lomax(c, loc=0.0, scale=scale), {"c": float(c), "loc": 0.0, "scale": float(scale)}, 2
    raise ValueError(f"unknown gap model {name}")


def fit_gap_models(x: np.ndarray, seed: int = 7, max_n: int = SUBSAMPLE_MAX) -> list[FitResult]:
    """Fit exponential, gamma, Weibull, lognormal and Lomax to positive gaps."""
    raw = _finite_positive(x)
    n = len(raw)
    if n < 30:
        return [_fit_fail(name, n, n, "too_few_observations") for name in GAP_MODELS]
    fit_x = subsample(raw, max_n, seed)
    ks_x = subsample(fit_x, min(KS_COMPARE_N, len(fit_x)), seed + 1)
    out = []
    for name in GAP_MODELS:
        try:
            dist, params, n_params = _frozen_and_params(name, fit_x)
            if any(not np.isfinite(v) or v <= 0 for k, v in params.items() if k != "loc"):
                out.append(_fit_fail(name, n, len(fit_x), f"nonpositive_params:{params}"))
                continue
            ll = _loglik(dist, fit_x)
            if not np.isfinite(ll):
                out.append(_fit_fail(name, n, len(fit_x), "nonfinite_loglik"))
                continue
            out.append(_fit_success(name, n, len(fit_x), n_params, params, ll, dist, ks_x, raw))
        except Exception as exc:  # noqa: BLE001
            out.append(_fit_fail(name, n, len(fit_x), f"{type(exc).__name__}:{exc}"))
    return out


def lr_statistic(full: FitResult, restricted: FitResult) -> dict[str, float]:
    """Likelihood-ratio statistic 2(ll_full - ll_rest) for nested models (same n_fit)."""
    if full.n_fit != restricted.n_fit or not np.isfinite(full.loglik) or not np.isfinite(restricted.loglik):
        return {"lr": float("nan"), "df": float("nan"), "pvalue": float("nan")}
    df = full.n_params - restricted.n_params
    if df <= 0:
        return {"lr": float("nan"), "df": float(df), "pvalue": float("nan")}
    lr = 2.0 * (full.loglik - restricted.loglik)
    p = float(stats.chi2.sf(max(lr, 0.0), df))
    return {"lr": float(lr), "df": float(df), "pvalue": p}


def best_fit(results: list[FitResult]) -> FitResult | None:
    ok = [r for r in results if r.status == "ok" and np.isfinite(r.aic)]
    if not ok:
        return None
    return min(ok, key=lambda r: (r.aic, r.bic))


def gap_threshold_diagnostics(x: np.ndarray, results: list[FitResult],
                              alpha: float = CURRENT_ALPHA) -> dict[str, float]:
    """Compare the current Poisson rule to the data and to each fitted tail."""
    x = _finite_positive(x)
    n = len(x)
    lam_hat = (1.0 / float(np.mean(x))) if n else float("nan")
    threshold = poisson_gap_threshold(lam_hat, alpha)
    out: dict[str, float] = {
        "n": n,
        "mean_s": float(np.mean(x)) if n else float("nan"),
        "lambda_hat": lam_hat,
        "poisson_threshold_s": threshold,
        "minus_ln_alpha_over_lambda": float(-np.log(alpha) / lam_hat) if lam_hat > 0 else float("nan"),
        "empirical_p_above_threshold": empirical_survival(x, threshold),
        "empirical_quantile_1_minus_alpha": empirical_quantile(x, 1.0 - alpha) if n >= 1 / alpha else float("nan"),
        "empirical_quantile_0.999": empirical_quantile(x, 0.999) if n >= 200 else float("nan"),
        "empirical_quantile_0.9999": empirical_quantile(x, 0.9999) if n >= 2_000 else float("nan"),
    }
    for result in results:
        if result.status != "ok":
            continue
        try:
            dist = _dist_from_params(result.model, result.params)
            out[f"{result.model}_p_above_threshold"] = float(dist.sf(threshold))
            out[f"{result.model}_quantile_1_minus_alpha"] = float(dist.ppf(1.0 - alpha))
        except Exception:
            out[f"{result.model}_p_above_threshold"] = float("nan")
            out[f"{result.model}_quantile_1_minus_alpha"] = float("nan")
    return out


def _dist_from_params(name: str, params: dict[str, float]):
    if name == "expon":
        return stats.expon(loc=params.get("loc", 0.0), scale=params["scale"])
    if name == "gamma":
        return stats.gamma(params["a"], loc=params.get("loc", 0.0), scale=params["scale"])
    if name == "weibull":
        return stats.weibull_min(params["c"], loc=params.get("loc", 0.0), scale=params["scale"])
    if name == "lognorm":
        return stats.lognorm(params["s"], loc=params.get("loc", 0.0), scale=params["scale"])
    if name == "lomax":
        return stats.lomax(params["c"], loc=params.get("loc", 0.0), scale=params["scale"])
    if name == "norm":
        return stats.norm(loc=params["loc"], scale=params["scale"])
    if name == "t":
        return stats.t(params["df"], loc=params["loc"], scale=params["scale"])
    if name == "laplace":
        return stats.laplace(loc=params["loc"], scale=params["scale"])
    raise ValueError(name)


def fit_return_models(x: np.ndarray, seed: int = 7, max_n: int = SUBSAMPLE_MAX) -> list[FitResult]:
    """Fit normal, Student-t and Laplace to log returns (location and scale free)."""
    raw = _finite(x)
    n = len(raw)
    if n < 50:
        return [_fit_fail(name, n, n, "too_few_observations") for name in RETURN_MODELS]
    fit_x = subsample(raw, max_n, seed)
    ks_x = subsample(fit_x, min(KS_COMPARE_N, len(fit_x)), seed + 1)
    out = []
    for name in RETURN_MODELS:
        try:
            if name == "norm":
                loc, scale = stats.norm.fit(fit_x)
                dist = stats.norm(loc=loc, scale=scale)
                params = {"loc": float(loc), "scale": float(scale)}
                n_params = 2
            elif name == "t":
                df, loc, scale = stats.t.fit(fit_x)
                dist = stats.t(df, loc=loc, scale=scale)
                params = {"df": float(df), "loc": float(loc), "scale": float(scale)}
                n_params = 3
            else:
                loc, scale = stats.laplace.fit(fit_x)
                dist = stats.laplace(loc=loc, scale=scale)
                params = {"loc": float(loc), "scale": float(scale)}
                n_params = 2
            if params.get("scale", 1) <= 0 or not all(np.isfinite(v) for v in params.values()):
                out.append(_fit_fail(name, n, len(fit_x), f"bad_params:{params}"))
                continue
            ll = _loglik(dist, fit_x)
            if not np.isfinite(ll):
                out.append(_fit_fail(name, n, len(fit_x), "nonfinite_loglik"))
                continue
            out.append(_fit_success(name, n, len(fit_x), n_params, params, ll, dist, ks_x, np.abs(raw) + 1e-16))
        except Exception as exc:  # noqa: BLE001
            out.append(_fit_fail(name, n, len(fit_x), f"{type(exc).__name__}:{exc}"))
    return out


def robust_z(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    med, mad = median_mad(x)
    scale = MAD_SCALE * max(mad, eps)
    if not np.isfinite(scale) or scale == 0:
        return np.full(len(x), np.nan)
    return (_finite(x) - med) / scale if len(_finite(x)) == len(x) else (x - med) / scale


def k12_false_alarm(x: np.ndarray, results: list[FitResult], k: float = CURRENT_K,
                    eps: float = 1e-4) -> dict[str, float]:
    """Empirical and model P(|z|>k) for the global robust-z used as a proxy for k=12.

    This is a *global* median/MAD standardisation, not the detector's trailing
    window of 500. The threshold on the return scale is
    :math:`\\mathrm{med} \\pm k\\cdot 1.4826\\cdot\\max(\\mathrm{MAD},\\varepsilon)`.
    """
    r = _finite(x)
    med, mad = median_mad(r)
    scale = MAD_SCALE * max(mad, eps)
    lo, hi = med - k * scale, med + k * scale
    out = {
        "n": len(r),
        "median": med,
        "mad": mad,
        "scale_1p4826_mad": scale,
        "threshold_lo": lo,
        "threshold_hi": hi,
        "empirical_p_abs_z_gt_k": float(np.mean((r < lo) | (r > hi))) if len(r) else float("nan"),
        "share_zero": float(np.mean(r == 0)) if len(r) else float("nan"),
        "share_abs_below_eps": float(np.mean(np.abs(r) <= eps)) if len(r) else float("nan"),
    }
    for result in results:
        if result.status != "ok":
            continue
        try:
            dist = _dist_from_params(result.model, result.params)
            out[f"{result.model}_p_abs_z_gt_k"] = float(dist.cdf(lo) + dist.sf(hi))
        except Exception:
            out[f"{result.model}_p_abs_z_gt_k"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Spreads (ticks)
# ---------------------------------------------------------------------------
def _geometric_shifted_p(x: np.ndarray) -> float:
    """P(S=k)=(1-p)^{k-1}p for k=1,2,...  MLE is p=1/mean."""
    return float(1.0 / np.mean(x))


def fit_spread_models(s: np.ndarray, seed: int = 7, max_n: int = SUBSAMPLE_MAX) -> list[FitResult]:
    """Empirical ticks plus shifted geometric, negative binomial, and lognormal."""
    raw = _finite_positive(s)
    n = len(raw)
    if n < 30:
        names = ("geometric_shifted", "nbinom", "lognorm")
        return [_fit_fail(name, n, n, "too_few_observations") for name in names]
    fit_x = subsample(raw, max_n, seed)
    ks_x = subsample(fit_x, min(KS_COMPARE_N, len(fit_x)), seed + 1)
    out: list[FitResult] = []

    # Shifted geometric on integer ticks (round if within 1e-6 of an integer).
    ticks = np.rint(fit_x)
    integerish = np.mean(np.abs(fit_x - ticks) < 1e-6)
    ticks = np.maximum(ticks, 1.0)
    try:
        p = _geometric_shifted_p(ticks)
        # scipy nbinom/geom: geom is number of trials until first success, support 1,2,...
        dist = stats.geom(p)
        ll = _loglik(dist, ticks, discrete=True)
        out.append(_fit_success("geometric_shifted", n, len(fit_x), 1, {"p": p}, ll, dist, ticks[: min(len(ticks), KS_COMPARE_N)], ticks,
                                detail=f"integerish={integerish:.4f}"))
    except Exception as exc:  # noqa: BLE001
        out.append(_fit_fail("geometric_shifted", n, len(fit_x), f"{type(exc).__name__}:{exc}"))

    try:
        # NB on S: scipy nbinom is failures before r successes, support 0,1,...
        # Fit to (S-1) so the minimum tick of 1 maps to 0.
        y = np.maximum(ticks - 1.0, 0.0)
        mean, var = float(np.mean(y)), float(np.var(y, ddof=1)) if len(y) > 1 else float(np.mean(y))
        if var > mean + 1e-12 and mean > 0:
            # method of moments, then polish with Nelder-Mead on loglik if needed
            r = mean * mean / (var - mean)
            p = r / (r + mean)
            r = max(r, 1e-3)
            p = min(max(p, 1e-9), 1 - 1e-9)

            def nll(theta):
                rr, pp = theta
                if rr <= 0 or not (0 < pp < 1):
                    return 1e300
                val = -np.sum(stats.nbinom.logpmf(y, rr, pp))
                return val if np.isfinite(val) else 1e300

            from scipy.optimize import minimize
            opt = minimize(nll, x0=(r, p), method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 1e-6, "fatol": 1e-6})
            r_hat, p_hat = float(opt.x[0]), float(opt.x[1])
            if r_hat <= 0 or not (0 < p_hat < 1):
                r_hat, p_hat = r, p
            dist = stats.nbinom(r_hat, p_hat)
            # For KS/tail we use the distribution of S = Y+1, approximated via a shifted frozen
            ll = _loglik(dist, y, discrete=True)
            # ppf/sf wrappers for S
            class _Shifted:
                def __init__(self, d):
                    self.d = d
                def cdf(self, t):
                    return self.d.cdf(np.asarray(t) - 1.0)
                def ppf(self, q):
                    return self.d.ppf(q) + 1.0
                def sf(self, t):
                    return self.d.sf(np.asarray(t) - 1.0)
            shifted = _Shifted(dist)
            out.append(_fit_success("nbinom", n, len(fit_x), 2, {"r": r_hat, "p": p_hat}, ll,
                                    shifted, ticks[: min(len(ticks), KS_COMPARE_N)], ticks,
                                    detail=f"fit_on_S_minus_1; integerish={integerish:.4f}"))
        else:
            out.append(_fit_fail("nbinom", n, len(fit_x), f"var_le_mean:mean={mean:.4g},var={var:.4g}"))
    except Exception as exc:  # noqa: BLE001
        out.append(_fit_fail("nbinom", n, len(fit_x), f"{type(exc).__name__}:{exc}"))

    try:
        s, loc, scale = stats.lognorm.fit(fit_x, floc=0)
        dist = stats.lognorm(s, loc=0.0, scale=scale)
        ll = _loglik(dist, fit_x)
        out.append(_fit_success("lognorm", n, len(fit_x), 2, {"s": float(s), "loc": 0.0, "scale": float(scale)},
                                ll, dist, ks_x, raw))
    except Exception as exc:  # noqa: BLE001
        out.append(_fit_fail("lognorm", n, len(fit_x), f"{type(exc).__name__}:{exc}"))
    return out


def spread_threshold_diagnostics(s: np.ndarray, results: list[FitResult]) -> dict[str, float]:
    s = _finite_positive(s)
    q95 = empirical_quantile(s, CURRENT_SPREAD_Q)
    rule = spread_rule_threshold(q95)
    out = {
        "n": len(s),
        "mean_ticks": float(np.mean(s)) if len(s) else float("nan"),
        "median_ticks": float(np.median(s)) if len(s) else float("nan"),
        "share_one_tick": float(np.mean(np.abs(s - 1.0) < 1e-6)) if len(s) else float("nan"),
        "q95_ticks": q95,
        "rule_threshold_ticks": rule,
        "empirical_p_above_rule": empirical_survival(s, rule),
        "empirical_q999": empirical_quantile(s, 0.999) if len(s) >= 200 else float("nan"),
        "empirical_q9999": empirical_quantile(s, 0.9999) if len(s) >= 2_000 else float("nan"),
    }
    for result in results:
        if result.status != "ok":
            continue
        try:
            if result.model == "geometric_shifted":
                dist = stats.geom(result.params["p"])
            elif result.model == "nbinom":
                dist = stats.nbinom(result.params["r"], result.params["p"])
                # S = Y+1
                out[f"{result.model}_p_above_rule"] = float(dist.sf(rule - 1.0))
                out[f"{result.model}_q95"] = float(dist.ppf(CURRENT_SPREAD_Q) + 1.0)
                out[f"{result.model}_q999"] = float(dist.ppf(0.999) + 1.0)
                continue
            elif result.model == "lognorm":
                dist = stats.lognorm(result.params["s"], loc=0.0, scale=result.params["scale"])
            else:
                continue
            out[f"{result.model}_p_above_rule"] = float(dist.sf(rule))
            out[f"{result.model}_q95"] = float(dist.ppf(CURRENT_SPREAD_Q))
            out[f"{result.model}_q999"] = float(dist.ppf(0.999))
        except Exception:
            out[f"{result.model}_p_above_rule"] = float("nan")
    return out
