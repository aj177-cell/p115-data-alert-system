r"""Bayesian missing-interval check (Milestone 2, add-only).

Model, per key (publisher_id, instrument_id), on in-session gaps $g_t$ (the same
construction as ``missing_intervals``: consecutive ``ts_recv`` of one key, closed
time removed by the v3 ``SessionClock``):

* Normal regime: $g_t \mid \lambda \sim \mathrm{Exp}(\lambda)$ with a Gamma prior
  $\lambda \sim \mathrm{Gamma}(a_0, b_0)$, $a_0 = m_0$, $b_0 = m_0/\lambda_0$
  (weakly informative: $m_0$ pseudo-gaps at the pooled historical rate $\lambda_0$).
* Conjugate update with a discount (forgetting) factor $\delta$, the steady
  Gamma-Poisson model of West & Harrison:
  $a_t = \delta a_{t-1} + w_t$, $b_t = \delta b_{t-1} + w_t g_t$, effective window
  $1/(1-\delta)$ gaps. $w_t = 1$ in the first pass; in the robust second pass
  $w_t = 1 - P(\text{outage}\mid g_t)$ so that outages do not drag the rate down.
* Posterior predictive of the next gap (uses $a_{t-1}, b_{t-1}$) is Lomax:
  $P(G>g) = (b/(b+g))^{a}$, density $f_N(g) = \frac{a}{b}(1+g/b)^{-(a+1)}$.
* Outage regime: log-uniform gap length on $[d_{\min}, d_{\max}]$,
  $f_O(g) = 1/(g \ln(d_{\max}/d_{\min}))$, prior probability $\pi$ per gap.
* $P(\text{outage}\mid g) = \pi f_O / (\pi f_O + (1-\pi) f_N)$. A gap is a
  *candidate* when this is at least ``posterior_threshold``.

Nothing here changes ``p115_detectors``; this module only reads its calendar and
``SessionClock``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lfilter

NS = 1_000_000_000


@dataclass
class BayesGapConfig:
    discount: float = 0.998
    """delta. Effective window 1/(1-delta) = 500 gaps, the same as rate_window."""
    prior_strength: float = 20.0
    """m0: pseudo-gaps in the Gamma prior (weak next to the ~500 effective gaps)."""
    prior_rate: float | None = None
    """lambda0 (arrivals per in-session second). None: the key's own first-pass mean rate."""
    outage_prior: float = 1e-6
    """pi: prior probability that a given gap is an outage."""
    min_outage_s: float = 5.0
    """d_min: shortest silence the outage hypothesis covers (in-session seconds)."""
    max_outage_s: float = 86_400.0
    """d_max: upper end of the log-uniform outage law."""
    posterior_threshold: float = 0.5
    """Flag a candidate when P(outage | g) >= this."""
    robust: bool = True
    """Second pass that down-weights likely outages in the conjugate update."""
    high_gap_s: float = 60.0
    """Severity 'high' above this (same meaning as max_gap_s in v3); no effect on flagging."""


@dataclass
class GapPosterior:
    gap_s: np.ndarray
    a: np.ndarray
    b: np.ndarray
    tail: np.ndarray
    p_outage: np.ndarray
    candidate: np.ndarray
    valid: np.ndarray


def _discounted(x: np.ndarray, delta: float, init: float) -> np.ndarray:
    """y_t = delta*y_{t-1} + x_t with y_{-1}=init; returns y (inclusive of x_t)."""
    y, _ = lfilter([1.0], [1.0, -delta], x, zi=[delta * init])
    return y


def _prior_state(gaps: np.ndarray, cfg: BayesGapConfig) -> tuple[float, float]:
    lam0 = cfg.prior_rate
    if lam0 is None or not np.isfinite(lam0) or lam0 <= 0:
        tot = float(np.sum(gaps))
        lam0 = len(gaps) / tot if tot > 0 else 1.0
    a0 = float(cfg.prior_strength)
    return a0, a0 / float(lam0)


def log_lomax_pdf(g, a, b):
    return np.log(a) - np.log(b) - (a + 1.0) * np.log1p(g / b)


def log_lomax_sf(g, a, b):
    return -a * np.log1p(g / b)


def outage_log_pdf(g, cfg: BayesGapConfig):
    g = np.asarray(g, dtype="float64")
    norm = np.log(np.log(cfg.max_outage_s / cfg.min_outage_s))
    with np.errstate(divide="ignore"):
        out = -np.log(np.maximum(g, 1e-300)) - norm
    return np.where(g >= cfg.min_outage_s, out, -np.inf)


def outage_posterior(g, a, b, cfg: BayesGapConfig):
    """P(outage | g) under the two-component mixture (vectorised, log-space)."""
    lo = outage_log_pdf(g, cfg)
    ln = log_lomax_pdf(g, a, b)
    with np.errstate(over="ignore", invalid="ignore"):
        logit = np.log(cfg.outage_prior) - np.log1p(-cfg.outage_prior) + lo - ln
        p = np.where(np.isneginf(lo), 0.0, 1.0 / (1.0 + np.exp(-np.clip(logit, -700, 700))))
    return p


def _one_key(gaps: np.ndarray, cfg: BayesGapConfig):
    """gaps: in-session gaps of one key in time order (first record excluded)."""
    n = len(gaps)
    if n == 0:
        z = np.array([], dtype="float64")
        return z, z, z, z
    a0, b0 = _prior_state(gaps, cfg)
    w = np.ones(n)
    for _ in range(2 if cfg.robust else 1):
        a_post = _discounted(w, cfg.discount, a0)
        b_post = _discounted(w * gaps, cfg.discount, b0)
        # predictive for gap t uses the posterior after t-1
        a_prev = np.r_[a0, a_post[:-1]]
        b_prev = np.r_[b0, b_post[:-1]]
        b_prev = np.maximum(b_prev, 1e-12)
        p = outage_posterior(gaps, a_prev, b_prev, cfg)
        w = 1.0 - p
    tail = np.exp(log_lomax_sf(gaps, a_prev, b_prev))
    return a_prev, b_prev, tail, p


def posterior_from_clock(s_ns: np.ndarray, key: np.ndarray, cfg: BayesGapConfig | None = None,
                         prior_rates: dict[Any, float] | None = None) -> GapPosterior:
    """Arrays sorted by key then time. ``s_ns``: in-session clock (int ns). ``key``: int id."""
    cfg = cfg or BayesGapConfig()
    n = len(s_ns)
    gap = np.r_[0.0, np.diff(s_ns).astype("float64")] / NS if n else np.array([])
    same = np.r_[False, key[1:] == key[:-1]] if n else np.array([], bool)
    a = np.full(n, np.nan); b = np.full(n, np.nan); tail = np.full(n, np.nan); p = np.zeros(n)
    if n:
        bounds = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi - lo < 2:
                continue
            c = cfg
            if prior_rates is not None and key[lo] in prior_rates:
                c = replace(cfg, prior_rate=prior_rates[key[lo]])
            aa, bb, tt, pp = _one_key(gap[lo + 1:hi], c)
            a[lo + 1:hi], b[lo + 1:hi], tail[lo + 1:hi], p[lo + 1:hi] = aa, bb, tt, pp
    cand = same & (p >= cfg.posterior_threshold)
    return GapPosterior(gap, a, b, tail, p, cand, same)


def bayes_missing_intervals(df: pd.DataFrame, cfg: BayesGapConfig | None = None, calendar=None,
                            v3=None) -> pd.DataFrame:
    """Bayesian counterpart of ``detectors.missing_intervals`` on a raw or normalized frame.

    Returns the v3 flag table (check='missing_intervals', subtype='gap_bayes'), with
    ``value`` = gap (s), ``threshold`` = P(outage | gap), and the tail probability in ``detail``.
    """
    from integration.combine import load_v3
    v3 = v3 or load_v3()
    det = v3.detectors
    cfg = cfg or BayesGapConfig()
    cal = calendar if calendar is not None else v3.config.MissingIntervalsConfig().calendar
    norm = det._ensure_norm(df)
    norm = norm[norm["ts_recv_ns"] != np.iinfo(np.int64).min]
    if norm.empty:
        return det.empty_flags()
    norm = norm.sort_values(det.KEY + ["ts_recv_ns"], kind="stable").reset_index(drop=True)
    t = norm["ts_recv_ns"].to_numpy()
    clock = det.SessionClock.build(cal, int(t.min()), int(t.max()))
    s = clock.clock(t)
    key = det._key_id(norm)
    post = posterior_from_clock(s, key, cfg)
    idx = np.flatnonzero(post.candidate)
    detail = [f"gap_start={pd.Timestamp(t[i - 1], unit='ns', tz='UTC').isoformat()} "
              f"lambda_post={post.a[i] / post.b[i]:.3g}/s shape={post.a[i]:.3g} "
              f"tail={post.tail[i]:.3g} p_outage={post.p_outage[i]:.3g}" for i in idx]
    sev = np.where(post.gap_s[idx] > cfg.high_gap_s, "high", "medium")
    return det.make_flags(norm.iloc[idx], "missing_intervals", "gap_bayes", sev,
                          post.gap_s[idx], post.p_outage[idx], detail)


def poisson_rule_from_clock(s_ns: np.ndarray, key: np.ndarray, alpha: float = 1e-6, window: int = 500,
                            min_gap_s: float = 5.0, max_gap_s: float = 60.0) -> np.ndarray:
    """Re-implementation of the v3 adaptive Poisson rule (model='poisson') for gap-level overlap.

    Same arithmetic as detectors.missing_intervals; checked against the stored v3 flag counts.
    """
    n = len(s_ns)
    if n == 0:
        return np.zeros(0, bool)
    gap = np.r_[0.0, np.diff(s_ns).astype("float64")] / NS
    same = np.r_[False, key[1:] == key[:-1]]
    s_series = pd.Series(s_ns.astype("float64"))
    prev = s_series.groupby(key).shift(1)
    prev_n = s_series.groupby(key).shift(window + 1)
    span = (prev - prev_n).to_numpy() / NS
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(span > 0, window / span, np.nan)
        raw = -np.log(alpha) / lam
    thr = np.where(np.isfinite(raw), np.clip(raw, min_gap_s, max_gap_s), max_gap_s)
    return same & (gap > thr)
