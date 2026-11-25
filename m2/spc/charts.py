r"""One-sided control charts on a standardised series $z_t$ and their in-control ARL.

All charts watch one direction ("up" = spread widening, "down" = arrival rate falling);
for "down" the charts run on $-z_t$. After a signal the statistic restarts at 0, so a
series yields a count of signals (Montgomery, *Introduction to Statistical Quality Control*,
ch. 9).

* Shewhart:  signal when $z_t > c$.
* CUSUM:     $C_t = \max(0, C_{t-1} + z_t - k)$, signal when $C_t > h$.
* EWMA:      $E_t = \lambda z_t + (1-\lambda) E_{t-1}$, signal when
             $E_t > L\,\sigma_{E,t}$, $\sigma_{E,t}^2 = \frac{\lambda}{2-\lambda}\bigl(1-(1-\lambda)^{2t}\bigr)$.

NaN inputs (no data in the bucket) leave the statistic unchanged and cannot signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class ChartParams:
    shewhart_c: float
    cusum_k: float
    cusum_h: float
    ewma_lambda: float
    ewma_L: float


def _prep(z, direction):
    z = np.asarray(z, dtype="float64")
    return -z if direction == "down" else z


def shewhart(z, c, direction="up"):
    x = _prep(z, direction)
    return np.nan_to_num(x, nan=-np.inf) > c


def cusum(z, k, h, direction="up", return_stat=False):
    x = _prep(z, direction)
    stat = np.zeros(len(x)); sig = np.zeros(len(x), bool); c = 0.0
    for i, v in enumerate(x):
        if v == v:
            c = max(0.0, c + v - k)
            if c > h:
                sig[i] = True; stat[i] = c; c = 0.0; continue
        stat[i] = c
    return (sig, stat) if return_stat else sig


def ewma(z, lam, L, direction="up", return_stat=False):
    x = _prep(z, direction)
    stat = np.zeros(len(x)); sig = np.zeros(len(x), bool); e = 0.0; n = 0
    for i, v in enumerate(x):
        if v == v:
            n += 1
            e = lam * v + (1 - lam) * e
            sd = np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * n)))
            if e > L * sd:
                sig[i] = True; stat[i] = e; e = 0.0; n = 0; continue
        stat[i] = e
    return (sig, stat) if return_stat else sig


def run_charts(z, params: ChartParams, direction):
    return {
        "shewhart": shewhart(z, params.shewhart_c, direction),
        "cusum": cusum(z, params.cusum_k, params.cusum_h, direction),
        "ewma": ewma(z, params.ewma_lambda, params.ewma_L, direction),
    }


# ---------------------------------------------------------------------------
# In-control ARL for i.i.d. N(0,1) input (Brook & Evans 1972 Markov chain)
def cusum_arl(k, h, mu=0.0, m=400):
    """ARL of the upper CUSUM with reference k and limit h for N(mu,1) data."""
    w = h / m
    centers = (np.arange(m) + 0.5) * w  # state j covers [j w, (j+1) w); state 0 includes the reset at 0
    centers[0] = 0.0
    edges = np.arange(m + 1) * w
    P = np.empty((m, m))
    for i, c in enumerate(centers):
        cdf = norm.cdf(edges - c + k - mu)
        P[i, 0] = cdf[1]
        P[i, 1:] = np.diff(cdf)[1:]
    return float(np.linalg.solve(np.eye(m) - P, np.ones(m))[0])


def ewma_arl(lam, L, mu=0.0, m=401, lower_sd=6.0):
    """ARL of the one-sided (upper) EWMA with asymptotic limit L*sigma_E (Lucas & Saccucci 1990 chain)."""
    s = np.sqrt(lam / (2 - lam))
    ucl, lcl = L * s, -lower_sd * s  # lower edge truncated far below (statistic is unrestricted)
    edges = np.linspace(lcl, ucl, m + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    P = np.empty((m, m))
    for i, c in enumerate(centers):
        z_edges = (edges - (1 - lam) * c) / lam
        cdf = norm.cdf(z_edges - mu)
        cdf[0] = 0.0  # mass below the truncation folds into the lowest cell
        P[i] = np.diff(cdf)
    start = np.argmin(np.abs(centers))
    return float(np.linalg.solve(np.eye(m) - P, np.ones(m))[start])


def calibrate(arl0=370.0, k=0.5, lam=0.2):
    """Limits giving the target in-control ARL for i.i.d. N(0,1) z (one-sided)."""
    c = float(norm.ppf(1 - 1 / arl0))
    h = brentq(lambda x: cusum_arl(k, x) - arl0, 1.0, 12.0)
    L = brentq(lambda x: ewma_arl(lam, x) - arl0, 1.0, 5.0)
    return ChartParams(c, k, h, lam, L)


def simulated_arl(chart, params: ChartParams, mu=0.0, n_series=400, length=20_000, seed=0):
    """Monte Carlo ARL check (used in tests): mean run length to first signal."""
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_series):
        z = rng.standard_normal(length) + mu
        if chart == "cusum":
            s = cusum(z, params.cusum_k, params.cusum_h)
        elif chart == "ewma":
            s = ewma(z, params.ewma_lambda, params.ewma_L)
        else:
            s = shewhart(z, params.shewhart_c)
        i = np.flatnonzero(s)
        runs.append(i[0] + 1 if len(i) else length)
    return float(np.mean(runs))
