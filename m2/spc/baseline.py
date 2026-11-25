r"""Phase I baselines and standardisation.

For bucket (time-of-day) $b$ and Phase I days $d \in D$ (in-control history: the normal-week
sessions of the same symbol, excluding the day being charted):

$$m_b = \operatorname{rollmed}_{\pm 7}\bigl(\operatorname{med}_{d\in D} y_{d,b}\bigr),\qquad
  s_b = \max\Bigl(1.4826\,\operatorname{rollmed}_{\pm 15}\bigl(\operatorname{med}_{d\in D}|y_{d,b}-m_b|\bigr),\; s_{\min}\Bigr),\qquad
  z_{t} = \frac{y_t - m_{b(t)}}{s_{b(t)}} .$$

Series: arrival rate $y = 2\sqrt{N + 3/8}$ (Anscombe; variance about 1 under Poisson, so
$s_{\min}=1$), spread $y = \ln S$ with $S$ the bucket-median quoted spread in ticks ($s_{\min}=0.3 \approx \ln(4/3)$, the smallest relative move a name quoting three ticks or fewer can make).

Minute buckets are autocorrelated (lag-1 correlation 0.2-0.9 on the normal week), so the charts
run on AR(1) residuals (Alwan & Roberts 1988; Montgomery ch. 10):
$e_t = (z_t - \hat\phi z_{t-1})/\sqrt{1-\hat\phi^2}$, with $\hat\phi$ the lag-1 correlation of the
Phase I $z$ (clipped at $\pm 5$).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

S_MIN = {"rate": 1.0, "spread": 0.3}
DIRECTION = {"rate": "down", "spread": "up"}


def transform(series: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "rate":
        return 2.0 * np.sqrt(series["n_records"].astype("float64") + 0.375)
    s = series["spread_ticks"].astype("float64")
    return np.log(s.where(s > 0))


def phase1(history: list[pd.DataFrame], kind: str, med_half=7, mad_half=15):
    """Baseline (m_b, s_b) indexed by bucket from a list of Phase I bucket series."""
    if not history:
        return None
    mat = pd.concat([transform(h, kind).set_axis(h["bucket"].to_numpy()) for h in history], axis=1)
    mat = mat.sort_index()
    med = mat.median(axis=1, skipna=True)
    m = med.rolling(2 * med_half + 1, center=True, min_periods=1).median()
    dev = mat.sub(m, axis=0).abs().median(axis=1, skipna=True)
    s = 1.4826 * dev.rolling(2 * mad_half + 1, center=True, min_periods=1).median()
    s = s.fillna(S_MIN[kind]).clip(lower=S_MIN[kind])
    return pd.DataFrame({"m": m, "s": s})


def standardise(series: pd.DataFrame, base: pd.DataFrame | None, kind: str) -> np.ndarray:
    y = transform(series, kind).to_numpy()
    if base is None:
        return np.full(len(y), np.nan)
    m = series["bucket"].map(base["m"]).to_numpy(dtype="float64")
    s = series["bucket"].map(base["s"]).to_numpy(dtype="float64")
    return (y - m) / s


def ar1_phi(zs: list[np.ndarray], clip=5.0) -> float:
    """Pooled lag-1 correlation of Phase I z (within days, clipped), limited to [0, 0.95]."""
    num = den = 0.0
    for z in zs:
        x = np.clip(np.asarray(z, dtype="float64"), -clip, clip)
        a, b = x[:-1], x[1:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            continue
        a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
        num += float((a * b).sum()); den += float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float(np.clip(num / den, 0.0, 0.95)) if den > 0 else 0.0


def prewhiten(z: np.ndarray, phi: float) -> np.ndarray:
    z = np.asarray(z, dtype="float64")
    prev = np.r_[np.nan, z[:-1]]
    e = (z - phi * np.where(np.isfinite(prev), prev, 0.0)) / np.sqrt(1 - phi * phi)
    return np.where(np.isfinite(prev), e, z)
