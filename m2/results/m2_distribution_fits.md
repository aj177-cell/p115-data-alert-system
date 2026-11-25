# Milestone 2: distribution fits for gaps, spreads and returns

Add-only study on a copy of the combined-plus tree. Detector thresholds in `p115_detectors/` were **not** changed. A flagged row remains a **candidate**, not a confirmed data error.

## Data and method

The fits use the 95 symbol-day Level-1 mbp-1 pull already on the box (`/workspace/p115/data_pipeline/`), the same list as `p115_detectors/analyze.py::tasks()`: ten XNAS names (AAPL, MSFT, NVDA, SPY, QQQ, CRWV, NBIS, LWAY, PLAB, ADTN) on eight sessions, plus ES.c.0 and CL.c.0 on the normal week and the CPI day, and ES around the degraded 2026-08-29 Saturday.

In-session time uses the v3 `SessionClock` / `SessionCalendar` presets (equities RTH 09:30–16:00 America/New_York, with the 2025-11-28 13:00 ET early close; CME Globex hours with the 16:00–17:00 CT break). Equities are fitted in one regime (RTH). CME sessions are split into **day** (08:30–16:00 CT) and **overnight** (the rest of the Globex window). Day types are: normal week (2026-10-05–09, pooled across the five days), CPI (2026-10-14), half-day (2025-11-28), VIX-high (2026-09-16), degraded Saturday (2026-08-29_weekend), and around-degraded (2026-08-28 and 08-31, also pooled when both exist).

Inter-arrival gaps are in-session clock differences between consecutive `F_LAST` records of one `(publisher_id, instrument_id)`, the same construction as `missing_intervals`. Spreads are $(a-b)/\delta$ on `F_LAST` two-sided quotes, with $\delta=0.01$ for XNAS and the definition tick for ES (0.25) and CL (0.01). Log mid returns use consecutive in-session mids; trade returns use `action=='T'` and drop side `N`. Zero and negative spreads are reported as a share and excluded from the wide-spread fits.

Each family is fitted by MLE (`scipy.stats`, loc fixed at 0 for positive gap models). We report AIC, BIC, the Kolmogorov–Smirnov $D$ (the $p$-value treats parameters as known, so it is anti-conservative after MLE and is not used as a formal test), and a tail mean-absolute error of log-quantiles in the upper 1%. Nested gamma vs exponential gaps also get a likelihood-ratio test. Series longer than 100,000 observations are randomly subsampled to 100,000 for MLE and KS (seed 7); empirical tail rates and quantiles use the full series, except that the $1-10^{-6}$ empirical quantile is left blank unless $n \ge 10^{6}$.

The current v3 rules are compared **statically** (global median, MAD, $Q_{0.95}$, and $\hat\lambda=1/\bar g$). The live detectors use a trailing window (500 arrivals, 15-minute spread quantile, rolling robust $z$). Suggestions below are therefore directional, not drop-in replacements.

All 95 symbol-days were loaded. That produces 126 instrument–regime units (equities have one RTH regime; futures have day and overnight). Two of those units are empty: the 2026-08-29_weekend ES window contains only 18 raw L1 rows, all on Sunday 2026-08-30, and almost all of them fall in the 16:00–17:00 CT Globex break. After SessionClock, there are no in-session prints, so the degraded-Saturday regime could not be fitted. The 08-28 and 08-31 around-degraded sessions were fitted (and pooled). The remaining 124 units have fits.

## 1. Inter-arrival times

The v3 missing-interval rule, under a homogeneous Poisson assumption (exponential gaps), flags

$$g > \min\bigl(\max(-\ln\alpha/\hat\lambda,\; 5\,\mathrm{s}),\; 60\,\mathrm{s}\bigr),\qquad \alpha=10^{-6}.$$

Under an exponential law, $-\ln\alpha/\hat\lambda$ is exactly the $1-\alpha$ quantile, and $P(G>g)=e^{-\lambda g}$. The 5-second floor and 60-second cap were chosen on the pilot, not from a fit.

On the 16 pooled (or single, if not pooled) instrument–regime–day-type units, AIC prefers exponential on 0, gamma on 0, Weibull on 0, lognormal on 3, and Lomax on 13.
 The gamma-vs-exponential likelihood-ratio test rejects a pure exponential at the 1% level in 16 of 16 units where both fits succeeded.

The unclamped Poisson quantile $-\ln\alpha/\hat\lambda$ has median 0.5814 s (range 0.07165–71.16 s). After the 5 s / 60 s clamp, the threshold has median 5 s; 13 units would sit on the 5 s floor and 1 on the 60 s cap.

At the current clamped threshold $T$, the **empirical** exceedance rate has median 7.09e-05 (mean 0.00339). That is far above $\alpha=10^{-6}$ whenever $T$ is sitting on the 5 s floor: the floor, not the Poisson tail, is doing the work on liquid names.

| symbol | day_type | regime | n | best | lambda_hat | T_s | emp_P(G>T) | exp_P(G>T) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CL.c.0 | normal_week | cme_day | 4232592 | lomax | 31.4 | 5 | 0.000132 | 9.99e-70 |
| CL.c.0 | normal_week | cme_overnight | 4494706 | lomax | 16.1 | 5 | 0.00106 | 2e-37 |
| ES.c.0 | around_degraded | cme_day | 10412491 | lomax | 193 | 5 | 8.64e-07 | 0 |
| ES.c.0 | normal_week | cme_day | 19089131 | lomax | 141 | 5 | 9.43e-07 | 1.33e-301 |
| ES.c.0 | around_degraded | cme_overnight | 2381625 | lomax | 22.8 | 5 | 3.65e-05 | 1.19e-50 |
| ES.c.0 | normal_week | cme_overnight | 9510529 | lomax | 34.1 | 5 | 6.32e-05 | 8.99e-73 |
| AAPL | normal_week | equity_rth | 2901457 | lomax | 24.8 | 5 | 1.28e-05 | 1.17e-53 |
| ADTN | normal_week | equity_rth | 130736 | lomax | 1.12 | 12.4 | 0.0146 | 8.73e-07 |
| CRWV | normal_week | equity_rth | 1464580 | lomax | 12.5 | 5 | 0.000188 | 1.03e-27 |
| LWAY | normal_week | equity_rth | 22714 | lognorm | 0.194 | 60 | 0.0207 | 8.73e-06 |
| MSFT | normal_week | equity_rth | 2326042 | lomax | 19.9 | 5 | 7.87e-05 | 3.32e-44 |
| NBIS | normal_week | equity_rth | 1824829 | lognorm | 15.6 | 5 | 0.000497 | 1.09e-34 |
| NVDA | normal_week | equity_rth | 8516872 | lomax | 72.8 | 5 | 0 | 1.31e-158 |
| PLAB | normal_week | equity_rth | 88863 | lomax | 0.76 | 18.2 | 0.0169 | 1e-06 |
| QQQ | normal_week | equity_rth | 11837018 | lognorm | 101 | 5 | 0 | 1.84e-218 |
| SPY | normal_week | equity_rth | 8081719 | lomax | 69.1 | 5 | 0 | 1.84e-151 |

**Is exponential adequate?** No, not as a complete law, once the sample is large. On the 16 pooled units above, AIC prefers Lomax 13 times and lognormal 3 times (never exponential, gamma, or Weibull). Across all 124 fitted units the same pattern holds (Lomax 100, lognormal 20, gamma 2, Weibull 2, exponential 0), and the gamma-vs-exponential LR test rejects shape $=1$ at the 1% level in 124 of 124. The exponential still describes the centre of liquid series better than the far tail: Lomax can chase the heaviest gaps but should not be adopted as the default detector model (see suggestion 3). Time-of-day variation, which the optional `model='tod'` already implements in the detector, is visible in the ES day vs overnight split (overnight gaps are longer, $\hat\lambda$ is smaller).

Where $n\ge 10^{6}$, the empirical $1-10^{-6}$ gap quantile can be compared directly to the 5 s floor. On the pooled normal week it is 6.91 s (AAPL), 11.0 s (MSFT), 3.06 s (SPY), 2.17 s (QQQ), 1.88 s (NVDA), and 4.93 s (ES day). So for the most liquid names the floor is in the same ballpark as a literal $10^{-6}$ empirical tail; for AAPL/MSFT the empirical $P(G>5\,\mathrm{s})$ is $1.3\times 10^{-5}$ and $7.9\times 10^{-5}$, above the nominal $\alpha$. Quiet names are a different regime: LWAY sits on the 60 s cap with empirical $P(G>60\,\mathrm{s})=0.021$, and PLAB/ADTN sit on 18 s / 12 s with exceedance rates around 1.5%. CPI and VIX-high days raise $\hat\lambda$ (AAPL $49.4$ and $30.0$ vs $24.8$ in the normal week; ES day $241$ vs $141$) but the clamped $T$ stays at 5 s on liquid names.

**Threshold suggestions (not applied):**

1. Keep the Poisson form as a baseline, but treat the 5 s floor as the actual policy for liquid RTH names: $-\ln(10^{-6})/\hat\lambda$ is often well below 5 s, so $\alpha=10^{-6}$ is not the operating false-alarm rate.
2. If a probability interpretation is wanted, set the floor first, then report the **empirical** $P(G>T)$ (and the fitted gamma/Lomax tail) rather than claiming $10^{-6}$. A data-driven floor could be a high quantile of liquid-name RTH gaps (for example around the 0.999–0.9999 quantile on AAPL/MSFT/SPY in the normal week); see the table and the gap tail-QQ plots.
3. Do not switch the default to `overdispersed` / Lomax on the basis of AIC alone: the v3 sensitivity run already found that this model missed 41–56% of injected 10 s / 20 s outages. Any new tail model should be re-checked on those injected outages before it is adopted.
4. Fit day and overnight CME sessions separately, or keep a time-of-day rate, rather than one $\hat\lambda$ for the whole Globex day.

## 2. Spreads in ticks

The live rule flags a *run* lasting at least 0.5 s when

$$S > \max(3\,Q_{0.95},\; Q_{0.95}+3),$$

and $Q_{0.95}$ is a trailing 15-minute quantile. Here we compare that formula on the **global** $Q_{0.95}$ of each unit (an upper bound on how aggressive the static version would be).

Across 16 pooled/comparable units, the global $Q_{0.95}$ has median 3.5 ticks. ES is typically 1 tick at the 95th percentile in the day session ($T=4$); large-cap XNAS names in this L1 sample are wider (AAPL normal-week median 3 ticks, $Q_{0.95}=9$, $T=27$; MSFT $Q_{0.95}=24$, $T=72$). The static rule threshold $\max(3Q,Q+3)$ has median 10.5 ticks. The empirical share of quotes above that static threshold has median 0.000325.

| symbol | day_type | regime | n | share_1tick | Q95 | T | emp_P(S>T) | best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | normal_week | equity_rth | 2901462 | 0.0824 | 9 | 27 | 0.000656 | lognorm |
| ADTN | normal_week | equity_rth | 130741 | 0.645 | 3 | 9 | 0.0016 | lognorm |
| CL.c.0 | normal_week | cme_day | 4236884 | 0.548 | 2 | 6 | 0.00123 | lognorm |
| CL.c.0 | normal_week | cme_overnight | 4499460 | 0.393 | 2 | 6 | 0.000546 | lognorm |
| CRWV | normal_week | equity_rth | 1464585 | 0.15 | 8 | 24 | 0.000531 | lognorm |
| ES.c.0 | around_degraded | cme_day | 10935617 | 0.952 | 1 | 4 | 9.41e-05 | lognorm |
| ES.c.0 | normal_week | cme_day | 20436794 | 0.954 | 1 | 4 | 3.09e-05 | lognorm |
| ES.c.0 | around_degraded | cme_overnight | 2442230 | 0.903 | 2 | 6 | 4.83e-05 | lognorm |
| ES.c.0 | normal_week | cme_overnight | 9791094 | 0.913 | 2 | 6 | 1.36e-05 | lognorm |
| LWAY | normal_week | equity_rth | 22719 | 0.00753 | 48 | 144 | 4.4e-05 | nbinom |
| MSFT | normal_week | equity_rth | 2326047 | 0.0116 | 24 | 72 | 0.000424 | lognorm |
| NBIS | normal_week | equity_rth | 1824834 | 0.00428 | 29 | 87 | 0.000224 | nbinom |
| NVDA | normal_week | equity_rth | 8516877 | 0.354 | 3 | 9 | 0.0011 | lognorm |
| PLAB | normal_week | equity_rth | 88868 | 0.0925 | 9 | 27 | 0.00132 | lognorm |
| QQQ | normal_week | equity_rth | 11837023 | 0.122 | 5 | 15 | 0.00016 | lognorm |
| SPY | normal_week | equity_rth | 8081724 | 0.281 | 4 | 12 | 0.000227 | lognorm |

AIC often prefers a continuous lognormal on spreads because the right tail is heavier than a shifted geometric; a geometric or negative binomial still matches the discrete mass at 1–3 ticks better, especially on ES (about 95% one-tick in the day session). Illiquid names (LWAY, PLAB, ADTN) have a much heavier spread distribution: $Q_{0.95}$ already sits well above one tick (LWAY 48 ticks in the normal week), so $\max(3Q,Q+3)$ becomes very large and the static rule almost never fires. That is conservative, not tight. On the CPI day, ES day-session $Q_{0.95}$ rises from 1 to 2 ticks ($T$ from 4 to 6) and AAPL $Q_{0.95}$ is 7 ticks vs 9 in the normal week; VIX-high SPY $Q_{0.95}$ is 5 vs 4. Those moves look like genuine wider books, which the literature appendix already flags as a caveat for announcement windows.

**Threshold suggestions (not applied):**

1. Keep the trailing 15-minute $Q_{0.95}$ (the global quantile is a poor substitute on mixed quiet/busy hours).
2. For liquid names with $Q_{0.95}\approx 1$, $T=\max(3,4)=4$ ticks is a mild bar; a duration rule of 0.5 s is doing a lot of the work. If the team wants fewer candidates on quiet names, consider also requiring a minimum tick excess that scales with liquidity, or a reference-spread (SPY/ES) move, rather than lowering $Q$ itself.
3. Do not tighten the multiplier on CPI / FOMC-like days without an announcement window: a wider spread can be genuine.

## 3. Log mid and trade returns

The v3 extreme-price rule flags

$$z_t=\frac{r_t-\widetilde r_W}{1.4826\,\max(\mathrm{MAD}_W(r),\varepsilon)},\qquad |z_t|>k=12,$$

with $W=500$, $\varepsilon=\max(\delta/p,10^{-4})$, and $r=\ln(p_t/p_{t-1})$ on trades and mids. The comparison here uses a **global** median/MAD (a simpler proxy) and also fits normal, Student-$t$ and Laplace laws to all mid returns and to the nonzero mid returns. Tick discreteness is material: a large share of consecutive mids do not move, so the distribution has a point mass at 0 that no continuous law can fit. Nonzero returns still sit on a discrete tick grid.

On nonzero mid returns, AIC prefers Student-$t$ in 11 units, Laplace in 0, and a normal in 5 of 16.
 The Hill tail index $\hat\alpha$ on $|r|$ (nonzero) has median 2.5 (a finite $\alpha$ near 2–4 is consistent with heavy tails; values should be read with the usual Hill-$k$ caution).
 Empirical $P(|z|>12)$ under the **global** robust $z$ (all mid returns, including zeros) has median 0. On nonzero mids only, the median is 0.
 The fitted $t$ law's $P(|z|>12)$ has median 1.6e-06.

| symbol | day_type | regime | share_r=0 | n_unique_|r| | best_nz | Hill_alpha | emp_P(|z|>12)_all | emp_P(|z|>12)_nz | t_P(|z|>12)_nz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | normal_week | equity_rth | 0.73 | 14295 | t | 3.514 | 0 | 0 | 0.000115 |
| ADTN | normal_week | equity_rth | 0.933 | 352 | t | 1.488 | 0.000872 | 0 | 1.21e-16 |
| CL.c.0 | normal_week | cme_day | 0.848 | 3634 | t | 2.285 | 4.48e-06 | 2.17e-05 | 7.69e-29 |
| CL.c.0 | normal_week | cme_overnight | 0.824 | 3845 | t | 3.673 | 4.44e-07 | 2.53e-06 | 4.74e-101 |
| CRWV | normal_week | equity_rth | 0.748 | 11878 | t | 5.432 | 0 | 0 | 6.75e-08 |
| ES.c.0 | around_degraded | cme_day | 0.945 | 1618 | norm | 1.478 | 0 | 0 | 9.04e-05 |
| ES.c.0 | normal_week | cme_day | 0.949 | 1986 | norm | 2.152 | 0 | 0 | 0 |
| ES.c.0 | around_degraded | cme_overnight | 0.925 | 1033 | norm | 2.703 | 0 | 0 | 8.96e-05 |
| ES.c.0 | normal_week | cme_overnight | 0.929 | 1802 | norm | 1.548 | 0 | 0 | 0 |
| LWAY | normal_week | equity_rth | 0.552 | 1440 | t | 2.051 | 0.0222 | 0.00354 | 0.00441 |
| MSFT | normal_week | equity_rth | 0.522 | 42135 | t | 3.297 | 0 | 0 | 7.67e-06 |
| NBIS | normal_week | equity_rth | 0.387 | 67805 | t | 4.042 | 1.15e-05 | 1.79e-05 | 0.000551 |
| NVDA | normal_week | equity_rth | 0.876 | 9637 | t | 2.473 | 0 | 0 | 3.55e-34 |
| PLAB | normal_week | equity_rth | 0.816 | 1391 | t | 1.904 | 0.000855 | 0.000367 | 3.12e-06 |
| QQQ | normal_week | equity_rth | 0.816 | 15314 | norm | 3.27 | 0 | 0 | 1.68e-05 |
| SPY | normal_week | equity_rth | 0.841 | 8009 | t | 2.488 | 0 | 0 | 7.81e-80 |

A Gaussian with $k=12$ implies a vanishing false-alarm rate ($\Phi(-12)$ is on the order of $10^{-33}$). That is why $k=12$ was chosen: tick-discrete, heavy-tailed returns are not normal, and a $3$–$6$ ‘sigma’ rule on a MAD scale would flood the output. Under a Student-$t$ with a small degrees of freedom, $P(|z|>12)$ is larger than the Gaussian value but still typically tiny at $k=12$; most mid-return candidates, if any, will come from genuine jumps or from the rolling window sitting on a quiet MAD floor, which the detector already floors with $\varepsilon$.

**Threshold suggestions (not applied):**

1. Keep $k=12$ as a conservative screen. The fits do **not** support lowering $k$ to a Gaussian $3$–$6$ without a different scale (for example a trailing quantile of $|r|$ or a $t$ scale).
2. Report, next to each extreme-price candidate, the local MAD floor status (whether $\varepsilon$ was binding) and the raw tick move. That is more useful than a probability under a poorly specified continuous law.
3. Continue to exclude auction side `N` prints. Zero mid changes should stay in the rolling MAD (they shrink it); they should not be treated as a continuous density at zero.

## Outputs and limits

- Fit rows: `m2/results/fits.csv` (and `fits_gaps.csv`, `fits_spreads.csv`, `fits_mid_returns.csv`, …)
- Per-unit diagnostics: `m2/results/summaries.csv`
- Plots: `m2/results/plots/` (gap histograms and upper-1% QQ, spread histograms with the static rule line, return histograms with normal/$t$/Laplace overlays)

Limits: no full exchange holiday calendar (same as v3); CME day vs overnight uses 08:30 CT as a documented convention, not a venue calendar API; global $Q_{0.95}$ and global MAD are not the live trailing windows; KS $p$-values are not formal after MLE; the $1-10^{-6}$ empirical gap quantile is unidentified when $n<10^{6}$; the degraded-Saturday window had no in-session data; MBO / Level-2 is not used. These fits describe the 95 symbol-day pilot pull, not a probability sample of all high-frequency data.

## How to rerun

```bash
python -m pip install -r requirements-m2.txt
python m2/run_distribution_fits.py --data-root /workspace/p115/data_pipeline --output-dir m2/results
python -m unittest integration.tests.test_m2_distributions m2.tests.test_distributions
```

