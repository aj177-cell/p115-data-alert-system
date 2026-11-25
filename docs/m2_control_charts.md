# Milestone 2: statistical process control charts (CUSUM, EWMA, Shewhart)

This is an add-only study on the combined-plus tree, after the distribution fits (`docs/m2_distribution_fits.md`) and the Bayesian gap check (`docs/m2_bayesian_gaps.md`). Nothing in Milestone 1, `p115_detectors/`, Yuyang's integration layer, the team pipeline files, or any existing threshold was changed. A chart signal is a **candidate**, not a confirmed data error.

Code: `m2/spc/` holds the following modules.

| Module | Contents |
| --- | --- |
| `series.py` | bucket series |
| `baseline.py` | Phase I baselines, standardisation, AR(1) prewhitening |
| `charts.py` | charts and the ARL calibration |
| `evaluate.py` | real-data signals and overlap |
| `inject.py` | injected faults |
| `report.py` | tables and plots |
| `run_spc.py` | command line |
| `tests/test_spc.py` | 13 tests, also loaded by `integration/tests/test_m2_spc.py` |

Results are in `m2/results/spc/`, with plots in `m2/results/spc/plots/`.

## Why

The v3 checks are *point* checks: they ask whether one record, one quote or one gap is unusual. The professor's "industrial engineering techniques" point to statistical process control, which asks whether the *process level* has moved. That is the right tool for faults that are partial or gradual: a feed that drops 30–70% of its messages without ever going fully silent, or a spread that widens by half and stays there. A point check can miss those because no single record looks extreme.

## Method

### Series

For each symbol and session, the in-session time is cut into one-minute buckets using the v3 session presets:

- equities: 09:30–16:00 ET, 390 buckets (210 on the 2025-11-28 half-day);
- CME: 17:00 CT to 16:00 CT, 1,380 buckets, with no buckets in the daily break.

The bucket index is the time-of-day key. Each bucket has two series:

- **Arrival rate:** $N_t$ = number of records in the bucket. Its variance-stabilising (Anscombe) transform is $y_t = 2\sqrt{N_t + 3/8}$, which has variance about 1 if $N_t$ is Poisson.
- **Quoted spread:** $S_t$ = median positive `F_LAST` spread in ticks within the bucket, carried forward over buckets with no quote (the last quote is still standing). The series is $y_t = \ln S_t$.

### Phase I baseline (robust, time-of-day adjusted)

Phase I uses in-control history: the same symbol's **normal-week** sessions (2026-10-05 to 10-09), excluding the day being charted. That leaves four days for a normal-week day and all five for the CPI, VIX-high, half-day and around-degraded days. For each bucket $b$:

$$m_b = \operatorname{rollmed}_{\pm 7}\Bigl(\operatorname*{med}_{d \in D} y_{d,b}\Bigr), \qquad
s_b = \max\Bigl(1.4826\,\operatorname{rollmed}_{\pm 15}\bigl(\operatorname*{med}_{d\in D}|y_{d,b}-m_b|\bigr),\ s_{\min}\Bigr), \qquad
z_t = \frac{y_t - m_{b(t)}}{s_{b(t)}} .$$

Medians and MADs make the baseline robust to Phase I outliers. The rolling windows smooth the four or five days across neighbouring minutes. Because each minute has its own $m_b$ and $s_b$, the open and close are compared with earlier opens and closes, so they are not flagged every day:

- the first and last 30 minutes hold 10–14% of the CUSUM and EWMA signals on equities (`open_close_share.csv`);
- if signals were spread evenly through the day, those 60 minutes would hold 15.4%.

The scale floors are:

- $s_{\min} = 1$ for the rate, the Poisson value after the Anscombe transform;
- $s_{\min} = 0.3 \approx \ln(4/3)$ for the spread. Spreads move in whole ticks, so for names quoting one to three ticks the MAD is often 0. The floor makes the smallest possible move about one standard deviation.

### Autocorrelation: AR(1) residual charts

On the normal week, minute buckets have a lag-1 autocorrelation of 0.2–0.9 (rate median about 0.5; spread up to 0.92 for LWAY because of the carry-forward). Classical limits assume independent points, so autocorrelation inflates false alarms. Following Alwan and Roberts (1988) and Montgomery (ch. 10), the charts run on AR(1) residuals:

$$e_t = \frac{z_t - \hat\phi\, z_{t-1}}{\sqrt{1-\hat\phi^2}},$$

where $\hat\phi$ is the pooled lag-1 correlation of the Phase I $z$ for that symbol and series (clipped at $\pm 5$ and limited to $[0, 0.95]$).

### Charts (one-sided)

Spread charts watch for **widening**; rate charts watch for **drops**, so they run on $-e_t$. After a signal the statistic restarts at 0.

$$\text{Shewhart: } e_t > c, \qquad
\text{CUSUM: } C_t = \max(0,\, C_{t-1} + e_t - k),\ C_t > h, \qquad
\text{EWMA: } E_t = \lambda e_t + (1-\lambda)E_{t-1},\ E_t > L\sqrt{\tfrac{\lambda}{2-\lambda}\bigl(1-(1-\lambda)^{2t}\bigr)} .$$

### Parameter choice and ARL

We use $k = 0.5$, the classical choice for detecting a shift of 1 standard deviation, and $\lambda = 0.2$. Limits target an in-control average run length of $\mathrm{ARL}_0 = 370$ buckets, the one-sided equivalent of the 3-sigma convention. At one-minute buckets, that is about one false signal per series per equity day, or about 3.7 per CME session.

1. **Nominal limits** assume independent $N(0,1)$ points. They are computed exactly with the Brook–Evans Markov chain for the CUSUM and the Lucas–Saccucci chain for the EWMA, and checked by simulation in the tests:
   - $c = 2.78$, $h = 4.10$, $L = 2.60$;
   - the CUSUM chain matches Siegmund's approximation within 3%;
   - for a 1-sigma shift, the run length is 8.6 buckets for the CUSUM and 8.2 for the EWMA.
2. **Phase I-calibrated limits** are used for all results below. Real minute data are heavier-tailed and still somewhat dependent after prewhitening, so the nominal limits give $\mathrm{ARL}_0 \approx 140$ on the normal week (7.1–7.3 signals per 1,000 buckets). We therefore re-solved $h$, $L$ and $c$ (keeping $k$ and $\lambda$) so that the pooled normal-week signal rate is exactly 1/370, using leave-one-day-out baselines:

| series | Shewhart $c$ | CUSUM $k$ / $h$ | EWMA $\lambda$ / $L$ |
| --- | --- | --- | --- |
| nominal (i.i.d.) | 2.78 | 0.5 / 4.10 | 0.2 / 2.60 |
| spread, Phase I | 2.92 | 0.5 / 7.75 | 0.2 / 3.36 |
| rate, Phase I | 2.77 | 0.5 / 8.11 | 0.2 / 3.17 |

Without prewhitening, the same calibration needs $h = 65.5$ for the spread and $24.5$ for the rate. That shows how strongly autocorrelation breaks the textbook limits. Both limit sets are in `m2/results/spc/params.json`.

By construction, the normal-week rate is 2.7 signals per 1,000 buckets. The other day types are out-of-sample tests.

## Results on the 95 symbol-days

### Signals per 1,000 buckets by day type (Phase I limits)

| day type | rate CUSUM | rate EWMA | rate Shewhart | spread CUSUM | spread EWMA | spread Shewhart |
| --- | --- | --- | --- | --- | --- | --- |
| normal week (calibration) | 2.70 | 2.70 | 2.70 | 2.70 | 2.70 | 2.64 |
| CPI 2026-10-14 | 0.00 | 0.00 | 1.80 | 2.25 | 1.80 | 2.55 |
| VIX-high 2026-09-16 | 7.18 | 10.77 | 2.82 | **16.41** | **18.21** | 9.74 |
| half-day 2025-11-28 | **12.86** | **17.62** | 3.81 | 2.38 | 2.38 | 9.52 |
| ES around the degraded weekend (08-28, 08-31) | **24.28** | **29.35** | 9.42 | 0.00 | 0.00 | 0.00 |

The median standardised (not prewhitened) level by day type is as follows.

| day type | median rate $z$ | median spread $z$ |
| --- | --- | --- |
| normal week | 0.06 | 0.00 |
| CPI | 1.26 | 0.00 |
| VIX-high | 0.09 | 0.00 |
| half-day | −0.13 | −1.35 |
| around degraded | −0.67 | 0.00 |

Per symbol and day, see `real_signals_by_symbol_day.csv` and `cusum_by_symbol_day_type.csv`.

**Do the special days show up as expected?**

- **CPI (Oct 14): yes, as a busier day, not a degraded one.** Activity is far above normal: the median rate $z$ is +1.26 overall, and +2.9 for AAPL and +3.9 for SPY (record counts 1.9× and 2.4× the normal week). So the rate-*drop* charts are silent (0 CUSUM and 0 EWMA signals), which is correct. RTH spreads are not wider on CPI day: the release is at 08:30 ET, before the equity open. A few names show spread episodes: NVDA 9, CL 3 and LWAY 3 CUSUM signals.
- **VIX-high (Sep 16): yes.** Spread-widening signals run at about 6–7× the normal-week rate (CUSUM 16.4 vs 2.7 per 1,000 buckets), led by NVDA 13, QQQ 12, CRWV 11, NBIS 9, MSFT 7 and SPY 7. The rate charts are also elevated (NBIS 25).
- **Half-day (Nov 28, 2025): yes.** Thanksgiving-Friday activity is lower in parts of the morning, giving 4.8× the normal rate-drop signal rate (NBIS 18, QQQ 4). Spreads are *tighter* than normal (median $z$ −1.35), so the spread CUSUM and EWMA are quiet. The Shewhart spread chart's 9.5 per 1,000 comes from single-minute spikes.
- **Degraded CME Saturday (2026-08-29): cannot be charted.** The pull holds 18 records for that window, all on Sunday and nearly all inside the 16:00–17:00 CT break, so there are no in-session buckets. The neighbouring ES sessions do stand out: 43 rate-drop CUSUM signals on 08-28 and 24 on 08-31, against 0–4 per normal-week day.
  - On 08-28 they cluster in the first two hours after the 17:00 CT open (on 08-27) and from about 20:00 CT to 08:30 CT.
  - On 08-31 they cluster between about 01:40 and 08:20 CT.
  - See `plots/es_rate_cusum.png`. Total records those days are 1.39× and 0.95× normal, so this is a *shift in when* messages arrive in the overnight session, not a uniformly quiet day. The Phase I history is the October normal week, so we cannot separate feed degradation from late-August seasonality. This is a candidate for a human to check, not a confirmed outage.

A sustained shift produces repeated signals, because the chart restarts after each one. Grouping signals less than 15 minutes apart, the 212 rate-CUSUM signals form 114 episodes and the 174 spread-CUSUM signals form 88.

### Overlap with the existing point checks (±5 minutes)

> **Correction (2026-11-22).** The Bayesian columns were first computed from a Bayesian run whose GLBX days after 2026-10-05 were copies of 10-05 (a cache-naming bug, see `docs/m2_bayesian_gaps.md` and `docs/m2_carryover_decisions.md`, item 6). They were recomputed; the Bayesian columns now track the Poisson columns closely. Chart signals, calibration and all injection results were unaffected.

| chart | SPC signals | near a v3 Poisson gap candidate | Poisson candidate minutes near an SPC signal | near a Bayesian gap candidate | Bayesian candidate minutes near an SPC signal |
| --- | --- | --- | --- | --- | --- |
| rate CUSUM | 212 | 116 (55%) | 551 of 8,768 (6.3%) | 115 (54%) | 544 of 7,975 (6.8%) |
| rate EWMA | 250 | 140 (56%) | 547 (6.2%) | 139 (56%) | 538 (6.7%) |
| rate Shewhart | 147 | 47 (32%) | 209 (2.4%) | 46 (31%) | 209 (2.6%) |

| chart | SPC signals | near a v3 spread candidate | spread-candidate minutes near an SPC signal |
| --- | --- | --- | --- |
| spread CUSUM | 174 | 5 | 5 of 9 |
| spread EWMA | 178 | 7 | 5 of 9 |
| spread Shewhart | 163 | 6 | 6 of 9 |

The charts and the point checks mostly see different things:

- About half of the rate-drop signals coincide with a gap candidate. These are slow periods that also contain a long gap.
- But 94% of gap-candidate minutes have no rate signal: a single 6 s gap does not move a one-minute count much.
- The ES around-degraded signals are mostly *not* near gap candidates (15 of 76 ES rate signals).
- The v3 spread rule fired in only 9 minutes on the whole pull (26 candidates, mostly SPY), so almost every spread signal is new information.

## Injected faults (what SPC catches that the point checks miss)

We used nine symbols on 2026-10-09: AAPL, MSFT, CRWV, NBIS, PLAB, ADTN, LWAY, CL and ES. The Phase I history is the other four normal-week days. Each fault run places three 30-minute windows at random (fixed seeds), away from the first and last 30 minutes, giving 27 windows per fault:

- **widen ×1.5 / ×2 / ×3:** the ask is raised so the spread is at least that many times wider (whole ticks);
- **ramp 1→3:** the factor rises linearly across the window;
- **thin 30% / 50% / 70%:** that share of *all* records in the window is removed at random. This is partial degradation, not an outage.

The comparisons are the v3 `unusual_spreads` rule (pilot preset) for spread faults, and the v3 Poisson gap rule and the Bayesian gap check for thinning faults.

A window counts as detected only by a signal or candidate that the unmodified day does not have. For SPC that means the same chart and bucket; for point checks, no clean candidate within 2 s. Background candidates that happen to fall inside a window are therefore not credited. SPC signals count up to 5 minutes after the window end; point candidates up to 60 s after.

| fault | v3 spread rule | Poisson gap | Bayesian gap | SPC CUSUM | SPC EWMA | SPC Shewhart |
| --- | --- | --- | --- | --- | --- | --- |
| widen ×1.5 | 0/27 | — | — | **16/27** (8 min) | 12/27 (5.5 min) | 3/27 (3.5 min) |
| widen ×2 | 0/27 | — | — | **25/27** (6 min) | 22/27 (4 min) | 16/27 (1.5 min) |
| widen ×3 | 1/27 (1.6 min) | — | — | 26/27 (4 min) | **27/27** (2 min) | 27/27 (1 min) |
| ramp 1→3 | 0/27 | — | — | **27/27** (13 min) | 26/27 (12 min) | 19/27 (17.5 min) |
| thin 30% | — | 19/27 (3.0 min) | 16/27 (2.2 min) | 11/27 (13.5 min) | 9/27 (9 min) | 1/27 |
| thin 50% | — | 23/27 (1.3 min) | 22/27 (1.3 min) | 16/27 (10 min) | 16/27 (7 min) | 5/27 |
| thin 70% | — | 24/27 (1.2 min) | 24/27 (1.2 min) | 23/27 (12 min) | 20/27 (7.5 min) | 8/27 |

Numbers in brackets are median detection delays from the fault start. Details are in `inject_summary.csv`, `inject_by_symbol.csv` and `plots/inject_detection.png` / `inject_delays.png`.

**False alarms.** On the faulted days, new SPC signals outside the windows (with a 15-minute guard after each window) total 0–9 per fault over nine symbol-days. The CUSUM is the noisiest on thinning (4, 9 and 5); the spread charts give 0–2. New point candidates outside the windows are 0 by construction of the comparison. On the clean days (nine symbol-days, 5,490 buckets), the counts are:

| method | clean-day count |
| --- | --- |
| rate CUSUM | 27 |
| rate EWMA | 29 |
| rate Shewhart | 22 |
| spread CUSUM | 13 |
| spread EWMA | 13 |
| spread Shewhart | 20 |
| Poisson gap | 1,858 |
| Bayesian gap | 1,746 |
| v3 spread rule | 0 |

The gap-check counts are background candidates; most come from CL, NBIS, ADTN and PLAB.

What this shows:

1. **Sustained spread widening is invisible to the v3 spread rule, and the CUSUM catches it.** The rolling 15-minute $Q_{0.95}$ adapts to the new level within minutes. A step to $f\times$ exceeds $\max(3Q_{0.95}, Q_{0.95}+3)$ only if roughly $f > 3Q_{0.95}/S \ge 3$ *and* the baseline has not caught up, so the rule found 1 of 81 step-widening windows and 0 of 27 ramps. The spread CUSUM found 67 of 81 steps and 27 of 27 ramps, with zero to two new false alarms per fault. This is the clearest case for SPC: the spike rule and the level-shift chart are complementary.
2. **Partial thinning is mostly caught by the gap checks on mid- and low-liquidity names; SPC adds the very liquid ones.** On names whose natural gaps sit near the 5 s floor (CRWV, NBIS, ADTN, PLAB), removing records quickly creates a gap over 5 s, so the point checks detect within 1–3 minutes. On **ES** the gap checks found 0/3, 0/3 and 1/3 windows at 30%, 50% and 70% thinning, while the rate CUSUM found 2/3, 3/3 and 3/3. Even at 70% thinning, ES gaps stay well under 5 s, so only an aggregate count notices.
3. **SPC is slower and weaker on very quiet names.**
   - Median delays are 7–14 minutes against 1–3 for the gap checks.
   - On LWAY (about 8 records a minute) and MSFT on this day, the rate charts miss most thinning: Phase I variation is large relative to the drop.
4. **CUSUM vs EWMA vs Shewhart.** The CUSUM has the best recall on small and gradual shifts. The EWMA is similar and a little faster when it fires. The Shewhart chart catches only large steps (×3, or thinning combined with single extreme minutes) and should be kept only as a backstop.

## Recommendation (not applied)

1. **Add the spread CUSUM (Phase I-calibrated, AR(1)-prewhitened) as a separate "sustained widening" candidate type** next to the v3 spread rule. It catches a failure mode the current rule cannot (16–27 of 27 sustained or ramping windows vs 0–1) at about one false signal per 370 minutes.
2. **Add the rate CUSUM or EWMA as a "partial degradation" candidate,** mainly for liquid instruments (ES, CL, large-cap equities), where the gap checks are blind to thinning. Keep the Poisson or Bayesian gap check as the fast detector for silences.
3. **Calibrate limits on Phase I data, not from textbook tables.** With minute buckets, the nominal $h = 4.1$ gives about 2.7× the intended false-alarm rate even after prewhitening. Use the calibrated $h \approx 7.8$–$8.1$ and $L \approx 3.2$–$3.4$, and refresh them when the Phase I history changes.
4. **Report episodes, not raw signals,** because a sustained shift re-signals after each restart. Show the ES around-degraded episodes to a human with the session time.
5. **For production, use a rolling Phase I history** of the previous 20 normal sessions per symbol, with holidays and event days excluded, rather than one fixed week.

## Limitations

- **Phase I is one week (four or five sessions).** The time-of-day baseline is smoothed but still noisy, and October is the only reference: late-August ES and the November half-day are compared with October behaviour.
- **One-minute buckets set the time resolution.** SPC delays are measured in minutes; faults shorter than a few minutes are for the point checks.
- **AR(1) prewhitening removes most, but not all, dependence.** The remaining dependence and heavy tails are why calibration on Phase I is needed. The calibrated $\mathrm{ARL}_0$ holds on the normal week by construction; on the clean injection day it was about 200 buckets for the rate CUSUM and 420 for the spread CUSUM.
- **Spread carry-forward on quiet names** (LWAY often has no quote for minutes) makes the spread series sticky and less sensitive.
- **Injected faults are synthetic:** a multiplicative ask shift and uniform random thinning, on one day for nine symbols. Real degradations may be bursty or affect only one side of the book.
- **The degraded Saturday has no in-session data,** so it cannot be charted; the around-degraded ES findings are unconfirmed candidates.
- **No holiday calendar or venue session API** (same as v3). FOMC and contract-roll days are not in the sample.

## How to rerun

```bash
python -m pip install -r requirements-m2.txt
python -m m2.spc.run_spc --mode series   # bucket series for 95 symbol-days (about 1 min, cached in m2/results/spc/series)
python -m m2.spc.run_spc --mode charts   # Phase I baselines, calibrated limits (params.json), signals, overlap
python -m m2.spc.run_spc --mode inject   # injected faults on 9 symbols (about 2 min; a few GB peak on ES, one symbol-day at a time)
python -m m2.spc.run_spc --mode plots    # summary.txt, tables and figures
python -m unittest m2.spc.tests.test_spc
python -m integration.verify --run-tests
python run_combined.py --stress
```
