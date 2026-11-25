# p115 Market Data Quality Detectors (v3)

These are seven vectorized checks for Databento **MBP-1** data. They read the Parquet written by
`team_pipeline/databento_pipeline.py::download_and_show()`, which is indexed on `ts_recv`.
v0.1 was built on synthetic data. **v2 (2026-10-30)** applies the threshold changes proposed after the real-data pilot
(`results_real/summary.md` §6) and was rerun on all 95 symbol-days (`results_real_v2/`). v1 thresholds are kept in `config_v1.py`.

## Files
| file | purpose |
|---|---|
| `config_v2.py` | The v2 thresholds, unchanged, for reproducing `results_real_v2/` |
| `config_v1.py` | The v1 thresholds, unchanged, for reproducing `results_real/` |
| `config.py` | All thresholds (v2), each with its rationale, plus the `equities_config()` and `cme_futures_config()` presets |
| `detectors.py` | One function per check, plus `run_all(df, config) -> (flags, summary)` |
| `synthetic.py` | `generate_mbp1()` makes a clean stream; `inject_faults()` adds faults with ground-truth labels |
| `evaluate.py` | Per-check precision/recall/F1 with a ±1 s matching tolerance, and the sensitivity sweep (writes `results/*.csv` and `sensitivity.png`) |
| `tests/` | 34 pytest cases (22 original + 6 for v2 + 6 for the v3 missing-interval rule): each check has a positive and a negative case, plus raw-vs-converted input, Parquet round trip, the F_LAST filter, `sequence==0`, and window masking |

## Usage with the team pipeline
```python
import sys; sys.path.insert(0, "detectors")
import pandas as pd
from databento_pipeline import download_and_show          # team_pipeline, unchanged
from config import DetectorConfig, cme_futures_config, equities_config
from detectors import run_all

download_and_show(...)                                     # writes .../data.parquet (mbp-1)
df = pd.read_parquet(".../data.parquet")                   # index = ts_recv (pretty ts, float px)
flags, summary = run_all(df, cme_futures_config())         # or equities_config()
# optional extras:
# run_all(df, cfg, status=status_df)                       # masks non-trading windows for crosses
# cfg.stale.reference = {"symbol": "ESZ6"}                 # reference instrument for stale-vs-quiet
```
Raw `DBNStore.to_df(price_type="fixed", pretty_ts=False)` frames work too. The detector handles
1e-9 fixed-point prices, `UNDEF_PRICE`/`UNDEF_TIMESTAMP` → NaN, and integer action/side codes.

Flag columns: `ts_recv, ts_event, publisher_id, instrument_id, symbol, check, subtype, severity, value, threshold, detail`.
`subtype` is an addition to the requested schema. Severity levels are `info < low < medium < high`.

### Conventions used by every check
* Rows are grouped by `(publisher_id, instrument_id)`, except sequence, which groups by `publisher_id` (see below).
* **Book-state checks** (crossed, stale, spreads, mid-price extremes) only look at rows with `F_LAST` (128) set.
  Intermediate rows within one event can legitimately show a transient book.
* `F_BAD_TS_RECV` (8) and `F_MAYBE_BAD_BOOK` (4) are used as features. A timestamp inversion on an
  `F_BAD_TS_RECV` row is downgraded to `info`. The flags are also reported directly by the `vendor_flags` check.

## Checks
Notation: $b_t, a_t$ are the best bid and ask, $m_t=(a_t+b_t)/2$, $\delta$ is the tick size,
$\tilde x$ is the median, and $\mathrm{MAD}$ is the median absolute deviation.

1. **Crossed / locked books** (`crossed_books`): a run of consecutive `F_LAST` rows with $b_t > a_t$ (crossed) or
   $b_t = a_t$ (locked, optional). The run is flagged if its duration is at least $\tau_{\min}$ (default 1 ms). Runs inside
   expected windows (CME 16:00–17:00 CT maintenance, or status `is_trading != 'Y'`) become `info`.
2. **Extreme prices** (`extreme_prices`): uses the log return $r_t=\ln(p_t/p_{t-1})$ on trades and mids (absolute
   difference for prices $\le 0$). With a trailing window of $W=500$:
   $$z_t=\frac{r_t-\tilde r}{1.4826\,\max(\mathrm{MAD}(r),\ \epsilon)},\qquad \epsilon=\max(\delta/p_t,\ 10^{-4}),\qquad |z_t|>k=12$$
   (v1: $k=8$, $\epsilon=\delta/p$ when the tick was known.) v2 only flags inside the session calendar (equities RTH incl. early closes; CME preset masks the 16:00–17:00 CT break).
   Optional static bands $p_t\notin[L,U]$ can be set per symbol.
3. **Timestamp inversions** (`timestamp_inversions`):
   * $ts^{recv}_t < ts^{recv}_{t-1}$ (high, or info if `F_BAD_TS_RECV`).
   * $ts^{event}_t < ts^{event}_{t-1}-\varepsilon$ (low, because venues can report out of order).
   * $ts^{recv}_t-ts^{event}_t<-1\,\mu s$ (negative latency).
   * Latency outlier: $\ell_t>\tilde\ell+k_\ell\cdot1.4826\,\mathrm{MAD}(\ell)$ and $\ell_t-\tilde\ell>\Delta_{\min}$, with $k_\ell=10$; $\Delta_{\min}$ = 5 ms equities, 20 ms CME (v1: 1 ms).
     v2 collapses outliers of one key less than 1 s apart into one episode flag (`detail` has `records=`).
   * `ts_in_delta` saturated at $\pm(2^{31}-1)$.
4. **Missing intervals** (`missing_intervals`): $g_t$ is the gap in in-session time, measured on a session clock that
   skips closed periods, holidays and early closes. The gap is flagged if
   $$g_t>\min\!\Big(\max\big(-\ln\alpha/\hat\lambda_t,\ g_{\min}\big),\ g_{\max}\Big)$$
   where $\hat\lambda_t$ is the trailing message rate over 500 messages. Under a Poisson arrival model,
   $P(G>g)=e^{-\lambda g}$, so $\alpha=10^{-6}$ is the per-gap false-alarm rate. Defaults: $g_{\min}=5$ s (v1: 1 s, v2: 2 s), $g_{\max}=60$ s, `model='poisson'`.
   v3 also implements, for comparison, `model='overdispersed'` (gamma-mixed Poisson / negative binomial, i.e. Lomax gaps
   $\Pr(G>g)=(1+g/\sigma)^{-\kappa}$ with $\kappa=\max(2CV^2/(CV^2-1),\kappa_{\min})$, $\sigma=\bar g(\kappa-1)$ from the trailing
   window, threshold $\sigma(\alpha^{-1/\kappa}-1)$) and `model='tod'` (rate of the key's 5-minute session bucket). On real data
   plus injected 10 s / 20 s outages, Poisson with a 5 s floor kept v2 recall (0.97) and cut medium flags 73%; the
   overdispersed model lost 41-56% of injected outages; TOD matched P5 but is not causal (see `results_real_v3/`).
   `bar_completeness()` checks derived bars by counting 1m bars per 5m/15m/30m bucket.
5. **Stale quotes** (`stale_quotes`): an unchanged $(b,a,q^b,q^a)$ on `F_LAST` rows for age $A\ge\theta$ (5 s). The run is
   *stale* if the instrument kept trading during it (at least 10 trades; v1: 3) or the reference mid moved
   $|\Delta m^{ref}|/m^{ref}\ge 5$ bp. Otherwise it is *quiet*, which is not reported by default.
6. **Unusual spreads** (`unusual_spreads`): $s_t=(a_t-b_t)/\delta$ in ticks. A run of $s_t\le 0$ is `zero_or_negative`.
   A run is `wide` if it lasts at least 0.5 s and $s_t>\max(c\cdot q_{0.95}^{15\min},\ q_{0.95}+3)$ with $c=3$. An alternative mode
   uses a time-of-day robust z: $(s_t-\tilde s_{\text{bucket}})/(1.4826\,\mathrm{MAD})>6$.
   v2: flags only in session (equities RTH, CME break masked); a NaN tick in `tick_size` is treated as missing and
   `default_tick` is used (`equities_config()` sets 0.01, because XNAS definitions carry NaN `min_price_increment`).
7. **Sequence discontinuities** (`sequence_discontinuities`): with $d_t=\text{seq}_t-\text{seq}_{t-1}$, a gap is
   $d_t>1$ (more precisely $d_t-1\ge$ `min_gap`) and a regression is $d_t<0$. $d_t=0$ is allowed. Rows with `sequence==0`
   are skipped (for example EQUS.MINI).
   v2 `gap_mode`: `flag` / `report_only` (gaps at `info`) / `skip` (regressions only). XNAS ITCH `sequence` is one feed-wide
   counter, so `equities_config()` uses `skip`. The CME preset uses `per_instrument=True, gap_mode='flag'`, which is only
   valid on full-channel pulls; the pilot rerun sets `skip` for its single-instrument GLBX pulls too (per-instrument steps
   are >1 for ~20% of ES and ~91% of CL records) and reports the gap count in `run_index.csv`.

## Synthetic evaluation (`python evaluate.py`)
The data is 1 h of 2 instruments (110,921 records) with 5 injected faults per check. A detected flag counts as a match
if it falls within ±1 s of a label. Flags caused by another fault (for example the spread change when a book is crossed)
are counted as *collateral* and do not count as false positives.

| check | P | R | F1 | flags on clean stream |
|---|---|---|---|---|
| crossed_books | 1.00 | 1.00 | 1.00 | 0 |
| extreme_prices | 0.80 | 1.00 | 0.89 | 1 |
| timestamp_inversions | 1.00 | 1.00 | 1.00 | 0 |
| missing_intervals | 1.00 | 1.00 | 1.00 | 0 |
| stale_quotes | 1.00 | 1.00 | 1.00 | 0 |
| unusual_spreads | 1.00 | 1.00 | 1.00 | 0 |
| sequence_discontinuities | 1.00 | 1.00 | 1.00 | 0 |

The injected faults are deliberately clear-cut, so these numbers are an **upper bound**. They show the plumbing is
correct, not how the checks perform in the field. The sensitivity sweep (`results/sensitivity.csv/.png`) is more informative:
* For extreme prices, precision falls to 0.32 at k=6 and 0.04 at k=4. The remaining false positives are natural
  1–2 tick mid moves where the MAD sits at its one-tick floor.
* Missing intervals picks up 1 false positive at α ≥ 1e-3.
* Stale quotes loses all recall once θ is above the 30 s fault length.
* Wide spreads lose all recall above c=3, because the injection is about 10 ticks on a 1–3 tick spread.
* Sequence loses recall once `min_gap` is above the size of the injected jump.

## Known limitations
* v2 thresholds were tuned on the 95-symbol-day pilot (see `results_real_v2/pilot_results_v2.pdf`); no check has a confirmed real error yet, so recall on real faults is untested.
* MAD comes from a trailing *rolling-window* median of absolute deviations from the rolling median. It is an
  approximation (it lags), not an exact rolling MAD.
* Extreme-price detection works on returns, so after a bad print the *reversion* print also fires. Both flags fall
  within tolerance of the label, so they count as a true positive and a collateral flag.
* The meaning of `sequence` may be per channel, not per instrument (open question in `databento_discovery_prep.md`).
  The default is to group per publisher. Symbol-filtered pulls will then show gaps that are really other
  instruments' messages. Set `per_instrument=True` if Databento confirms per-instrument sequencing.
* Session calendars are hand-coded: the equities RTH preset, a CME Globex preset, and the 2025-11-28 13:00 ET early
  close. There is no full exchange holiday calendar yet.
* There are no off-tick-price, trade-outside-NBBO, or cross-venue consistency checks yet.
* Only the top of book (`*_00`) is used, and `bid_ct_00`/`ask_ct_00` are not used yet.
* Status-based masking needs a separately downloaded `status` schema DataFrame.

## Next steps: real-data validation
1. **2026-08-29 GLBX (marked `degraded`; note it is a Saturday)**: check that missing intervals and sequence flags
   line up with the degraded windows, and that the weekend calendar suppresses false alarms.
2. **2026-10-14 (CPI volatility day)**: a false-positive stress test for extreme prices, spreads and latency.
   Tune `k` and `multiplier` so the 08:30 ET release itself is not reported as a data error.
3. **2025-11-28 (half day)**: confirm the 13:00 ET early close stops missing-interval and stale flags after the close.
4. Hand-label a sample of flags on each day, re-run `evaluate.py`-style metrics, and freeze per-dataset presets.

## v2 changes (old -> new)
| setting | v1 | v2 |
|---|---|---|
| `timestamps.latency_min_excess_ns` | 1 ms | 5 ms equities, 20 ms CME |
| `timestamps.latency_episode_gap_s` | (per record) | 1.0 s episodes |
| `missing.min_gap_s` | 1.0 | 2.0 |
| `sequence.gap_mode` | flag (per publisher) | XNAS `skip`; CME preset per-instrument `flag` (pilot run: `skip`, counted) |
| XNAS tick | NaN (spreads off, floor 1e-4) | 0.01 via `spreads.default_tick`; NaN lookup -> default |
| `extreme.k` | 8 | 12 |
| `extreme` floor | $\delta/p$ else $10^{-4}$ | $\max(\delta/p, 10^{-4})$ |
| `extreme.calendar`, `spreads.calendar` | none (all hours) | RTH 09:30–16:00 ET (13:00 on 2025-11-28); CME Globex hours |
| `stale.min_trades` | 3 | 10 |

Rerun: `PYTHONPATH=detectors python analyze_v2.py` (≤2 workers, big symbol-days alone), then `python compare_v2.py`.

## v3 change (2026-10-30)
Only `missing.min_gap_s` 2.0 -> 5.0 (Poisson model kept). All 95 symbol-days, medium flags 70,382 -> 13,308
(CL 24,237 -> 4,967, NBIS 11,892 -> 1,465, ES 9,916 -> 777); the 821 flags of 60 s or more are unchanged.
Rerun: `PYTHONPATH=detectors:. python analyze_v3.py` (missing_intervals only), `python compare_v3.py`;
sensitivity `sens_v3.py`, synthetic `synth_v3.py`.
