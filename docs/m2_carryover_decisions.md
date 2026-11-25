# Milestone 2: carryover decisions from `docs/integration.md`

`docs/integration.md` ends with a list of open decisions. Aryan took all of them, and this page is for the team to review. For each item it gives the evidence, the options with their trade-offs, and a recommendation. Where a fix was safe and additive, it is implemented as an **opt-in sidecar** in `m2/carryover/`. The final choice stays with the team.

Nothing in Milestone 1, `p115_detectors/`, Yuyang's integration layer, the team pipeline files or any existing threshold was changed. `python run_combined.py --stress` still catches 41/41. All flags are **candidates**, not confirmed data errors. Times are given in ET (America/New_York) unless marked otherwise.

## Decision table

| # | Item | What we found | Options | Recommendation | Sidecar (opt-in) |
| --- | --- | --- | --- | --- | --- |
| 1 | Engine `crossed_quote` on records without `F_LAST` | Stress clean stream: all 3,273 engine alerts are on non-`F_LAST` records. Real data: **0 crossed records** in 79.0 M XNAS records. GLBX has 183 crossed records (171 with `F_LAST`); 17 of its 18 crossed `F_LAST` runs fall in the 16:45–17:00 CT pre-open. | (a) keep, marked; (b) demote to `info`; (c) drop; (d) also require 1 ms persistence on `F_LAST` | **(b) demote.** Clean-stream findings go 3,273 → 0, recall stays 5/5 and 41/41, and the rows stay for audit. | `crossed.flast_aware_crossed(unified, mode)` |
| 2 | Real-time versions of the batch checks | Most v3 checks are causal already. Two are not: `latency_outlier` uses the whole-day median/MAD, and `stale_quotes` uses a whole-day quantile. The SPC baseline uses other days of the same week. | (a) document; (b) live classes for gaps, crossed and latency; (c) rewrite v3 | **(b), with batch kept as the reference.** The live gap monitor gives the same candidates as stored v3 on real days. An open-gap alarm would fire a median 2.4 s earlier. | `streaming.StreamingGapMonitor`, `StreamingCrossed`, `TrailingLatencyOutlier`, `detected_at()` |
| 3 | Severity alignment | The same evidence gets different levels in the two systems. Crossed quote is engine `warning` (2) but v3 `high` (3); time order is engine `warning` (2) but v3 `low` (1). Bayes and SPC have no severity at all. | (a) leave native; (b) add a mapped `severity_m2` column next to the native one; (c) change native severities | **(b).** One table covers all four sources, the native `severity`/`level` columns are unchanged, and 41/41 faults stay above `info`. | `severity.apply()`, `results/severity_map.csv` |
| 4 | Two ≈5.2 s pauses in the bundled trades sample | Trade-to-trade pauses over 5 s are routine. AAPL averages 626 per normal-week session and SPY 455, and 98.7% of all 42,375 such pauses in the pull have quote records inside them. | (a) keep as `medium`; (b) on trades input, reclassify as `trade_pause` (`info`) when quotes are active, or below a per-symbol floor otherwise; (c) raise the floor for trades globally | **(b).** Both bundled pauses become `info` under the per-symbol floor (AAPL 16 s, SPY 15 s). The bundled quotes sample (14:30:00–14:30:29 UTC) does not cover them. | `trade_pauses.reclassify(unified, schema, quotes, floor_s)` |
| 5 | `metrics.csv` not regenerated with v3 thresholds | Regenerating gives 12 `extreme_prices` flags (stored 15), 2 collateral flags (stored 5) and 0 on the clean stream (stored 1). Precision 0.8 and recall 1.0 are unchanged. The stored file is reproduced exactly with k = 8 and no log floor, i.e. v1-era settings. | (a) leave; (b) ship the regenerated file as a sidecar; (c) overwrite (needs a `verify` manifest change) | **(b) now, (c) by the v3 owners.** The stored file is checksummed by `integration.verify`, and `p115_detectors/` is off-limits here. | `results/v3_metrics_regenerated/` |
| 6 | "v3 summary vs candidate files: CL 5,100 vs 4,973, ES 738 vs 777" | **This was our own bug, not a v3 inconsistency.** The Bayesian run's per-day cache used `Path.with_suffix`, which treated `.0_2026-10-05` in `GLBX_ES.c.0_2026-10-05` as a suffix, so every GLBX day reused 10-05 (9 × 82 = 738; 6 × 850 = 5,100). The stored v3 files were always consistent. | — | **Fixed.** `_p()` replaces `with_suffix`, the GLBX days were rerun, and the Bayes and SPC docs and results were corrected. A regression test was added. | `m2/bayes/run_bayes_gaps.py::_p` |

Rerun everything with `python -m m2.carryover.run_carryover --mode all`. It is read-only on the data and writes only to `m2/carryover/results/`.

## 1. Crossed quotes on records without `F_LAST`

**Background.** In MBP-1, a packet that changes the book in several steps carries `F_LAST` only on its last record. The intermediate records can show a book that never existed at a packet boundary. v3 `crossed_books` evaluates only `F_LAST`, non-snapshot records. The engine checks every record. `integration.combine.engine_table` already marks these alerts (`f_last = False`).

**Evidence.**

- **Stress stream** (`results/crossed_modes_stress.csv`). Engine `crossed_quote` findings with level > 0:

  | mode | faulted stream | clean stream | crossed labels found by the engine | all labels found (either system) |
  | --- | ---: | ---: | ---: | ---: |
  | mark (today) | 3,364 | 3,273 | 5/5 | 41/41 |
  | demote | 150 | 0 | 5/5 | 41/41 |
  | drop | 150 | 0 | 5/5 | 41/41 |
  | persist (≥ 1 ms on `F_LAST`) | 150 | 0 | 5/5 | 41/41 |

- **Real pull, every symbol-day** (`results/crossed_census_real.csv`). The census uses the engine's own predicate: $b > a$, both prices > 0, both sizes ≠ 0.
  - XNAS: 80 symbol-days, 78,962,559 records, **0 crossed records**. The 3,273 is therefore a property of the synthetic generator, which deliberately crosses intermediate states, and not of the Nasdaq feed in this pull.
  - GLBX: 15 symbol-days, 67,519,781 records, 183 crossed records: 171 with `F_LAST` and 12 without. They form 18 crossed `F_LAST` runs of at least 1 ms, and 17 of them start between 16:45 and 17:00 CT (5:45–6:00 PM ET), in the Globex pre-open before the 17:00 CT reopen. The v3 CME preset already treats 16:00–17:00 CT as an expected window and reports those runs as `info`.
  - One run is intraday: **CL.c.0 on 2026-10-08 at 12:17:11 PM ET, 4.94 s, 40 records.** v3 `crossed_books` flags it when run on that slice, but no stored real run contains it, because the stored v3 real run (`results_real_v3`) executed `missing_intervals` only. It is worth a look by whoever owns the futures review.
- The engine does not run on GLBX (Milestone 1 is equities only), so on the real pull the engine's crossed rule would raise nothing at all.

**Options.**
- (a) keep as today: the noise stays in any engine-only view.
- (b) demote to `info` / level 0: the rows stay for audit, and dashboards that filter on level > 0 lose the noise.
- (c) drop: the cleanest table, but the evidence is lost.
- (d) persist: the same as v3's definition, but it needs the raw frame, and it added nothing over (b) and (c) here.

**Recommendation: (b)** for the unified report. If the engine is ever extended to futures, it also needs the pre-open window that v3 uses.

## 2. Real-time versions of the batch checks

`results/streaming_causality.csv` audits each check. Summary:

| check | causal in batch? | earliest live detection | live version |
| --- | --- | --- | --- |
| crossed / locked books | yes | run start + `min_duration_s` (1 ms) | `StreamingCrossed` |
| extreme prices (robust z, static band) | yes (window shifted by one) | the record | not needed |
| ts_recv / ts_event inversion, negative latency | yes | the record | not needed |
| latency outlier | **no**: whole-day median/MAD per key | the record | `TrailingLatencyOutlier` |
| missing intervals (Poisson) | yes (rate from the previous 500 gaps) | gap start + threshold; batch raises at the gap end | `StreamingGapMonitor` (`poll` = open-gap alarm) |
| stale quotes | **no**: adaptive $\theta$ is a whole-day quantile | book change + $\theta$, once `min_trades` prints arrive | trailing quantile (not built) |
| unusual spreads | yes (rolling quantile shifted by one) | the record | not needed |
| sequence | yes | the next record | not needed |
| Bayesian gaps (`m2/bayes`) | yes (discounted state) | gap end | not needed |
| SPC charts | per minute yes; the Phase I baseline used other days of the same week | bucket close | build the baseline from earlier days only |

**Live vs batch** (`results/streaming_vs_batch.csv`):

| data | check | batch | live | identical |
| --- | --- | ---: | ---: | --- |
| stress stream | missing intervals | 5 | 5 | yes (times and thresholds) |
| stress stream | crossed books | 5 | 5 | yes (run starts); raised a median 0.09 s after the start when driven by updates, 1 ms with `poll` |
| stress stream | latency outlier (trailing window 5,000) | 2 | 2 | yes |
| real AAPL 2026-10-08 (622,007 records) | latency outlier | 0 | 0 | yes |
| real AAPL 2026-10-08 | missing intervals | 8 | 8 | yes, against the stored v3 flags |
| real LWAY 2026-10-14 | missing intervals | 118 | 118 | yes |
| real CL.c.0 2026-10-08 (2.08 M records) | missing intervals | 627 | 627 | yes |

**Open-gap alarm.** A live monitor can alarm as soon as the silence exceeds the threshold the batch rule will use, without waiting for the next record. Over all 14,129 stored v3 gap candidates (95 symbol-days), the lead over batch is $g - \theta$: median **2.4 s**, 90th percentile 16.5 s, maximum 763.5 s.

**Recommendation.**
- Use `StreamingGapMonitor`, with its open-gap alarm, as the real-time gap check. Keep batch v3 as the reference that it must match; the tests check this.
- Use the trailing latency window. `stale_quotes` needs a trailing-quantile $\theta$ before it can run live; we have not built that.
- Throughput of these pure-Python classes is about 65k records/s (CL day: 2.08 M records in 32 s). That is fine for a study, not for a production feed handler.

## 3. Severity alignment

The common scale reuses the `level` numbers from `integration.combine.LEVEL`:

| level | meaning |
| --- | --- |
| 3 high | the record or interval is wrong as delivered; exclude it until reviewed |
| 2 medium | probably a data problem; review before use |
| 1 low | unusual but often legitimate market or venue behaviour |
| 0 info | expected behaviour, kept for audit |

The full mapping is `results/severity_map.csv`, 29 rows. The first matching row wins. The rows that change something are:

| source / rule / subtype | native | proposed | why |
| --- | --- | --- | --- |
| engine `crossed_quote`, record without `F_LAST` | warning | **info** | intermediate book state (item 1) |
| engine `crossed_quote`, `F_LAST` | warning | **high** | same evidence as v3 `crossed` (high) |
| engine `time_order` | warning | **low** | same evidence as v3 `ts_event_inversion` (low); `ts_event` is not guaranteed to be monotonic |
| engine `integrity` | critical | high | the scale has no separate "critical" |
| v3 `crossed_books/locked` | medium | **low** | locked books occur in fast markets |
| v3 `stale_quotes/stale` | high | **medium** | a stale book is usually a quiet name; review it rather than exclude it |
| v3 `unusual_spreads/wide` | medium | **low** | wide spreads at the open, the close and on news are market behaviour |
| v3 `missing_intervals/trade_pause*` | medium | info | item 4 |
| Bayes candidate, $P(\text{outage}) \ge 0.99$ and gap > 60 s | — | high | same 60 s line as v3 `high` |
| Bayes candidate, otherwise | — | medium | |
| SPC Shewhart signal, or a CUSUM/EWMA episode under 15 minutes | — | low | context for the point checks |
| SPC CUSUM/EWMA episode of 15 or more one-minute buckets | — | medium | a persistent shift |

**Effect** (`results/severity_counts.csv`):
- Stress stream: 3,214 engine alerts become `info` and the other 150 engine crossed alerts become `high`. Five stale candidates go to `medium` and five wide-spread candidates go to `low`. All 41 labelled faults still have `level_m2` > 0.
- Real Bayesian candidates (95 symbol-days): 390 `high` and 12,481 `medium`.
- Real SPC episodes: for example, the rate CUSUM gives 96 `low` and 18 `medium`.

**Recommendation: adopt the column, and treat the three v3 demotions (locked, stale, wide) as separate team calls.** They are judgement calls, not corrections.

## 4. The two ≈5.2 s pauses in the trades sample

The bundled `XNAS.ITCH` trades sample (AAPL and SPY, 2025-09-15) produces two v3 gap candidates and no engine alerts:
- SPY at 10:32:21 AM ET: gap 5.219 s, $\lambda$ = 6.79/s, $p = 3.96\times10^{-16}$.
- AAPL at 10:35:33 AM ET: gap 5.152 s, $\lambda$ = 5.57/s, $p = 3.5\times10^{-13}$.

Both are `medium`. With trades only, a pause in prints cannot be told apart from a pause in the feed.

**Evidence from the real pull** (`results/trade_pause_census.csv`). The data is XNAS mbp-1, trades = `action == 'T'`, regular hours only, 80 symbol-days. Normal week, per symbol:

| symbol | trades/day | pauses > 5 s per day | share with quote records inside | median records inside | longest pause | 0.9999 quantile of trade gaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AAPL | 120,200 | 626 | 99.9% | 41 | 28.1 s | 15.0 s |
| SPY | 65,465 | 455 | 100% | 161 | 22.4 s | 14.4 s |
| QQQ | 68,553 | 467 | 100% | 291 | 23.3 s | 16.0 s |
| NVDA | 195,535 | 197 | 100% | 105 | 15.1 s | 10.7 s |
| MSFT | 78,040 | 888 | 99.9% | 40 | 25.2 s | 20.3 s |
| LWAY | 698 | 154 | 82.9% | 5 | 1,877 s | 1,823 s |

Across all 80 symbol-days, 41,807 of 42,375 pauses over 5 s (98.7%) have quotes inside. These are Nasdaq-only prints, which is the same feed as the bundled sample. A 5.2 s trade pause on AAPL or SPY is therefore an ordinary event: Poisson-rare given the average print rate, but common because prints cluster.

**Options.**
- (a) keep `medium`.
- (b) opt-in policy for trades-schema runs: reclassify as `trade_pause` / `info` when a quotes or mbp-1 frame shows records inside the pause; with no quotes, reclassify below a per-symbol floor. The floor is the normal-week maximum of the 0.9999 quantile, rounded up: AAPL 16 s, SPY 15 s, NVDA 11 s, QQQ 17 s, MSFT 21 s; illiquid names have much higher floors (`results/trade_pause_floors.json`).
- (c) a global 15 s floor for trades.

**Result on the bundled sample** (`results/trade_pause_bundled.csv`). Both candidates become `trade_pause_below_floor` / `info` under (b) and (c). With the bundled quotes sample they stay `gap`, because that sample only covers 14:30:00–14:30:29 UTC.

**Recommendation: (b).** Rows are never removed. The floors come from a single month of data and should be refreshed when the sample grows.

## 5. Regenerating `metrics.csv`

`metrics_regen.regenerate()` runs v3 `evaluate.main()` with its output folder pointed at `m2/carryover/results/v3_metrics_regenerated/`. It writes metrics, labels, flags, summary, the sensitivity CSV and the PNG. The stored file is untouched; a test checks its SHA-256.

The only differences (`results/v3_metrics_diff.csv`) are in `extreme_prices`:

| column | stored | regenerated |
| --- | ---: | ---: |
| flags | 15 | 12 |
| collateral flags | 5 | 2 |
| clean-stream flags | 1 | 0 |

Precision (0.8), recall (1.0) and the two false positives are unchanged. The three flags that disappear are robust-z flags on small returns ($-3.0\times10^{-4}$ and $-1.4\times10^{-3}$) next to injected faults (`results/v3_metrics_extreme_flag_diff.csv`).

**Why the stored file differs** (`results/v3_metrics_extreme_settings.csv`). It is reproduced exactly with $k = 8$ and `mad_floor_log` = 0, the v1-era settings. With v3's $k = 12$ the result is 12 flags whatever the floor. The stored results are dated 2026-10-30 at 2:56 PM ET, and `config.py` and `detectors.py` were changed later that day, at 9:26 and 9:31 PM ET.

**Recommendation.** Use the sidecar numbers in any Milestone 2 table. The v3 owners should replace the stored file, update the synthetic table in the `p115_detectors` README and refresh the `integration.verify` manifest in a single commit.

## 6. CL 5,100 vs 4,973 and ES 738 vs 777: our bug, now fixed

The mismatch came from Aryan's Milestone 2 Bayesian run (`m2/bayes/run_bayes_gaps.py`), not from v3:
- The per-day cache was named with `stem.with_suffix(".json")`. For `GLBX_ES.c.0_2026-10-06`, Python treats `.0_2026-10-06` as the suffix, so every GLBX day was written to and read from `GLBX_ES.c.json` / `GLBX_CL.c.json`. Every GLBX day after 2026-10-05 therefore repeated the 10-05 row: ES 9 × 82 = 738 and CL 6 × 850 = 5,100.
- XNAS stems contain no dot, so XNAS was never affected. Injected-outage results do not use the cache and were not affected either.
- The stored v3 `run_index.csv`, the per-day JSON and the Parquet files all agree: CL 4,973 = 850 + 980 + 1,040 + 627 + 693 + 783, and ES 777.

**Fix.**
- `_p(stem, ext)` appends the extension instead of using `with_suffix`.
- The GLBX cache files were deleted and the real run was repeated. The Poisson re-implementation now equals stored v3 on all 95 symbol-days (checked by a test against `run_index.csv`).
- The SPC series were rebuilt, because their Bayesian candidate columns came from the same run, and the charts and plots were rerun.

**Corrected numbers.**

| | before | after |
| --- | --- | --- |
| CL Poisson / Bayes / both | 5,100 / 5,220 / 5,100 | **4,973 / 5,028 / 4,856** (117 Poisson-only, 172 Bayes-only) |
| ES Poisson / Bayes / both | 738 / 738 / 738 | **777 / 778 / 774** |
| 95-day total, Poisson / Bayes | 14,217 / 13,023 | **14,129 / 12,871** |
| SPC rate CUSUM signals near a Bayesian candidate | 91 (43%) | **115 (54%)** |

`docs/m2_bayesian_gaps.md` and `docs/m2_control_charts.md` carry a dated correction note.

## Representative-sample extension (not pulled)

The pilot has 95 symbol-days. It has no FOMC day *labelled* as one, no contract roll, and one holiday-type day (the 2025-11-28 half day). Notes on the current pull:
- 2026-09-16, labelled `vix_high`, **is an FOMC decision day** (the 15–16 September 2026 meeting), so that label mixes two effects.
- In the October pull, CL.c.0 is CLX6 (last trade 2026-10-20) and ES.c.0 is ESZ6 (expiry 2026-12-18), so no roll falls inside it.

**Calendars** (`results/calendars_2025_2026.json`, `results/roll_calendar_2025_2026.csv`):

- **FOMC decision days.**
  - 2025: Jan 29, Mar 19, May 7, Jun 18, Jul 30, Sep 17, Oct 29, Dec 10.
  - 2026: Jan 28, Mar 18, Apr 29, Jun 17, Jul 29, Sep 16, Oct 28, Dec 9.
  - Source: federalreserve.gov FOMC calendar, opened 2026-11-22.
- **ES roll.**
  - Expiry is the third Friday of Mar/Jun/Sep/Dec; the CME roll date is 8 days earlier.
  - Databento `ES.c.0` is calendar-ranked and switches on the session after expiry (Databento symbology docs).
  - 2025: expiries Mar 21, Jun 20, Sep 19, Dec 19; roll dates Mar 13, Jun 12, Sep 11, Dec 11.
  - 2026: expiries Mar 20, Jun 19, Sep 18, Dec 18; roll dates Mar 12, Jun 11, Sep 10, Dec 10.
- **CL last trade.**
  - Rule: 3 business days before the 25th of the month before delivery, or before the business day preceding the 25th when the 25th is not a business day.
  - Examples: CLZ5 2025-11-20, CLF6 2025-12-19, CLX6 2026-10-20, CLZ6 2026-11-20.
  - The full list for 2025–2026 is in the CSV. It is computed with the NYSE holiday list; CME clearing holidays should be confirmed before use.
- **Holidays.**
  - NYSE closed: 21 weekdays in 2025–2026, including 2025-01-09 (National Day of Mourning).
  - NYSE early closes: 2025-07-03, 2025-11-28, 2025-12-24, 2026-11-27, 2026-12-24.
  - CME holiday sessions with early closes: 18 days. Source: `exchange_calendars` XNYS/CMES.

**Proposed extension and estimate** (`results/sample_extension_estimate.csv`, `results/sample_extension_summary.csv`):

| block | days | symbol-days | records (est.) | DBN size | cost (est.) |
| --- | --- | ---: | ---: | ---: | ---: |
| FOMC | 2025-12-10, 2026-01-28, 2026-10-28; all 12 symbols | 36 | 84.1 M | 6.3 GiB | $9.13 |
| ES roll | Dec 2025 and Sep 2026: roll date (c.0 + c.1), expiry, first c.0 switch session | 8 | 45.0 M | 3.4 GiB | $6.03 |
| CL expiry | CLV6 and CLX6 last trade (c.0 + c.1) and the next session | 6 | 9.2 M | 0.7 GiB | $1.24 |
| Holidays | 2025-12-24 early close and 2025-12-26 (all 12 symbols); 2026-01-19 and 2026-07-03 (NYSE closed: ES and CL only) | 28 | 33.0 M | 2.5 GiB | $3.76 |
| **total** | | **78** | **171 M** | **12.8 GiB** | **≈ $20** |

**How the estimate was built.**
- Mean records per symbol-day by day type come from this pull. The CPI day stands in for FOMC days, the half day for holidays, and the normal week for roll days.
- Pricing: 80 B per MBP-1 record at the verified GiB prices, XNAS $1.20 and GLBX $1.80 (`/workspace/p115/pilot_cost_check.md`). The same method gives $3.04 for one CPI-type day of all 12 symbols; the actual pull was $1.43 + $1.39 + $0.22.
- Assumptions:
  - c.1 on a roll day has half the records of c.0.
  - A futures holiday session has 40% of a normal day.
  - Allow roughly ±30% ($15–28). Roll days and FOMC afternoons can run hotter than these proxies.
- Definition and status pulls for the same days cost fractions of a cent.
- On disk this is about 9 GB of Parquet at the current ratio of 7.6 GB per 146 M records. The big futures days should be processed one at a time, as now.

**Exact cost.** `python -m m2.carryover.sample_extension --get-cost` calls only `metadata.get_cost` and `metadata.get_record_count`, which are free, and never a data endpoint. It was **not run**, because `DATABENTO_API_KEY` is not set in this environment. Nothing was pulled and nothing was spent.

## Limitations

- The real-data evidence is the 95 symbol-day pilot only (Nasdaq-only prints and book, ES and CL front months).
- The trades-pause floors come from one month of data.
- The severity proposals for locked, stale and wide are judgement calls.
- The live classes are reference implementations, tested against batch, not tuned for throughput.
- The extension estimate is a model, not a quote; run `--get-cost` with the key for exact numbers.

## Files

| path | contents |
| --- | --- |
| `m2/carryover/crossed.py` | `flast_aware_crossed`, `crossed_census`, `census_all` |
| `m2/carryover/streaming.py` | `CAUSALITY`, `StreamingGapMonitor`, `StreamingCrossed`, `TrailingLatencyOutlier`, `detected_at` |
| `m2/carryover/severity.py` | `SEVERITY_MAP`, `apply`, `bayes_rows`, `spc_rows` |
| `m2/carryover/trade_pauses.py` | `pause_stats`, `census_all`, `reclassify` |
| `m2/carryover/metrics_regen.py` | `regenerate`, `compare`, `flag_diff` |
| `m2/carryover/sample_extension.py` | calendars, roll rules, extension plan, offline estimate, optional free `get_cost` |
| `m2/carryover/run_carryover.py` | produces every file in `m2/carryover/results/` |
| `m2/carryover/tests/test_carryover.py` | 24 tests, also loaded by `integration/tests/test_m2_carryover.py` |
