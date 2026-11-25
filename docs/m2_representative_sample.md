# Milestone 2: representative-sample extension

The 95 symbol-day pilot (06–09 Oct 2026 plus a few special days) has no futures roll and no clean FOMC day. It also has only one early close. This extension pulls **85 more symbol-days** across the strata that were missing. It then runs every check on them: the seven v3 checks, Yuyang's combined engine (XNAS), the Bayesian gap model and the SPC charts. Each stratum is compared with the pilot normal week.

Everything here is add-only. Nothing changed in Milestone 1, `p115_detectors/`, the integration layer, the team pipeline files, any stored result or any threshold. `python run_combined.py --stress` still catches 41/41. `python -m integration.verify --run-tests` passes: 41 Milestone 1, 34 v3, and 86 integration tests, 9 of which are new (`m2/sample/tests`). Every flag is a **candidate**, not a confirmed data error. Times are ET (America/New_York) unless marked CT (America/Chicago; CT = ET − 1 h).

Code is in `m2/sample/`, results in `m2/sample/results/`, and tests in `m2/sample/tests/test_sample.py`, which is loaded into the integration suite. Rerun with:

```
python -m m2.sample.run_sample --mode {strata,v3,engine,bayes,spc,volume,report} [--source sample|pilot|all]
```

Every mode writes a per-symbol-day checkpoint under `results/cache/<mode>/` and skips days already done.

## 1. Data pull and cost

The data was downloaded on 2026-11-22 between 11:57 and 13:20 ET, through the **unmodified team notebook**. The method is the same as the 95-day rerun. A copy of `Databento_Data_Pipeline.ipynb` is executed with only the parameter values of one section cell changed. So `databento_pipeline.py`'s own download function makes every request. The changed values are recorded in each notebook's `metadata.p115_run`, and the executed notebooks are saved. The API key reached the pipeline through a FIFO and was never written to disk. A scan of the run folders, the data, the repo and the logs finds it in 0 files. The data is on the box only (`/workspace/p115/data_sample`, 9.6 GB) and is not in the zip.

| item | value |
| --- | --- |
| 78-symbol-day plan (`m2/carryover/results/sample_extension_get_cost.csv`) | **\$16.6271**, 144,195,784 records |
| 7 additions (ES.c.1 on 5 roll days, CL.c.1 on 2 expiry days, normal-week c.1) | \$3.9128 |
| status + definition schemas | \$0.0010 |
| **pre-download `metadata.get_cost` total, all 72 requests** | **\$20.5409** (cap \$28) |
| mbp-1 records downloaded | 173,371,913. Every request's count equals its quoted bytes / 80 B exactly. |
| requests that hit a read timeout and were retried | 3: GLBX 2026-09-21 ES.c.0 (\$0.6537), XNAS 2026-01-28 (\$1.4836), GLBX 2025-12-11 (\$1.5785) |
| worst case if each failed partial stream was billed in full | \$24.2567 |

**Record counts that looked too high.** On ES expiry days the log shows 6,051,260 (2026-09-18) and 6,683,218 (2025-12-19) against quoted 338,437 and 320,506. These do not overrun the quote. The quoted figures are the plan's **ES.c.0-only** rows. The download requests ES.c.0 **and** ES.c.1, because c.1 is one of the 7 additions. The free `metadata.get_record_count` gives 338,437 + 5,712,823 = 6,051,260 and 320,506 + 6,362,712 = 6,683,218. Both match the downloads exactly. They are already inside the \$20.5409 precheck, which priced both symbols. It is the same request window, schema and dates. In short, the shift of volume to the back month makes the c.1 half large.

**Actual spend.** Databento's API has no endpoint for account balance or invoices, so the charge cannot be read without the portal. If the 3 timed-out attempts were not billed, or were billed only for the bytes actually streamed, the charge is \$20.54 to about \$21. The hard upper bound is \$24.26. If the believed \$88 of team credit is right, about \$63.7–\$67.5 is left. Please check the Databento portal's usage page to confirm.

## 2. Strata and the 2026-09-16 relabel

`m2/sample/strata.py` tags every symbol-day without editing stored results. `m2.extract.DAY_TYPE["2026-09-16"]` is still `vix_high`. In the new metadata (`results/strata.csv`) 2026-09-16 carries **both** `vix_high` and `fomc`, because the FOMC decision was that day. In the pilot those 10 XNAS rows move to the FOMC stratum, and the stored label is kept in the `stored_day_type` column.

| stratum | days | symbol-days |
| --- | --- | ---: |
| FOMC | 2025-12-10, 2026-01-28, 2026-09-16 (GLBX CL), 2026-10-28 | 37 (30 XNAS + 7 GLBX) |
| ES roll | roll Thursday 2025-12-11 / 2026-09-10, FOMC-roll Wednesday 2026-09-16, expiry 2025-12-19 / 2026-09-18, first day after the switch 2025-12-22 / 2026-09-21 | 12 (7 c.0 + 5 c.1) |
| CL expiry | last trade 2026-09-22 / 2026-10-20 (c.0 + c.1), day after 2026-09-23 / 2026-10-21 | 6 |
| holiday / early close | 2025-12-24 (NYSE 13:00 close; CME halt), 2025-12-26 (normal hours, thin), 2026-01-19 (MLK: NYSE closed, CME halt), 2026-07-03 (Independence Day observed: NYSE closed, CME halt) | 28 |
| normal week, back month | 2026-10-08 ES.c.1, CL.c.1 | 2 |
| **baseline** | pilot normal week 2026-10-05..09 | 60 (50 XNAS + 10 GLBX) |

Priority for multi-tag days: holiday, then ES roll, then CL expiry, then FOMC. So ES.c.0 on 2026-09-16 counts as ES roll, and CL.c.0 on that day counts as FOMC. **Caveat:** CL.c.0 on 2026-01-19 is also the day before CLG6 expires. Only 3,503 records are left on the front month that day, so it behaves like an expiry day.

## 3. Results by stratum: v3 checks

Mean candidates per symbol-day, with the pilot preset (`cfg_for` from `analyze_v2.py`: v3 presets, tick table, stale reference). **Validation:** recomputed on the 60 pilot normal-week symbol-days, `missing_intervals` equals the stored `results_real_v3/run_index.csv` on 60/60. `crossed_books` is 0 everywhere.

| group | stratum | n | extreme | ts inversions | missing | stale | spreads | sequence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XNAS | normal week | 50 | 1.1 | 0.4 | 107.8 | 1.4 | 0.1 | 0.0 |
| XNAS | FOMC | 30 | 12.2 | 1.2 | 136.2 | 4.6 | 2.2 | 1.0 |
| XNAS | holiday | 20 | 4.8 | 0.5 | 154.7 | 1.4 | 2.4 | 0.5 |
| GLBX c.0 | normal week | 10 | 2.3 | 103.2 | 480.0 | 0.1 | 0.1 | 0.5 |
| GLBX c.0 | FOMC | 7 | 3.1 | 149.7 | 641.3 | 0.3 | 0.4 | 0.3 |
| GLBX c.0 | ES roll | 7 | 0.0 | 9.4 | 351.1 | 0.0 | 0.7 | 0.1 |
| GLBX c.0 | CL expiry | 4 | **335.0** | 177.2 | 344.5 | 4.0 | **14.8** | 0.5 |
| GLBX c.0 | holiday | 8 | 3.2 | 25.1 | **1205.1** | 2.5 | 1.0 | 0.5 |
| GLBX c.1 | ES roll | 5 | 0.0 | 11.0 | 154.4 | 0.0 | 1.0 | 0.6 |
| GLBX c.1 | CL expiry | 2 | 2.5 | 398.5 | 736.5 | 0.0 | 0.0 | 0.0 |
| GLBX c.1 | normal week | 2 | 82.0 | 91.5 | 616.5 | 0.0 | 2.0 | 0.0 |

Per-symbol-day counts against each symbol's normal-week mean are in `v3_vs_normal_by_symbol_day.csv`.

- **FOMC.** On XNAS, FOMC days add extreme-price candidates: 365 against 32 expected from the same symbols' normal-week means. Spread candidates go 66 vs 2, stale 139 vs 41, and sequence 30 vs 0. Of the 431 extreme or spread candidates on the three XNAS FOMC days, 38% fall between 13:55 and 15:30 ET, the window of the statement and press conference. Those are most likely market moves. The rest are spread through the session. GLBX missing intervals are higher on CL FOMC days (585–1,330 against a CL normal mean of 838 over a 24 h window), but that is within the CL range. ES is 96–378 against 122.
- **Engine (XNAS).** **0 alerts on every symbol-day** in every stratum, and the same 0 on the 50 normal-week days. So the engine is quiet on real Nasdaq data through FOMC and the Christmas Eve early close. Neither its `crossed_quote` nor its gap rules fire. This matches the bundled AAPL sample (0) in `docs/integration.md`.

## 4. Rolls: front vs back month

`ES.c.0`/`CL.c.0` are **calendar-ranked**: c.0 is the nearest expiry until it expires, and the switch happens on the next session. Volume moves to the next contract days before that. `futures_volume.csv` shows the exchange volume per contract:

| day | ES.c.0 volume | ES.c.1 volume | note |
| --- | ---: | ---: | --- |
| 2025-12-11 (roll Thursday) | 1,511,522 | 65,351 | front still dominant |
| 2026-09-10 (roll Thursday) | 1,510,857 | 62,919 | front still dominant |
| 2026-09-16 (FOMC Wednesday) | 158,661 | 1,424,863 | **back month carries 90%** |
| 2026-09-18 (expiry Friday) | 7,742 | 1,275,838 | c.0 trades only until the 08:30 CT settlement |
| 2025-12-19 (expiry Friday) | 8,326 | 1,257,206 | same |
| 2026-09-21 / 2025-12-22 (after switch) | 1,289,899 / 932,188 | — | c.0 is the new contract |

CL behaves the same way. On the last trade day (2026-09-22, 2026-10-20) CL.c.0 has 26,237 and 22,062 records, against 2.86 M and 2.55 M on CL.c.1.

How the rules behave:

- **Missing-interval rule.** It follows liquidity, not the label. ES.c.0 counts are 55 (12-11) and 206 (09-10) on roll Thursdays, where normal is 122. They rise to 213 on 09-16 and **480 / 850 on expiry Fridays**, then fall back to 107 the day after the switch. 12-22 has 547, but it is Christmas week. Expiry-day gaps on c.0 are candidates on an expiring, near-empty contract, not feed loss. The back month c.1 has 72–255 on the same days.
- **Spread rule.** ES spreads stay at 0–3 candidates/day through the roll, because ES keeps a one-tick book even when thin. The CL front month on its **last trade day** is where spreads and prices break down: `unusual_spreads` 38 and 20 against 0.2 normal, and `extreme_prices` **601 and 732** against 4.6. The next day (c.0 = new contract) is back to 0–1 and 3–4. These are candidates produced by an expiring contract, not data errors.
- **Control charts.** Phase-1 charts are built on the c.0 normal-week history. Mean signals per symbol-day per chart are 1.5 on the normal week (by calibration). They rise to 70–137 for the rate charts and 78–121 for the spread charts on ES roll days, and 97–206 / 268–431 on CL expiry days. Back-month series compared with the front-month history give 152–709. The rate charts correctly show that activity on the contract has collapsed. **SPC should be fed the most-active contract (volume-ranked `.v.0`, or a roll-adjusted series), not calendar `.c.0`, if roll weeks are not meant to alarm.** No baseline was changed here.

## 5. Holidays and the opt-in holiday calendar

`m2/sample/holidays.py` is an **opt-in sidecar**. `holiday_config(cfg, venue, root)` returns a deep copy of a v3 config in which every session calendar (missing, stale, extreme, spreads) has its holidays and early closes filled in. The v3 presets themselves are unchanged (tested). It contains NYSE closures and 13:00 ET early closes for 2025–26. It also has CME equity and energy holiday halts in CT: on Monday holidays and Thanksgiving, ES halts at 12:00 and CL at 13:30. On Black Friday and Christmas Eve, ES halts at 12:15 and CL at 12:45. On Juneteenth and Independence Day, both halt at 12:00. Sources are CME holiday calendar notices. `halt_check` confirms the times against the data:

| symbol-day | calendar close | last trade |
| --- | --- | --- |
| ES 2025-12-24 | 12:15 CT | 12:14:59.51 CT |
| CL 2025-12-24 | 12:45 CT | 12:44:59.00 CT |
| ES 2026-01-19 | 12:00 CT | 11:59:59.87 CT (a few quote records until 15:10 CT) |
| CL 2026-01-19 | 13:30 CT | 12:56:31 CT (thin expiring contract) |
| ES / CL 2026-07-03 | 12:00 CT | 11:59:59.66 / 11:59:59.04 CT |
| NYSE 2025-12-24 | 13:00 ET | regular session ends 13:00. Thin names (LWAY, ADTN, PLAB) trade last at 13:00:00.1. The large names keep trading after hours until 17:00. |

**Does the calendar suppress expected-quiet false candidates?** Yes on XNAS. It helps little on GLBX, where the cause is thinness rather than a missing close time.

| system (holiday symbol-days) | default calendar | holiday calendar |
| --- | ---: | ---: |
| XNAS v3 `missing_intervals` (10 sd, 2025-12-24) | 1,548 | 972 (SPY, QQQ, NVDA → 0; AAPL 86 → 3) |
| XNAS v3 `unusual_spreads` / `extreme_prices` / `stale` | 48 / 82 / 14 | 0 / 9 / 4 |
| XNAS Bayes candidates / Poisson | 1,518 / 1,548 | 885 / 972 (gaps > 60 s: 258 → 45) |
| XNAS SPC signals (rate + spread) / episodes | 3,098 / 170 | 164 / 56 |
| GLBX v3 `missing_intervals` (6 sd) | 5,955 | 5,910 (ES 01-19 163 → 118) |
| GLBX SPC signals (rate + spread) / episodes | 8,179 / 206 | 5,389 / 186 |

The XNAS gain comes from the post-13:00 ET buckets, which the RTH preset expects to be busy. What is left on 2025-12-24 sits in genuinely thin small caps before 13:00: NBIS 416, CRWV 189, PLAB 127, ADTN 94 and LWAY 65 gap candidates. On CME, the feed simply goes silent after the halt. A gap needs a record after it, so there is almost nothing to suppress except ES on 01-19, where scattered quotes continue until 15:10 CT. The large GLBX holiday counts are thin-market gaps inside the open session: ES 1,810 (12-24) and 1,948 (12-26) against a normal mean of 122, and CL 2,036 and 1,738 against 838. Only the per-day-type prior or rate (the Bayes prior, a holiday-specific SPC baseline) can address those. On SPC the calendar removes the post-halt buckets, which brings rate signals from 4,241 to 3,096.

**Recommendation.** Offer `holiday_config` as an opt-in for production runs. Do not make it the default until the team owns the calendar data and its yearly update. Pair it with a holiday-specific baseline for the thin-market effect.

**Known limitation.** v3 `SessionCalendar.early_closes` cuts every window on that local date. On CME this also removes the 17:00 CT holiday-evening reopen. The sample windows end at 16:00 CT, before that reopen, so they are not affected. A full-day CME holiday run would need a per-window close.

## 6. Bayesian gaps

Default clock. Poisson-rule and posterior candidates are mean per symbol-day.

| group | stratum | n | Poisson | Bayes |
| --- | --- | ---: | ---: | ---: |
| XNAS | normal week | 50 | 107.8 | 93.6 |
| XNAS | FOMC | 30 | 136.2 | 118.9 |
| XNAS | holiday | 20 | 154.7 | 143.7 |
| GLBX c.0 | normal week | 10 | 480.0 | 484.6 |
| GLBX c.0 | FOMC | 7 | 641.3 | 644.6 |
| GLBX c.0 | ES roll | 7 | 351.1 | 323.1 |
| GLBX c.0 | CL expiry | 4 | 344.5 | 286.0 |
| GLBX c.0 | holiday | 8 | 1205.1 | 1200.4 |

The rate prior is the pilot's normal-week rate per symbol. With a gamma-Poisson prior $\lambda \sim \Gamma(\alpha,\beta)$ and a gap of length $g$ the posterior probability of an outage moves in proportion to $\exp(-\lambda g)$. A thin day with a normal-day $\lambda$ gives many candidates whose posterior is near 1. This is the expected effect on holidays and expiries. A day-type prior $\lambda_{\text{holiday}} \approx \lambda/k$ would rescale it.

## 7. Limitations

- One or two days per stratum, and the 10 pilot symbols plus ES and CL. The counts describe these days and should not be treated as rates.
- Engine results are XNAS only, as before. The engine sees no futures-specific inputs.
- The SPC back-month comparison uses front-month history. It shows the mismatch on purpose.
- The actual charge depends on how Databento bills the three timed-out partial streams (see §1).
