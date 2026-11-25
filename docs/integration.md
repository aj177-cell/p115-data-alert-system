# Combining Milestone 1 with detectors v3

This branch runs two independent detector sets on the same Databento records and merges their output, **without editing either one**.

| Layer | Where | Status |
| --- | --- | --- |
| Milestone 1: pipeline, normalization, online alert engine (3 rules) | `README.md`, `requirements.txt`, `run_checks.py`, `databento_pipeline.py`, `alert_system/`, `tests/`, `sample_data/`, `docs/research.md` | unchanged (tag `milestone-1`) |
| detectors v3: 7 vectorized checks + vendor flags, presets, synthetic faults, 95 symbol-day pilot | `p115_detectors/` (contents of `p115_detectors_v3_final.zip`) | unchanged |
| Integration layer | `integration/`, `run_combined.py`, `docs/integration.md`, `requirements-combined.txt` | new |

## Quick start (merged package `p115_combined.zip`)

Everything is already in place: Milestone 1 at the repository root, v3 in `p115_detectors/`, and the integration layer.

```bash
python -m pip install -r requirements-combined.txt
python -m integration.verify --run-tests   # checksums + all three test suites
python run_combined.py                     # or --sample trades, --stress
```

To stay small, the merged zip leaves out one v3 file: `p115_detectors/results_real/allflags.parquet` (v1 flags, 38 MB, read only by `make_charts_v1.py`). `verify` therefore lists it as missing under "detectors v3 results", which does not fail the check. Copy it from `p115_detectors_v3_final.zip` if you need it.

## Setting up the branch (add-on `p115_combine_addon.zip`)

From the repository root, on a clean checkout of the Milestone 1 commit:

```bash
git tag milestone-1                      # freeze the submitted state
git switch -c combine-detectors
unzip -n p115_combine_addon.zip          # adds integration/, run_combined.py, this file, requirements-combined.txt
unzip p115_detectors_v3_final.zip -d p115_detectors
python -m pip install -r requirements-combined.txt
python -m integration.verify --run-tests
```

`unzip -n` never overwrites an existing file; none of the add-on paths exist in Milestone 1.

## Why the two fit

- **Same input.** v3 reads the `data.parquet` written by `databento_pipeline.download_and_show()` (`ts_recv` index, float prices, UTC timestamps). The bundled samples in `sample_data/` have the same layout, so both systems read the same file.
- **Same environment.** The 34 v3 tests pass under the Milestone 1 pins (pandas 3.0.6, pyarrow 26.0.0; numpy 2.5.4).
- **No import clash.** v3 imports the top-level names `config`, `detectors` and `synthetic` from `p115_detectors/detectors/`. Milestone 1 uses `alert_system.config` and `alert_system.detectors`. `integration.combine.load_v3()` raises if a different top-level `config`, `detectors` or `synthetic` is already imported.

## Proof that the combination is add-only

```bash
python -m integration.verify              # SHA-256: 16 Milestone 1 files, 20 v3 code files (+403 v3 result files)
python -m integration.verify --run-tests  # also: 41 Milestone 1 tests, 34 v3 tests, 13 integration tests
git diff --name-status --diff-filter=MDRT milestone-1   # prints nothing: no file modified, deleted, renamed
```

The v3 result files are reported but not required, because re-running the v3 scripts rewrites them legitimately.

## How to run

```bash
python -m pip install -r requirements-combined.txt
python run_checks.py                      # Milestone 1 alone, exactly as before
python run_combined.py                    # both systems, bundled AAPL quotes
python run_combined.py --sample trades
python run_combined.py --local <path under data/>/data.parquet --symbol AAPL --start "2026-10-14 09:30" --end "2026-10-14 16:00"
python run_combined.py --stress           # v3 synthetic stream with labeled faults, both systems scored
python run_combined.py --output-dir out/  # engine_m1_alerts.json, detectors_v3_flags.parquet, combined_findings.csv, overlap.csv
```

The v3 package still runs on its own from `p115_detectors/` (`python -m pytest detectors`, `python analyze_v2.py`, ...), as its README describes.

## What the integration layer does

- Calls Milestone 1 only through `normalize_frame` and `AlertEngine.process`, in file order, as `run_checks.py` does. For a bundled sample the events and alerts are identical to `run_checks.py` (tested).
- Files above `MAX_EVENTS` (250,000) are passed in chunks to the same engine instance; event ids are renumbered to the row in the whole file. Chunked and single runs give identical alerts (tested).
- Calls v3 only through `run_all`, with the presets the pilot used: `equities_config()` for equity datasets; `cme_futures_config()` with sequence gaps skipped for `GLBX.MDP3` (symbol-filtered pulls).
- Milestone 1 supports equity `trades` and `mbp-1` only. For other data the engine is skipped with a message and v3 still runs.
- Keeps both native outputs and adds one table, `combined_findings.csv`:

| Column | Meaning |
| --- | --- |
| `source` | `engine_m1` or `detectors_v3` |
| `mode` | `online` (record by record) or `batch` (whole frame) |
| `rule`, `subtype` | engine `rule_id`; v3 `check` and `subtype` |
| `severity` | native value (`critical`/`warning`; `info`/`low`/`medium`/`high`) |
| `level` | adapter convention: info 0, low 1, medium or warning 2, high or critical 3 |
| `ts_event`, `ts_recv` | evidence time; for the engine, `ts_recv` is `detected_at` |
| `symbol`, `publisher_id`, `instrument_id` | record identity |
| `value`, `threshold`, `detail` | v3 values as emitted; for the engine, the headline observed value and threshold, and the explanation |
| `f_last` | engine only: whether the record that raised the alert had `F_LAST` set |

The engine's full alert objects (observed values, possible causes, next checks, references) stay in `engine_m1_alerts.json`.

## Coverage of the task

| Task item | engine_m1 (Milestone 1) | detectors_v3 |
| --- | --- | --- |
| Crossed books | `crossed_quote`: any record with bid > ask | `crossed_books`: F_LAST records, runs of at least 1 ms, locked books as a separate subtype |
| Extreme prices | planned for Milestone 2 | `extreme_prices`: robust MAD z-score of log returns, k = 12 |
| Timestamp inversions | `time_order`: ts_event below the running maximum of its scope | `timestamp_inversions`: ts_recv inversion, ts_event inversion, negative latency, latency outlier, saturated ts_in_delta |
| Missing intervals | none | `missing_intervals`: Poisson rule on a session clock, 5 s floor |
| Stale quotes | none | `stale_quotes` |
| Unusual spreads | planned for Milestone 2 | `unusual_spreads` |
| Sequence gaps | none | `sequence_discontinuities` (gaps skipped on XNAS, where `sequence` is a feed-wide counter) |
| Record integrity | `integrity` | none: undefined prices become NaN and are skipped |
| Databento flags | none | `vendor_flags` (`F_BAD_TS_RECV`, `F_MAYBE_BAD_BOOK`) |
| Explanations | possible causes and next checks per alert | `detail` string |

## Stress test (`python run_combined.py --stress`)

The stream and the seven fault types come unchanged from `p115_detectors/detectors/synthetic.py` (1 h, 2 instruments, 109,714 records, 5 faults per type). The adapter adds 6 `integrity` faults (zero bid or undefined ask on a nonempty side), because the v3 injector has none that the integrity rule targets. Matching follows `evaluate.py` (same publisher and instrument, ±1 s, `info` ignored); on the v3 flags it reproduces `evaluate.match()` exactly (tested). The engine is scored with cooldown off.

| Fault family | Labels | engine_m1 | detectors_v3 | Either |
| --- | ---: | ---: | ---: | ---: |
| crossed_books | 5 | 5 | 5 | 5 |
| extreme_prices | 5 | no rule | 5 | 5 |
| timestamp_inversions | 5 | 1 | 5 | 5 |
| missing_intervals | 5 | no rule | 5 | 5 |
| stale_quotes | 5 | no rule | 5 | 5 |
| unusual_spreads | 5 | no rule | 5 | 5 |
| sequence_discontinuities | 5 | no rule | 5 | 5 |
| integrity (adapter) | 6 | 6 | no rule | 6 |

Together the two systems find all 41 labels; alone, the engine finds 12 and v3 finds 35. On the clean stream v3 raises no findings. The engine raises 3,273 `crossed_quote` alerts there, all on records without `F_LAST`, which the generator deliberately makes crossed to show that book checks must skip them.

## Differences for the team to decide in Milestone 2

1. **Crossed quotes.** The engine checks every record; v3 checks only `F_LAST`, non-snapshot records and requires a 1 ms run. Where both apply they agree: each of the 5 injected crosses has engine alerts within ±1 s of the v3 flag. Records without `F_LAST` are the only source of engine false positives on the synthetic stream; on the bundled AAPL sample both systems report 0, and the effect on real data is untested. The adapter only marks such alerts (`f_last = False`).
2. **Timestamp order.** On the overlapping definition the two agree: engine `time_order` and v3 `ts_event_inversion` both catch the one synthetic fault that actually reorders ts_event. The other four (ts_recv moved back, ts_event after ts_recv, a latency spike that does not reorder records) are caught only by v3's other subtypes.
3. **Timing.** The engine is online and past-only (`docs/research.md`). v3 is batch: `latency_outlier` uses the median and MAD of the whole key, and run-based flags (crossed, stale, wide spread) are stamped at the run start but known only at its end. An online replay needs a `detected_at` per v3 flag or past-only variants.
4. **Severity.** The scales differ; `level` is only an adapter convention.
5. **Trades schema.** v3 was tuned on mbp-1. On the bundled trades sample, `missing_intervals` flags two pauses of about 5.2 s in trade prints (medium); these are probably normal pauses and should be reviewed before they are reported as data gaps.
6. **v3 housekeeping.** `p115_detectors/detectors/results/metrics.csv` and the synthetic table in its README were not regenerated with the v3 thresholds: re-running `evaluate.py` gives 12 `extreme_prices` flags (stored: 15) and 0 on the clean stream (stored: 1); precision and recall are unchanged.
