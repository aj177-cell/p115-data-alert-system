# Batch data-quality checks

These checks read the Parquet written by `databento_pipeline.py` (`data.parquet`, indexed on `ts_recv`, the `DBNStore.to_df()` layout). The same files are what `alert_system.data.load_data` normalizes into events. The batch checks use the pipeline table, not the event dicts: normalization renames `bid_px_00` to `bid`, drops `flags`, and stores a quote midpoint in `price`.

Streaming rules stay in `alert_system/detectors.py`. They are not a second copy of these checks. See the README for the two overlaps.

Thresholds live in `alert_system/quality/config.py`. The values there are the current set: v2 settings, with the missing-interval floor at 5 seconds. `equities_config()` is the XNAS preset (sequence gaps skipped, default tick 0.01). `cme_futures_config()` is the GLBX preset.

## Run

On a bundled sample, together with the streaming rules:

```bash
python run_checks.py --mode both
python run_checks.py --mode both --sample trades
```

On a file the pipeline already wrote (no API call):

```bash
python run_checks.py --mode batch --path data/downloads/<run>/data.parquet
python run_checks.py --mode batch --preset cme --path data/downloads/<run>/data.parquet
```

A directory is scanned for `data.parquet` only, so aggregated bar files in the same folder are left alone.

```python
from alert_system.quality import equities_config, run_all
from alert_system.quality.inputs import read_pipeline_parquet

flags, summary = run_all(read_pipeline_parquet("data.parquet"), equities_config())
```

Synthetic scoring (writes `alert_system/quality/results/`, which is local output and not part of the repo):

```bash
python -m alert_system.quality.evaluate
```

Flag columns: `ts_recv, ts_event, publisher_id, instrument_id, symbol, check, subtype, severity, value, threshold, detail`. Severity order is `info < low < medium < high`.

Rows are grouped by `(publisher_id, instrument_id)`, except sequence, which groups by `publisher_id` unless `per_instrument` is set. Book-state checks use records with `F_LAST` (128) set and `F_SNAPSHOT` (32) clear. `F_BAD_TS_RECV` (8) downgrades a receive-time inversion to `info`. `vendor_flags` also reports `F_BAD_TS_RECV` and `F_MAYBE_BAD_BOOK` (4).

## Checks

1. **Crossed / locked books** (`crossed_books`). On `F_LAST` rows with both prices defined, a run with bid above ask, or bid equal to ask, is flagged when it lasts at least `min_duration_s` (0.001 s). One flag per run. Windows listed on the config, and optional status rows where `is_trading != 'Y'`, are severity `info`.
2. **Extreme prices** (`extreme_prices`). Log return of trade prints (`action == 'T'`, side `N` excluded) and of the book mid. With a trailing window of 500 and a 50-observation warm-up, flag `|z| > k` with `k = 12` and scale `1.4826 * max(MAD, max(tick/price, 1e-4))`. Only inside the session calendar.
3. **Timestamp inversions** (`timestamp_inversions`). Consecutive `ts_recv` decrease (high, or info if `F_BAD_TS_RECV`); consecutive `ts_event` decrease (low); `ts_recv - ts_event < -1 µs` (negative latency); latency above `median + 10 * 1.4826 * MAD` and at least 5 ms above the median on the equities preset (20 ms on the CME preset). Outliers of one key less than 1 s apart collapse to one episode. Saturated `ts_in_delta` (`±(2^31-1)`) is reported at low severity.
4. **Missing intervals** (`missing_intervals`). In-session gap `g` between consecutive `ts_recv` values. Flag when

   `g > min(max(-ln(alpha) / lambda_hat, 5 s), 60 s)`

   with `alpha = 1e-6` and `lambda_hat` the trailing rate over 500 arrivals. The 5 s floor and 60 s cap are `min_gap_s` and `max_gap_s`. `bar_completeness` counts 1-minute bars inside a 5m/15m/30m bucket and is separate from `run_all`.
5. **Stale quotes** (`stale_quotes`). Best bid, ask, and sizes unchanged for at least 5 s in session. The interval is stale when the instrument prints at least 10 trades during it, or a configured reference mid moves at least 5 bp. Quiet intervals are not reported by default.
6. **Unusual spreads** (`unusual_spreads`). Spread in ticks. A non-positive spread is `zero_or_negative` (also visible to the crossed-book check; both reports are kept). A wide run must last at least 0.5 s and exceed `max(3 * q95, q95 + 3)` ticks on a trailing 15-minute window. Session gated. `equities_config()` uses a 0.01 default tick when the definition tick is missing.
7. **Sequence discontinuities** (`sequence_discontinuities`). `delta = seq_i - seq_{i-1}`. `delta` of 0 or 1 is allowed. `delta > 1` is a gap; `delta < 0` is a regression. Rows with `sequence == 0` are skipped. On XNAS the counter is feed-wide, so a symbol-filtered pull shows gaps that are other symbols' messages. `equities_config()` sets `gap_mode='skip'` and still reports regressions.

`vendor_flags` is the extra feed-condition check, not one of the seven.

## Pilot

`docs/results/flags_by_check.csv` and `docs/results/run_index.csv` summarize a 95 symbol-day Level-1 run (XNAS.ITCH and GLBX.MDP3) on this config. Those rows are candidates. No check in that run has a confirmed vendor error. Per-record flag tables and the chart inputs that needed them are not in the repo. `python -m alert_system.quality.plot_pilot --output pilot.png` charts the two summary files.

## What the synthetic scores mean

`evaluate.py` injects clear faults into an hour of synthetic MBP-1. A flag matches a label inside ±1 s. On that set the checks score at the top of the range (extreme prices F1 0.89, the others 1.00), with a handful of clean-stream extreme-price flags. That shows the plumbing, not field precision. Lowering `k`, `alpha`, or the spread multiplier admits many more flags; the config comments record why the current numbers were left where they are.
