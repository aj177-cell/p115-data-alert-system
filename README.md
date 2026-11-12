# P115 Data Alert System

## Project goal

Build a system that reads historical stock and ETF market events (individual trades and level-one quotes) from Databento in their original order, flags records or market changes that merit review, and explains each alert with the observed values, possible causes, and next checks.

A flagged row is a **candidate for verification**. It is not a confirmed data error. The 95 symbol-day pilot did not confirm a vendor error, and a clean sample that raises nothing does not measure detection accuracy.

## What is in the system

Two check families share the pipeline output and stay separate where they overlap. `databento_pipeline.py` and the loader are unchanged.

| Piece | Module | Author |
| --- | --- | --- |
| Download, DBN store, Parquet export, manifests | `databento_pipeline.py` | Yuyang An |
| Validation and normalization into one event format | `alert_system/data.py` | Yuyang An |
| Streaming checks: record integrity, event-time order, immediate crossed quote | `alert_system/detectors.py` | Yuyang An |
| Streaming research notes | `docs/research.md` | Yuyang An |
| Seven batch checks, vendor flags, threshold presets, synthetic faults | `alert_system/quality/` | Aryan Jhaveri |
| Threshold notes and check definitions | `docs/quality_checks.md`, `alert_system/quality/config.py` | Aryan Jhaveri |
| 95 symbol-day Level-1 pilot summaries | `docs/results/` | Aryan Jhaveri |
| Literature mapping for the four papers in the research notes | `docs/literature_appendix.md` | Aryan Jhaveri |

The streaming engine walks normalized events one at a time and can suppress repeat alerts. The batch checks are vectorized over the pipeline Parquet (`ts_recv` index, `bid_px_00`, `flags`, `sequence`). That is the table `download_and_show` writes and the table the loader reads. The loader's event dicts are a different shape (quote `price` becomes a midpoint, and `flags` is dropped), so the batch checks are not pointed at the event list.

### Streaming rules

| Rule | What it checks |
| --- | --- |
| Record integrity | Invalid timestamps, nonfinite fields, nonpositive equity prices, or negative sizes |
| Event time moved backwards | An event timestamp earlier than the latest event time already seen in the same source scope |
| Crossed quote | Bid above ask on one valid two-sided quote (both prices positive, both sizes nonzero) |

Design notes: [docs/research.md](docs/research.md).

### Batch checks

| Check | What it checks |
| --- | --- |
| `crossed_books` | A run of `F_LAST` book updates with bid above ask, or bid equal to ask, lasting at least 1 ms |
| `extreme_prices` | Robust z-score of log returns on trades and mids, `|z| > 12`, inside the session |
| `timestamp_inversions` | Consecutive `ts_recv` or `ts_event` decreases, negative latency, latency episodes, saturated `ts_in_delta` |
| `missing_intervals` | In-session gap longer than `min(max(-ln(alpha) / lambda_hat, 5 s), 60 s)` |
| `stale_quotes` | A best bid/ask that sits still for at least 5 s while trades print or a reference mid moves |
| `unusual_spreads` | A non-positive spread, or a wide spread that persists at least 0.5 s |
| `sequence_discontinuities` | Sequence regressions, and gaps when the preset asks for them |
| `vendor_flags` (extra) | Databento `F_BAD_TS_RECV` and `F_MAYBE_BAD_BOOK` |

`equities_config()` skips sequence gaps. On a symbol-filtered XNAS pull the sequence counter is feed-wide, so a gap between two AAPL records is usually other symbols' messages, not a hole in AAPL. Regressions are still reported. Definitions, presets, and the synthetic scorecard: [docs/quality_checks.md](docs/quality_checks.md).

### Overlap

**Crossed quote and crossed books both stay.** The streaming rule fires on the current normalized quote as soon as bid exceeds ask, including a cross that lasts one update, and it does not flag a locked quote (`bid == ask`). The batch rule keeps only `F_LAST` non-snapshot book rows, ignores a cross shorter than 1 ms, emits one flag for the whole run, and reports locked books as their own subtype. `unusual_spreads` also reports a non-positive spread, which Aryan's check documents as an intentional second view of the same book state. Collapsing these into one rule would drop either the immediate per-record alert or the duration, locked-book, and `F_LAST` logic.

**Event-time order and timestamp inversions both stay.** The streaming rule compares event time with the high-water mark in a scope of dataset, symbol, publisher, schema, and kind, so a record can be late relative to an earlier record that is not its neighbor. The batch rule compares each record with the previous record in file order inside `(publisher_id, instrument_id)`. It also flags receive-time reversals, `ts_recv` earlier than `ts_event` by more than 1 µs, latency episodes, and a saturated `ts_in_delta`. Those signals are not in the streaming rule.

Record integrity has no batch twin. It remains the structural screen on the normalized event.

### Pilot results

[docs/results/flags_by_check.csv](docs/results/flags_by_check.csv) and [docs/results/run_index.csv](docs/results/run_index.csv) summarize 95 symbol-days of Level-1 data (XNAS.ITCH equities and GLBX.MDP3 futures) on the current thresholds. Counts by check:

| Check | Flags |
| --- | ---: |
| crossed_books | 25 |
| extreme_prices | 148 |
| missing_intervals | 14,129 |
| sequence_discontinuities | 5 |
| stale_quotes | 158 |
| timestamp_inversions | 1,252 |
| unusual_spreads | 26 |
| vendor_flags | 15 |

Treat every count as a candidate. Per-record flag tables are large and are not in the repo. `python -m alert_system.quality.plot_pilot --output pilot.png` charts the two summary files.

How the four papers in the research notes relate to these checks: [docs/literature_appendix.md](docs/literature_appendix.md).

## Included data

| Sample | Dataset and schema | Interval (New York time) | Events |
| --- | --- | --- | ---: |
| AAPL quotes | `XNAS.ITCH / mbp-1` | September 15, 2025, 10:30:00-10:30:30 | 2,109 |
| AAPL + SPY trades | `XNAS.ITCH / trades` | September 15, 2025, 10:30:00-10:45:00 | 11,596 |

## How to run

Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_checks.py                       # streaming checks, AAPL quotes
python run_checks.py --sample trades       # streaming checks, AAPL + SPY trades
python run_checks.py --mode both           # streaming checks and the seven batch checks on the quotes sample
python run_checks.py --mode both --sample trades
python run_checks.py --mode batch --path path/to/data.parquet
python run_checks.py --output alerts.json  # streaming alerts
```

`--preset equities` is the default for batch runs (XNAS pilot settings). Use `--preset cme` for the GLBX futures preset and `--preset base` for `DetectorConfig()` with sequence gaps enabled. `--path` reads a local Parquet file or a directory of pipeline `data.parquet` files. It does not call Databento.

No API key is needed for the bundled samples. Tests:

```bash
python -m unittest discover -s tests -v
python -m pytest alert_system/quality/tests -q
```

The pytest file is the batch-detector suite (34 tests) on synthetic streams. The unittest suite covers the pipeline, the loader, the streaming rules, and a combined run on the AAPL/SPY samples. Neither suite downloads data.

### Initial result on the bundled samples

Both samples pass the three streaming checks; no streaming alerts are raised. That is expected for short, clean exchange samples. The unit tests use constructed records to confirm that each streaming rule fires when it should.

On the equities batch preset, the 30-second AAPL quote sample raises no flags. The 15-minute AAPL + SPY trade sample raises two `missing_intervals` candidates, one per symbol, each an in-session gap of about 5.2 seconds (the floor is 5 seconds). A pause between trades can be a quiet stretch of a liquid name. Those two rows are candidates for verification, not a finding that records were dropped.

### Optional: downloading more data

Downloading additional history requires your own Databento credentials and may incur charges. Place a `.env` file one directory above this folder:

```dotenv
DATABENTO_API_KEY=your_databento_api_key
DATABENTO_HISTORICAL_BASE_URL=https://hist.databento.com
DATABENTO_TIMESERIES_ENDPOINT=https://hist.databento.com/v0/timeseries.get_range
```

After `download_and_show` writes a run directory, point `--mode batch --path` at its `data.parquet`.

## Still open

1. **Dashboard**: a local browser view of the chart, the event stream, and the alert explanations.
2. **Event replay** on a simulated clock, and **stress testing** that injects known anomalies into the streaming path end to end.
3. **One alert document** for both families. Streaming alerts carry an explanation, possible causes, and next checks. Batch output is a flag table. They are printed together and not yet stored as one schema.
4. **Sequence semantics** on XNAS and GLBX (feed-wide versus per instrument) are still a question for the sponsor. The equities preset skips gaps for that reason.
5. **Holiday calendar.** The session clock has the regular 09:30–16:00 America/New_York window and the 2025-11-28 13:00 early close. It does not yet use a full exchange calendar.
6. **Unusual trade size** and **queue-imbalance prediction** are not checks. The literature appendix says queue imbalance predicts the next mid move and is out of scope for data-quality detection.
