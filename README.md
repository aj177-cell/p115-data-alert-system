# P115 Data Alert System — Milestone 1

## Project goal

Build a system that reads historical stock and ETF market events (individual trades and level-one quotes) from Databento in their original order, flags records or market changes that merit review, and explains each alert with the observed values, possible causes, and next checks.

## Milestone 1 scope

This milestone covers the data foundation and the first data-quality checks.

Completed:

- **Data acquisition** (`databento_pipeline.py`): Databento historical client configuration from `.env`, time-zone-aware request handling, download to DBN with Parquet export, and a credential-free manifest for each download. Also includes preview and plotting helpers used for initial data exploration.
- **Validation and normalization** (`alert_system/data.py`): checks the dataset and schema, rejects unsupported asset classes and schemas, and converts `trades` and `mbp-1` records into one event format. Nanosecond timestamps and receive time (`ts_recv`) are preserved.
- **Three data-quality rules** (`alert_system/detectors.py`):

| Rule | What it checks |
| --- | --- |
| Record integrity | Invalid timestamps, nonfinite fields, nonpositive equity prices, or negative sizes |
| Event time moved backwards | An event timestamp earlier than the previous one in the same source scope |
| Crossed quote | Bid above ask within one valid two-sided quote |

- **Sample data** with provenance (request parameters and SHA-256 checksums).
- **Unit tests** for the pipeline, normalization, and rules.

Design notes for the rules are in [docs/research.md](docs/research.md).

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
python run_checks.py                  # AAPL quotes sample
python run_checks.py --sample trades  # AAPL + SPY trades sample
python run_checks.py --output alerts.json
```

No API key is needed for the bundled samples. Run the tests with:

```bash
python -m unittest discover -s tests -v
```

### Initial result

Both original samples pass all three checks; no alerts were raised. This is expected for short, clean exchange samples and does not measure detection accuracy. The unit tests use constructed records to confirm that each rule fires when it should.

### Optional: downloading more data

Downloading additional history requires your own Databento credentials and may incur charges. Place a `.env` file one directory above this folder:

```dotenv
DATABENTO_API_KEY=your_databento_api_key
DATABENTO_HISTORICAL_BASE_URL=https://hist.databento.com
DATABENTO_TIMESERIES_ENDPOINT=https://hist.databento.com/v0/timeseries.get_range
```

## Next steps

1. **Statistical market rules**: unusual price move (robust median/MAD score), unusual individual trade size, spread widening, and persistent queue imbalance. All baselines must use past data only.
2. **Event replay**: replay records on a simulated clock in arrival order, so alerts appear as they would in real time.
3. **Stress testing**: inject known anomalies into an in-memory copy to check the end-to-end alert path.
4. **Dashboard**: a local browser interface to view the chart, event stream, and alert explanations.
5. **Calibration**: choose thresholds on earlier dates and validate on later dates.
