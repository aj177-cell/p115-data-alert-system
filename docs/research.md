# Rule Design Notes — Milestone 1

This project feeds historical Databento equity trades and top-of-book events into an online detector in their original order, without aggregating them into bars or sampling events before detection. The research question is how to identify records or market changes that merit review and provide evidence that can be checked. Each alert separates what was observed, possible causes, and the next checks to make.

Milestone 1 covers the data pipeline and the first three data-quality rules. Statistical market rules are planned for the next milestone (see the end of this file).

## Data Definitions and Causal Timing

- `timestamp` retains the original event time. `available_timestamp` represents when the record becomes available to the detector, using Databento's receive time, `ts_recv`, when available for trades and quotes. Alerts retain both the evidence time, `timestamp`, and the detection time, `detected_at`. Each incoming event is evaluated directly, without conversion into minute-level statistics.
- Arrival order is preserved. Globally sorting by event time before plotting or detection would conceal timestamp reversals. Databento documents that `ts_event` can be non-monotonic because of differences between exchange gateway clocks; a reversal alone does not establish a transmission error. The system does not infer packet loss without the original sequence/channel semantics. [Databento timestamp conventions](https://databento.com/docs/conventions/common-fields-enums-types)
- For trades, `price` is the execution price. For quotes, it is the midpoint of a valid two-sided quote. The original MBP-1 `price` is an order-event price and must not be used directly as the midpoint. This schema describes events at the best price level, not an order-by-order stream covering every price level. [Databento MBP-1](https://databento.com/docs/schemas/mbp-1)
- Each check uses only the current record and records processed before it. Nothing is evaluated with full-file statistics or future data.
- State is isolated by `dataset / symbol / publisher_id / schema / kind`, preventing different venues or trade-price and midpoint series from being mixed.
- Multiple records may legitimately share a timestamp. Time comparisons retain nanosecond precision rather than first converting timestamps into lower-precision floating-point epoch values.
- A missing quote side or an empty queue with size 0 is allowed. Related market signals are temporarily ineligible; an incomplete two-sided book is not automatically treated as bad data. Databento's `UNDEF_PRICE` should be converted into a missing value during normalization. [Databento missing-value and price conventions](https://databento.com/docs/conventions/common-fields-enums-types)

## Implemented Rules

### 1. Record Integrity — `integrity`

This rule immediately checks that timestamps include a valid UTC offset, supplied numeric values are finite, quantities are nonnegative, trades have a price, and equity prices are positive. Nonempty equity quote sides must also have positive prices. A missing quote side temporarily makes the related rules ineligible; absence alone is not classified as invalid data.

A violation produces a `critical` alert and marks the record for review. Here, `critical` means that the record should not continue to serve as a calculation input; it does not mean the entire exchange feed is unusable. The trader should check the raw record, instrument definition, and normalization logic. The data owner can then decide whether to quarantine or correct it.

### 2. Event Time Moved Backwards — `time_order`

If the current `timestamp` is earlier than the largest event time previously observed in the same scope, the rule produces a `warning` with the reversal in seconds and the previous time high-water mark. Equal timestamps are allowed. Reordered records are kept in the output. Possible causes include late events, corrections, or clock differences between exchange gateways.

The next step is to compare `ts_event`, `ts_recv`, and the source's semantics. This observation alone does not establish packet loss, excessive network latency, or fabricated market data.

### 3. Crossed Quote — `crossed_quote`

A record with valid, nonempty quote sides and `bid > ask` produces a `warning`. A locked quote, `bid == ask`, does not trigger this rule.

The project uses the publisher scope of the original record; it does not construct crosses from different instruments or unrelated records. For MBP-1, the publisher identifies the dataset and venue. Supporting consolidated schemas in the future would also require preserving the separate publisher attribution of the bid and ask. An observed cross is not evidence of an arbitrage opportunity that can be executed simultaneously. Review quote conditions, subsequent quotes, and the source's venue coverage. [Databento MBP-1 and CMBP-1 fields](https://databento.com/docs/schemas/mbp-1)

## Alert Suppression

| Parameter | Default | Meaning |
| --- | --- | --- |
| `cooldown_events` | 15 | Minimum arriving records between emitted alerts for the same scope and rule |
| `cooldown_seconds` | 30 | Minimum seconds of market availability time between alerts for the same scope and rule |

Both conditions must be satisfied before the same rule emits another alert. Suppression reduces repeated messages; the detector still processes every record.

## Literature Being Reviewed for the Next Milestone

- Brownlees & Gallo (2006), *Financial Econometric Analysis at Ultra-High Frequency: Data Handling Concerns* — cleaning and screening of high-frequency records. [Research record](https://flore.unifi.it/handle/2158/210321)
- Cont, Kukanov & Stoikov (2014), *The Price Impact of Order Book Events* — order book events and short-term price changes. [Paper](https://arxiv.org/abs/1011.6402)
- Gould & Bonart (2015 preprint), *Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book*. [Paper](https://arxiv.org/abs/1512.03492)
- Barndorff-Nielsen & Shephard (2006), *Econometrics of Testing for Jumps in Financial Economics Using Bipower Variation* — boundary between formal jump tests and simple screening rules.

The batch checks in [`alert_system/quality`](../alert_system/quality/config.py) now screen price moves (robust median/MAD on log returns) and spread widening, with the thresholds recorded in that config. They run on pipeline Parquet, separately from the three streaming rules above. The mapping from these four papers onto those checks is in [literature_appendix.md](literature_appendix.md). Unusual trade size and queue-imbalance prediction are not data-quality checks. Queue imbalance stays out of scope: it predicts the next mid move, which is a different question from whether a record is bad. A dashboard, event replay, and a full exchange holiday calendar are still open.
