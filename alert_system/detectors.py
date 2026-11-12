"""Streaming data-quality checks on normalized events.

Three rules: record integrity, event time against the high-water mark in the
same source scope, and an immediate crossed quote. The batch checks in
``alert_system.quality`` keep their own book-state, clock, gap, spread, and
sequence logic. Both implementations stay; see the README.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite

import pandas as pd


DEFAULT_CONFIG = {
    "cooldown_events": 15,  # Integer same-scope arrivals, >= 0; 0 disables suppression
    "cooldown_seconds": 30.0,  # Nonnegative market-time seconds; 0 disables time suppression
}

BROWNLEES = {"title": "Brownlees & Gallo (2006): High-frequency data handling", "url": "https://flore.unifi.it/handle/2158/210321"}
DATABENTO = {"title": "Databento: MBP-1 schema", "url": "https://databento.com/docs/schemas/mbp-1"}
TIMESTAMPS = {"title": "Databento: Timestamps and common fields", "url": "https://databento.com/docs/conventions/common-fields-enums-types"}


def rule_catalog():
    definitions = [
        ("integrity", "Record integrity", ["trade", "quote"],
         "Check timestamp, finite values, nonnegative sizes and equity prices.",
         ["timestamp", "price"], "Immediate; absent quote sides are allowed.", [BROWNLEES, DATABENTO]),
        ("time_order", "Event time moved backwards", ["trade", "quote"],
         "Compare event time with the latest observed time in the same source scope.",
         ["timestamp"], "Immediate after one timestamp; ties are allowed.", [TIMESTAMPS]),
        ("crossed_quote", "Crossed quote", ["quote"],
         "Flag bid above ask within one quote record; source scope is retained.",
         ["bid", "ask", "publisher_id"], "Two valid, nonempty quote sides in one record; locked quotes are allowed.", [DATABENTO]),
    ]
    return [{"id": key, "title": title, "kinds": kinds, "description": description,
             "required_fields": required, "eligibility": eligibility,
             "references": deepcopy(references)}
            for key, title, kinds, description, required, eligibility, references in definitions]


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _timestamp(value):
    try:
        result = pd.Timestamp(value)
        return result.tz_convert("UTC") if not pd.isna(result) and result.tzinfo else None
    except (ValueError, TypeError, OverflowError):
        return None


@dataclass
class _State:
    arrivals: int = 0
    latest_timestamp: datetime | None = None
    latest_available_timestamp: datetime | None = None
    emitted: dict = field(default_factory=dict)


class AlertEngine:
    def __init__(self, config=None):
        config = config or {}
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown detector settings: {', '.join(sorted(unknown))}")
        self.config = {**DEFAULT_CONFIG, **config}
        for name, value in self.config.items():
            if not isinstance(value, (int, float)) or _number(value) is None or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if not isinstance(self.config["cooldown_events"], int) or isinstance(self.config["cooldown_events"], bool):
            raise ValueError("cooldown_events must be an integer")
        self.rules = {rule["id"]: rule for rule in rule_catalog()}
        self.reset()

    def reset(self):
        self.states = {}
        self.alert_count = 0

    def _emit(self, state, event, rule_id, severity, observed, thresholds, explanation, causes, actions):
        market_time = state.latest_available_timestamp
        previous = state.emitted.get(rule_id)
        if previous is not None:
            arrival, previous_time = previous
            if state.arrivals - arrival < self.config["cooldown_events"]:
                return []
            if market_time is not None and previous_time is not None and (market_time - previous_time).total_seconds() < self.config["cooldown_seconds"]:
                return []
        state.emitted[rule_id] = (state.arrivals, market_time)
        self.alert_count += 1
        return [{"id": f"alert-{self.alert_count:06d}", "rule_id": rule_id,
                 "title": self.rules[rule_id]["title"], "severity": severity,
                 "symbol": event.get("symbol", "UNKNOWN"), "timestamp": str(event.get("timestamp", "")),
                 "detected_at": str(event.get("available_timestamp") or event.get("timestamp", "")),
                 "dataset": event.get("dataset"), "schema": event.get("schema"),
                 "publisher_id": event.get("publisher_id"), "event_id": event.get("id"),
                 "observed": observed, "thresholds": thresholds, "explanation": explanation,
                 "possible_causes": causes, "next_checks": actions,
                 "references": deepcopy(self.rules[rule_id]["references"])}]

    def process(self, event):
        scope = tuple(event.get(name) for name in ("dataset", "symbol", "publisher_id", "schema", "kind"))
        state = self.states.setdefault(scope, _State())
        state.arrivals += 1
        timestamp = _timestamp(event.get("timestamp"))
        available = _timestamp(event.get("available_timestamp")) if event.get("available_timestamp") else timestamp
        if available is not None and (state.latest_available_timestamp is None or available > state.latest_available_timestamp):
            state.latest_available_timestamp = available
        numbers = {name: _number(event.get(name)) for name in ("price", "size", "bid", "ask", "bid_size", "ask_size")}
        issues = [f"{name} contains invalid numeric data ({value})" for name, value in event.get("invalid_fields", {}).items()]
        if timestamp is None:
            issues.append("timestamp must contain a valid UTC offset")
        if event.get("kind") not in ("trade", "quote"):
            issues.append("kind must be trade or quote")
        if event.get("kind") == "trade" and numbers["price"] is None:
            issues.append("trade price is required")
        for name, value in numbers.items():
            if event.get(name) is not None and value is None:
                issues.append(f"{name} is not finite numeric data")
            elif value is not None and name in ("size", "bid_size", "ask_size") and value < 0:
                issues.append(f"{name} is negative")
            elif value is not None and name == "price" and value <= 0 and event.get("asset_class", "equity") == "equity":
                issues.append(f"{name} must be positive for equities")
        if event.get("asset_class", "equity") == "equity":
            for side in ("bid", "ask"):
                if numbers[side] is not None and numbers[side] <= 0 and (numbers[f"{side}_size"] or 0) > 0:
                    issues.append(f"{side} must be positive for a nonempty equity quote")
        alerts = []
        if issues:
            alerts += self._emit(state, event, "integrity", "critical", {"issues": issues}, {"violations_allowed": 0},
                                 "This record fails structural checks and should be reviewed before further use.",
                                 ["Malformed normalization or an inconsistent source record", "Instrument conventions require verification"],
                                 ["Inspect the raw record and instrument definition", "Exclude the record from later analysis until reviewed"])
        backwards = timestamp is not None and state.latest_timestamp is not None and timestamp < state.latest_timestamp
        if backwards:
            alerts += self._emit(state, event, "time_order", "warning", {"previous_latest_timestamp": state.latest_timestamp.isoformat(), "backward_seconds": (state.latest_timestamp - timestamp).value / 1e9},
                                 {"backward_seconds_allowed": 0}, "Event time is earlier than an already observed event in the same source scope.",
                                 ["Late or reordered event delivery", "Independent exchange gateway clocks or a correction"],
                                 ["Compare event and receive timestamps in the source", "Review ordering before using sequence-sensitive analysis; this does not prove packet loss"])
        if timestamp is not None and not backwards:
            state.latest_timestamp = timestamp
        if issues or backwards:
            return alerts
        if event["kind"] == "quote":
            bid, ask = numbers["bid"], numbers["ask"]
            sides = bid is not None and ask is not None and bid > 0 and ask > 0 and numbers["bid_size"] != 0 and numbers["ask_size"] != 0
            if sides and bid > ask:
                alerts += self._emit(state, event, "crossed_quote", "warning", {"bid": bid, "ask": ask, "cross_amount": bid - ask}, {"maximum_cross": 0},
                                     "Bid exceeds ask in this source record. This is not a claim of executable arbitrage.",
                                     ["Transient quote alignment or source normalization issue", "Market state or venue aggregation needs verification"],
                                     ["Check the source's venue scope and quote condition", "Verify against subsequent quotes"])
        return alerts
