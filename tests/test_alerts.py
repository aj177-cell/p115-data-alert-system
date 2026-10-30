from datetime import datetime, timedelta, timezone
import json
import unittest

from alert_system.detectors import AlertEngine, rule_catalog


START = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)


def trade(index, **changes):
    event = {"id": f"trade-{index}", "timestamp": (START + timedelta(seconds=index)).isoformat(),
             "symbol": "AAPL", "dataset": "XNAS.ITCH", "schema": "trades", "publisher_id": 2,
             "kind": "trade", "price": 100.0, "size": 100, "asset_class": "equity"}
    return {**event, **changes}


def quote(index, **changes):
    event = {"id": f"quote-{index}", "timestamp": (START + timedelta(seconds=index)).isoformat(),
             "symbol": "AAPL", "dataset": "XNAS.ITCH", "schema": "mbp-1", "publisher_id": 2,
             "kind": "quote", "price": 100.0, "bid": 99.99, "ask": 100.01,
             "bid_size": 100, "ask_size": 100, "asset_class": "equity"}
    return {**event, **changes}


def ids(alerts):
    return {alert["rule_id"] for alert in alerts}


class AlertTests(unittest.TestCase):
    def setUp(self):
        self.engine = AlertEngine({"cooldown_events": 0, "cooldown_seconds": 0})

    def test_clean_records_produce_no_alerts(self):
        for index in range(5):
            self.assertEqual(self.engine.process(trade(index)), [])
            self.assertEqual(self.engine.process(quote(index)), [])

    def test_invalid_price_and_negative_size(self):
        alerts = self.engine.process(trade(0, price=-1, size=-2))
        self.assertEqual(ids(alerts), {"integrity"})
        self.assertEqual(len(alerts[0]["observed"]["issues"]), 2)

    def test_reordered_records_are_flagged(self):
        for index in range(3):
            self.engine.process(trade(index))
        self.assertEqual(ids(self.engine.process(trade(1))), {"time_order"})
        self.assertEqual(self.engine.process(trade(3)), [])

    def test_nanosecond_reversal_is_detected(self):
        self.assertEqual(self.engine.process(quote(0, timestamp="2024-01-02T15:00:00.000000009Z")), [])
        alert = self.engine.process(quote(1, timestamp="2024-01-02T15:00:00.000000001Z"))[0]
        self.assertEqual(alert["rule_id"], "time_order")
        self.assertEqual(alert["observed"]["backward_seconds"], 8e-9)

    def test_scopes_are_isolated(self):
        self.engine.process(trade(5))
        for changes in ({"symbol": "MSFT"}, {"publisher_id": 3}, {"dataset": "XNYS.PILLAR"}):
            self.assertEqual(self.engine.process(trade(1, **changes)), [])

    def test_missing_empty_and_locked_quotes_are_not_crossed(self):
        for event in (quote(0, bid=None, bid_size=0, price=None), quote(1, bid=0, bid_size=0, price=None),
                      quote(2, bid=100, ask=100), quote(3, bid=102, bid_size=0, price=None)):
            self.assertEqual(self.engine.process(event), [])
        self.assertEqual(ids(self.engine.process(quote(4, bid=101))), {"crossed_quote"})

    def test_nonfinite_fields_and_bad_timestamps_serialize_safely(self):
        alerts = self.engine.process(trade(0, price=float("nan"), timestamp="bad-time"))
        self.assertEqual(ids(alerts), {"integrity"})
        json.dumps(alerts, allow_nan=False)
        self.assertEqual(ids(self.engine.process(trade(1, timestamp="2024-01-02T15:01:00"))), {"integrity"})

    def test_cooldown_suppresses_repeats(self):
        engine = AlertEngine({"cooldown_events": 0, "cooldown_seconds": 30})
        self.assertEqual(ids(engine.process(quote(0, bid=101))), {"crossed_quote"})
        for second in (1, 2, 20, 29):
            self.assertEqual(engine.process(quote(second, bid=101)), [])
        self.assertEqual(ids(engine.process(quote(30, bid=101))), {"crossed_quote"})

    def test_catalog_and_configuration_validation(self):
        self.assertEqual([rule["id"] for rule in rule_catalog()], ["integrity", "time_order", "crossed_quote"])
        for config in ({"unknown": 2}, {"cooldown_events": 1.5}, {"cooldown_seconds": -1}):
            with self.assertRaises(ValueError):
                AlertEngine(config)


if __name__ == "__main__":
    unittest.main()
