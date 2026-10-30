from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import pandas as pd

from alert_system.data import _read_file, load_data, local_path, normalize_frame, validate_request
from alert_system.detectors import AlertEngine


def fixture_data(count=4):
    frame = pd.DataFrame({"symbol": ["AAPL"] * count, "publisher_id": [2] * count, "price": [100.0] * count, "size": [1000] * count}, index=pd.date_range("2025-09-15T13:30:00Z", periods=count, freq="min", name="ts_event"))
    return normalize_frame(frame, {"source": "test", "dataset": "XNAS.ITCH", "schema": "trades"})


class DataTests(unittest.TestCase):
    def test_bundled_samples_load_in_arrival_order(self):
        for sample, symbols in (("quotes", ["AAPL"]), ("trades", ["AAPL", "SPY"])):
            data = load_data({"source": "bundled", "sample": sample})
            self.assertEqual(data.metadata["symbols"], symbols)
            self.assertEqual(len(data.events), data.metadata["provenance"]["records"])
            self.assertEqual([event["id"] for event in data.events], sorted(event["id"] for event in data.events))

    def test_rejects_path_escape_and_non_market_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            (root / ".env").write_text("secret")
            with self.assertRaises(ValueError):
                local_path("../.env", root)

    def test_local_csv_and_parquet_preserve_values(self):
        data = fixture_data(3)
        frame = pd.DataFrame(data.events)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            for extension in ("csv", "parquet"):
                target = root / "data" / f"market.{extension}"
                getattr(frame, f"to_{extension}")(target, index=False)
                result = load_data({"source": "local", "path": target.name}, root)
                self.assertEqual(len(result.events), 3)
                self.assertEqual(result.events[0]["price"], 100)
                self.assertEqual(result.metadata["dataset"], "XNAS.ITCH")
                self.assertEqual(result.metadata["schema"], "trades")

    def test_rejects_non_equity_and_derived_bar_schema(self):
        frame = pd.DataFrame(fixture_data().events)
        with self.assertRaisesRegex(ValueError, "Conflicting dataset"):
            normalize_frame(frame, {"dataset": "GLBX.MDP3", "schema": "trades"})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "data"
            directory.mkdir()
            frame.to_parquet(directory / "bars_5m.parquet")
            (directory / "manifest.json").write_text(json.dumps({"request": {"dataset": "XNAS.ITCH", "schema": "ohlcv-1m"}, "output_schema": "ohlcv-5m"}))
            with self.assertRaisesRegex(ValueError, "schema"):
                load_data({"source": "local", "path": "bars_5m.parquet"}, root)

    def test_limit_rejects_without_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "data"
            directory.mkdir()
            frame = pd.DataFrame(fixture_data(3).events)
            for extension in ("csv", "parquet"):
                target = directory / f"market.{extension}"
                getattr(frame, f"to_{extension}")(target, index=False)
                with patch("alert_system.data.MAX_EVENTS", 2):
                    with self.assertRaisesRegex(ValueError, "up to 2 events"):
                        load_data({"source": "local", "path": target.name}, root)

    def test_context_required_and_declared_asset_schema_must_agree(self):
        frame = pd.DataFrame({"ts_event": ["2025-09-15T14:30:00Z"], "symbol": ["AAPL"], "price": [100], "size": [10]})
        with self.assertRaisesRegex(ValueError, "manifest"):
            normalize_frame(frame, {"source": "local"})
        declared = frame.assign(dataset="GLBX.MDP3", schema="trades", asset_class="futures")
        with self.assertRaisesRegex(ValueError, "equity"):
            normalize_frame(declared, {"source": "local"})
        declared = frame.assign(dataset="XNAS.ITCH", schema="trades", asset_class="futures")
        with self.assertRaisesRegex(ValueError, "equity"):
            normalize_frame(declared, {"source": "local"})
        with self.assertRaisesRegex(ValueError, "Conflicting dataset"):
            normalize_frame(frame.assign(dataset="GLBX.MDP3", schema="trades"), {"dataset": "XNAS.ITCH", "schema": "trades"})
        with self.assertRaisesRegex(ValueError, "Conflicting schema"):
            normalize_frame(frame.assign(dataset="XNAS.ITCH", schema="mbp-1"), {"dataset": "XNAS.ITCH", "schema": "trades"})

    def test_record_types_and_deeper_book_fields_cannot_masquerade(self):
        frame = pd.DataFrame({"ts_event": ["2025-09-15T14:30:00Z"], "symbol": ["AAPL"], "price": [100], "size": [10]})
        for altered in (frame.assign(rtype=160, action="A", order_id=123), frame.assign(rtype=10), frame.assign(depth=1)):
            with self.assertRaises(ValueError):
                normalize_frame(altered, {"dataset": "XNAS.ITCH", "schema": "trades"})
        quoted = frame.assign(rtype=1, bid_px_00=99, ask_px_00=100, bid_sz_00=10, ask_sz_00=10, bid_px_01=98, ask_px_01=101)
        with self.assertRaisesRegex(ValueError, "deeper-book"):
            normalize_frame(quoted, {"dataset": "XNAS.ITCH", "schema": "mbp-1"})
        with self.assertRaisesRegex(ValueError, "Order-level"):
            normalize_frame(frame.assign(rtype=0, order_id=123), {"dataset": "XNAS.ITCH", "schema": "trades"})

    def test_dbn_and_parquet_metadata_conflicts_are_rejected(self):
        store = SimpleNamespace(dataset="GLBX.MDP3", schema="trades", reader=SimpleNamespace(close=Mock()))
        with patch("alert_system.data.db.DBNStore.from_file", return_value=store):
            with self.assertRaisesRegex(ValueError, "Conflicting dataset"):
                _read_file(Path("test.dbn"), {"dataset": "XNAS.ITCH", "schema": "trades"})
        store.reader.close.assert_called_once()
        frame = pd.DataFrame({"ts_event": ["2025-09-15T14:30:00Z"], "symbol": ["AAPL"], "price": [100], "size": [10]})
        frame.attrs = {"dataset": "GLBX.MDP3", "schema": "trades"}
        with self.assertRaisesRegex(ValueError, "Conflicting dataset"):
            normalize_frame(frame, {"dataset": "XNAS.ITCH", "schema": "trades"})

    def test_empty_quote_sides_have_no_midpoint_or_chart_price(self):
        for bid in (0, None, float("nan")):
            frame = pd.DataFrame({"ts_event": ["2025-09-15T14:30:00Z"], "symbol": ["AAPL"], "bid_px_00": [bid], "bid_sz_00": [0], "ask_px_00": [100], "ask_sz_00": [100]})
            data = normalize_frame(frame, {"dataset": "XNAS.ITCH", "schema": "mbp-1"})
            self.assertIsNone(data.events[0]["price"])
            self.assertNotIn("invalid_fields", data.events[0])
            self.assertEqual(AlertEngine().process(data.events[0]), [])

    def test_missing_quote_side_csv_nan_matches_none(self):
        text = "ts_event,symbol,dataset,schema,bid_px_00,ask_px_00,bid_sz_00,ask_sz_00\n2025-09-15T14:30:00Z,AAPL,XNAS.ITCH,mbp-1,100,,100,\n"
        csv_frame = pd.read_csv(StringIO(text))
        none_frame = pd.DataFrame({"ts_event": ["2025-09-15T14:30:00Z"], "symbol": ["AAPL"], "dataset": ["XNAS.ITCH"], "schema": ["mbp-1"], "bid_px_00": [100], "ask_px_00": [None], "bid_sz_00": [100], "ask_sz_00": [None]})
        from_csv = normalize_frame(csv_frame, {"source": "local"}).events[0]
        from_none = normalize_frame(none_frame, {"source": "local"}).events[0]
        self.assertEqual(from_csv, from_none)
        self.assertIsNone(from_csv["price"])
        self.assertNotIn("invalid_fields", from_csv)
        self.assertEqual(AlertEngine().process(from_csv), [])
        csv_frame["ask_sz_00"] = float("inf")
        infinite_size = normalize_frame(csv_frame, {"source": "local"}).events[0]
        self.assertEqual(infinite_size["invalid_fields"]["ask_sz_00"], "Infinity")
        self.assertEqual(AlertEngine().process(infinite_size)[0]["rule_id"], "integrity")

    def test_nonfinite_numbers_retain_json_safe_integrity_evidence(self):
        quote = pd.DataFrame({"ts_event": ["2025-09-15T14:30:00Z"], "symbol": ["AAPL"], "bid_px_00": [float("inf")], "bid_sz_00": [100], "ask_px_00": [100], "ask_sz_00": [100]})
        trade = pd.DataFrame({"ts_event": ["2025-09-15T14:30:00Z"], "symbol": ["AAPL"], "price": [100], "size": [float("inf")]})
        for frame, schema, field in ((quote, "mbp-1", "bid_px_00"), (trade, "trades", "size")):
            event = normalize_frame(frame, {"dataset": "XNAS.ITCH", "schema": schema}).events[0]
            self.assertEqual(event["invalid_fields"][field], "Infinity")
            alerts = AlertEngine().process(event)
            self.assertEqual(alerts[0]["rule_id"], "integrity")
            self.assertIn(field, alerts[0]["observed"]["issues"][0])
            json.dumps({"event": event, "alerts": alerts}, allow_nan=False)

    def test_databento_parameters_use_utc_and_reject_future(self):
        result = validate_request({"source": "databento", "symbols": ["AAPL"], "start": "2025-09-15T09:30", "end": "2025-09-15T16:00"})
        self.assertEqual(result["start"], "2025-09-15T13:30:00+00:00")
        with self.assertRaises(ValueError):
            validate_request({"source": "databento", "symbols": ["AAPL"], "start": "2099-01-01", "end": "2099-01-02"})

    def test_quote_receive_time_and_event_nanoseconds_are_preserved(self):
        frame = pd.DataFrame({"symbol": ["AAPL"], "ts_event": ["2025-09-15T14:30:00.000000001Z"], "bid_px_00": [100], "ask_px_00": [101], "bid_sz_00": [1], "ask_sz_00": [1]}, index=pd.DatetimeIndex(["2025-09-15T14:30:00.000000999Z"], name="ts_recv"))
        event = normalize_frame(frame, {"dataset": "XNAS.ITCH", "schema": "mbp-1"}).events[0]
        self.assertTrue(event["timestamp"].endswith("000000001+00:00"))
        self.assertTrue(event["available_timestamp"].endswith("000000999+00:00"))


if __name__ == "__main__":
    unittest.main()
