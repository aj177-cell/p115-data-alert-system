from datetime import date
from contextlib import redirect_stdout
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
from pathlib import Path
import tempfile
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import databento as db
from databento_dbn import Action, Metadata, OHLCVMsg, RType, Schema, Side, SType, TradeMsg
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import databento_pipeline as pipeline


START = pd.Timestamp("2024-01-02T14:30:00Z")


def write_dbn(path, schema="trades", count=7, missing_prices=(), start=START, simultaneous=False):
    mappings = [
        SimpleNamespace(
            raw_symbol=symbol,
            intervals=[SimpleNamespace(start_date=date(2024, 1, 2), end_date=date(2024, 1, 4), symbol=str(instrument_id))],
        )
        for instrument_id, symbol in [(1, "AAPL"), (2, "MSFT")]
    ]
    metadata = Metadata(
        dataset="XNAS.ITCH",
        start=start.value,
        stype_in=SType.RAW_SYMBOL,
        stype_out=SType.INSTRUMENT_ID,
        schema=Schema(schema),
        symbols=["AAPL", "MSFT"],
        mappings=mappings,
    )
    records = []
    for index in range(count):
        timestamp = start.value + (index // 2 if simultaneous else index) * 60_000_000_000
        instrument_id = 1 + index % 2
        price = (100 * instrument_id + index) * 1_000_000_000
        if index in missing_prices:
            price = 2 ** 63 - 1
        common = {"publisher_id": 2, "instrument_id": instrument_id, "ts_event": timestamp}
        if schema == "trades":
            record = TradeMsg(**common, ts_recv=timestamp, price=price, size=index + 1, action=Action.TRADE, side=Side.BID, depth=0)
        else:
            record = OHLCVMsg(**common, rtype=RType.OHLCV_1M, open=price, high=price + 1_000_000_000, low=price - 1_000_000_000, close=price, volume=index + 1)
        records.append(bytes(record))
    path.write_bytes(metadata.encode() + b"".join(records))
    return path


class LocalTimeseries:
    def __init__(self, source, fail=False):
        self.source = source
        self.fail = fail
        self.requests = []

    def get_range(self, path, **request):
        self.requests.append(request)
        if self.fail:
            path.write_bytes(b"partial response")
            raise ConnectionError("Interrupted download")
        store = db.DBNStore.from_bytes(self.source.read_bytes())
        store.to_file(path, compression="zstd", mode="x")
        return db.DBNStore.from_bytes(path.read_bytes())



class ConfigurationTests(unittest.TestCase):
    def test_parent_environment_and_gateway_are_used_without_exposing_key(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "P115"
            root.mkdir()
            key = "db-" + "x" * 29
            (parent / ".env").write_text(
                f"DATABENTO_API_KEY={key}\nDATABENTO_HISTORICAL_BASE_URL=https://hist.databento.com/v0/\nDATABENTO_TIMESERIES_ENDPOINT=https://hist.databento.com/v0/timeseries.get_range\n"
            )
            (root / ".env").write_text("DATABENTO_API_KEY=wrong-local-key\n")
            with patch.object(pipeline, "__file__", str(root / "databento_pipeline.py")):
                config = pipeline.load_config()
            self.assertEqual(config.project_root, root.resolve())
            self.assertEqual(config.data_dir, root.resolve() / "data")
            self.assertEqual(config.api_key, key)
            self.assertEqual(config.historical_gateway, "https://hist.databento.com")
            self.assertNotIn(key, repr(config))
            client = pipeline.create_historical_client(config)
            self.assertEqual(client.timeseries._base_url, "https://hist.databento.com/v0/timeseries")

    def test_mismatched_endpoint_rejects_configuration_without_exposing_key(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "P115"
            root.mkdir()
            key = "db-" + "x" * 29
            (parent / ".env").write_text(
                f"DATABENTO_API_KEY={key}\nDATABENTO_HISTORICAL_BASE_URL=https://hist.databento.com/v0\nDATABENTO_TIMESERIES_ENDPOINT=https://invalid.example/v0/timeseries.get_range\n"
            )
            with self.assertRaises(ValueError) as raised:
                pipeline.load_config(root)
            self.assertNotIn(key, str(raised.exception))


class DownloadTests(unittest.TestCase):
    def test_sdk_http_download_has_no_authentication_deprecation_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_dbn(root / "source.dbn", count=7)
            store = db.DBNStore.from_bytes(source.read_bytes())
            try:
                store.to_file(root / "response.dbn.zst", compression="zstd", mode="x")
            finally:
                store.reader.close()
            payload = (root / "response.dbn.zst").read_bytes()
            received = []

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    received.append((self.path, self.headers.get("Authorization")))
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                def log_message(self, format, *args):
                    pass

            with patch("socket.getfqdn", return_value="localhost"), ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
                thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
                thread.start()
                key = "db-" + "x" * 29
                client = db.Historical(key=key, gateway=f"http://127.0.0.1:{server.server_port}")
                client.timeseries._base_url = f"http://127.0.0.1:{server.server_port}/v0/timeseries"
                request = {"dataset": "XNAS.ITCH", "schema": "trades", "symbols": ["AAPL", "MSFT"], "start": START.isoformat(), "end": (START + pd.Timedelta(minutes=7)).isoformat(), "limit": None}
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", DeprecationWarning)
                        result = pipeline.download_historical(client, request, root / "data", preview_rows=2, chunk_rows=3, export_formats=("parquet",))
                finally:
                    server.shutdown()
                    thread.join()
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0][0], "/v0/timeseries.get_range")
            scheme, token = received[0][1].split(" ", 1)
            self.assertEqual(scheme, "Basic")
            self.assertEqual(base64.b64decode(token).decode(), f"{key}:")
            self.assertEqual(result.dbn_path.read_bytes(), payload)
            self.assertEqual(result.summary["record_count"], 7)
            self.assertEqual(len(result.preview), 2)
            self.assertEqual(len(pd.read_parquet(result.run_dir / "data.parquet")), 7)
            self.assertEqual(json.loads(result.manifest_path.read_text())["status"], "complete")
            self.assertNotIn(key, result.manifest_path.read_text())

    def test_request_times_use_the_configured_zone_and_keep_explicit_offsets(self):
        cases = [
            ("2025-07-15T09:30:00", "2025-07-15T09:31:00", "2025-07-15T13:30:00Z"),
            ("2025-01-15T09:30:00", "2025-01-15T09:31:00", "2025-01-15T14:30:00Z"),
            ("2025-07-15T14:30:00Z", "2025-07-15T14:31:00Z", "2025-07-15T14:30:00Z"),
            ("2025-07-15T09:30:00+02:00", "2025-07-15T09:31:00+02:00", "2025-07-15T07:30:00Z"),
        ]
        client = SimpleNamespace(metadata=SimpleNamespace(list_schemas=lambda dataset: ["trades"]))
        for start, end, expected in cases:
            with self.subTest(start=start), patch.object(pipeline, "TIMEZONE", "America/New_York"), patch.object(pipeline, "load_config", return_value=SimpleNamespace(data_dir=Path("unused"))), patch.object(pipeline, "create_historical_client", return_value=client), patch.object(pipeline, "download_historical", new_callable=Mock, side_effect=RuntimeError("Transport boundary")) as download, redirect_stdout(StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Transport boundary"):
                    pipeline.download_and_show(dataset="XNAS.ITCH", symbols=["AAPL"], start=start, end=end)
                request = download.call_args.args[1]
                self.assertEqual(pd.Timestamp(request["start"]), pd.Timestamp(expected))
                self.assertEqual(pd.Timestamp(request["end"]), pd.Timestamp(expected) + pd.Timedelta(minutes=1))
                self.assertEqual(pd.Timestamp(request["start"]).utcoffset(), pd.Timedelta(0))

    def test_ambiguous_and_nonexistent_local_times_fail_before_download(self):
        cases = [
            ("2025-11-02T01:30:00", "2025-11-02T01:31:00"),
            ("2025-03-09T02:30:00", "2025-03-09T02:31:00"),
        ]
        client = SimpleNamespace(metadata=SimpleNamespace(list_schemas=lambda dataset: ["trades"]))
        for start, end in cases:
            with self.subTest(start=start), patch.object(pipeline, "TIMEZONE", "America/New_York"), patch.object(pipeline, "load_config", return_value=SimpleNamespace(data_dir=Path("unused"))), patch.object(pipeline, "create_historical_client", return_value=client), patch.object(pipeline, "download_historical", new_callable=Mock) as download:
                with self.assertRaises(ValueError) as raised:
                    pipeline.download_and_show(dataset="XNAS.ITCH", symbols=["AAPL"], start=start, end=end)
                self.assertIn("America/New_York", str(raised.exception))
                self.assertRegex(str(raised.exception).lower(), "ambiguous|nonexistent|non-existent|daylight|dst")
                download.assert_not_called()

    def test_local_inputs_are_recorded_while_downloads_and_aggregated_files_stay_utc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_dbn(root / "source.dbn", schema="ohlcv-1m", count=30, simultaneous=True)
            timeseries = LocalTimeseries(source)
            client = SimpleNamespace(timeseries=timeseries, metadata=SimpleNamespace(list_schemas=lambda dataset: ["ohlcv-1m"]))
            start, end = "2024-01-02T09:30:00", "2024-01-02T09:45:00"
            with patch.object(pipeline, "TIMEZONE", "America/New_York"), patch.object(pipeline, "load_config", return_value=SimpleNamespace(data_dir=root / "data")), patch.object(pipeline, "create_historical_client", return_value=client), patch.object(pipeline, "display") as display, redirect_stdout(StringIO()):
                result = pipeline.download_and_show(dataset="XNAS.ITCH", symbols=["AAPL", "MSFT"], start=start, end=end, data_type="ohlcv", bar_interval="5m")
            request = timeseries.requests[0]
            self.assertEqual(pd.Timestamp(request["start"]), START)
            self.assertEqual(pd.Timestamp(request["end"]), START + pd.Timedelta(minutes=15))
            self.assertIsNone(request["limit"])
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["timezone"], "America/New_York")
            self.assertEqual(manifest["input_times"], {"start": start, "end": end})
            self.assertEqual(pd.Timestamp(manifest["request"]["start"]), START)
            self.assertEqual(manifest["aggregation"]["timezone"], "UTC")
            bars = pd.read_parquet(result.run_dir / "bars_5m.parquet")
            self.assertEqual(str(bars.index.tz), "UTC")
            self.assertEqual(str(result.preview.index.tz), "UTC")
            self.assertEqual(bars.index.min(), START)
            self.assertEqual(result.summary["record_count"], 6)
            self.assertEqual(pd.Timestamp(result.summary["time_start"]), START)
            store = db.DBNStore.from_file(result.dbn_path)
            try:
                raw = store.to_df()
                self.assertEqual(str(raw.index.tz), "UTC")
                self.assertEqual(len(raw), 30)
            finally:
                store.reader.close()
            frames = [call.args[0] for call in display.call_args_list if isinstance(call.args[0], pd.DataFrame)]
            summary = next(frame for frame in frames if "Value" in frame.columns)
            self.assertEqual(pd.Timestamp(summary.loc["First timestamp (America/New_York)", "Value"]).hour, 9)
            self.assertIn("-05:00", str(summary.loc["First timestamp (America/New_York)", "Value"]))
            shown = next(frame for frame in frames if "instrument_id" in frame.columns)
            self.assertEqual(str(shown.index.tz), "America/New_York")
            self.assertEqual(shown.index[0].hour, 9)

    def test_one_step_download_has_no_record_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_dbn(root / "source.dbn", count=1207)
            timeseries = LocalTimeseries(source)
            client = SimpleNamespace(timeseries=timeseries, metadata=SimpleNamespace(list_schemas=lambda dataset: ["trades", "ohlcv-1m"]))
            config = SimpleNamespace(data_dir=root / "data")
            with patch.object(pipeline, "load_config", return_value=config), patch.object(pipeline, "create_historical_client", return_value=client), patch.object(pipeline, "show_download") as show, redirect_stdout(StringIO()):
                result = pipeline.download_and_show(dataset="XNAS.ITCH", symbols=["AAPL", "MSFT"], start=START.isoformat(), end="2024-01-04T00:00:00Z", preview_rows=3)
            self.assertIsNone(timeseries.requests[0]["limit"])
            self.assertEqual(result.summary["record_count"], 1207)
            self.assertEqual(len(result.preview), 3)
            self.assertEqual(len(pd.read_parquet(result.run_dir / "data.parquet")), 1207)
            self.assertIsNone(json.loads(result.manifest_path.read_text())["request"]["limit"])
            show.assert_called_once_with(result)

    def test_five_minute_bars_preserve_instruments_across_chunks_and_midnight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = pd.Timestamp("2024-01-02T23:55:00Z")
            source = write_dbn(root / "source.dbn", schema="ohlcv-1m", count=20, start=start, simultaneous=True)
            client = SimpleNamespace(timeseries=LocalTimeseries(source))
            request = {"dataset": "XNAS.ITCH", "schema": "ohlcv-1m", "symbols": ["AAPL", "MSFT"], "start": start.isoformat(), "end": "2024-01-03T00:05:00Z", "limit": None}
            result = pipeline.download_historical(client, request, root / "data", preview_rows=2, chunk_rows=3)
            result = pipeline._aggregate_bars(result, request, "5m", preview_rows=2, chunk_rows=3, export_formats=("csv",))
            bars = pd.read_parquet(result.run_dir / "bars_5m.parquet")
            self.assertEqual(result.summary["source_records"], 20)
            self.assertEqual(result.summary["record_count"], 4)
            self.assertEqual(result.summary["output_schema"], "ohlcv-5m")
            self.assertEqual(len(result.preview), 2)
            self.assertEqual(len(pd.read_csv(result.run_dir / "bars_5m.csv")), 4)
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(set(manifest["export_formats"]), {"csv", "parquet"})
            self.assertEqual(manifest["preview_rows_limit"], 2)
            self.assertEqual(manifest["chunk_rows"], 3)
            self.assertEqual(manifest["aggregation"]["group_by"], ["publisher_id", "instrument_id"])
            self.assertFalse(bars["partial_from_request"].any())
            self.assertNotIn("rtype", bars.columns)
            expected = {
                1: [[100, 109, 99, 108, 25], [110, 119, 109, 118, 75]],
                2: [[201, 210, 200, 209, 30], [211, 220, 210, 219, 80]],
            }
            for instrument, values in expected.items():
                group = bars[bars["instrument_id"] == instrument].sort_index()
                self.assertEqual(list(group.index), [start, start + pd.Timedelta(minutes=5)])
                self.assertEqual(group[["open", "high", "low", "close", "volume"]].values.tolist(), values)

    def test_partial_and_empty_bar_outputs_are_explicit(self):
        for count in [0, 18]:
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                start = pd.Timestamp("2024-01-02T23:56:00Z")
                source = write_dbn(root / "source.dbn", schema="ohlcv-1m", count=count, start=start, simultaneous=True)
                client = SimpleNamespace(timeseries=LocalTimeseries(source))
                request = {"dataset": "XNAS.ITCH", "schema": "ohlcv-1m", "symbols": ["AAPL", "MSFT"], "start": start.isoformat(), "end": "2024-01-03T00:05:00Z", "limit": None}
                result = pipeline.download_historical(client, request, root / "data", chunk_rows=3)
                result = pipeline._aggregate_bars(result, request, "5m", preview_rows=2, chunk_rows=3)
                bars = pd.read_parquet(result.run_dir / "bars_5m.parquet")
                self.assertEqual(result.summary["source_records"], count)
                self.assertEqual(len(bars), 4 if count else 0)
                if count:
                    self.assertEqual(int(bars["partial_from_request"].sum()), 2)
                    self.assertEqual(int(bars["volume"].sum()), 171)
                else:
                    self.assertTrue(result.preview.empty)

    def test_unsupported_depth_is_rejected_before_download(self):
        for dataset, data_type, schemas in [("OPRA.PILLAR", "l2", ["cmbp-1", "trades", "ohlcv-1m"]), ("OPRA.PILLAR", "order_book", ["cmbp-1", "trades"]), ("DBIX.CORE", "l2", ["mbp-1", "trades"])]:
            with self.subTest(dataset=dataset, data_type=data_type), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                timeseries = LocalTimeseries(root / "unused.dbn")
                client = SimpleNamespace(timeseries=timeseries, metadata=SimpleNamespace(list_schemas=lambda dataset: schemas))
                with patch.object(pipeline, "load_config", return_value=SimpleNamespace(data_dir=root / "data")), patch.object(pipeline, "create_historical_client", return_value=client), self.assertRaises(ValueError) as raised:
                    pipeline.download_and_show(dataset=dataset, symbols=["AAPL.OPT"], start=START.isoformat(), end="2024-01-02T14:31:00Z", data_type=data_type)
                self.assertIn(dataset, str(raised.exception))
                self.assertEqual(timeseries.requests, [])
                self.assertFalse((root / "data").exists())

    def test_l1_uses_the_available_top_of_book_schema(self):
        for dataset, schemas, expected in [("XNAS.BASIC", ["cmbp-1", "trades"], "cmbp-1"), ("OPRA.PILLAR", ["cmbp-1", "trades"], "cmbp-1"), ("GLBX.MDP3", ["cmbp-1", "mbp-1"], "mbp-1")]:
            with self.subTest(dataset=dataset):
                client = SimpleNamespace(metadata=SimpleNamespace(list_schemas=lambda dataset: schemas))
                with patch.object(pipeline, "load_config", return_value=SimpleNamespace(data_dir=Path("unused"))), patch.object(pipeline, "create_historical_client", return_value=client), patch.object(pipeline, "download_historical", new_callable=Mock, side_effect=RuntimeError("Transport boundary")) as download, redirect_stdout(StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "Transport boundary"):
                        pipeline.download_and_show(dataset=dataset, symbols=["AAPL"], start=START.isoformat(), end="2024-01-02T14:31:00Z", data_type="l1")
                self.assertEqual(download.call_args.args[1]["schema"], expected)
                self.assertIsNone(download.call_args.args[1]["limit"])

    def test_chunked_statistics_and_exports_cover_all_records(self):
        for schema in ["trades", "ohlcv-1m"]:
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = write_dbn(root / "source.dbn", schema=schema, missing_prices=(5,) if schema == "trades" else ())
                client = SimpleNamespace(timeseries=LocalTimeseries(source))
                request = {"dataset": "XNAS.ITCH", "schema": schema, "symbols": ["AAPL", "MSFT"], "start": START.isoformat(), "limit": None}
                result = pipeline.download_historical(client, request, root / "data", preview_rows=3, chunk_rows=2, export_formats=("csv", "parquet"))
                self.assertEqual(result.summary["record_count"], 7)
                self.assertEqual(len(result.preview), 3)
                self.assertEqual(result.summary["preview_rows"], 3)
                self.assertIsNone(client.timeseries.requests[0]["limit"])
                if schema == "trades":
                    self.assertFalse(result.preview["price"].isna().any())
                    self.assertEqual(result.summary["missing_values"]["price"], 1)
                self.assertEqual(pd.Timestamp(result.summary["time_end"]), START + pd.Timedelta(minutes=6))
                self.assertEqual(len(pd.read_csv(result.run_dir / "data.csv")), 7)
                exported = pd.read_parquet(result.run_dir / "data.parquet")
                self.assertEqual(len(exported), 7)
                self.assertEqual(set(exported["symbol"]), {"AAPL", "MSFT"})
                self.assertEqual(exported["close" if schema.startswith("ohlcv") else "price"].iloc[-1], 106.0)
                manifest = json.loads(result.manifest_path.read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["summary"]["record_count"], 7)
                self.assertEqual(set(result.summary["files"]), {"data.dbn.zst", "data.csv", "data.parquet"})

    def test_empty_download_and_zero_preview_are_supported(self):
        for count in [0, 7]:
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = write_dbn(root / "source.dbn", count=count)
                client = SimpleNamespace(timeseries=LocalTimeseries(source))
                result = pipeline.download_historical(client, {"dataset": "XNAS.ITCH", "schema": "trades", "start": START.isoformat()}, root / "data", preview_rows=0, chunk_rows=2, export_formats=("csv", "parquet"))
                self.assertEqual(result.summary["record_count"], count)
                self.assertTrue(result.preview.empty)
                self.assertEqual(len(pd.read_csv(result.run_dir / "data.csv")), count)
                self.assertEqual(len(pd.read_parquet(result.run_dir / "data.parquet")), count)
                self.assertEqual(json.loads(result.manifest_path.read_text())["status"], "complete")
                figure = pipeline.plot_preview(result.preview, "trades")
                self.assertEqual(len(figure.axes[0].lines), 0)
                plt.close(figure)

    def test_failed_download_cannot_produce_success_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = SimpleNamespace(timeseries=LocalTimeseries(root / "unused.dbn", fail=True))
            with self.assertRaises(ConnectionError):
                pipeline.download_historical(client, {"dataset": "XNAS.ITCH", "schema": "trades", "start": START.isoformat()}, root / "data")
            paths = list((root / "data").rglob("manifest.json"))
            self.assertEqual(len(paths), 1)
            manifest = json.loads(paths[0].read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error_type"], "ConnectionError")
            self.assertNotIn("summary", manifest)
            self.assertFalse((paths[0].parent / "data.dbn.zst").exists())



class VisualizationTests(unittest.TestCase):
    def test_display_and_plot_use_new_york_without_modifying_utc_records(self):
        index = pd.date_range("2025-07-15T13:30:00Z", periods=2, freq="5min")
        frame = pd.DataFrame({"instrument_id": [1, 1], "symbol": ["AAPL", "AAPL"], "price": [100.0, 101.0], "ts_recv": index}, index=index)
        with patch.object(pipeline, "TIMEZONE", "America/New_York"):
            shown = pipeline._display_preview(frame)
            self.assertEqual(str(shown.index.tz), "America/New_York")
            self.assertEqual(str(shown["ts_recv"].dt.tz), "America/New_York")
            self.assertEqual(shown.index[0].hour, 9)
            self.assertEqual(shown["ts_recv"].iloc[0].hour, 9)
            self.assertEqual(str(frame.index.tz), "UTC")
            self.assertEqual(str(frame["ts_recv"].dt.tz), "UTC")
            figure = pipeline.plot_preview(frame, "trades")
        self.assertEqual(figure.axes[0].get_xlabel(), "Time (America/New_York)")
        labels = figure.axes[0].xaxis.get_major_formatter().format_ticks(mdates.date2num(index))
        self.assertEqual(labels, ["09:30", "09:35"])
        self.assertEqual(pd.Timestamp(figure.axes[0].lines[0].get_xdata()[0]), index[0])
        plt.close(figure)

    def test_indices_plot_levels_whether_quote_fields_are_present_or_empty(self):
        for quotes in [[float("nan"), float("nan")], [4999.0, 5000.0]]:
            with self.subTest(quotes=quotes):
                frame = pd.DataFrame({"instrument_id": [1, 1], "symbol": ["SPX", "SPX"], "price": [5000.0, 5001.0], "bid_px_00": quotes, "ask_px_00": [value + 2 for value in quotes]}, index=pd.date_range(START, periods=2, freq="s"))
                figure = pipeline.plot_preview(frame, "mbp-1", dataset="DBIX.CORE")
                self.assertEqual(figure.axes[0].get_ylabel(), "Index value")
                self.assertEqual(len(figure.axes[0].lines), 1)
                self.assertEqual(list(figure.axes[0].lines[0].get_ydata()), [5000.0, 5001.0])
                plt.close(figure)
        bars = pd.DataFrame({"instrument_id": [1, 1], "symbol": ["SPX", "SPX"], "close": [5000.0, 5001.0], "volume": [0, 0]}, index=pd.date_range(START, periods=2, freq="5min"))
        figure = pipeline.plot_preview(bars, "ohlcv-5m", dataset="DBIX.CORE")
        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(figure.axes[0].get_ylabel(), "Index value")
        self.assertEqual(list(figure.axes[0].lines[0].get_ydata()), [5000.0, 5001.0])
        plt.close(figure)

    def test_each_symbol_has_its_own_price_and_volume_series(self):
        for schema in ["trades", "ohlcv-1m"]:
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as directory:
                source = write_dbn(Path(directory) / "source.dbn", schema=schema)
                frame = db.DBNStore.from_bytes(source.read_bytes()).to_df()
                for mapped in [True, False]:
                    preview = frame if mapped else frame.assign(symbol=None)
                    figure = pipeline.plot_preview(preview, schema)
                    self.assertEqual(len(figure.axes), 2 if schema.startswith("ohlcv") else 1)
                    self.assertEqual(len(figure.axes[0].lines), 2)
                    for instrument_id, line in zip([1, 2], figure.axes[0].lines):
                        expected = frame[frame["instrument_id"] == instrument_id]
                        self.assertEqual(list(line.get_ydata()), list(expected["close" if schema.startswith("ohlcv") else "price"]))
                        self.assertEqual(len(line.get_xdata()), len(expected))
                        if mapped:
                            self.assertIn(expected["symbol"].iloc[0], line.get_label())
                    if schema.startswith("ohlcv"):
                        self.assertEqual(len(figure.axes[1].lines), 2)
                    plt.close(figure)

    def test_mbo_visualization_counts_events_instead_of_connecting_order_prices(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_dbn(Path(directory) / "source.dbn")
            frame = db.DBNStore.from_bytes(source.read_bytes()).to_df()
            frame.index = pd.DatetimeIndex([START + pd.Timedelta(milliseconds=index) for index in range(len(frame))])
            figure = pipeline.plot_preview(frame, "mbo")
            self.assertTrue(figure.axes[0].get_ylabel().startswith("Events"))
            self.assertEqual(len(figure.axes[0].lines), 2)
            self.assertTrue(all(line.get_drawstyle() == "steps-mid" for line in figure.axes[0].lines))
            left, right = figure.axes[0].get_xlim()
            self.assertLess((right - left) * 86400, 3)
            plt.close(figure)

    def test_single_ohlcv_bar_has_a_short_time_axis_and_visible_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_dbn(Path(directory) / "source.dbn", schema="ohlcv-1m", count=1)
            frame = db.DBNStore.from_bytes(source.read_bytes()).to_df()
            figure = pipeline.plot_preview(frame, "ohlcv-1m")
            left, right = figure.axes[0].get_xlim()
            self.assertAlmostEqual((right - left) * 86400, 120, places=3)
            self.assertEqual(figure.axes[1].lines[0].get_marker(), ".")
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
