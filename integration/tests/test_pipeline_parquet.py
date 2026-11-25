"""Plus-layer tests: pipeline Parquet reader and batch runner on bundled samples.

Adapted from Aryan's tests/test_integration.py. The batch checks are p115_detectors
via integration.pipeline_parquet, not a relocated copy. No Databento client and no download.
"""
import tempfile
import unittest
from pathlib import Path

from integration.combine import load_v3
from integration.pipeline_parquet import (
    pipeline_frame,
    pipeline_parquets,
    read_pipeline_parquet,
    run_batch_frame,
    run_batch_sample,
)
from run_checks import run


class PipelineParquetTests(unittest.TestCase):
    def test_streaming_and_batch_checks_share_the_sample_files(self):
        v3 = load_v3()
        for sample, symbols in (("quotes", ["AAPL"]), ("trades", ["AAPL", "SPY"])):
            data, alerts = run(sample)
            frame, flags, summary = run_batch_sample(sample, "equities")
            self.assertEqual(data.metadata["symbols"], symbols)
            self.assertEqual(len(data.events), data.metadata["provenance"]["records"])
            self.assertEqual(len(frame), len(data.events))
            self.assertEqual(sorted(frame["symbol"].unique()), symbols)
            self.assertEqual(alerts, [])
            self.assertEqual(list(flags.columns), v3.detectors.FLAG_COLUMNS)
            sequence_gaps = flags[(flags["check"] == "sequence_discontinuities") & (flags["subtype"] == "gap")]
            self.assertTrue(sequence_gaps.empty)
            if sample == "quotes":
                self.assertTrue(flags.empty)
                self.assertTrue(summary.empty)
            else:
                self.assertEqual(sorted(flags["symbol"]), ["AAPL", "SPY"])
                self.assertEqual(list(flags["check"]), ["missing_intervals", "missing_intervals"])
                self.assertTrue((flags["value"] > 5).all() and (flags["value"] < 6).all())
                self.assertEqual(int(summary["n_flags"].sum()), 2)

    def test_quotes_frame_keeps_pipeline_book_columns(self):
        frame, flags, summary = run_batch_sample("quotes", "equities")
        for column in ("bid_px_00", "ask_px_00", "flags", "sequence", "ts_event"):
            self.assertIn(column, frame.columns)
        self.assertEqual(frame.index.name, "ts_recv")
        self.assertTrue((frame["bid_px_00"] > 0).any())
        self.assertTrue((frame["ask_px_00"] > 0).any())
        self.assertEqual(int(summary["n_flags"].sum()) if len(summary) else 0, len(flags))

    def test_directory_of_pipeline_parquet_runs_without_a_download(self):
        source = pipeline_frame({"source": "bundled", "sample": "quotes"})
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "download"
            run_dir.mkdir()
            source.to_parquet(run_dir / "data.parquet")
            paths = pipeline_parquets(run_dir)
            self.assertEqual(len(paths), 1)
            frame = read_pipeline_parquet(paths[0])
            flags, summary = run_batch_frame(frame, "equities")
            self.assertEqual(len(frame), len(source))
            v3 = load_v3()
            self.assertEqual(list(flags.columns), v3.detectors.FLAG_COLUMNS)
            self.assertEqual(int(summary["n_flags"].sum()) if len(summary) else 0, len(flags))

    def test_databento_source_is_refused_before_any_download(self):
        with self.assertRaisesRegex(ValueError, "does not download"):
            pipeline_frame({
                "source": "databento",
                "symbols": ["AAPL"],
                "start": "2025-09-15T14:30:00Z",
                "end": "2025-09-15T14:31:00Z",
            })

    def test_unknown_preset_is_rejected(self):
        frame, _flags, _summary = run_batch_sample("quotes", "equities")
        with self.assertRaisesRegex(ValueError, "preset must be equities, cme, or base"):
            run_batch_frame(frame, "not-a-preset")


if __name__ == "__main__":
    unittest.main()
