"""Tests for the add-only integration layer (Milestone 1 tests in tests/ are untouched).

Run from the repository root: python -m unittest discover -s integration/tests -t .
"""
import unittest
from pathlib import Path

import pandas as pd

import run_checks
from alert_system.data import load_data, normalize_frame
from integration import combine, stress, verify


class UntouchedFilesTests(unittest.TestCase):
    def test_milestone1_files_are_byte_identical(self):
        total, changed, missing = verify.check("milestone1.sha256")
        self.assertEqual(total, 16)
        self.assertEqual((changed, missing), ([], []))

    def test_v3_code_is_byte_identical(self):
        total, changed, missing = verify.check("detectors_v3_code.sha256")
        self.assertGreater(total, 0)
        self.assertEqual((changed, missing), ([], []))


class ModuleIsolationTests(unittest.TestCase):
    def test_v3_modules_load_from_their_own_folder(self):
        import alert_system.config
        import alert_system.detectors
        v3 = combine.load_v3()
        for module in (v3.config, v3.detectors, v3.synthetic):
            self.assertEqual(Path(module.__file__).resolve().parent, combine.V3_DIR.resolve())
        self.assertIsNot(v3.detectors, alert_system.detectors)
        self.assertIsNot(v3.config, alert_system.config)
        self.assertTrue(hasattr(alert_system.detectors, "AlertEngine"))
        self.assertTrue(hasattr(v3.detectors, "run_all"))


class EngineAdapterTests(unittest.TestCase):
    def test_bundled_samples_reproduce_run_checks(self):
        for sample in ("quotes", "trades"):
            with self.subTest(sample=sample):
                data, expected = run_checks.run(sample)
                frame, metadata = combine.bundled(sample)
                self.assertEqual(normalize_frame(frame, metadata).events, data.events)
                alerts, _ = combine.run_engine(frame, metadata)
                self.assertEqual(alerts, expected)

    def test_chunked_run_equals_single_run(self):
        v3 = combine.load_v3()
        frame = v3.synthetic.to_databento_df(v3.synthetic.generate_mbp1(duration_s=300, seed=3))
        whole, _ = combine.run_engine(frame, stress.SYNTHETIC_META)
        chunked, _ = combine.run_engine(frame, stress.SYNTHETIC_META, chunk_rows=997)
        self.assertGreater(len(whole), 0)
        self.assertEqual(chunked, whole)

    def test_non_equity_data_skips_engine_but_runs_v3(self):
        frame, metadata = combine.bundled("quotes")
        result = combine.run_combined(frame, {**metadata, "dataset": "GLBX.MDP3"})
        self.assertEqual(result.alerts, [])
        self.assertIn("engine_m1 skipped", result.engine_note)
        self.assertEqual(list(result.unified.columns), combine.UNIFIED_COLUMNS)


class UnifiedTableTests(unittest.TestCase):
    def test_bundled_trades(self):
        frame, metadata = combine.bundled("trades")
        result = combine.run_combined(frame, metadata)
        self.assertEqual(list(result.unified.columns), combine.UNIFIED_COLUMNS)
        self.assertEqual(len(result.unified), len(result.alerts) + len(result.flags))
        self.assertEqual(set(result.overlap["check"]), {name for name, _, _ in combine.OVERLAPS})


class StressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = stress.stress_test(duration_s=1800, n_per_type=3, integrity_per_type=2)

    def test_unified_columns_and_levels(self):
        unified = self.result.unified
        self.assertEqual(list(unified.columns), combine.UNIFIED_COLUMNS)
        self.assertEqual(set(unified["source"]), {combine.ENGINE, combine.BATCH})
        self.assertTrue(unified["level"].isin([0, 1, 2, 3]).all())

    def test_every_label_is_found_by_at_least_one_system(self):
        detection = self.result.detection
        self.assertTrue((detection["either"] == detection["labels"]).all(), detection)

    def test_each_system_adds_coverage(self):
        detection = self.result.detection
        self.assertEqual(detection.loc["integrity", combine.ENGINE], detection.loc["integrity", "labels"])
        self.assertTrue(pd.isna(detection.loc["integrity", combine.BATCH]))
        for family in ("missing_intervals", "stale_quotes", "sequence_discontinuities"):
            self.assertTrue(pd.isna(detection.loc[family, combine.ENGINE]))
            self.assertEqual(detection.loc[family, combine.BATCH], detection.loc[family, "labels"])

    def test_engine_crosses_on_clean_stream_are_intermediate_records(self):
        clean = self.result.clean
        crosses = clean[(clean["source"] == combine.ENGINE) & (clean["rule"] == "crossed_quote")]
        self.assertGreater(len(crosses), 0)
        self.assertTrue(crosses["f_last"].eq(False).all())
        v3_crosses = clean[(clean["source"] == combine.BATCH) & (clean["rule"] == "crossed_books") & (clean["level"] > 0)]
        self.assertEqual(len(v3_crosses), 0)

    def test_overlap_report_on_injected_crosses(self):
        crossed = combine.overlap_report(self.result.unified).set_index("check").loc["crossed quote"]
        self.assertGreater(crossed["detectors_v3"], 0)
        self.assertEqual(crossed["detectors_v3_with_engine_match"], crossed["detectors_v3"])
        self.assertGreater(crossed["engine_m1"], crossed["engine_m1_with_v3_match"])

    def test_scoring_reproduces_v3_evaluate(self):
        v3 = combine.load_v3(names=("config", "detectors", "synthetic", "evaluate"))
        faulted, labels = v3.synthetic.inject_faults(v3.synthetic.generate_mbp1(duration_s=1800, seed=7),
                                                     n_per_type=3, seed=11)
        flags, _ = v3.detectors.run_all(v3.synthetic.to_databento_df(faulted), stress.v3_eval_config(v3))
        theirs = v3.evaluate.match(flags, labels).set_index("check")
        detection, breakdown = stress.score(combine.v3_table(flags), labels)
        for check in theirs.index:
            self.assertEqual(detection.loc[check, combine.BATCH], theirs.loc[check, "detected"])
            if (combine.BATCH, check) in breakdown.index:
                mine = breakdown.loc[(combine.BATCH, check)]
                self.assertEqual((mine["on_label"], mine["collateral"], mine["unlabeled"]),
                                 tuple(int(theirs.loc[check, name]) for name in ("tp_flags", "collateral_flags", "fp_flags")))


if __name__ == "__main__":
    unittest.main()
