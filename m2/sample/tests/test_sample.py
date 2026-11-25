"""Tests for the representative-sample extension: strata metadata, holiday-calendar sidecar, runner helpers."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from integration.combine import load_v3
from m2.extract import DAY_TYPE
from m2.sample import holidays as H
from m2.sample.strata import sample_tasks, stratum_of, strata_table, tags_for

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "m2" / "sample" / "results"


def ns(s, tz):
    return pd.Timestamp(s, tz=tz).tz_convert("UTC").value


class StrataTests(unittest.TestCase):
    def test_sept16_is_vix_high_and_fomc_without_touching_stored_label(self):
        self.assertEqual(DAY_TYPE["2026-09-16"], "vix_high")
        t = tags_for("XNAS", "AAPL", "2026-09-16")
        self.assertIn("vix_high", t); self.assertIn("fomc", t)
        st = strata_table()
        row = st[(st["source"] == "pilot") & (st["day"] == "2026-09-16")].iloc[0]
        self.assertEqual(row["stored_day_type"], "vix_high")
        self.assertEqual(set(row["tags"].split("|")), {"vix_high", "fomc"})

    def test_strata(self):
        self.assertEqual(len(sample_tasks()), 85)
        self.assertEqual(stratum_of(tags_for("GLBX", "ES.c.0", "2026-09-16")), "es_roll")
        self.assertEqual(stratum_of(tags_for("GLBX", "CL.c.0", "2026-09-16")), "fomc")
        self.assertEqual(stratum_of(tags_for("GLBX", "CL.c.1", "2026-10-20")), "cl_expiry")
        self.assertEqual(stratum_of(tags_for("GLBX", "ES.c.1", "2026-10-08")), "normal_week_c1")
        self.assertEqual(stratum_of(tags_for("XNAS", "SPY", "2025-12-24")), "holiday")
        self.assertEqual(stratum_of(tags_for("XNAS", "SPY", "2026-10-07")), "normal_week")


class HolidayCalendarTests(unittest.TestCase):
    def setUp(self):
        self.v3 = load_v3()

    def test_presets_unchanged(self):
        before = self.v3.config.equities_config().missing.calendar
        H.holiday_config(self.v3.config.equities_config(), "XNAS", v3=self.v3)
        after = self.v3.config.equities_config().missing.calendar
        self.assertEqual(before, after)
        self.assertEqual(after.holidays, [])
        self.assertEqual(self.v3.config.cme_calendar().early_closes, {})

    def test_xnas_early_close_and_closure(self):
        cal = H.holiday_calendar("XNAS", v3=self.v3)
        D = self.v3.detectors
        t = np.array([ns("2025-12-24 12:30", "America/New_York"), ns("2025-12-24 13:30", "America/New_York"),
                      ns("2026-01-19 11:00", "America/New_York")])
        clk = D.SessionClock.build(cal, int(t.min()), int(t.max()))
        self.assertEqual(list(clk.in_session(t)), [True, False, False])
        base = D.SessionClock.build(self.v3.config.equities_config().missing.calendar, int(t.min()), int(t.max()))
        self.assertEqual(list(base.in_session(t)), [True, True, True])

    def test_cme_halt_by_root(self):
        D = self.v3.detectors
        tz = "America/Chicago"
        t = np.array([ns("2026-01-18 18:00", tz), ns("2026-01-19 11:30", tz), ns("2026-01-19 12:30", tz), ns("2026-01-19 14:00", tz)])
        es = D.SessionClock.build(H.holiday_calendar("GLBX", "ES", self.v3), int(t.min()), int(t.max()))
        cl = D.SessionClock.build(H.holiday_calendar("GLBX", "CL", self.v3), int(t.min()), int(t.max()))
        self.assertEqual(list(es.in_session(t)), [True, True, False, False])
        self.assertEqual(list(cl.in_session(t)), [True, True, True, False])

    def test_holiday_config_reaches_every_calendar_check(self):
        cfg = H.holiday_config(self.v3.config.cme_futures_config(), "GLBX", "CL", self.v3)
        for s in ("missing", "stale", "extreme", "spreads"):
            self.assertEqual(getattr(cfg, s).calendar.early_closes.get("2026-07-03"), "12:00")
        self.assertEqual(H.is_special("GLBX", "2026-01-19", "ES"), "halt")
        self.assertEqual(H.is_special("XNAS", "2026-01-19"), "closed")
        self.assertIsNone(H.is_special("XNAS", "2026-10-08"))

    def test_gap_spanning_halt_is_not_in_session(self):
        D = self.v3.detectors
        tz = "America/Chicago"
        t = np.array([ns("2026-01-19 11:59:50", tz), ns("2026-01-19 15:59:00", tz)])
        es = D.SessionClock.build(H.holiday_calendar("GLBX", "ES", self.v3), int(t.min()), int(t.max()))
        base = D.SessionClock.build(self.v3.config.cme_calendar(), int(t.min()), int(t.max()))
        self.assertLess((es.clock(t)[1] - es.clock(t)[0]) / 1e9, 11)
        self.assertGreater((base.clock(t)[1] - base.clock(t)[0]) / 1e9, 14000)


class RunnerHelperTests(unittest.TestCase):
    def test_cache_names(self):
        from m2.sample.run_sample import _p, _episodes
        self.assertEqual(_p(Path("/c/GLBX_ES.c.1_2026-09-18"), ".json").name, "GLBX_ES.c.1_2026-09-18.json")
        self.assertEqual(_episodes([1, 5, 40, 41, 100]), 3)
        self.assertEqual(_episodes([]), 0)

    def test_saved_results_if_present(self):
        p = RES / "v3_by_stratum.csv"
        if not p.is_file():
            self.skipTest("sample results not built")
        t = pd.read_csv(p)
        self.assertTrue({"normal_week", "fomc", "es_roll", "cl_expiry", "holiday"} <= set(t["stratum"]))


if __name__ == "__main__":
    unittest.main()
