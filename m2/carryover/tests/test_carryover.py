"""Tests for the Milestone 2 carryover sidecars (m2/carryover) and the Bayes cache-name fix."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from integration import stress
from integration.combine import V3_DIR, load_v3, v3_table
from m2.carryover import severity as S
from m2.carryover._util import ns
from m2.carryover import streaming as L
from m2.carryover.crossed import crossed_census, flast_aware_crossed
from m2.carryover.trade_pauses import pause_stats, reclassify

ROOT = Path(__file__).resolve().parents[3]
NS = 1_000_000_000
_R = None


def run():
    global _R
    if _R is None:
        _R = stress.stress_test()
    return _R


def synthetic():
    v3 = load_v3()
    raw = v3.synthetic.generate_mbp1(seed=7)
    faulted, labels = v3.synthetic.inject_faults(raw, seed=11)
    return v3, v3.synthetic.to_databento_df(raw), v3.synthetic.to_databento_df(faulted), labels


def engine_crossed(t):
    return (t["source"] == "engine_m1") & (t["rule"] == "crossed_quote") & (t["level"] > 0)


class CrossedTests(unittest.TestCase):
    def test_mark_is_identity(self):
        r = run()
        pd.testing.assert_frame_equal(flast_aware_crossed(r.unified, "mark"), r.unified)

    def test_demote_and_drop_clear_clean_stream_and_keep_recall(self):
        r = run()
        self.assertEqual(int(engine_crossed(r.clean).sum()), 3273)
        for mode in ("demote", "drop"):
            self.assertEqual(int(engine_crossed(flast_aware_crossed(r.clean, mode)).sum()), 0)
            det, _ = stress.score(flast_aware_crossed(r.unified, mode), r.labels)
            self.assertEqual(int(det.loc["crossed_books", "engine_m1"]), 5)
            self.assertEqual(int(det["either"].sum()), 41)

    def test_demote_keeps_rows(self):
        r = run()
        self.assertEqual(len(flast_aware_crossed(r.unified, "demote")), len(r.unified))

    def test_persist_keeps_injected_crosses(self):
        r = run()
        _, _, faulted, _ = synthetic()
        det, _ = stress.score(flast_aware_crossed(r.unified, "persist", frame=faulted), r.labels)
        self.assertEqual(int(det.loc["crossed_books", "engine_m1"]), 5)

    def test_census_on_clean_stream(self):
        _, raw, _, _ = synthetic()
        c = crossed_census(raw)
        self.assertEqual(c["crossed_f_last"], 0)
        self.assertEqual(c["crossed_no_f_last"], 3273)


class SeverityTests(unittest.TestCase):
    def test_native_columns_untouched_and_all_rows_mapped(self):
        r = run()
        a = S.apply(r.unified)
        pd.testing.assert_series_equal(a["level"], r.unified["level"])
        pd.testing.assert_series_equal(a["severity"], r.unified["severity"])
        self.assertFalse((a["map_key"] == "native").any())
        self.assertTrue(set(a["severity_m2"]) <= set(S.LEVELS))

    def test_overlapping_checks_get_the_same_level(self):
        r = run()
        a = S.apply(r.unified)
        eng = a[(a["rule"] == "crossed_quote") & (a["f_last"] == True)]  # noqa: E712
        v3x = a[(a["rule"] == "crossed_books") & (a["subtype"] == "crossed")]
        self.assertEqual(set(eng["level_m2"]), set(v3x["level_m2"]))
        self.assertTrue((a.loc[(a["rule"] == "crossed_quote") & (a["f_last"] == False), "severity_m2"] == "info").all())  # noqa: E712

    def test_every_injected_fault_still_above_info(self):
        r = run()
        a = S.apply(r.unified)
        a = a[a["level_m2"] > 0].copy()
        a["level"] = a["level_m2"]
        det, _ = stress.score(a, r.labels)
        self.assertEqual(int(det["either"].sum()), 41)

    def test_bayes_and_spc_rows(self):
        g = pd.DataFrame({"venue": "X", "symbol": "A", "day": "d", "ts_recv_ns": [1, 2, 3], "gap_s": [6.0, 90.0, 90.0],
                          "a": 1, "b": 1, "tail": 0, "p_outage": [0.6, 0.999, 0.7], "bayes": True, "poisson": True})
        self.assertEqual(list(S.apply(S.bayes_rows(g))["severity_m2"]), ["medium", "high", "medium"])
        sig = pd.DataFrame({"symbol": "A", "day": "d", "kind": "rate", "chart": "cusum", "bucket": [1, 5, 9, 100]})
        e = S.apply(S.spc_rows(sig))
        self.assertEqual(list(e["value"]), [9.0, 1.0])
        self.assertEqual(list(e["severity_m2"]), ["low", "low"])


class StreamingTests(unittest.TestCase):
    def test_gap_monitor_equals_batch(self):
        v3, _, faulted, _ = synthetic()
        cfg = v3.config.DetectorConfig().missing
        batch = v3.detectors.missing_intervals(faulted, cfg)
        t = ns(faulted.index).to_numpy()
        clk = v3.detectors.SessionClock.build(cfg.calendar, int(t.min()), int(t.max()))
        live = L.replay_gaps(faulted, L.StreamingGapMonitor(cfg.alpha, cfg.rate_window, cfg.min_gap_s, cfg.max_gap_s, clock=clk.clock))
        self.assertEqual(sorted(ns(batch["ts_recv"])), sorted(live["ts_recv_ns"]))
        np.testing.assert_allclose(np.sort(batch["threshold"].astype(float)), np.sort(live["threshold_s"]))

    def test_crossed_equals_batch(self):
        v3, _, faulted, _ = synthetic()
        cfg = v3.config.DetectorConfig().crossed
        b = v3.detectors.crossed_books(faulted, cfg)
        b = b[b["subtype"] == "crossed"]
        live = L.replay_crossed(faulted, L.StreamingCrossed(cfg.min_duration_s))
        self.assertEqual(sorted(ns(b["ts_recv"])), sorted(live["start_ns"]))
        self.assertTrue((live["detected_ns"] - live["start_ns"] >= cfg.min_duration_s * NS).all())

    def test_open_gap_alarm_fires_once_before_gap_end(self):
        m = L.StreamingGapMonitor(rate_window=10, min_gap_s=1.0, max_gap_s=60.0)
        for i in range(20):
            m.update(i * NS // 10)
        last = 19 * NS // 10
        self.assertEqual(m.poll(last + NS // 2), [])
        alarms = m.poll(last + 2 * NS)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(m.poll(last + 3 * NS), [])
        self.assertIsNotNone(m.update(last + 4 * NS))

    def test_trailing_latency_flags_spike(self):
        tr = L.TrailingLatencyOutlier(k=10, min_excess_ns=1_000_000, window=1000, min_periods=100)
        rng = np.random.default_rng(0)
        hits = [tr.update(int(x))[0] for x in rng.normal(200_000, 5_000, 500)]
        self.assertFalse(any(hits))
        self.assertTrue(tr.update(50_000_000)[0])

    def test_detected_at_gap_lead(self):
        u = pd.DataFrame({"source": ["detectors_v3"], "rule": ["missing_intervals"], "subtype": ["gap"],
                          "ts_recv": [pd.Timestamp("2026-10-08T14:00:20Z")], "threshold": [5.0], "value": [20.0],
                          "detail": ["gap_start=2026-10-08T14:00:00+00:00 wall=20.000s"]})
        self.assertAlmostEqual(float(L.detected_at(u)["live_lead_s"].iloc[0]), 15.0)


class TradePauseTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame({"source": "detectors_v3", "mode": "batch", "rule": "missing_intervals", "subtype": "gap",
                             "severity": "medium", "level": 2.0, "ts_event": [pd.Timestamp("2025-09-15T14:32:21Z")],
                             "ts_recv": [pd.Timestamp("2025-09-15T14:32:21Z")], "symbol": "SPY", "publisher_id": 2,
                             "instrument_id": 1, "value": [5.2], "threshold": [5.0],
                             "detail": ["gap_start=2025-09-15T14:32:15.800000+00:00 wall=5.200s"], "f_last": None})

    def test_quotes_inside_make_it_a_pause(self):
        q = pd.DataFrame({"ts_recv": pd.to_datetime(["2025-09-15T14:32:17Z", "2025-09-15T14:32:19Z"], utc=True), "symbol": "SPY"})
        out = reclassify(self.table(), "trades", quotes=q)
        self.assertEqual(out["subtype"].iloc[0], "trade_pause")
        self.assertEqual(out["level"].iloc[0], 0.0)

    def test_no_quotes_inside_keeps_candidate(self):
        q = pd.DataFrame({"ts_recv": pd.to_datetime(["2025-09-15T14:40:00Z"], utc=True), "symbol": "SPY"})
        out = reclassify(self.table(), "trades", quotes=q)
        self.assertEqual(out["subtype"].iloc[0], "gap")

    def test_floor_and_schema_guard(self):
        self.assertEqual(reclassify(self.table(), "trades", floor_s={"SPY": 15.0})["subtype"].iloc[0], "trade_pause_below_floor")
        self.assertEqual(reclassify(self.table(), "trades", floor_s=5.0)["subtype"].iloc[0], "gap")
        pd.testing.assert_frame_equal(reclassify(self.table(), "mbp-1"), self.table())

    def test_bundled_trades_sample(self):
        from integration.combine import bundled, run_combined
        fr, meta = bundled("trades")
        u = run_combined(fr, meta).unified
        gaps = u[u["rule"] == "missing_intervals"]
        self.assertEqual(len(gaps), 2)
        out = reclassify(u, "trades")
        self.assertEqual(len(out), len(u))
        self.assertTrue((out.loc[gaps.index, "severity"] == "info").all())

    def test_pause_stats(self):
        t = pd.Timestamp("2026-10-08 13:30:01Z").value + np.array([0, 1, 2, 9, 10]) * NS
        q = pd.Timestamp("2026-10-08 13:30:01Z").value + np.array([4, 5]) * NS
        fr = pd.DataFrame({"ts_recv": pd.to_datetime(np.r_[t, q], utc=True), "action": ["T"] * 5 + ["A"] * 2})
        s = pause_stats(fr, "2026-10-08")
        self.assertEqual((s["trades"], s["pauses_gt"], s["pauses_with_quotes"]), (5, 1, 1))


class MetricsRegenTests(unittest.TestCase):
    def test_sidecar_reproduces_known_difference_and_leaves_stored_file(self):
        from m2.carryover.metrics_regen import compare, regenerate
        stored = Path(V3_DIR) / "results" / "metrics.csv"
        before = hashlib.sha256(stored.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            regenerate(Path(d), sweep=False)
            diff = compare(Path(d)).set_index(["check", "column"])["regenerated"]
        self.assertEqual(hashlib.sha256(stored.read_bytes()).hexdigest(), before)
        self.assertEqual(int(diff[("extreme_prices", "flags")]), 12)
        self.assertEqual(int(diff[("extreme_prices", "clean_stream_flags")]), 0)
        self.assertEqual(set(diff.index.get_level_values(0)), {"extreme_prices"})


class BayesCacheRegressionTests(unittest.TestCase):
    def test_cache_name_keeps_dotted_symbols(self):
        from m2.bayes.run_bayes_gaps import _p
        a = _p(Path("/c/GLBX_ES.c.0_2026-10-05"), ".json")
        b = _p(Path("/c/GLBX_ES.c.0_2026-10-06"), ".json")
        self.assertNotEqual(a, b)
        self.assertEqual(a.name, "GLBX_ES.c.0_2026-10-05.json")

    def test_saved_real_run_has_distinct_days_and_matches_v3(self):
        path = ROOT / "m2/results/bayes/real_symbol_days.csv"
        if not path.is_file():
            self.skipTest("real run not saved")
        d = pd.read_csv(path)
        self.assertFalse(d.duplicated(["symbol", "day"]).any())
        self.assertTrue((d["poisson"] == d["v3_stored_flags"]).all())
        idx = pd.read_csv(ROOT / "p115_detectors/results_real_v3/run_index.csv")
        m = d.merge(idx, on=["symbol", "day"], suffixes=("", "_idx"))
        self.assertEqual(len(m), len(d))
        self.assertTrue((m["poisson"] == m["flags"]).all())


class SampleExtensionTests(unittest.TestCase):
    def test_calendar_rules(self):
        from m2.carryover.sample_extension import cl_last_trade, third_friday
        self.assertEqual(str(third_friday(2026, 9).date()), "2026-09-18")
        self.assertEqual(str(cl_last_trade(2025, 12).date()), "2025-11-20")  # CLZ5
        self.assertEqual(str(cl_last_trade(2026, 1).date()), "2025-12-19")   # CLF6
        self.assertEqual(str(cl_last_trade(2026, 6).date()), "2026-05-19")   # 25 May 2026 is Memorial Day

    def test_estimate_offline(self):
        from m2.carryover.sample_extension import estimate, requests
        census = pd.DataFrame([dict(symbol=s, day_type=t, records=1e6) for s in
                               ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "CRWV", "NBIS", "LWAY", "PLAB", "ADTN", "ES.c.0", "CL.c.0"]
                               for t in ("normal_week", "cpi", "half_day")])
        e = estimate(census)
        self.assertEqual(len(e), len(requests()))
        self.assertTrue((e["usd_est"] > 0).all())


if __name__ == "__main__":
    unittest.main()
