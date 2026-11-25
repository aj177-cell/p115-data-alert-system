"""Tests for the Milestone 2 control charts (m2/spc)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from integration.combine import load_v3
from m2.spc.baseline import ar1_phi, phase1, prewhiten, standardise
from m2.spc.charts import ChartParams, calibrate, cusum, cusum_arl, ewma, ewma_arl, run_charts, shewhart, simulated_arl
from m2.spc.evaluate import signals, zscores
from m2.spc.series import build_series, session_buckets

NS = 1_000_000_000


class ChartMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = calibrate(370.0)

    def test_cusum_arl_matches_siegmund(self):
        # Siegmund: ARL0 ~ (exp(2 k b) - 2 k b - 1) / (2 k^2), b = h + 1.166
        for h in (3.0, 4.0, 5.0):
            b = h + 1.166
            sieg = (np.exp(2 * 0.5 * b) - 2 * 0.5 * b - 1) / (2 * 0.25)
            self.assertLess(abs(cusum_arl(0.5, h) / sieg - 1), 0.03)

    def test_calibrated_limits_hit_target(self):
        self.assertAlmostEqual(cusum_arl(self.p.cusum_k, self.p.cusum_h), 370.0, delta=1.0)
        self.assertAlmostEqual(ewma_arl(self.p.ewma_lambda, self.p.ewma_L), 370.0, delta=1.0)
        for chart in ("cusum", "ewma", "shewhart"):
            self.assertLess(abs(simulated_arl(chart, self.p, n_series=300, seed=3) / 370.0 - 1), 0.15, chart)

    def test_one_sigma_shift_detected_fast(self):
        self.assertLess(cusum_arl(self.p.cusum_k, self.p.cusum_h, mu=1.0), 12)
        self.assertLess(ewma_arl(self.p.ewma_lambda, self.p.ewma_L, mu=1.0), 12)

    def test_direction_nan_and_restart(self):
        z = np.r_[np.zeros(20), np.full(10, -3.0), [np.nan] * 5, np.zeros(10)]
        self.assertFalse(cusum(z, 0.5, 4.0, "up").any())
        s = cusum(z, 0.5, 4.0, "down")
        self.assertTrue(s[20:30].any())
        self.assertFalse(s[30:].any())
        self.assertFalse(shewhart(np.array([np.nan, 5.0]), 3.0)[0])
        self.assertTrue(ewma(np.r_[np.zeros(5), np.full(10, 3.0)], 0.2, 2.6).any())


class BaselineTests(unittest.TestCase):
    def _day(self, rng, shift=0.0):
        b = np.arange(390)
        lam = 200 * (1 + 0.8 * np.exp(-b / 30) + 0.6 * np.exp((b - 389) / 20))  # U-shaped intraday rate
        n = rng.poisson(lam * (1 + shift))
        return pd.DataFrame({"bucket": b, "n_records": n, "spread_ticks": np.full(390, 2.0)})

    def test_tod_baseline_standardises_in_control_day(self):
        rng = np.random.default_rng(1)
        hist = [self._day(rng) for _ in range(4)]
        z = standardise(self._day(rng), phase1(hist, "rate"), "rate")
        self.assertLess(abs(np.median(z)), 0.25)
        self.assertLess(abs(1.4826 * np.median(np.abs(z - np.median(z))) - 1), 0.35)
        # open and close are not systematically flagged
        self.assertLess(abs(np.median(z[:30])), 0.6)
        self.assertLess(abs(np.median(z[-30:])), 0.6)

    def test_spread_scale_floor(self):
        rng = np.random.default_rng(2)
        hist = [self._day(rng) for _ in range(4)]
        base = phase1(hist, "spread")
        self.assertTrue((base["s"] >= 0.3).all())

    def test_prewhitening_removes_ar1(self):
        rng = np.random.default_rng(3)
        x = np.zeros(5000)
        for i in range(1, 5000):
            x[i] = 0.6 * x[i - 1] + rng.standard_normal() * np.sqrt(1 - 0.36)
        phi = ar1_phi([x])
        self.assertAlmostEqual(phi, 0.6, delta=0.05)
        e = prewhiten(x, phi)
        self.assertLess(abs(np.corrcoef(e[1:-1], e[2:])[0, 1]), 0.05)
        self.assertAlmostEqual(np.std(e), 1.0, delta=0.06)


class SeriesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v3 = load_v3()

    def test_session_bucket_grid(self):
        self.assertEqual(len(session_buckets("XNAS", "2026-10-08", self.v3)), 390)
        self.assertEqual(len(session_buckets("XNAS", "2025-11-28", self.v3)), 210)
        self.assertEqual(len(session_buckets("GLBX", "2026-10-08", self.v3)), 1380)
        self.assertEqual(len(session_buckets("GLBX", "2026-08-29_weekend", self.v3)), 0)

    def test_counts_and_carry_forward(self):
        t0 = pd.Timestamp("2026-10-08 09:30", tz="America/New_York").value
        t = np.array([t0 + 1 * NS, t0 + 2 * NS, t0 + 61 * NS, t0 + 3 * 60 * NS + 5 * NS, t0 - 60 * NS])
        flags = np.full(5, 128)
        bid = np.array([10.00, 10.00, 10.00, 10.00, 10.00]); ask = np.array([10.02, 10.04, 10.01, 10.05, 10.50])
        s = build_series(t, flags, bid, ask, 0.01, "XNAS", "2026-10-08", self.v3).set_index("bucket")
        self.assertEqual(list(s["n_records"].iloc[:4]), [2, 1, 0, 1])  # pre-open record excluded
        self.assertAlmostEqual(s.loc[0, "spread_ticks"], 3.0, places=6)
        self.assertAlmostEqual(s.loc[2, "spread_ticks"], 1.0, places=6)  # carried forward
        self.assertFalse(s.loc[2, "spread_obs"])
        self.assertAlmostEqual(s.loc[3, "spread_ticks"], 5.0, places=6)


class SyntheticIntegrationTests(unittest.TestCase):
    """End to end on the v3 synthetic generator: Phase I from four clean seeds."""

    @classmethod
    def setUpClass(cls):
        cls.v3 = load_v3()
        cls.hist = [cls._series(cls._raw(s)) for s in (1, 2, 3, 4)]
        cls.params = {"rate": calibrate(370.0), "spread": calibrate(370.0)}

    @classmethod
    def _raw(cls, seed):
        r = cls.v3.synthetic.generate_mbp1(duration_s=3600, seed=seed)
        r = r[r["instrument_id"] == 101].reset_index(drop=True)
        return r

    @classmethod
    def _series(cls, r):
        return build_series(r["ts_recv"].to_numpy().astype(np.int64), r["flags"].to_numpy().astype(np.int64),
                            r["bid_px_00"].to_numpy() / 1e9, r["ask_px_00"].to_numpy() / 1e9, 0.01, "XNAS",
                            "2026-10-08", cls.v3)

    def _window(self, r, a, b):
        t0 = pd.Timestamp("2026-10-08 09:30", tz="America/New_York").value
        t = r["ts_recv"].to_numpy().astype(np.int64)
        return (t >= t0 + a * 60 * NS) & (t < t0 + b * 60 * NS)

    def test_clean_day_is_quiet(self):
        z = zscores(self._series(self._raw(98)), self.hist)
        sig = signals(z, self.params)
        self.assertLessEqual(sum(int(m[:60].sum()) for m in sig.values()), 2)

    def test_thinning_found_by_rate_cusum(self):
        r = self._raw(99)
        rng = np.random.default_rng(0)
        drop = self._window(r, 20, 40) & (rng.random(len(r)) < 0.5)
        sig = signals(zscores(self._series(r[~drop]), self.hist), self.params)
        hit = np.flatnonzero(sig[("rate", "cusum")])
        self.assertTrue(((hit >= 20) & (hit < 40)).any())
        self.assertFalse(sig[("spread", "cusum")][:60].any())

    def test_spread_widening_found_by_spread_charts(self):
        r = self._raw(97).copy()
        w = self._window(r, 10, 30)
        r.loc[w, "ask_px_00"] = r.loc[w, "ask_px_00"] + 2 * 10_000_000  # +2 ticks
        sig = signals(zscores(self._series(r), self.hist), self.params)
        for c in ("cusum", "ewma"):
            hit = np.flatnonzero(sig[("spread", c)])
            self.assertTrue(((hit >= 10) & (hit < 30)).any(), c)

    def test_v3_defaults_untouched(self):
        cfg = self.v3.config.DetectorConfig()
        self.assertEqual((cfg.spreads.quantile, cfg.spreads.multiplier, cfg.spreads.min_excess_ticks, cfg.spreads.window),
                         (0.95, 3.0, 3.0, "15min"))
        self.assertEqual((cfg.missing.alpha, cfg.missing.min_gap_s, cfg.missing.max_gap_s), (1e-6, 5.0, 60.0))


if __name__ == "__main__":
    unittest.main()
