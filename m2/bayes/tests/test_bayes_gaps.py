"""Tests for the Milestone 2 Bayesian missing-interval check (m2/bayes/model.py)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from integration import stress
from integration.combine import load_v3
from m2.bayes.model import (
    BayesGapConfig,
    _discounted,
    bayes_missing_intervals,
    log_lomax_sf,
    outage_posterior,
    poisson_rule_from_clock,
    posterior_from_clock,
)

NS = 1_000_000_000


def clock_from_gaps(gaps):
    s = np.r_[0.0, np.cumsum(gaps)]
    return (s * NS).astype(np.int64), np.zeros(len(s), np.int64)


class ConjugateMathTests(unittest.TestCase):
    def test_discount_filter_matches_loop(self):
        rng = np.random.default_rng(3)
        x = rng.exponential(1.0, 1000)
        y = _discounted(x, 0.97, 5.0)
        ref, prev = [], 5.0
        for v in x:
            prev = 0.97 * prev + v
            ref.append(prev)
        np.testing.assert_allclose(y, ref, rtol=1e-12)

    def test_posterior_predictive_is_lomax(self):
        # G | lambda ~ Exp(lambda), lambda ~ Gamma(a, rate=b)  =>  P(G > g) = (b / (b + g))^a
        rng = np.random.default_rng(5)
        a, b = 4.0, 2.0
        lam = rng.gamma(a, 1.0 / b, 400_000)
        g = rng.exponential(1.0 / lam)
        for q in (0.5, 2.0, 6.0):
            emp = (g > q).mean()
            self.assertAlmostEqual(emp, np.exp(log_lomax_sf(q, a, b)), delta=0.004)

    def test_steady_shape_is_effective_window(self):
        s, k = clock_from_gaps(np.random.default_rng(1).exponential(0.1, 20_000))
        post = posterior_from_clock(s, k, BayesGapConfig(discount=0.998))
        self.assertAlmostEqual(post.a[-1], 500.0, delta=1.0)
        self.assertAlmostEqual(post.a[-1] / post.b[-1], 10.0, delta=1.0)

    def test_outage_posterior_zero_below_dmin_and_monotone(self):
        cfg = BayesGapConfig(min_outage_s=5.0)
        g = np.array([1.0, 4.99, 5.0, 6.0, 8.0, 12.0, 30.0])
        p = outage_posterior(g, 500.0, 500.0 / 1.5, cfg)  # rate 1.5/s
        self.assertEqual(p[0], 0.0)
        self.assertEqual(p[1], 0.0)
        self.assertTrue(np.all(np.diff(p[2:]) >= 0))
        self.assertGreater(p[-1], 0.99)


class SyntheticStreamTests(unittest.TestCase):
    def test_flags_injected_gaps_and_nothing_else_on_poisson_stream(self):
        rng = np.random.default_rng(11)
        g = rng.exponential(0.1, 200_000)
        g[[50_000, 120_000, 180_000]] = [6.0, 12.0, 25.0]
        s, k = clock_from_gaps(g)
        post = posterior_from_clock(s, k)
        self.assertEqual(list(np.flatnonzero(post.candidate)), [50_001, 120_001, 180_001])
        self.assertEqual(list(np.flatnonzero(poisson_rule_from_clock(s, k))), [50_001, 120_001, 180_001])

    def test_robust_update_keeps_rate_after_outage(self):
        rng = np.random.default_rng(2)
        g = rng.exponential(0.2, 5000); g[2000] = 300.0
        s, k = clock_from_gaps(g)
        robust = posterior_from_clock(s, k, BayesGapConfig(robust=True))
        plain = posterior_from_clock(s, k, BayesGapConfig(robust=False))
        i = 2002
        self.assertAlmostEqual(robust.a[i] / robust.b[i], 5.0, delta=0.8)
        self.assertLess(plain.a[i] / plain.b[i], 2.0)

    def test_keys_are_independent(self):
        rng = np.random.default_rng(4)
        s1, _ = clock_from_gaps(rng.exponential(0.1, 3000))
        s2, _ = clock_from_gaps(rng.exponential(2.0, 3000))
        s = np.r_[s1, s2]; key = np.r_[np.zeros(len(s1), int), np.ones(len(s2), int)]
        post = posterior_from_clock(s, key)
        self.assertFalse(post.valid[len(s1)])
        self.assertAlmostEqual(post.a[len(s1) - 1] / post.b[len(s1) - 1], 10.0, delta=1.5)
        self.assertAlmostEqual(post.a[-1] / post.b[-1], 0.5, delta=0.1)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v3 = load_v3()

    def test_v3_missing_defaults_untouched(self):
        cfg = self.v3.config.MissingIntervalsConfig()
        self.assertEqual((cfg.model, cfg.alpha, cfg.min_gap_s, cfg.max_gap_s, cfg.rate_window, cfg.adaptive),
                         ("poisson", 1e-6, 5.0, 60.0, 500, True))

    def test_poisson_reimplementation_matches_detector_on_synthetic_stream(self):
        v3 = self.v3
        raw = v3.synthetic.generate_mbp1(duration_s=1800, seed=7)
        faulted, _ = v3.synthetic.inject_faults(raw, n_per_type=3, seed=11)
        df = v3.synthetic.to_databento_df(faulted)
        cfg = stress.v3_eval_config(v3).missing
        det = v3.detectors
        norm = det.normalize(df).sort_values(det.KEY + ["ts_recv_ns"], kind="stable").reset_index(drop=True)
        t = norm["ts_recv_ns"].to_numpy()
        s = det.SessionClock.build(cfg.calendar, int(t.min()), int(t.max())).clock(t)
        mine = poisson_rule_from_clock(s, det._key_id(norm))
        self.assertEqual(int(mine.sum()), len(det.missing_intervals(df, cfg)))

    def test_stress_stream_outages_found_and_clean_stream_quiet(self):
        v3 = self.v3
        raw = v3.synthetic.generate_mbp1(duration_s=3600, seed=7)
        faulted, labels = v3.synthetic.inject_faults(raw, n_per_type=5, seed=11)
        cal = stress.v3_eval_config(v3).missing.calendar
        gaps = labels[labels["check"] == "missing_intervals"].reset_index(drop=True)
        flags = bayes_missing_intervals(v3.synthetic.to_databento_df(faulted), calendar=cal, v3=v3)
        hits = stress.match_matrix(flags, gaps)
        self.assertEqual(int(hits.any(axis=0).sum()), len(gaps))
        self.assertEqual(int((~hits.any(axis=1)).sum()), 0)
        clean = bayes_missing_intervals(v3.synthetic.to_databento_df(raw), calendar=cal, v3=v3)
        self.assertEqual(len(clean), 0)
        self.assertEqual(set(flags["subtype"]), {"gap_bayes"})
        self.assertTrue((flags["threshold"] >= 0.5).all())  # threshold column holds P(outage | gap)

    def test_bundled_pipeline_sample_runs(self):
        from m2.extract import load_bundled_sample
        frame = load_bundled_sample("quotes")
        cal = self.v3.config.equities_config().missing.calendar
        flags = bayes_missing_intervals(frame, calendar=cal, v3=self.v3)
        self.assertEqual(list(flags.columns), list(self.v3.detectors.FLAG_COLUMNS))
        self.assertTrue((flags["value"] >= BayesGapConfig().min_outage_s).all())


if __name__ == "__main__":
    unittest.main()
