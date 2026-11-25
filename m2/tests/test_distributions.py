"""Unit tests for Milestone 2 distribution fits, on synthetic draws with known parameters."""
from __future__ import annotations

import unittest

import numpy as np

from m2.distributions import (
    CURRENT_ALPHA,
    best_fit,
    empirical_survival,
    fit_gap_models,
    fit_return_models,
    fit_spread_models,
    hill_estimator,
    k12_false_alarm,
    lr_statistic,
    poisson_gap_threshold,
    spread_rule_threshold,
    subsample,
)


class GapFitTests(unittest.TestCase):
    def test_exponential_mle_recovers_scale(self):
        rng = np.random.default_rng(7)
        scale = 0.25  # mean gap 0.25 s, lambda=4
        x = rng.exponential(scale, size=20_000)
        fits = {r.model: r for r in fit_gap_models(x, seed=7)}
        self.assertEqual(fits["expon"].status, "ok")
        self.assertAlmostEqual(fits["expon"].params["scale"], scale, delta=0.02)
        self.assertIs(best_fit(list(fits.values())).model, fits[best_fit(list(fits.values())).model].model)
        # Exponential should be among the best (AIC within 15 of the winner)
        winner = best_fit(list(fits.values()))
        self.assertLess(fits["expon"].aic - winner.aic, 15)

    def test_lr_rejects_exponential_when_data_are_gamma(self):
        rng = np.random.default_rng(11)
        x = rng.gamma(shape=3.0, scale=0.1, size=15_000)
        fits = {r.model: r for r in fit_gap_models(x, seed=11)}
        self.assertEqual(fits["gamma"].status, "ok")
        self.assertEqual(fits["expon"].status, "ok")
        lr = lr_statistic(fits["gamma"], fits["expon"])
        self.assertGreater(lr["lr"], 20)
        self.assertLess(lr["pvalue"], 1e-6)
        self.assertEqual(best_fit(list(fits.values())).model, "gamma")

    def test_poisson_threshold_formula(self):
        lam = 2.0
        raw = -np.log(CURRENT_ALPHA) / lam
        self.assertTrue(5.0 < raw < 60.0)
        self.assertAlmostEqual(poisson_gap_threshold(lam), raw)
        self.assertEqual(poisson_gap_threshold(1e-4), 60.0)  # clamps at 60 s
        self.assertEqual(poisson_gap_threshold(100.0), 5.0)  # clamps at 5 s

    def test_empirical_survival_on_exponential(self):
        rng = np.random.default_rng(3)
        x = rng.exponential(1.0, size=50_000)
        # P(X>3) = exp(-3) ≈ 0.0498
        self.assertAlmostEqual(empirical_survival(x, 3.0), np.exp(-3.0), delta=0.01)


class SpreadFitTests(unittest.TestCase):
    def test_shifted_geometric_recovers_p(self):
        rng = np.random.default_rng(5)
        p = 0.6
        s = rng.geometric(p, size=15_000).astype(float)  # support 1,2,...
        fits = {r.model: r for r in fit_spread_models(s, seed=5)}
        self.assertEqual(fits["geometric_shifted"].status, "ok")
        self.assertAlmostEqual(fits["geometric_shifted"].params["p"], p, delta=0.03)

    def test_spread_rule_threshold(self):
        self.assertEqual(spread_rule_threshold(1.0), 4.0)  # max(3*1, 1+3)
        self.assertEqual(spread_rule_threshold(2.0), 6.0)  # max(6, 5)
        self.assertEqual(spread_rule_threshold(0.5), 3.5)  # max(1.5, 3.5)


class ReturnFitTests(unittest.TestCase):
    def test_normal_preferred_on_gaussian_draws(self):
        rng = np.random.default_rng(9)
        x = rng.normal(0.0, 0.01, size=20_000)
        fits = {r.model: r for r in fit_return_models(x, seed=9)}
        self.assertEqual(fits["norm"].status, "ok")
        self.assertAlmostEqual(fits["norm"].params["scale"], 0.01, delta=0.002)
        # t should estimate a large df (lighter than a fixed small-df t)
        if fits["t"].status == "ok":
            self.assertGreater(fits["t"].params["df"], 8)
        winner = best_fit(list(fits.values()))
        self.assertIn(winner.model, ("norm", "t"))  # large-df t can tie
        diag = k12_false_alarm(x, list(fits.values()), k=12, eps=1e-8)
        # Under a Gaussian, P(|z|>12) is essentially 0
        self.assertLess(diag["empirical_p_abs_z_gt_k"], 1e-4)
        if np.isfinite(diag.get("norm_p_abs_z_gt_k", np.nan)):
            self.assertLess(diag["norm_p_abs_z_gt_k"], 1e-10)

    def test_student_t_recovers_heavy_tails(self):
        rng = np.random.default_rng(13)
        x = stats_t_draw(rng, df=3.0, loc=0.0, scale=0.002, size=20_000)
        fits = {r.model: r for r in fit_return_models(x, seed=13)}
        self.assertEqual(fits["t"].status, "ok")
        self.assertLess(fits["t"].params["df"], 8.0)
        self.assertGreater(fits["t"].params["df"], 1.2)
        self.assertLess(fits["t"].aic, fits["norm"].aic)

    def test_hill_positive_on_lomax(self):
        rng = np.random.default_rng(2)
        # Lomax with shape c=3, scale=1; tail index is c
        x = rng.pareto(3.0, size=30_000)  # Pareto I; Hill still recovers a finite alpha
        hill = hill_estimator(x, k=800)
        self.assertGreater(hill["alpha"], 1.0)
        self.assertLess(hill["alpha"], 6.0)


class HelperTests(unittest.TestCase):
    def test_subsample_is_reproducible_and_capped(self):
        x = np.arange(10_000)
        a = subsample(x, max_n=200, seed=7)
        b = subsample(x, max_n=200, seed=7)
        c = subsample(x, max_n=200, seed=8)
        self.assertEqual(len(a), 200)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        y = np.arange(50)
        self.assertEqual(len(subsample(y, max_n=200, seed=7)), 50)


def stats_t_draw(rng, df, loc, scale, size):
    return loc + scale * rng.standard_t(df, size=size)



class ExtractBundledTests(unittest.TestCase):
    def test_bundled_quotes_yield_positive_in_session_gaps(self):
        from m2.extract import extract_series, load_bundled_sample
        frame = load_bundled_sample("quotes")
        bundles = extract_series(frame, venue="XNAS", symbol="AAPL", day="2025-09-15", tick=0.01)
        self.assertEqual(len(bundles), 1)
        b = bundles[0]
        self.assertEqual(b.regime, "equity_rth")
        self.assertGreater(b.n_flast, 100)
        self.assertTrue((b.gaps_s > 0).all())
        self.assertLess(float(np.median(b.gaps_s)), 1.0)  # liquid name, sub-second updates
        self.assertTrue((b.spreads_ticks > 0).all())
        self.assertGreater(len(b.mid_returns), 50)


if __name__ == "__main__":
    unittest.main()
