# test_chi2_distance.py
import math
import unittest

import numpy as np

# Adjust import if your module name/path differs
from automated_lbp_benchmarking.distance_metrics import (
    ChiSquareDistance,
    CosineDistance,
    HellingerDistance,
    get_distance_metric,
    chi2_distance,
)


class TestHistogramDistanceMetrics(unittest.TestCase):
    def setUp(self):
        self.chi2 = ChiSquareDistance()
        self.cos = CosineDistance()
        self.hel = HellingerDistance()

        # A few handy histograms (non-negative)
        self.a = np.array([0.0, 1.0, 2.0, 3.0])
        self.b = np.array([0.0, 2.0, 1.0, 4.0])
        self.c = np.array([5.0, 0.0, 0.0, 0.0])  # sparse / one-hot-ish
        self.zeros = np.zeros(4, dtype=float)

    # ---------- Shared invariants / input validation ----------

    def test_identical_vectors_distance_zero(self):
        for metric in (self.chi2, self.cos, self.hel):
            self.assertAlmostEqual(metric(self.a, self.a), 0.0, places=12)

    def test_symmetry(self):
        # D(a,b) == D(b,a) for these implementations
        for metric in (self.chi2, self.cos, self.hel):
            dab = metric(self.a, self.b)
            dba = metric(self.b, self.a)
            self.assertAlmostEqual(dab, dba, places=12)

    def test_shape_mismatch_raises(self):
        a = [0, 1, 2]
        b = [0, 1, 2, 3]
        for metric in (self.chi2, self.cos, self.hel):
            with self.assertRaises(ValueError):
                metric(a, b)

    def test_negative_values_raises(self):
        a = [0, 1, -2, 3]
        b = [0, 1, 2, 3]
        for metric in (self.chi2, self.cos, self.hel):
            with self.assertRaises(ValueError):
                metric(a, b)

    def test_outputs_are_finite(self):
        # Should never return NaN/Inf for valid inputs
        for metric in (self.chi2, self.cos, self.hel):
            val = metric(self.a, self.b)
            self.assertTrue(math.isfinite(val))

    # ---------- Chi-square distance specifics ----------

    def test_chi2_nonnegative(self):
        self.assertGreaterEqual(self.chi2(self.a, self.b), 0.0)
        self.assertGreaterEqual(self.chi2(self.a, self.c), 0.0)

    def test_chi2_matches_manual_computation(self):
        # Matches: 0.5 * sum((a-b)^2 / (a+b+epsilon))
        a = np.array([0.0, 1.0, 2.0, 3.0])
        b = np.array([0.0, 2.0, 1.0, 4.0])
        epsilon = self.chi2.epsilon

        manual = 0.5 * np.sum(((a - b) ** 2) / (a + b + epsilon))
        self.assertAlmostEqual(self.chi2(a, b), float(manual), places=12)

    def test_chi2_wrapper_matches_class(self):
        # chi2_distance() wraps ChiSquareDistance(epsilon=epsilon)(a,b)
        a = [0, 1, 2, 3]
        b = [0, 2, 1, 4]
        self.assertAlmostEqual(chi2_distance(a, b), self.chi2(a, b), places=12)

        # also verify custom epsilon changes are respected
        epsilon = 1e-6
        self.assertAlmostEqual(
            chi2_distance(a, b, epsilon=epsilon),
            ChiSquareDistance(epsilon=epsilon)(a, b),
            places=12,
        )

    # ---------- Cosine distance specifics ----------

    def test_cosine_range_for_nonnegative_vectors(self):
        # For non-negative vectors, cosine similarity is in [0,1], so distance in [0,1]
        for x, y in ((self.a, self.b), (self.a, self.c), (self.c, self.b)):
            d = self.cos(x, y)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_cosine_all_zero_both_returns_zero(self):
        # If both norms are ~0 => return 0.0
        self.assertEqual(self.cos(self.zeros, self.zeros), 0.0)

    def test_cosine_one_zero_other_nonzero_returns_one(self):
        # If one norm is ~0 and the other isn't => return 1.0
        self.assertEqual(self.cos(self.zeros, self.a), 1.0)
        self.assertEqual(self.cos(self.a, self.zeros), 1.0)

    def test_cosine_scaling_invariance(self):
        # Scaling either vector by positive constant does not change cosine distance
        d1 = self.cos(self.a, self.b)
        d2 = self.cos(10.0 * self.a, self.b)
        d3 = self.cos(self.a, 0.1 * self.b)
        self.assertAlmostEqual(d1, d2, places=12)
        self.assertAlmostEqual(d1, d3, places=12)

    # ---------- Hellinger distance specifics ----------

    def test_hellinger_range(self):
        # Hellinger distance is bounded in [0,1] in this normalized form
        for x, y in ((self.a, self.b), (self.a, self.c), (self.c, self.b)):
            d = self.hel(x, y)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_hellinger_both_zero_sums_returns_zero(self):
        # If both sums are ~0 => return 0.0
        self.assertEqual(self.hel(self.zeros, self.zeros), 0.0)

    def test_hellinger_one_zero_sum_other_nonzero_returns_one(self):
        # If one sum is ~0 and the other isn't => return 1.0
        self.assertEqual(self.hel(self.zeros, self.a), 1.0)
        self.assertEqual(self.hel(self.a, self.zeros), 1.0)

    def test_hellinger_scaling_invariance_due_to_normalization(self):
        # Hellinger normalizes to distributions p=a/sum(a), q=b/sum(b),
        # so positive scaling should not change the result.
        d1 = self.hel(self.a, self.b)
        d2 = self.hel(10.0 * self.a, self.b)
        d3 = self.hel(self.a, 0.1 * self.b)
        self.assertAlmostEqual(d1, d2, places=12)
        self.assertAlmostEqual(d1, d3, places=12)

    # ---------- Registry / factory behavior ----------

    def test_get_distance_metric_known_names(self):
        # Exact names
        self.assertEqual(get_distance_metric("chi2").name, "chi2")
        self.assertEqual(get_distance_metric("cosine").name, "cosine")
        self.assertEqual(get_distance_metric("hellinger").name, "hellinger")

    def test_get_distance_metric_aliases(self):
        # Aliases for chi2
        self.assertEqual(get_distance_metric("chisq").name, "chi2")
        self.assertEqual(get_distance_metric("chi-square").name, "chi2")

    def test_get_distance_metric_trims_and_lowercases(self):
        self.assertEqual(get_distance_metric("  CHI2  ").name, "chi2")

    def test_get_distance_metric_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_distance_metric("euclidean")


if __name__ == "__main__":
    unittest.main()