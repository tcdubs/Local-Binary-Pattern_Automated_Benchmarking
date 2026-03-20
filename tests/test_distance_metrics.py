
import math
import unittest

import numpy as np

from automated_lbp_benchmarking.distance_metrics import (
    ChiSquareDistance,
    CosineDistance,
    HellingerDistance,
    get_distance_metric,
    chi2_distance,
)
from automated_lbp_benchmarking.main import center_crop, ImageRecord, NearestNeighborMatcher
from PIL import Image


class TestHistogramDistanceMetrics(unittest.TestCase):
    def setUp(self):
        self.chi2 = ChiSquareDistance()
        self.cos = CosineDistance()
        self.hel = HellingerDistance()

        self.a = np.array([0.0, 1.0, 2.0, 3.0])
        self.b = np.array([0.0, 2.0, 1.0, 4.0])
        self.c = np.array([5.0, 0.0, 0.0, 0.0])  
        self.zeros = np.zeros(4, dtype=float)

    def test_identical_vectors_distance_zero(self):
        for metric in (self.chi2, self.cos, self.hel):
            self.assertAlmostEqual(metric(self.a, self.a), 0.0, places=12)

    def test_symmetry(self):
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
        for metric in (self.chi2, self.cos, self.hel):
            val = metric(self.a, self.b)
            self.assertTrue(math.isfinite(val))

    # ---------- Chi-square distance specifics ----------

    def test_chi2_nonnegative(self):
        self.assertGreaterEqual(self.chi2(self.a, self.b), 0.0)
        self.assertGreaterEqual(self.chi2(self.a, self.c), 0.0)

    def test_chi2_matches_manual_computation(self):
        # Match manual chi2 computation for known inputs with chi2 stragegy object
        a = np.array([0.1, 0.2, 0.3, 0.4])
        b = np.array([0.2, 0.4, 0.1, 0.3])
        epsilon = self.chi2.epsilon

        manual = 0.5 * np.sum(((a - b) ** 2) / (a + b + epsilon))
        self.assertAlmostEqual(self.chi2(a, b), float(manual))

    def test_chi2_wrapper_matches_class(self):

        a = [0, 1, 2, 3]
        b = [0, 2, 1, 4]
        self.assertAlmostEqual(chi2_distance(a, b), self.chi2(a, b))

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
        # If both norms are about 0 => return 0.0
        self.assertEqual(self.cos(self.zeros, self.zeros), 0.0)

    def test_cosine_one_zero_other_nonzero_returns_one(self):
        # If one norm is ~0 and the other isn't => return 1.0
        self.assertEqual(self.cos(self.zeros, self.a), 1.0)
        self.assertEqual(self.cos(self.a, self.zeros), 1.0)

    def test_cosine_scaling_invariance(self):
        # Scaling either vector by positive constant doesnt change cosine distance
        d1 = self.cos(self.a, self.b)
        d2 = self.cos(10.0 * self.a, self.b)
        d3 = self.cos(self.a, 0.1 * self.b)
        self.assertAlmostEqual(d1, d2, places=12)
        self.assertAlmostEqual(d1, d3, places=12)

    # ---------- Hellinger distance specifics ----------

    def test_hellinger_range(self):
        for x, y in ((self.a, self.b), (self.a, self.c), (self.c, self.b)):
            d = self.hel(x, y)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_hellinger_both_zero_sums_returns_zero(self):
        self.assertEqual(self.hel(self.zeros, self.zeros), 0.0)

    def test_hellinger_one_zero_sum_other_nonzero_returns_one(self):
        self.assertEqual(self.hel(self.zeros, self.a), 1.0)
        self.assertEqual(self.hel(self.a, self.zeros), 1.0)

    def test_hellinger_scaling_invariance_due_to_normalization(self):
        d1 = self.hel(self.a, self.b)
        d2 = self.hel(10.0 * self.a, self.b)
        d3 = self.hel(self.a, 0.1 * self.b)
        self.assertAlmostEqual(d1, d2, places=12)
        self.assertAlmostEqual(d1, d3, places=12)

    # ---------- Registry / factory behavior ----------

    def test_get_distance_metric_known_names(self):
        self.assertEqual(get_distance_metric("chi2").name, "chi2")
        self.assertEqual(get_distance_metric("cosine").name, "cosine")
        self.assertEqual(get_distance_metric("hellinger").name, "hellinger")

    def test_get_distance_metric_aliases(self):
        self.assertEqual(get_distance_metric("chisq").name, "chi2")
        self.assertEqual(get_distance_metric("chi-square").name, "chi2")

    def test_get_distance_metric_trims_and_lowercases(self):
        self.assertEqual(get_distance_metric("  CHI2  ").name, "chi2")

    def test_get_distance_metric_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_distance_metric("euclidean")

class TestImageCropper(unittest.TestCase):
    def test_center_crop_extracts_exact_center_region(self):
        # Mock image known dimesnions
        original_image_height = 10
        original_image_width = 12

        # Mock center patch values
        requested_crop_width = 2
        requested_crop_height = 2

        # Get 10 x 12 array with values 0-119 to verify proper cropping
        input_array = np.arange(
            original_image_height * original_image_width
        ).reshape(original_image_height, original_image_width)

        # Compute expected center region indices
        expected_row_start_index = (original_image_height - requested_crop_height) // 2
        expected_column_start_index = (original_image_width - requested_crop_width) // 2

        # Manually slice out middle 2x2 region for verification
        expected_center_region = input_array[
            expected_row_start_index:expected_row_start_index + requested_crop_height,
            expected_column_start_index:expected_column_start_index + requested_crop_width
        ]

        # Call the center_crop function
        cropped_array = center_crop(
            input_array,
            requested_crop_width,
            requested_crop_height
        )
        # Verify the cropped array matches the expected center region
        assert np.array_equal(cropped_array, expected_center_region)

    def test_distance_metrics_and_matcher_integration(self):
    # Create three known histograms
    # Record A is closest to Record B under chi-square distance.
        hist_A = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        hist_B = np.array([0.1, 0.2, 0.34, 0.36], dtype=np.float64)  # very close to A
        hist_C = np.array([0.8, 0.1, 0.05, 0.05], dtype=np.float64)  # far from A

        # Dummy image to pack into ImageRecord object
        dummy_image = Image.new("RGB", (8, 8))
        record_A = ImageRecord(
            instance="0001",
            category="catA",
            distance="10cm",
            rotation="0deg",
            lighting="bright",
            image=dummy_image,
            lbp_hist=hist_A,
        )
        record_B = ImageRecord(
            instance="0002",
            category="catA",  # same category as A so "correct" should become True when matched
            distance="10cm",
            rotation="0deg",
            lighting="bright",
            image=dummy_image,
            lbp_hist=hist_B,
        )
        record_C = ImageRecord(
            instance="0003",
            category="catB",
            distance="10cm",
            rotation="0deg",
            lighting="bright",
            image=dummy_image,
            lbp_hist=hist_C,
        )

        records = [record_A, record_B, record_C]
        matcher = NearestNeighborMatcher(metric_name="chi2")
        matched_records = matcher(records)

        #for record_A, nearest neighbor should be record_B
        assert matched_records[0].matched_index == 1
        assert matched_records[0].matched_category == "catA"
        assert matched_records[0].correct is True

        # Distance should be a positive float
        assert isinstance(matched_records[0].nn_distance, float)
        assert np.isfinite(matched_records[0].nn_distance)
        assert matched_records[0].nn_distance >= 0.0


if __name__ == "__main__":
    unittest.main()