from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Sequence

import numpy as np


class DistanceMetric(Protocol):
    """Strategy interface for histogram featue vector distance metrics."""

    name: str

    def __call__(
        self,
        first_vector: Sequence[float],
        second_vector: Sequence[float]
    ) -> float: ...


def _validate_and_convert_vectors(
    first_vector: Sequence[float],
    second_vector: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert inputs to float 64 numpy arrays and validate:
    """
    first_array = np.asarray(first_vector, dtype=np.float64)
    second_array = np.asarray(second_vector, dtype=np.float64)

    if first_array.shape != second_array.shape:
        raise ValueError(
            f"Input vectors must have the same shape: "
            f"{first_array.shape} != {second_array.shape}"
        )

    if np.any(first_array < 0) or np.any(second_array < 0):
        raise ValueError(
            "Distance expects non-negative vectors"
        )

    return first_array, second_array


@dataclass(frozen=True)
class ChiSquareDistance:
    """
    Chi-square distance metric
    """

    epsilon: float = 1e-10
    name: str = "chi2"

    def __call__(
        self,
        first_vector: Sequence[float],
        second_vector: Sequence[float]
    ) -> float:

        first_array, second_array = _validate_and_convert_vectors(
            first_vector,
            second_vector
        )

        squared_difference = (first_array - second_array) ** 2
        denominator = first_array + second_array + self.epsilon

        chi_square_value = 0.5 * np.sum(squared_difference / denominator)

        return float(chi_square_value)


@dataclass(frozen=True)
class CosineDistance:
    """
    Cosine distance: = 1 - cosine_similarity
    """

    epsilon: float = 1e-12
    name: str = "cosine"

    def __call__(
        self,
        first_vector: Sequence[float],
        second_vector: Sequence[float]
    ) -> float:

        first_array, second_array = _validate_and_convert_vectors(
            first_vector,
            second_vector
        )

        first_norm = np.linalg.norm(first_array)
        second_norm = np.linalg.norm(second_array)

        # Handle zeroed out histograms safely
        if first_norm < self.epsilon and second_norm < self.epsilon:
            return 0.0

        if first_norm < self.epsilon or second_norm < self.epsilon:
            return 1.0

        cosine_similarity = float(
            np.dot(first_array, second_array) /
            (first_norm * second_norm)
        )

        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        cosine_distance_value = 1.0 - cosine_similarity

        return cosine_distance_value


@dataclass(frozen=True)
class HellingerDistance:
    """
    Hellinger distance metric
    """

    epsilon: float = 1e-12
    name: str = "hellinger"

    def __call__(
        self,
        first_vector: Sequence[float],
        second_vector: Sequence[float]
    ) -> float:

        first_array, second_array = _validate_and_convert_vectors(
            first_vector,
            second_vector
        )

        first_sum = float(first_array.sum())
        second_sum = float(second_array.sum())

        if first_sum < self.epsilon and second_sum < self.epsilon:
            return 0.0

        if first_sum < self.epsilon or second_sum < self.epsilon:
            return 1.0

        first_distribution = first_array / first_sum
        second_distribution = second_array / second_sum

        difference_vector = (
            np.sqrt(first_distribution) -
            np.sqrt(second_distribution)
        )

        hellinger_value = np.sqrt(
            0.5 * np.sum(difference_vector * difference_vector)
        )

        return float(hellinger_value)


# Strategy table
_DISTANCE_METRICS: Dict[str, DistanceMetric] = {
    "chi2": ChiSquareDistance(),
    "chisq": ChiSquareDistance(),
    "chi-square": ChiSquareDistance(),
    "cosine": CosineDistance(),
    "hellinger": HellingerDistance(),
}


def get_distance_metric(metric_name: str) -> DistanceMetric:
    """
    Return a distance-metric strategy by name.
    """

    normalized_key = (metric_name or "").strip().lower()

    if normalized_key not in _DISTANCE_METRICS:
        raise ValueError(
            f"Unknown distance metric '{metric_name}'. "
            f"Available: {sorted(set(_DISTANCE_METRICS.keys()))}"
        )

    return _DISTANCE_METRICS[normalized_key]


def chi2_distance(
    first_vector: Sequence[float],
    second_vector: Sequence[float],
    epsilon: float = 1e-10
) -> float:
    return ChiSquareDistance(epsilon=epsilon)(
        first_vector,
        second_vector
    )