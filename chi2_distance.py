from typing import Sequence
import numpy as np


def chi2_distance(a: Sequence[float], b: Sequence[float], eps: float = 1e-10) -> float:
    """Compute the Chi-squared distance between two non-negative vectors.

    Formula:
        D = 0.5 * sum( (a_i - b_i)^2 / (a_i + b_i + eps) )

    Parameters:
    - a, b: 1D sequences (lists or numpy arrays) of the same length. Typically
      these are histogram vectors derived from LBP codes.
    - eps: small constant to avoid division by zero (default 1e-10)

    Returns:
    - float distance
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Input vectors must have the same shape: {a_arr.shape} != {b_arr.shape}")

    if np.any(a_arr < 0) or np.any(b_arr < 0):
        raise ValueError("Chi-squared distance expects non-negative vectors (e.g., histograms)")

    num = (a_arr - b_arr) ** 2
    den = a_arr + b_arr + eps
    return 0.5 * np.sum(num / den)


if __name__ == '__main__':
    # Quick smoke test
    v1 = [0, 1, 2, 3]
    v2 = [0, 2, 1, 4]
    print("chi2(v1,v2)=", chi2_distance(v1, v2))
