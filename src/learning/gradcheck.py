from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numerical_gradient(
    function: Callable[[np.ndarray], float],
    values: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Estimate a scalar function gradient with central finite differences."""
    if epsilon <= 0.0:
        raise ValueError("Epsilon must be positive.")

    result = np.zeros_like(values, dtype=np.float64)
    iterator = np.nditer(values, flags=["multi_index"], op_flags=["readwrite"])

    while not iterator.finished:
        index = iterator.multi_index
        original = float(values[index])

        values[index] = original + epsilon
        positive = function(values)

        values[index] = original - epsilon
        negative = function(values)

        values[index] = original
        result[index] = (positive - negative) / (2.0 * epsilon)
        iterator.iternext()

    return result


def relative_error(first: np.ndarray, second: np.ndarray) -> float:
    """Return max element-wise relative error."""
    denominator = np.maximum(1e-12, np.abs(first) + np.abs(second))
    return float(np.max(np.abs(first - second) / denominator))
