import numpy as np
import pytest

from src.learning.gradcheck import numerical_gradient, relative_error


def test_numerical_gradient_matches_quadratic() -> None:
    values = np.array([1.0, -2.0, 3.0])

    def function(current: np.ndarray) -> float:
        return float(np.sum(current * current))

    gradient = numerical_gradient(function, values.copy())

    assert np.allclose(gradient, 2.0 * values, atol=1e-7)


def test_relative_error() -> None:
    assert relative_error(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


def test_numerical_gradient_rejects_invalid_epsilon() -> None:
    with pytest.raises(ValueError, match="positive"):
        numerical_gradient(lambda values: float(np.sum(values)), np.ones(2), epsilon=0.0)
