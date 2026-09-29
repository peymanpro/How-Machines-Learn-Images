import numpy as np
import pytest

from src.learning.losses import (
    classification_accuracy,
    cross_entropy_from_logits,
    cross_entropy_gradient_from_logits,
    mean_squared_error,
    mean_squared_error_gradient,
)


def test_mean_squared_error() -> None:
    predictions = np.array([[1.0, 3.0]])
    targets = np.array([[1.0, 1.0]])

    assert mean_squared_error(predictions, targets) == pytest.approx(2.0)


def test_mean_squared_error_gradient() -> None:
    predictions = np.array([[1.0, 3.0]])
    targets = np.array([[1.0, 1.0]])

    assert np.allclose(
        mean_squared_error_gradient(predictions, targets),
        np.array([[0.0, 2.0]]),
    )


def test_cross_entropy_known_case() -> None:
    logits = np.array([[2.0, 0.0]])
    labels = np.array([0], dtype=np.int64)

    assert cross_entropy_from_logits(logits, labels) == pytest.approx(
        np.log(1.0 + np.exp(-2.0))
    )


def test_cross_entropy_gradient_sums_to_zero() -> None:
    logits = np.array([[2.0, 0.0, -1.0]])
    labels = np.array([1], dtype=np.int64)

    gradient = cross_entropy_gradient_from_logits(logits, labels)

    assert np.sum(gradient) == pytest.approx(0.0)


def test_accuracy() -> None:
    logits = np.array([[3.0, 1.0], [0.0, 4.0]])
    labels = np.array([0, 1])

    assert classification_accuracy(logits, labels) == 1.0
