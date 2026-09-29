import numpy as np
import pytest

from src.learning.activations import (
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
    tanh,
    tanh_derivative,
)


def test_relu_clips_negative_values() -> None:
    assert np.array_equal(relu(np.array([-2.0, 0.0, 3.0])), np.array([0.0, 0.0, 3.0]))


def test_relu_derivative() -> None:
    assert np.array_equal(
        relu_derivative(np.array([-1.0, 0.0, 2.0])),
        np.array([0.0, 0.0, 1.0]),
    )


def test_sigmoid_is_bounded() -> None:
    result = sigmoid(np.array([-20.0, 0.0, 20.0]))
    assert np.all((result > 0.0) & (result < 1.0))


def test_sigmoid_derivative_is_maximal_at_zero() -> None:
    assert sigmoid_derivative(np.array([0.0]))[0] == pytest.approx(0.25)


def test_tanh_derivative_at_zero() -> None:
    assert tanh_derivative(np.array([0.0]))[0] == pytest.approx(1.0)


def test_tanh_matches_numpy() -> None:
    values = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(tanh(values), np.tanh(values))


def test_softmax_rows_sum_to_one() -> None:
    probabilities = softmax(np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]]))
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)


def test_softmax_is_shift_invariant() -> None:
    logits = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(softmax(logits), softmax(logits + 100.0))


def test_softmax_rejects_non_batched_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        softmax(np.array([1.0, 2.0]))
