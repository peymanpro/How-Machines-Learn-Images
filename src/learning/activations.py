from __future__ import annotations

import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    """Return max(0, x) element-wise."""
    return np.maximum(values, 0.0)


def relu_derivative(values: np.ndarray) -> np.ndarray:
    """Derivative of ReLU, using zero at the non-positive branch."""
    return (values > 0.0).astype(np.float64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Return the logistic sigmoid with numerically stable branches."""
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_values = values[~positive]
    exp_values = np.exp(negative_values)
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def sigmoid_derivative(values: np.ndarray) -> np.ndarray:
    """Return the sigmoid derivative evaluated from pre-activation values."""
    activated = sigmoid(values)
    return activated * (1.0 - activated)


def tanh(values: np.ndarray) -> np.ndarray:
    """Return hyperbolic tangent values."""
    return np.tanh(values)


def tanh_derivative(values: np.ndarray) -> np.ndarray:
    """Return the derivative of tanh."""
    activated = np.tanh(values)
    return 1.0 - activated * activated


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits to row-wise probabilities."""
    if logits.ndim != 2:
        raise ValueError("Softmax expects a 2D array of batched logits.")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)
