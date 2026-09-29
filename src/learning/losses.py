from __future__ import annotations

from typing import cast

import numpy as np

from src.learning.activations import softmax


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the mean squared error across all elements."""
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    return float(np.mean((predictions - targets) ** 2))


def mean_squared_error_gradient(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Return the gradient of mean squared error with respect to predictions."""
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    return cast(np.ndarray, 2.0 * (predictions - targets) / predictions.size)


def cross_entropy_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Return mean multiclass cross-entropy from unnormalized logits."""
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D array.")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("Labels must be a 1D array matching the batch size.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Class labels must be integers.")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("Class labels are outside the logit class range.")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1)) + np.max(
        logits,
        axis=1,
    )
    selected = logits[np.arange(logits.shape[0]), labels]
    return float(np.mean(log_sum_exp - selected))


def cross_entropy_gradient_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return dL/dlogits for mean multiclass cross-entropy."""
    probabilities = softmax(logits)
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("Labels must be a 1D array matching the batch size.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Class labels must be integers.")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("Class labels are outside the logit class range.")

    gradient = probabilities.copy()
    gradient[np.arange(logits.shape[0]), labels] -= 1.0
    return cast(np.ndarray, gradient / logits.shape[0])


def classification_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Return fraction of samples whose argmax class is correct."""
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D array.")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("Labels must be a 1D array matching the batch size.")
    return float(np.mean(np.argmax(logits, axis=1) == labels))
