from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.learning.layers import Layer, Parameter
from src.learning.losses import (
    classification_accuracy,
    cross_entropy_from_logits,
    cross_entropy_gradient_from_logits,
)


@dataclass
class History:
    """Training metrics collected after each epoch."""

    loss: list[float]
    accuracy: list[float]


class Sequential:
    """A small sequential neural network with explicit forward/backward passes."""

    def __init__(self, layers: list[Layer]) -> None:
        if not layers:
            raise ValueError("Sequential model requires at least one layer.")
        self.layers = layers

    def forward(self, values: np.ndarray) -> np.ndarray:
        result = values
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backward(self, gradient: np.ndarray) -> None:
        result = gradient
        for layer in reversed(self.layers):
            result = layer.backward(result)

    def parameters(self) -> tuple[Parameter, ...]:
        return tuple(parameter for layer in self.layers for parameter in layer.parameters())

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def evaluate(self, values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        """Return cross-entropy loss and accuracy without changing parameters."""
        logits = self.forward(values)
        return (
            cross_entropy_from_logits(logits, labels),
            classification_accuracy(logits, labels),
        )

    def forward_trace(self, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return the output tensor produced by every layer in order."""
        result = values
        trace: list[np.ndarray] = []
        for layer in self.layers:
            result = layer.forward(result)
            trace.append(result)
        return tuple(trace)

    def predict(self, values: np.ndarray) -> np.ndarray:
        logits = self.forward(values)
        return np.argmax(logits, axis=1)

    def train_batch(self, values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        self.zero_grad()
        logits = self.forward(values)
        loss = cross_entropy_from_logits(logits, labels)
        gradient = cross_entropy_gradient_from_logits(logits, labels)
        self.backward(gradient)
        accuracy = classification_accuracy(logits, labels)
        return loss, accuracy
