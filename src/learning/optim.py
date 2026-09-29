from __future__ import annotations

import numpy as np

from src.learning.layers import Parameter
from src.learning.model import History, Sequential


class SGD:
    """Vanilla stochastic gradient descent."""

    def __init__(
        self,
        parameters: tuple[Parameter, ...],
        learning_rate: float = 0.01,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("Learning rate must be positive.")
        if not parameters:
            raise ValueError("SGD requires at least one trainable parameter.")
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self) -> None:
        for parameter in self.parameters:
            parameter.value[...] -= self.learning_rate * parameter.gradient


def train(
    model: Sequential,
    inputs: np.ndarray,
    labels: np.ndarray,
    optimizer: SGD,
    epochs: int,
) -> History:
    """Train a classification model on a full in-memory dataset."""
    if epochs <= 0:
        raise ValueError("Epochs must be positive.")
    if inputs.shape[0] != labels.shape[0]:
        raise ValueError("Inputs and labels must have matching batch sizes.")

    history = History(loss=[], accuracy=[])
    for _ in range(epochs):
        loss, accuracy = model.train_batch(inputs, labels)
        optimizer.step()
        history.loss.append(loss)
        history.accuracy.append(accuracy)

    return history
