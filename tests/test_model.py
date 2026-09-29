import numpy as np

from src.learning.layers import Dense, ReLU
from src.learning.model import Sequential


def test_evaluate_returns_loss_and_accuracy() -> None:
    model = Sequential([Dense(2, 2, seed=1)])
    inputs = np.eye(2)
    labels = np.array([0, 1], dtype=np.int64)

    loss, accuracy = model.evaluate(inputs, labels)

    assert loss > 0.0
    assert 0.0 <= accuracy <= 1.0


def test_forward_trace_contains_each_layer_output() -> None:
    model = Sequential([Dense(2, 3, seed=1), ReLU(), Dense(3, 2, seed=2)])

    trace = model.forward_trace(np.ones((2, 2)))

    assert len(trace) == 3
    assert trace[0].shape == (2, 3)
    assert trace[1].shape == (2, 3)
    assert trace[2].shape == (2, 2)
