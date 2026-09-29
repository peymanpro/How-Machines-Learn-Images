import numpy as np

from src.learning.inspection import activation_statistics, layer_summary, parameter_statistics
from src.learning.layers import Dense, ReLU
from src.learning.model import Sequential


def test_activation_statistics() -> None:
    stats = activation_statistics(np.array([[-1.0, 0.0, 1.0]]))

    assert stats["min"] == -1.0
    assert stats["max"] == 1.0
    assert stats["mean"] == 0.0


def test_parameter_statistics() -> None:
    model = Sequential([Dense(2, 3, seed=1)])

    stats = parameter_statistics(model)

    assert stats["count"] == 9.0
    assert stats["l2"] > 0.0


def test_layer_summary() -> None:
    model = Sequential([Dense(2, 3, seed=1), ReLU()])

    assert layer_summary(model) == ["Dense", "ReLU"]
