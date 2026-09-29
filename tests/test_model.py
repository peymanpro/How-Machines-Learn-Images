import numpy as np

from src.learning.layers import Dense
from src.learning.model import Sequential


def test_sequential_forwards_through_layers() -> None:
    model = Sequential([Dense(2, 3, seed=1), Dense(3, 2, seed=2)])
    result = model.forward(np.ones((4, 2)))

    assert result.shape == (4, 2)


def test_sequential_exposes_trainable_parameters() -> None:
    model = Sequential([Dense(2, 3, seed=1), Dense(3, 2, seed=2)])

    assert len(model.parameters()) == 4
