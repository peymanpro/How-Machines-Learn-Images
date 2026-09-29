import numpy as np

from src.learning.inspection import trace_layer_outputs
from src.learning.layers import Dense, ReLU
from src.learning.model import Sequential


def test_trace_layer_outputs_reports_statistics() -> None:
    model = Sequential([Dense(2, 2, seed=1), ReLU()])
    result = trace_layer_outputs(model, np.ones((1, 2)))

    assert [name for name, _ in result] == ["Dense", "ReLU"]
    assert result[0][1]["std"] >= 0.0
