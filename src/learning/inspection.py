from __future__ import annotations

import numpy as np

from src.learning.layers import Layer
from src.learning.model import Sequential


def activation_statistics(values: np.ndarray) -> dict[str, float]:
    """Summarize a tensor for inspecting what a layer produces."""
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "l2": float(np.sqrt(np.sum(values * values))),
    }


def parameter_statistics(model: Sequential) -> dict[str, float]:
    """Aggregate simple magnitude statistics across all trainable parameters."""
    parameters = model.parameters()
    if not parameters:
        return {"count": 0.0, "l2": 0.0}

    squared = sum(float(np.sum(parameter.value * parameter.value)) for parameter in parameters)
    count = sum(int(parameter.value.size) for parameter in parameters)
    return {"count": float(count), "l2": float(np.sqrt(squared))}


def layer_summary(model: Sequential) -> list[str]:
    """Return one human-readable line per layer."""
    return [layer.__class__.__name__ for layer in model.layers if isinstance(layer, Layer)]
