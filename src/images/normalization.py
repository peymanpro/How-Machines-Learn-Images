from __future__ import annotations

import numpy as np


def normalize_image(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        raise ValueError("Image data cannot be empty.")

    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Image data must be numeric.")

    values = data.astype(np.float64)

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if minimum < 0 or maximum > 255:
        raise ValueError("Image values must be between 0 and 255.")

    return values / 255.0
