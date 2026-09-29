from __future__ import annotations

import numpy as np


def compute_histogram(data: np.ndarray) -> np.ndarray:
    """Compute the 256-bin intensity histogram of a grayscale image."""
    if data.size == 0:
        raise ValueError("Image data cannot be empty.")

    if data.ndim != 2:
        raise ValueError("Histogram requires a 2D grayscale image.")

    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError("Image data must use an integer dtype.")

    if np.any(data < 0) or np.any(data > 255):
        raise ValueError("Image values must be between 0 and 255.")

    return np.bincount(data.ravel(), minlength=256).astype(np.int64)


def compute_rgb_histograms(
    data: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute a 256-bin histogram for each RGB channel."""
    if data.size == 0:
        raise ValueError("Image data cannot be empty.")

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("RGB histogram requires a 3D RGB image.")

    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError("Image data must use an integer dtype.")

    if np.any(data < 0) or np.any(data > 255):
        raise ValueError("Image values must be between 0 and 255.")

    names = ("red", "green", "blue")

    return {
        name: np.bincount(data[:, :, index].ravel(), minlength=256).astype(np.int64)
        for index, name in enumerate(names)
    }
