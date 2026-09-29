from __future__ import annotations

import numpy as np


def max_pool2d(
    image: np.ndarray,
    pool_size: int = 2,
    stride: int | None = None,
) -> np.ndarray:
    """Downsample a 2D image by taking local maxima."""
    if image.ndim != 2:
        raise ValueError("Image must be a 2D array.")
    if image.size == 0:
        raise ValueError("Image cannot be empty.")
    if pool_size <= 0:
        raise ValueError("Pool size must be positive.")
    if stride is None:
        stride = pool_size
    if stride <= 0:
        raise ValueError("Stride must be positive.")
    if image.shape[0] < pool_size or image.shape[1] < pool_size:
        raise ValueError("Pool size cannot exceed image dimensions.")

    output_height = (image.shape[0] - pool_size) // stride + 1
    output_width = (image.shape[1] - pool_size) // stride + 1
    output = np.empty((output_height, output_width), dtype=image.dtype)

    for row in range(output_height):
        for column in range(output_width):
            row_start = row * stride
            column_start = column * stride
            window = image[
                row_start : row_start + pool_size,
                column_start : column_start + pool_size,
            ]
            output[row, column] = np.max(window)

    return output
