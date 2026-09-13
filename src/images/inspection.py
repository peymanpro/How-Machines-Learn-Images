from __future__ import annotations

import numpy as np

from src.images.pixel import Pixel
from src.images.rgb import RGBPixel


def inspect_pixel(
    data: np.ndarray,
    x: int,
    y: int,
) -> Pixel | RGBPixel:
    """Inspect a single pixel using x/y image coordinates."""
    if data.size == 0:
        raise ValueError("Image data cannot be empty.")

    if data.ndim not in (2, 3):
        raise ValueError("Image data must be a 2D grayscale or 3D RGB array.")

    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError("Image data must use an integer dtype.")

    if np.any(data < 0) or np.any(data > 255):
        raise ValueError("Image values must be between 0 and 255.")

    if data.ndim == 3 and data.shape[2] != 3:
        raise ValueError("RGB image data must have exactly 3 channels.")

    if x < 0 or x >= data.shape[1]:
        raise IndexError("Pixel x coordinate is out of bounds.")

    if y < 0 or y >= data.shape[0]:
        raise IndexError("Pixel y coordinate is out of bounds.")

    if data.ndim == 2:
        return Pixel(value=int(data[y, x]))

    red, green, blue = (int(value) for value in data[y, x])
    return RGBPixel(red=red, green=green, blue=blue)