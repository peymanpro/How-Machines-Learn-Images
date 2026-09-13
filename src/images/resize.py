from __future__ import annotations

import numpy as np
from PIL import Image


def resize_image(
    data: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size

    if width <= 0 or height <= 0:
        raise ValueError("Target width and height must be positive.")

    if data.size == 0:
        raise ValueError("Image data cannot be empty.")

    if data.ndim not in (2, 3):
        raise ValueError("Image data must be a 2D or 3D array.")

    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError("Image data must use an integer dtype.")

    if np.any(data < 0) or np.any(data > 255):
        raise ValueError("Image values must be between 0 and 255.")

    image = Image.fromarray(data.astype(np.uint8))
    resized = image.resize((width, height), Image.Resampling.NEAREST)

    return np.asarray(resized)
