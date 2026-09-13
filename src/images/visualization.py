from __future__ import annotations

import numpy as np
from PIL import Image


def visualize_image(
    data: np.ndarray,
    scale: int = 1,
) -> Image.Image:
    """Convert an image array into a PIL image for visualization."""
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

    if scale <= 0:
        raise ValueError("Scale must be positive.")

    image = Image.fromarray(data.astype(np.uint8))

    if scale == 1:
        return image

    width, height = image.size

    return image.resize(
        (width * scale, height * scale),
        Image.Resampling.NEAREST,
    )