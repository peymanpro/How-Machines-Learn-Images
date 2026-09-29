from __future__ import annotations

import numpy as np
from PIL import Image

CHANNEL_NAMES = ("red", "green", "blue")


def visualize_channel(
    data: np.ndarray,
    channel: int,
    scale: int = 1,
) -> Image.Image:
    """Visualize one RGB channel as a grayscale image."""
    if data.size == 0:
        raise ValueError("Image data cannot be empty.")

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("Channel visualization requires an RGB image.")

    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError("Image data must use an integer dtype.")

    if np.any(data < 0) or np.any(data > 255):
        raise ValueError("Image values must be between 0 and 255.")

    if channel < 0 or channel >= 3:
        raise ValueError("Channel index must be 0, 1, or 2.")

    if scale <= 0:
        raise ValueError("Scale must be positive.")

    channel_data = data[:, :, channel].astype(np.uint8)
    image = Image.fromarray(channel_data, mode="L")

    if scale == 1:
        return image

    width, height = image.size

    return image.resize(
        (width * scale, height * scale),
        Image.Resampling.NEAREST,
    )


def visualize_channels(
    data: np.ndarray,
    scale: int = 1,
) -> dict[str, Image.Image]:
    """Visualize all RGB channels as grayscale images."""
    return {
        name: visualize_channel(data, channel=index, scale=scale)
        for index, name in enumerate(CHANNEL_NAMES)
    }
