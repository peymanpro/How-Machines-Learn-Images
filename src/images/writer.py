from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_image(data: np.ndarray, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if data.size == 0:
        raise ValueError("Image data cannot be empty.")

    if data.ndim not in (2, 3):
        raise ValueError("Image data must be a 2D or 3D array.")

    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError("Image data must use an integer dtype.")

    if np.any(data < 0) or np.any(data > 255):
        raise ValueError("Image values must be between 0 and 255.")

    Image.fromarray(data.astype(np.uint8)).save(output_path)
