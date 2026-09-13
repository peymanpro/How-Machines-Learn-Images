from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as image:
        return np.asarray(image)
