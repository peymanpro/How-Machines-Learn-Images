from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.images.loader import load_image


def test_load_image(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    source = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    Image.fromarray(source, mode="RGB").save(path)

    result = load_image(path)

    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8
    assert np.array_equal(result, source)


def test_load_image_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")
