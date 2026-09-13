from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.images.writer import save_image


def test_save_grayscale_image(tmp_path: Path) -> None:
    path = tmp_path / "output.png"
    data = np.array([[0, 128], [200, 255]], dtype=np.uint8)

    save_image(data, path)

    assert path.is_file()
    with Image.open(path) as image:
        assert image.size == (2, 2)
        assert np.array_equal(np.asarray(image), data)


def test_save_rgb_image(tmp_path: Path) -> None:
    path = tmp_path / "output.png"
    data = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    save_image(data, path)

    assert path.is_file()
    with Image.open(path) as image:
        assert image.size == (2, 2)
        assert np.array_equal(np.asarray(image), data)


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "output.png"
    data = np.zeros((2, 2), dtype=np.uint8)

    save_image(data, path)

    assert path.is_file()


def test_save_rejects_empty_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_image(np.array([], dtype=np.uint8), tmp_path / "output.png")


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((2, 2, 1, 1), dtype=np.uint8),
        np.zeros((2,), dtype=np.uint8),
    ],
)
def test_save_rejects_invalid_dimensions(tmp_path: Path, data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        save_image(data, tmp_path / "output.png")


def test_save_rejects_non_integer_data(tmp_path: Path) -> None:
    data = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(TypeError):
        save_image(data, tmp_path / "output.png")


@pytest.mark.parametrize(
    "data",
    [
        np.array([[-1]], dtype=np.int16),
        np.array([[256]], dtype=np.int16),
    ],
)
def test_save_rejects_out_of_range_values(
    tmp_path: Path, data: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        save_image(data, tmp_path / "output.png")
