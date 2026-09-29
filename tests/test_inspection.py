import numpy as np
import pytest

from src.images.inspection import inspect_pixel
from src.images.pixel import Pixel
from src.images.rgb import RGBPixel


def test_inspect_grayscale_pixel() -> None:
    data = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
        ],
        dtype=np.uint8,
    )

    result = inspect_pixel(data, x=1, y=0)

    assert result == Pixel(value=20)


def test_inspect_rgb_pixel() -> None:
    data = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )

    result = inspect_pixel(data, x=1, y=0)

    assert result == RGBPixel(red=40, green=50, blue=60)


def test_inspect_pixel_uses_x_as_width_and_y_as_height() -> None:
    data = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.uint8,
    )

    result = inspect_pixel(data, x=2, y=1)

    assert result == Pixel(value=6)


def test_inspect_pixel_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        inspect_pixel(np.array([], dtype=np.uint8), x=0, y=0)


def test_inspect_pixel_rejects_invalid_dimensions() -> None:
    data = np.zeros((2, 2, 1, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="2D grayscale or 3D RGB"):
        inspect_pixel(data, x=0, y=0)


def test_inspect_pixel_rejects_non_rgb_channel_count() -> None:
    data = np.zeros((2, 2, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="exactly 3 channels"):
        inspect_pixel(data, x=0, y=0)


def test_inspect_pixel_rejects_non_integer_data() -> None:
    data = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(TypeError, match="integer dtype"):
        inspect_pixel(data, x=0, y=0)


def test_inspect_pixel_rejects_out_of_range_data() -> None:
    data = np.array([[0, 256]], dtype=np.int16)

    with pytest.raises(ValueError, match="between 0 and 255"):
        inspect_pixel(data, x=0, y=0)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (-1, 0),
        (2, 0),
        (0, -1),
        (0, 2),
    ],
)
def test_inspect_pixel_rejects_out_of_bounds_coordinates(x: int, y: int) -> None:
    data = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(IndexError, match="out of bounds"):
        inspect_pixel(data, x=x, y=y)
