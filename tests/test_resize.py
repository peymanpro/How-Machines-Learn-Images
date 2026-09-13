import numpy as np
import pytest

from src.images.resize import resize_image


def test_resize_grayscale_image() -> None:
    data = np.array([[0, 255], [128, 64]], dtype=np.uint8)

    result = resize_image(data, (4, 3))

    assert result.shape == (3, 4)
    assert result.dtype == np.uint8


def test_resize_rgb_image() -> None:
    data = np.zeros((2, 2, 3), dtype=np.uint8)

    result = resize_image(data, (4, 5))

    assert result.shape == (5, 4, 3)
    assert result.dtype == np.uint8


def test_resize_uses_nearest_neighbor() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    result = resize_image(data, (4, 4))

    expected = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(result, expected)


@pytest.mark.parametrize("size", [(0, 2), (2, 0), (-1, 2)])
def test_resize_rejects_invalid_size(size: tuple[int, int]) -> None:
    data = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError):
        resize_image(data, size)


def test_resize_rejects_empty_data() -> None:
    with pytest.raises(ValueError):
        resize_image(np.array([], dtype=np.uint8), (2, 2))


def test_resize_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        resize_image(np.zeros((2, 2, 1, 1), dtype=np.uint8), (2, 2))


def test_resize_rejects_non_integer_data() -> None:
    with pytest.raises(TypeError):
        resize_image(np.zeros((2, 2), dtype=np.float32), (2, 2))


def test_resize_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        resize_image(np.array([[256]], dtype=np.int16), (2, 2))
