import numpy as np
import pytest
from PIL import Image

from src.images.visualization import visualize_image


def test_visualize_grayscale_image() -> None:
    data = np.array([[0, 127], [200, 255]], dtype=np.uint8)

    result = visualize_image(data)

    assert isinstance(result, Image.Image)
    assert result.size == (2, 2)
    assert result.mode == "L"
    assert np.array_equal(np.asarray(result), data)


def test_visualize_rgb_image() -> None:
    data = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    result = visualize_image(data)

    assert isinstance(result, Image.Image)
    assert result.size == (2, 2)
    assert result.mode == "RGB"
    assert np.array_equal(np.asarray(result), data)


def test_visualize_image_scales_with_nearest_neighbor() -> None:
    data = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    result = visualize_image(data, scale=2)

    expected = np.array(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 0],
            [255, 255, 0, 0],
        ],
        dtype=np.uint8,
    )

    assert result.size == (4, 4)
    assert np.array_equal(np.asarray(result), expected)


def test_visualize_image_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        visualize_image(np.array([], dtype=np.uint8))


def test_visualize_image_rejects_invalid_dimensions() -> None:
    data = np.zeros((2, 2, 2, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="2D grayscale or 3D RGB"):
        visualize_image(data)


def test_visualize_image_rejects_non_rgb_channel_count() -> None:
    data = np.zeros((2, 2, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="exactly 3 channels"):
        visualize_image(data)


def test_visualize_image_rejects_non_integer_data() -> None:
    data = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(TypeError, match="integer dtype"):
        visualize_image(data)


def test_visualize_image_rejects_out_of_range_data() -> None:
    data = np.array([[0, 256]], dtype=np.int16)

    with pytest.raises(ValueError, match="between 0 and 255"):
        visualize_image(data)


def test_visualize_image_rejects_non_positive_scale() -> None:
    data = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive"):
        visualize_image(data, scale=0)