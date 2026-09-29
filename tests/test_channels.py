import numpy as np
import pytest
from PIL import Image

from src.images.channels import visualize_channel, visualize_channels


def test_visualize_red_channel() -> None:
    data = np.array(
        [
            [[255, 10, 20], [128, 30, 40]],
            [[64, 50, 60], [0, 70, 80]],
        ],
        dtype=np.uint8,
    )

    result = visualize_channel(data, channel=0)

    assert isinstance(result, Image.Image)
    assert result.mode == "L"
    assert result.size == (2, 2)
    assert np.array_equal(
        np.asarray(result),
        np.array([[255, 128], [64, 0]], dtype=np.uint8),
    )


def test_visualize_channels_returns_red_green_blue() -> None:
    data = np.array(
        [
            [[10, 20, 30]],
            [[40, 50, 60]],
        ],
        dtype=np.uint8,
    )

    result = visualize_channels(data)

    assert set(result) == {"red", "green", "blue"}
    assert np.array_equal(np.asarray(result["red"]), [[10], [40]])
    assert np.array_equal(np.asarray(result["green"]), [[20], [50]])
    assert np.array_equal(np.asarray(result["blue"]), [[30], [60]])


def test_visualize_channel_scales_with_nearest_neighbor() -> None:
    data = np.array(
        [
            [[0, 100, 200], [255, 150, 50]],
        ],
        dtype=np.uint8,
    )

    result = visualize_channel(data, channel=1, scale=2)

    expected = np.array(
        [
            [100, 100, 150, 150],
            [100, 100, 150, 150],
        ],
        dtype=np.uint8,
    )

    assert result.size == (4, 2)
    assert np.array_equal(np.asarray(result), expected)


def test_visualize_channel_rejects_grayscale_data() -> None:
    data = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="requires an RGB image"):
        visualize_channel(data, channel=0)


def test_visualize_channel_rejects_invalid_channel() -> None:
    data = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="0, 1, or 2"):
        visualize_channel(data, channel=3)


def test_visualize_channel_rejects_negative_channel() -> None:
    data = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="0, 1, or 2"):
        visualize_channel(data, channel=-1)


def test_visualize_channel_rejects_non_positive_scale() -> None:
    data = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive"):
        visualize_channel(data, channel=0, scale=0)


def test_visualize_channel_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        visualize_channel(np.array([], dtype=np.uint8), channel=0)


def test_visualize_channel_rejects_non_integer_data() -> None:
    data = np.zeros((2, 2, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="integer dtype"):
        visualize_channel(data, channel=0)


def test_visualize_channel_rejects_out_of_range_data() -> None:
    data = np.array([[[0, 0, 256]]], dtype=np.int16)

    with pytest.raises(ValueError, match="between 0 and 255"):
        visualize_channel(data, channel=2)
