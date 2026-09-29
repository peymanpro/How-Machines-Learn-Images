import numpy as np
import pytest

from src.images.histogram import compute_histogram, compute_rgb_histograms


def test_compute_grayscale_histogram() -> None:
    data = np.array(
        [
            [0, 0, 1],
            [1, 2, 255],
        ],
        dtype=np.uint8,
    )

    result = compute_histogram(data)

    assert result.shape == (256,)
    assert result.dtype == np.int64
    assert result[0] == 2
    assert result[1] == 2
    assert result[2] == 1
    assert result[255] == 1
    assert int(result.sum()) == data.size


def test_compute_grayscale_histogram_contains_zero_for_missing_values() -> None:
    data = np.array([[10, 10, 20]], dtype=np.uint8)

    result = compute_histogram(data)

    assert result[0] == 0
    assert result[10] == 2
    assert result[20] == 1
    assert int(result.sum()) == 3


def test_compute_rgb_histograms() -> None:
    data = np.array(
        [
            [[255, 0, 10], [255, 20, 10]],
            [[0, 0, 20], [128, 20, 20]],
        ],
        dtype=np.uint8,
    )

    result = compute_rgb_histograms(data)

    assert set(result) == {"red", "green", "blue"}
    assert result["red"][0] == 1
    assert result["red"][128] == 1
    assert result["red"][255] == 2
    assert result["green"][0] == 2
    assert result["green"][20] == 2
    assert result["blue"][10] == 2
    assert result["blue"][20] == 2

    for histogram in result.values():
        assert histogram.shape == (256,)
        assert int(histogram.sum()) == data.shape[0] * data.shape[1]


def test_compute_histogram_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_histogram(np.array([], dtype=np.uint8))


def test_compute_histogram_rejects_non_grayscale_data() -> None:
    data = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="2D grayscale"):
        compute_histogram(data)


def test_compute_rgb_histograms_rejects_grayscale_data() -> None:
    data = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="3D RGB"):
        compute_rgb_histograms(data)


def test_compute_histogram_rejects_non_integer_data() -> None:
    data = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(TypeError, match="integer dtype"):
        compute_histogram(data)


def test_compute_rgb_histograms_rejects_non_integer_data() -> None:
    data = np.zeros((2, 2, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="integer dtype"):
        compute_rgb_histograms(data)


def test_compute_histogram_rejects_out_of_range_data() -> None:
    data = np.array([[0, 256]], dtype=np.int16)

    with pytest.raises(ValueError, match="between 0 and 255"):
        compute_histogram(data)


def test_compute_rgb_histograms_rejects_out_of_range_data() -> None:
    data = np.array([[[0, 0, 256]]], dtype=np.int16)

    with pytest.raises(ValueError, match="between 0 and 255"):
        compute_rgb_histograms(data)
