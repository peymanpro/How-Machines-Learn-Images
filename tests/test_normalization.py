import numpy as np
import pytest

from src.images.normalization import normalize_image


def test_normalize_image() -> None:
    data = np.array([[0, 128], [255, 64]], dtype=np.uint8)

    result = normalize_image(data)

    assert result.dtype == np.float64
    assert result == pytest.approx(data / 255.0)


def test_normalize_image_preserves_shape() -> None:
    data = np.zeros((2, 3, 3), dtype=np.uint8)

    assert normalize_image(data).shape == data.shape


def test_normalize_rejects_empty_data() -> None:
    with pytest.raises(ValueError):
        normalize_image(np.array([], dtype=np.uint8))


@pytest.mark.parametrize(
    "data",
    [
        np.array([[-1]], dtype=np.int16),
        np.array([[256]], dtype=np.int16),
    ],
)
def test_normalize_rejects_out_of_range_values(data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize_image(data)


def test_normalize_rejects_non_numeric_data() -> None:
    data = np.array([["red"]])

    with pytest.raises(TypeError):
        normalize_image(data)
