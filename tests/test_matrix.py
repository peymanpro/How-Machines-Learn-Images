import numpy as np
import pytest

from src.images.matrix import ImageMatrix


def test_image_matrix_accepts_2d_array() -> None:
    data = np.array([[0, 1], [2, 3]])

    image = ImageMatrix(data)

    assert image.shape == (2, 2)
    assert image.height == 2
    assert image.width == 2


def test_image_matrix_preserves_data() -> None:
    data = np.array([[10, 20], [30, 40]])

    image = ImageMatrix(data)

    assert np.array_equal(image.data, data)


@pytest.mark.parametrize(
    "data",
    [
        np.array([]),
        np.array([1, 2, 3]),
        np.zeros((2, 2, 1)),
        np.zeros((1, 2, 2, 1)),
    ],
)
def test_image_matrix_rejects_invalid_shape(data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ImageMatrix(data)
