import numpy as np
import pytest

from src.images.tensor import ImageTensor


def test_grayscale_image_tensor() -> None:
    data = np.zeros((28, 28))

    tensor = ImageTensor(data)

    assert tensor.shape == (28, 28)
    assert tensor.dimensions == 2


def test_rgb_image_tensor() -> None:
    data = np.zeros((32, 32, 3))

    tensor = ImageTensor(data)

    assert tensor.shape == (32, 32, 3)
    assert tensor.dimensions == 3


@pytest.mark.parametrize(
    "data",
    [
        np.array([]),
        np.zeros((2,)),
        np.zeros((2, 2, 2, 1)),
    ],
)
def test_image_tensor_rejects_invalid_shape(data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ImageTensor(data)
