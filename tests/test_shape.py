import pytest

from src.images.shape import ImageShape


def test_grayscale_image_shape() -> None:
    shape = ImageShape(height=28, width=28)

    assert shape.dimensions == (28, 28)
    assert shape.size == 784


def test_rgb_image_shape() -> None:
    shape = ImageShape(height=32, width=32, channels=3)

    assert shape.dimensions == (32, 32, 3)
    assert shape.size == 3072


@pytest.mark.parametrize(
    ("height", "width", "channels"),
    [
        (0, 10, 1),
        (-1, 10, 1),
        (10, 0, 1),
        (10, -1, 1),
        (10, 10, 0),
        (10, 10, -1),
    ],
)
def test_image_shape_rejects_invalid_values(height: int, width: int, channels: int) -> None:
    with pytest.raises(ValueError):
        ImageShape(height=height, width=width, channels=channels)
