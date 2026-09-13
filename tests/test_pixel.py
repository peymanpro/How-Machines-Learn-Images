import pytest

from src.images.pixel import Pixel


def test_pixel_accepts_valid_value() -> None:
    pixel = Pixel(128)

    assert pixel.value == 128


@pytest.mark.parametrize("value", [0, 255])
def test_pixel_accepts_boundaries(value: int) -> None:
    assert Pixel(value).value == value


@pytest.mark.parametrize("value", [-1, 256])
def test_pixel_rejects_invalid_value(value: int) -> None:
    with pytest.raises(ValueError):
        Pixel(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (128, 128 / 255),
        (255, 1.0),
    ],
)
def test_pixel_intensity(value: int, expected: float) -> None:
    assert Pixel(value).intensity == pytest.approx(expected)
