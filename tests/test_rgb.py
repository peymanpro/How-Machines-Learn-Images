import pytest

from src.images.rgb import RGBPixel


def test_rgb_pixel_accepts_valid_channels() -> None:
    pixel = RGBPixel(10, 20, 30)

    assert pixel.red == 10
    assert pixel.green == 20
    assert pixel.blue == 30


@pytest.mark.parametrize(
    ("red", "green", "blue"),
    [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (256, 0, 0),
        (0, 256, 0),
        (0, 0, 256),
    ],
)
def test_rgb_pixel_rejects_invalid_channels(
    red: int, green: int, blue: int
) -> None:
    with pytest.raises(ValueError):
        RGBPixel(red, green, blue)


def test_rgb_pixel_normalized() -> None:
    pixel = RGBPixel(0, 128, 255)

    assert pixel.normalized == pytest.approx((0.0, 128 / 255, 1.0))
