import numpy as np
import pytest

from src.vision.convolution import (
    convolve2d,
    cross_correlate2d,
    edge_kernels,
    pad_image,
)


def test_pad_image() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])

    padded = pad_image(image, padding=1)

    assert padded.shape == (4, 4)
    assert padded[1, 1] == 1.0
    assert padded[0, 0] == 0.0


def test_cross_correlation_identity_kernel() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    kernel = np.array([[1.0]])

    result = cross_correlate2d(image, kernel)

    assert np.array_equal(result, image)


def test_cross_correlation_matches_hand_calculation() -> None:
    image = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    kernel = np.array([[1.0, 0.0], [0.0, -1.0]])

    result = cross_correlate2d(image, kernel)

    assert np.array_equal(
        result,
        np.array([[-4.0, -4.0], [-4.0, -4.0]]),
    )


def test_mathematical_convolution_flips_kernel() -> None:
    image = np.arange(1.0, 10.0).reshape(3, 3)
    kernel = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = convolve2d(image, kernel)

    expected = cross_correlate2d(image, np.flip(kernel, axis=(0, 1)))

    assert np.array_equal(result, expected)


def test_stride_changes_output_geometry() -> None:
    image = np.ones((5, 5))
    kernel = np.ones((3, 3))

    result = cross_correlate2d(image, kernel, stride=2)

    assert result.shape == (2, 2)


def test_padding_preserves_geometry_for_odd_kernel() -> None:
    image = np.ones((5, 5))
    kernel = np.ones((3, 3))

    result = cross_correlate2d(image, kernel, padding=1)

    assert result.shape == image.shape


def test_kernel_cannot_exceed_padded_image() -> None:
    with pytest.raises(ValueError, match="larger"):
        cross_correlate2d(np.ones((2, 2)), np.ones((3, 3)))


def test_edge_kernels_are_named() -> None:
    kernels = edge_kernels()

    assert set(kernels) == {"horizontal", "vertical", "sharpen"}
    assert kernels["horizontal"].shape == (3, 3)
