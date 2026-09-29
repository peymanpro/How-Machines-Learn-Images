from __future__ import annotations

import numpy as np


def _validate_image_2d(image: np.ndarray, name: str = "Image") -> None:
    if image.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if image.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"{name} must be numeric.")


def _validate_kernel(kernel: np.ndarray) -> None:
    _validate_image_2d(kernel, "Kernel")


def _validate_geometry(stride: int, padding: int) -> None:
    if stride <= 0:
        raise ValueError("Stride must be positive.")
    if padding < 0:
        raise ValueError("Padding must be non-negative.")


def _output_size(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    effective = input_size + 2 * padding - kernel_size
    if effective < 0:
        raise ValueError("Kernel cannot be larger than the padded image.")
    return effective // stride + 1


def pad_image(image: np.ndarray, padding: int) -> np.ndarray:
    """Pad a 2D image with zeros around all borders."""
    _validate_image_2d(image)
    _validate_geometry(1, padding)
    return np.pad(image, ((padding, padding), (padding, padding)), mode="constant")


def cross_correlate2d(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Apply a kernel without flipping it, matching CNN cross-correlation."""
    _validate_image_2d(image)
    _validate_kernel(kernel)
    _validate_geometry(stride, padding)

    padded = pad_image(image, padding)
    output_height = _output_size(padded.shape[0], kernel.shape[0], stride, 0)
    output_width = _output_size(padded.shape[1], kernel.shape[1], stride, 0)
    output = np.empty(
        (output_height, output_width),
        dtype=np.result_type(image, kernel, np.float64),
    )

    for row in range(output_height):
        row_start = row * stride
        for column in range(output_width):
            column_start = column * stride
            patch = padded[
                row_start : row_start + kernel.shape[0],
                column_start : column_start + kernel.shape[1],
            ]
            output[row, column] = np.sum(patch * kernel)

    return output


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Perform mathematical 2D convolution by flipping the kernel first."""
    _validate_kernel(kernel)
    flipped = np.flip(kernel, axis=(0, 1))
    return cross_correlate2d(image, flipped, stride=stride, padding=padding)


def edge_kernels() -> dict[str, np.ndarray]:
    """Return small kernels commonly used to expose edges and gradients."""
    return {
        "horizontal": np.array(
            [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        ),
        "vertical": np.array(
            [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
        ),
        "sharpen": np.array(
            [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]
        ),
    }
