from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod, sqrt

import numpy as np

from src.learning.activations import relu
from src.vision.pooling import max_pool2d


@dataclass
class Parameter:
    """A trainable array and its current gradient."""

    value: np.ndarray
    gradient: np.ndarray

    def zero_grad(self) -> None:
        self.gradient.fill(0.0)


class Layer(ABC):
    """Base interface for differentiable network layers."""

    @abstractmethod
    def forward(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> tuple[Parameter, ...]:
        return ()

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()


class Dense(Layer):
    """Fully connected affine layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        seed: int = 0,
    ) -> None:
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Dense layer sizes must be positive.")

        rng = np.random.default_rng(seed)
        scale = sqrt(2.0 / input_size)
        weights = rng.normal(0.0, scale, size=(input_size, output_size))
        self.weights = Parameter(weights, np.zeros_like(weights))
        self.bias = Parameter(
            np.zeros(output_size, dtype=np.float64),
            np.zeros(output_size, dtype=np.float64),
        )
        self._input: np.ndarray | None = None

    def forward(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 2:
            raise ValueError("Dense input must be a 2D batch.")
        if values.shape[1] != self.weights.value.shape[0]:
            raise ValueError("Dense input width must match input_size.")

        self._input = values
        return values @ self.weights.value + self.bias.value

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        if self._input is None:
            raise RuntimeError("Dense backward called before forward.")
        if gradient.ndim != 2 or gradient.shape[1] != self.bias.value.shape[0]:
            raise ValueError("Dense gradient shape does not match layer output.")

        self.weights.gradient[...] = self._input.T @ gradient
        self.bias.gradient[...] = np.sum(gradient, axis=0)
        return gradient @ self.weights.value.T

    def parameters(self) -> tuple[Parameter, ...]:
        return (self.weights, self.bias)


class ReLU(Layer):
    """Rectified linear activation."""

    def __init__(self) -> None:
        self._input: np.ndarray | None = None

    def forward(self, values: np.ndarray) -> np.ndarray:
        self._input = values
        return relu(values)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        if self._input is None:
            raise RuntimeError("ReLU backward called before forward.")
        return gradient * (self._input > 0.0)


class Flatten(Layer):
    """Flatten all non-batch dimensions into a feature vector."""

    def __init__(self) -> None:
        self._input_shape: tuple[int, ...] | None = None

    def forward(self, values: np.ndarray) -> np.ndarray:
        if values.ndim < 2:
            raise ValueError("Flatten expects a batch dimension.")
        self._input_shape = tuple(int(size) for size in values.shape)
        return values.reshape(values.shape[0], -1)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        if self._input_shape is None:
            raise RuntimeError("Flatten backward called before forward.")
        return gradient.reshape(self._input_shape)


class MaxPool2D(Layer):
    """Two-dimensional max pooling over batched channel maps."""

    def __init__(
        self,
        pool_size: int = 2,
        stride: int | None = None,
    ) -> None:
        if pool_size <= 0:
            raise ValueError("Pool size must be positive.")
        if stride is not None and stride <= 0:
            raise ValueError("Stride must be positive.")
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self._input: np.ndarray | None = None
        self._argmax: np.ndarray | None = None

    def forward(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 4:
            raise ValueError("MaxPool2D input must have shape (batch, channels, height, width).")
        batch, channels, height, width = values.shape
        if height < self.pool_size or width < self.pool_size:
            raise ValueError("Pool size cannot exceed spatial dimensions.")

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        output = np.empty(
            (batch, channels, out_height, out_width),
            dtype=values.dtype,
        )
        argmax = np.empty(
            (batch, channels, out_height, out_width),
            dtype=np.int64,
        )

        for batch_index in range(batch):
            for channel in range(channels):
                channel_output = max_pool2d(
                    values[batch_index, channel],
                    pool_size=self.pool_size,
                    stride=self.stride,
                )
                output[batch_index, channel] = channel_output
                for row in range(out_height):
                    for column in range(out_width):
                        row_start = row * self.stride
                        column_start = column * self.stride
                        window = values[
                            batch_index,
                            channel,
                            row_start : row_start + self.pool_size,
                            column_start : column_start + self.pool_size,
                        ]
                        argmax[batch_index, channel, row, column] = int(
                            np.argmax(window)
                        )

        self._input = values
        self._argmax = argmax
        return output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        if self._input is None or self._argmax is None:
            raise RuntimeError("MaxPool2D backward called before forward.")

        batch, channels, height, width = self._input.shape
        output_height, output_width = gradient.shape[2:]
        result = np.zeros_like(self._input)

        for batch_index in range(batch):
            for channel in range(channels):
                for row in range(output_height):
                    for column in range(output_width):
                        flat_index = int(self._argmax[batch_index, channel, row, column])
                        local_row = flat_index // self.pool_size
                        local_column = flat_index % self.pool_size
                        target_row = row * self.stride + local_row
                        target_column = column * self.stride + local_column
                        result[batch_index, channel, target_row, target_column] += gradient[
                            batch_index,
                            channel,
                            row,
                            column,
                        ]

        return result


class Conv2D(Layer):
    """A trainable 2D cross-correlation layer implemented with NumPy loops."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        seed: int = 0,
    ) -> None:
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("Channel counts must be positive.")
        if kernel_size <= 0:
            raise ValueError("Kernel size must be positive.")
        if stride <= 0:
            raise ValueError("Stride must be positive.")
        if padding < 0:
            raise ValueError("Padding must be non-negative.")

        rng = np.random.default_rng(seed)
        scale = sqrt(2.0 / (input_channels * kernel_size * kernel_size))
        weights = rng.normal(
            0.0,
            scale,
            size=(output_channels, input_channels, kernel_size, kernel_size),
        )
        self.weights = Parameter(weights, np.zeros_like(weights))
        self.bias = Parameter(
            np.zeros(output_channels, dtype=np.float64),
            np.zeros(output_channels, dtype=np.float64),
        )
        self.stride = stride
        self.padding = padding
        self._input: np.ndarray | None = None
        self._padded_input: np.ndarray | None = None

    def _output_shape(self, height: int, width: int) -> tuple[int, int]:
        kernel = self.weights.value.shape[2]
        effective_height = height + 2 * self.padding - kernel
        effective_width = width + 2 * self.padding - kernel
        if effective_height < 0 or effective_width < 0:
            raise ValueError("Kernel cannot be larger than the padded input.")
        return (
            effective_height // self.stride + 1,
            effective_width // self.stride + 1,
        )

    def forward(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 4:
            raise ValueError("Conv2D input must have shape (batch, channels, height, width).")
        if values.shape[1] != self.weights.value.shape[1]:
            raise ValueError("Conv2D input channel count does not match layer.")

        batch, _, height, width = values.shape
        out_height, out_width = self._output_shape(height, width)
        padded = np.pad(
            values,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
        )
        output = np.empty(
            (batch, self.weights.value.shape[0], out_height, out_width),
            dtype=np.float64,
        )

        for batch_index in range(batch):
            for output_channel in range(self.weights.value.shape[0]):
                kernel = self.weights.value[output_channel]
                for row in range(out_height):
                    row_start = row * self.stride
                    for column in range(out_width):
                        column_start = column * self.stride
                        patch = padded[
                            batch_index,
                            :,
                            row_start : row_start + kernel.shape[1],
                            column_start : column_start + kernel.shape[2],
                        ]
                        output[batch_index, output_channel, row, column] = (
                            np.sum(patch * kernel) + self.bias.value[output_channel]
                        )

        self._input = values
        self._padded_input = padded
        return output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        if self._input is None or self._padded_input is None:
            raise RuntimeError("Conv2D backward called before forward.")
        if gradient.ndim != 4:
            raise ValueError("Conv2D gradient must be a 4D array.")

        batch, channels, height, width = self._input.shape
        output_channels, _, kernel_height, kernel_width = self.weights.value.shape
        _, _, out_height, out_width = gradient.shape

        self.weights.gradient.fill(0.0)
        self.bias.gradient.fill(0.0)
        padded_gradient = np.zeros_like(self._padded_input, dtype=np.float64)

        for batch_index in range(batch):
            for output_channel in range(output_channels):
                kernel = self.weights.value[output_channel]
                for row in range(out_height):
                    row_start = row * self.stride
                    for column in range(out_width):
                        column_start = column * self.stride
                        grad_value = gradient[
                            batch_index,
                            output_channel,
                            row,
                            column,
                        ]
                        patch = self._padded_input[
                            batch_index,
                            :,
                            row_start : row_start + kernel_height,
                            column_start : column_start + kernel_width,
                        ]
                        self.weights.gradient[output_channel] += grad_value * patch
                        self.bias.gradient[output_channel] += grad_value
                        padded_gradient[
                            batch_index,
                            :,
                            row_start : row_start + kernel_height,
                            column_start : column_start + kernel_width,
                        ] += grad_value * kernel

        if self.padding == 0:
            return padded_gradient

        return padded_gradient[
            :,
            :,
            self.padding : height + self.padding,
            self.padding : width + self.padding,
        ]

    def parameters(self) -> tuple[Parameter, ...]:
        return (self.weights, self.bias)


def parameter_count(layers: list[Layer]) -> int:
    """Return the total number of scalar trainable parameters."""
    return sum(int(prod(parameter.value.shape)) for layer in layers for parameter in layer.parameters())
