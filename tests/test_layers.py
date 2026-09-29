import numpy as np
import pytest

from src.learning.layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU


def test_dense_forward_shape() -> None:
    layer = Dense(3, 2, seed=1)
    result = layer.forward(np.ones((4, 3)))

    assert result.shape == (4, 2)


def test_dense_backward_shapes() -> None:
    layer = Dense(3, 2, seed=1)
    inputs = np.ones((4, 3))
    gradient = np.ones((4, 2))

    layer.forward(inputs)
    result = layer.backward(gradient)

    assert result.shape == inputs.shape
    assert layer.weights.gradient.shape == layer.weights.value.shape
    assert layer.bias.gradient.shape == layer.bias.value.shape


def test_dense_backward_matches_linear_gradient() -> None:
    layer = Dense(2, 1, seed=1)
    inputs = np.array([[2.0, 3.0], [4.0, 5.0]])
    gradient = np.ones((2, 1))

    layer.forward(inputs)
    layer.backward(gradient)

    assert np.allclose(layer.weights.gradient.ravel(), [6.0, 8.0])
    assert np.allclose(layer.bias.gradient, [2.0])


def test_relu_backward() -> None:
    layer = ReLU()
    inputs = np.array([[-1.0, 2.0]])
    layer.forward(inputs)

    assert np.array_equal(layer.backward(np.ones_like(inputs)), np.array([[0.0, 1.0]]))


def test_flatten_round_trip() -> None:
    layer = Flatten()
    inputs = np.arange(24.0).reshape(2, 3, 4)
    output = layer.forward(inputs)

    assert output.shape == (2, 12)
    assert np.array_equal(layer.backward(output), inputs)


def test_max_pool_backward_routes_to_maximum() -> None:
    layer = MaxPool2D()
    inputs = np.array([[[[1.0, 3.0], [2.0, 4.0]]]])
    layer.forward(inputs)

    gradient = np.ones((1, 1, 1, 1))
    assert np.array_equal(
        layer.backward(gradient),
        np.array([[[[0.0, 0.0], [0.0, 1.0]]]]),
    )


def test_conv_forward_shape() -> None:
    layer = Conv2D(1, 2, kernel_size=3, padding=1, seed=1)
    result = layer.forward(np.ones((4, 1, 8, 8)))

    assert result.shape == (4, 2, 8, 8)


def test_conv_backward_shapes() -> None:
    layer = Conv2D(1, 2, kernel_size=3, padding=1, seed=1)
    inputs = np.ones((2, 1, 5, 5))
    gradient = np.ones((2, 2, 5, 5))

    layer.forward(inputs)
    result = layer.backward(gradient)

    assert result.shape == inputs.shape
    assert layer.weights.gradient.shape == layer.weights.value.shape
    assert layer.bias.gradient.shape == layer.bias.value.shape


def test_conv_rejects_wrong_channels() -> None:
    layer = Conv2D(2, 1, kernel_size=3)

    with pytest.raises(ValueError, match="channel count"):
        layer.forward(np.ones((1, 1, 5, 5)))


def test_conv_rejects_kernel_geometry() -> None:
    layer = Conv2D(1, 1, kernel_size=5)

    with pytest.raises(ValueError, match="larger"):
        layer.forward(np.ones((1, 1, 3, 3)))


def test_parameter_count() -> None:
    from src.learning.layers import parameter_count

    model = [
        Dense(3, 4),
        Conv2D(1, 2, kernel_size=3),
    ]

    assert parameter_count(model) == 3 * 4 + 4 + 2 * 1 * 3 * 3 + 2
