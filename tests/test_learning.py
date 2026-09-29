import numpy as np
import pytest

from src.data.synthetic import generate_line_dataset
from src.learning.gradcheck import numerical_gradient, relative_error
from src.learning.layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Tanh
from src.learning.losses import mean_squared_error, mean_squared_error_gradient
from src.learning.model import Sequential
from src.learning.optim import SGD, train


def test_tanh_backward_at_zero() -> None:
    layer = Tanh()
    values = np.array([[0.0]])
    layer.forward(values)

    assert layer.backward(np.ones_like(values))[0, 0] == pytest.approx(1.0)


def test_dense_gradient_matches_finite_difference() -> None:
    model = Sequential([Dense(2, 1, seed=3)])
    inputs = np.array([[1.0, 2.0], [2.0, -1.0]])
    targets = np.array([[0.5], [-0.25]])

    def loss_for_weights(weights: np.ndarray) -> float:
        model.layers[0].weights.value[...] = weights
        predictions = model.forward(inputs)
        return mean_squared_error(predictions, targets)

    predictions = model.forward(inputs)
    gradient = mean_squared_error_gradient(predictions, targets)
    model.zero_grad()
    model.backward(gradient)
    analytic = model.layers[0].weights.gradient.copy()

    numeric = numerical_gradient(loss_for_weights, model.layers[0].weights.value.copy())

    assert relative_error(analytic, numeric) < 1e-7


def test_conv_gradient_matches_finite_difference() -> None:
    model = Sequential([Conv2D(1, 1, kernel_size=2, seed=4), Flatten(), Dense(4, 1, seed=5)])
    inputs = np.array([[[[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [2.0, 1.0, 0.0]]]])
    targets = np.array([[0.25]])

    def loss_for_weights(weights: np.ndarray) -> float:
        model.layers[0].weights.value[...] = weights
        predictions = model.forward(inputs)
        return mean_squared_error(predictions, targets)

    predictions = model.forward(inputs)
    gradient = mean_squared_error_gradient(predictions, targets)
    model.zero_grad()
    model.backward(gradient)
    analytic = model.layers[0].weights.gradient.copy()

    numeric = numerical_gradient(loss_for_weights, model.layers[0].weights.value.copy())

    assert relative_error(analytic, numeric) < 1e-6


def test_sgd_moves_parameter_down_gradient() -> None:
    layer = Dense(1, 1, seed=1)
    parameter = layer.weights
    parameter.gradient[...] = 2.0
    before = parameter.value.copy()

    SGD((parameter,), learning_rate=0.1).step()

    assert np.allclose(parameter.value, before - 0.2)


def test_tiny_network_learns_xor() -> None:
    inputs = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    model = Sequential(
        [
            Dense(2, 4, seed=1),
            Tanh(),
            Dense(4, 2, seed=2),
        ]
    )
    optimizer = SGD(model.parameters(), learning_rate=0.5)

    history = train(model, inputs, labels, optimizer, epochs=1000)

    assert history.loss[-1] < history.loss[0] * 0.25
    assert float(np.mean(model.predict(inputs) == labels)) >= 0.75


def test_cnn_backprop_changes_loss_on_image_task() -> None:
    inputs, labels = generate_line_dataset(samples=8, size=8, noise=0.0, seed=7)
    model = Sequential(
        [
            Conv2D(1, 2, kernel_size=3, padding=1, seed=1),
            ReLU(),
            MaxPool2D(),
            Flatten(),
            Dense(2 * 4 * 4, 2, seed=2),
        ]
    )
    assert model.layers[2].pool_size == 2
    optimizer = SGD(model.parameters(), learning_rate=0.05)

    history = train(model, inputs, labels, optimizer, epochs=5)

    assert history.loss[-1] < history.loss[0]
