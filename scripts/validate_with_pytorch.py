from __future__ import annotations

import numpy as np

from src.learning.gradcheck import relative_error
from src.learning.layers import Conv2D


def main() -> None:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install it separately to run this validation."
        ) from exc

    rng = np.random.default_rng(42)
    inputs = rng.normal(size=(1, 1, 4, 4))
    layer = Conv2D(1, 1, kernel_size=3, padding=1, seed=1)

    torch_input = torch.tensor(inputs, dtype=torch.float64, requires_grad=True)
    torch_weight = torch.tensor(layer.weights.value, dtype=torch.float64, requires_grad=True)
    torch_bias = torch.tensor(layer.bias.value, dtype=torch.float64, requires_grad=True)
    torch_output = functional.conv2d(
        torch_input,
        torch_weight,
        torch_bias,
        stride=layer.stride,
        padding=layer.padding,
    )
    torch_output.sum().backward()

    numpy_output = layer.forward(inputs)
    numpy_input_gradient = layer.backward(np.ones_like(numpy_output))

    output_error = relative_error(numpy_output, torch_output.detach().numpy())
    weight_error = relative_error(
        layer.weights.gradient,
        torch_weight.grad.detach().numpy(),
    )
    input_error = relative_error(
        numpy_input_gradient,
        torch_input.grad.detach().numpy(),
    )

    print(f"forward relative error: {output_error:.3e}")
    print(f"weight gradient relative error: {weight_error:.3e}")
    print(f"input gradient relative error: {input_error:.3e}")

    if max(output_error, weight_error, input_error) > 1e-10:
        raise SystemExit("Framework validation failed.")


if __name__ == "__main__":
    main()
