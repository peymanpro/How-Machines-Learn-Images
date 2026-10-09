# How Machines Learn Images

[![Quality](https://github.com/peymanpro/How-Machines-Learn-Images/actions/workflows/quality.yml/badge.svg?branch=main)](https://github.com/peymanpro/How-Machines-Learn-Images/actions/workflows/quality.yml)
[![Latest Release](https://img.shields.io/github/v/release/peymanpro/How-Machines-Learn-Images?display_name=tag)](https://github.com/peymanpro/How-Machines-Learn-Images/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A from-scratch, NumPy-based path from image pixels to a trainable convolutional neural network.

The project is designed to answer one question:

> What actually happens between a matrix of pixels and a machine that can learn visual patterns?

Instead of beginning with PyTorch or TensorFlow, the core learning pipeline is implemented explicitly with Python and NumPy. The repository then validates the implementation with tests, gradient checks, controlled experiments, and an optional comparison against PyTorch.

## Learning path

```
Pixels
  ↓
Image representation
  ↓
Vectors / matrices / geometry
  ↓
Linear transformations
  ↓
Convolution / cross-correlation
  ↓
Features
  ↓
Neurons / activations
  ↓
Prediction / loss
  ↓
Gradients
  ↓
Backpropagation
  ↓
Gradient descent
  ↓
Tiny neural network
  ↓
CNN
  ↓
Image classification
  ↓
Inspection / experiments
  ↓
Framework validation
```

## What is implemented from scratch

- image pixels, intensity, grayscale, RGB, shape, matrices, tensors, normalization, resizing and visualization
- vectors, dot product, matrix multiplication, norms, distance, cosine similarity and linear transformations
- 2D padding, cross-correlation, mathematical convolution, stride and edge kernels
- max pooling
- dense, ReLU, tanh, flatten, convolutional and max-pooling layers
- softmax, mean squared error and multiclass cross-entropy
- analytical gradients and central-difference numerical gradient checking
- explicit backpropagation through the network
- vanilla SGD
- deterministic synthetic image datasets
- an MNIST IDX loader and training entry point
- activation/parameter inspection
- controlled learning-rate experiments
- an optional PyTorch validation script
- a dependency-free interactive convolution demo

The core training implementation does not depend on a machine-learning framework.

## Repository layout

```
src/
  data/          Dataset generators and MNIST IDX loader
  images/        Image representations and utilities
  learning/      Layers, losses, gradients, model and optimizer
  math/          Vector, matrix and coordinate primitives
  vision/        Convolution and pooling primitives

tests/           Unit, integration and gradient-check tests
scripts/         Reproducible training and validation entry points
demo/            Browser-only interactive convolution demo
docs/            Architecture and validation documentation
```

## Quick start

Python 3.12 is the supported development version.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

python -m pytest -q
ruff check .
ruff format --check .
mypy src
```

Run the complete synthetic image pipeline:

```bash
python -m scripts.train_lines
python -m scripts.train_shapes
python -m scripts.run_experiments
```

## MNIST

The repository includes an IDX parser rather than storing the MNIST dataset in Git.

Provide the four standard IDX files and run:

```bash
python -m scripts.train_mnist \
  --train-images path/to/train-images-idx3-ubyte \
  --train-labels path/to/train-labels-idx1-ubyte \
  --test-images path/to/t10k-images-idx3-ubyte \
  --test-labels path/to/t10k-labels-idx1-ubyte
```

The default limits keep the example small enough for an educational from-scratch implementation. They can be changed with `--train-limit`, `--test-limit`, `--epochs` and `--learning-rate`.

## Gradient correctness

The repository does not treat "the loss went down" as sufficient evidence.

Analytical gradients for dense and convolutional layers are compared against central finite-difference numerical gradients. The same approach is exposed as a reusable utility in `src/learning/gradcheck.py`.

For an optional framework cross-check:

```bash
python -m scripts.validate_with_pytorch
```

PyTorch is intentionally not a core dependency.

## Interactive demo

Open `demo/index.html` in a browser.

The demo lets you draw a small image and observe the response of a vertical-edge kernel. It mirrors the sliding-window multiplication-and-summation operation used by the Python implementation.

## Verification

Continuous integration runs:

1. the full pytest suite
2. synthetic training smoke tests
3. Ruff linting
4. Ruff format checking
5. strict Mypy checking

See [docs/VALIDATION.md](docs/VALIDATION.md) for the evidence and limitations of each check.

## Design principles

The project follows a few strict rules:

- understand the mathematics before hiding it behind an abstraction
- keep the core implementation small and inspectable
- verify gradients rather than trusting them
- prefer deterministic experiments
- separate framework validation from the framework-free core
- document limitations instead of presenting toy results as production benchmarks

This repository is an educational implementation, not a performance-optimized deep-learning library.

## Releases

The automated [Quality workflow](.github/workflows/quality.yml) checks pushes and pull requests against the test suite, synthetic training smoke tests, Ruff, formatting, and strict Mypy. For a published version, the latest successful checks are reviewed first, then the release is created manually from GitHub so the account publishing the release is explicit. See [Release Process](docs/RELEASING.md).

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
