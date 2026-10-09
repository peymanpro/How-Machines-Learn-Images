# How Machines Learn Images v1.0.0

The first stable release of **How Machines Learn Images**, a Python and NumPy implementation that exposes the path from image pixels to a trainable convolutional neural network.

## Included

- Image, color-channel, matrix, vector, tensor, normalization, resizing, and inspection primitives
- Convolution and cross-correlation, padding, stride, edge kernels, and max pooling
- Dense, convolutional, activation, flatten, loss, and SGD components
- Explicit forward propagation and backpropagation with analytical gradients
- Central finite-difference gradient checks for dense and convolutional operations
- Deterministic synthetic-image datasets and training examples
- An MNIST IDX loader and reproducible training entry point
- Activation and parameter inspection, plus controlled learning-rate experiments
- Optional numerical validation against PyTorch
- Interactive browser-based convolution demonstration
- Automated tests, synthetic training smoke checks, Ruff lint/format checks, and strict Mypy

## Verification

This release is gated on the release workflow completing all of the following successfully:

- Full pytest suite
- Synthetic training smoke commands
- Ruff linting and formatting checks
- Strict Mypy checking of `src`

The latest completed quality run on the project reported **212 passing tests**. The release workflow re-runs the full quality sequence on the tagged release commit before publishing.

## Scope and limitations

The core implementation intentionally uses explicit NumPy operations for transparency rather than high-performance kernels. The MNIST IDX files are not bundled in the repository and must be supplied by the user. Synthetic accuracy results are educational checks, not production benchmarks or evidence of generalization to real-world data.

## License

MIT License.
