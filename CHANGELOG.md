# Changelog

## 1.0.0 — Release-ready

### Added

- complete from-scratch image representation path
- vector, matrix and coordinate mathematics
- convolution and pooling primitives
- activations, losses, trainable layers and SGD
- explicit backpropagation
- analytical/numerical gradient checks
- synthetic image datasets
- MNIST IDX loader and training entry point
- activation and parameter inspection
- controlled experiments
- optional PyTorch numerical validation
- interactive convolution demo
- automated quality workflow
- architecture and validation documentation

### Notes

The final CI baseline is 212 passing tests with lint, formatting, strict typing and training smoke checks green.

This release is educational by design. The implementation favors transparency and testability over optimized kernels and large-scale training infrastructure.
