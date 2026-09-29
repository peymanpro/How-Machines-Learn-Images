# Project State

Current milestone: M3 — Teach a Machine to Learn Images
Current phase: Phase 18 — Release Preparation
Status: Complete / release-ready

## Completed

- Phase 0 — Foundation
- Phase 1 — Image representation and inspection
- Phase 2 — Mathematical foundations and coordinates
- Phase 3 — Convolution and pooling from scratch
- Phase 4 — Neurons and trainable affine layers
- Phase 5 — Activations, prediction and losses
- Phase 6 — Gradient descent
- Phase 7 — Backpropagation
- Phase 8 — Tiny neural network
- Phase 9 — First synthetic image-learning task
- Phase 10 — CNN from scratch
- Phase 11 — MNIST loader and training entry point
- Phase 12 — Activation and parameter inspection
- Phase 13 — Controlled experiments
- Phase 14 — More difficult synthetic images
- Phase 15 — Optional PyTorch validation
- Phase 16 — Interactive convolution demo
- Phase 17 — Documentation and portfolio polish
- Phase 18 — Release preparation

## Verification policy

The repository's source of truth is Git history plus the required CI workflow.

Every final state must satisfy:

- pytest passes
- Ruff lint passes
- Ruff format check passes
- strict Mypy passes
- synthetic training smoke tests execute successfully

## Important implementation boundary

The educational core is framework-free:

`Python + NumPy`

PyTorch is used only by the optional validation script and is not required to run the main project or its tests.

## Known limitations

- MNIST data files are not stored in the repository; the loader expects external IDX files.
- The convolution and pooling implementations use explicit loops intentionally for clarity, not maximum performance.
- The project is designed to expose the mechanics of learning rather than compete with optimized ML libraries.
- Published accuracy should be interpreted as an educational experiment unless reproduced with the same data, seed and hyperparameters.

Repository HEAD is the authoritative version identifier; see Git history for the exact latest commit.
