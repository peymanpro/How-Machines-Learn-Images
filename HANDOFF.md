# Handoff

Project: How-Machines-Learn-Images
Milestone: M3 — Teach a Machine to Learn Images
Phase: Phase 18 — Release Preparation
Status: Complete / release-ready

The repository is intended to be understandable from first principles:

Image → Mathematics → Convolution → Neuron → Loss → Gradient → Backpropagation → CNN → Experiments

## Verified capabilities

- framework-free NumPy implementation of the learning core
- analytical and numerical gradient checks
- synthetic image classification
- MNIST IDX loading and training entry point
- activation/parameter inspection
- optional PyTorch cross-validation
- interactive browser demo
- automated CI for tests, linting, formatting and strict typing

## Final verification

Run:

```bash
python -m pytest -q
ruff check .
ruff format --check .
mypy src
python -m scripts.train_lines
python -m scripts.run_experiments
```

Repository is the source of truth.
