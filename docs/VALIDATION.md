# Validation

The project treats validation as part of the implementation, not as a final decoration.

## Required checks

### Tests

```bash
python -m pytest -q
```

The test suite covers image primitives, mathematical operations, convolution/pooling, activations, losses, layers, model composition, data loading, synthetic datasets and learning behavior.

### Ruff

```bash
ruff check .
ruff format --check .
```

Linting and formatting are required for every main-branch change.

### Mypy

```bash
mypy src
```

The source tree is checked in strict mode.

## Learning-specific evidence

### Dense gradient check

The analytical dense-layer gradient is compared with a central finite-difference numerical gradient.

### Convolution gradient check

The convolution weights are checked against finite differences through a loss that includes the complete forward path.

### Learning behavior

The repository verifies that:

- SGD changes parameters in the expected direction
- the XOR network reduces loss
- a small CNN can reduce loss on a synthetic image task

These are behavioral tests, not claims of production-level model quality.

## Smoke experiments

The CI workflow also executes:

```bash
python -m scripts.train_lines
python -m scripts.run_experiments
```

This protects the user-facing examples from silently becoming stale while the unit tests remain green.

## Optional framework validation

```bash
python -m scripts.validate_with_pytorch
```

This script is deliberately outside the required dependency set.

It checks:

- convolution forward output
- convolution weight gradients
- input gradients

against PyTorch for the same values and settings.

## Reproducibility

Experiments use explicit random seeds.

The goal is not to claim that a single run is scientifically definitive. The goal is to make behavior inspectable and repeatable.
