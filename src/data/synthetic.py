from __future__ import annotations

import numpy as np


def _split_indices(
    sample_count: int,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    indices = np.arange(sample_count)
    rng.shuffle(indices)
    split = int(sample_count * train_fraction)
    return indices[:split], indices[split:]


def generate_line_dataset(
    samples: int = 200,
    size: int = 16,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate horizontal-vs-vertical line images for the first learning task."""
    if samples < 2:
        raise ValueError("At least two samples are required.")
    if size < 5:
        raise ValueError("Image size must be at least 5.")
    if noise < 0.0:
        raise ValueError("Noise must be non-negative.")

    rng = np.random.default_rng(seed)
    inputs = rng.normal(0.0, noise, size=(samples, 1, size, size))
    labels = np.arange(samples, dtype=np.int64) % 2
    center = size // 2

    for index, label in enumerate(labels):
        if label == 0:
            inputs[index, 0, center - 1 : center + 2, 2:-2] += 1.0
        else:
            inputs[index, 0, 2:-2, center - 1 : center + 2] += 1.0

    inputs = np.clip(inputs, 0.0, 1.0)
    shuffle = rng.permutation(samples)
    return inputs[shuffle], labels[shuffle]


def generate_shape_dataset(
    samples: int = 200,
    size: int = 20,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a harder circle-vs-square classification dataset."""
    if samples < 2:
        raise ValueError("At least two samples are required.")
    if size < 9:
        raise ValueError("Image size must be at least 9.")
    if noise < 0.0:
        raise ValueError("Noise must be non-negative.")

    rng = np.random.default_rng(seed)
    inputs = rng.normal(0.0, noise, size=(samples, 1, size, size))
    labels = np.arange(samples, dtype=np.int64) % 2
    center = (size - 1) / 2.0
    radius = size * 0.27

    yy, xx = np.mgrid[:size, :size]
    distance = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)

    for index, label in enumerate(labels):
        if label == 0:
            inputs[index, 0] += (np.abs(distance - radius) <= 0.8).astype(float)
        else:
            half_width = size * 0.24
            dx = np.abs(xx - center)
            dy = np.abs(yy - center)
            inside = (dx <= half_width) & (dy <= half_width)
            boundary = inside & (
                (np.abs(dx - half_width) <= 0.8) | (np.abs(dy - half_width) <= 0.8)
            )
            inputs[index, 0] += boundary.astype(float)

    inputs = np.clip(inputs, 0.0, 1.0)
    shuffle = rng.permutation(samples)
    return inputs[shuffle], labels[shuffle]


def train_test_split(
    inputs: np.ndarray,
    labels: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split aligned tensors and labels into deterministic train/test partitions."""
    if inputs.shape[0] != labels.shape[0]:
        raise ValueError("Inputs and labels must have matching sample counts.")
    rng = np.random.default_rng(seed)
    train_indices, test_indices = _split_indices(inputs.shape[0], train_fraction, rng)
    return (
        inputs[train_indices],
        inputs[test_indices],
        labels[train_indices],
        labels[test_indices],
    )
