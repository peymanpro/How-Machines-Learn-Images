from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049


def _read_header(handle: BinaryIO, format_string: str) -> tuple[int, ...]:
    size = struct.calcsize(format_string)
    raw = handle.read(size)
    if len(raw) != size:
        raise ValueError("MNIST file ended before its header was complete.")
    return struct.unpack(format_string, raw)


def load_mnist_images(path: str | Path) -> np.ndarray:
    """Load an IDX3-ubyte MNIST image file as float64 NCHW data."""
    image_path = Path(path)
    with image_path.open("rb") as handle:
        magic, count, rows, columns = _read_header(handle, ">IIII")
        if magic != IMAGE_MAGIC:
            raise ValueError("Invalid MNIST image magic number.")
        raw = handle.read(count * rows * columns)
        if len(raw) != count * rows * columns:
            raise ValueError("MNIST image file is truncated.")

    values = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) / 255.0
    return values.reshape(count, 1, rows, columns)


def load_mnist_labels(path: str | Path) -> np.ndarray:
    """Load an IDX1-ubyte MNIST label file."""
    label_path = Path(path)
    with label_path.open("rb") as handle:
        magic, count = _read_header(handle, ">II")
        if magic != LABEL_MAGIC:
            raise ValueError("Invalid MNIST label magic number.")
        raw = handle.read(count)
        if len(raw) != count:
            raise ValueError("MNIST label file is truncated.")

    return np.frombuffer(raw, dtype=np.uint8).astype(np.int64)


def load_mnist(
    image_path: str | Path,
    label_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a normalized MNIST image tensor and integer labels."""
    images = load_mnist_images(image_path)
    labels = load_mnist_labels(label_path)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("MNIST image and label counts do not match.")
    return images, labels
