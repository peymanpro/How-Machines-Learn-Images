from pathlib import Path
import struct

import numpy as np
import pytest

from src.data.mnist import load_mnist


def _write_idx_fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "images.idx3-ubyte"
    label_path = tmp_path / "labels.idx1-ubyte"

    image_bytes = struct.pack(">IIII", 2051, 2, 2, 2) + bytes([0, 255, 128, 64, 10, 20, 30, 40])
    label_bytes = struct.pack(">II", 2049, 2) + bytes([3, 7])

    image_path.write_bytes(image_bytes)
    label_path.write_bytes(label_bytes)
    return image_path, label_path


def test_load_mnist_fixture(tmp_path: Path) -> None:
    image_path, label_path = _write_idx_fixture(tmp_path)

    images, labels = load_mnist(image_path, label_path)

    assert images.shape == (2, 1, 2, 2)
    assert labels.tolist() == [3, 7]
    assert images[0, 0, 0, 1] == pytest.approx(1.0)


def test_load_mnist_rejects_invalid_magic(tmp_path: Path) -> None:
    image_path, label_path = _write_idx_fixture(tmp_path)
    image_path.write_bytes(struct.pack(">IIII", 0, 2, 2, 2))

    with pytest.raises(ValueError, match="magic"):
        load_mnist(image_path, label_path)
