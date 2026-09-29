import numpy as np
import pytest

from src.data.mnist import LABEL_MAGIC, load_mnist
from src.data.synthetic import generate_line_dataset, generate_shape_dataset, train_test_split


def test_line_dataset_shape_and_labels() -> None:
    inputs, labels = generate_line_dataset(samples=12, size=10, seed=1)

    assert inputs.shape == (12, 1, 10, 10)
    assert labels.shape == (12,)
    assert set(labels.tolist()) == {0, 1}
    assert np.all((inputs >= 0.0) & (inputs <= 1.0))


def test_shape_dataset_is_two_class() -> None:
    inputs, labels = generate_shape_dataset(samples=12, size=12, seed=1)

    assert inputs.shape == (12, 1, 12, 12)
    assert set(labels.tolist()) == {0, 1}


def test_train_test_split_preserves_alignment() -> None:
    inputs = np.arange(20.0).reshape(10, 2)
    labels = np.arange(10, dtype=np.int64)

    train_x, test_x, train_y, test_y = train_test_split(inputs, labels, seed=1)

    assert train_x.shape[0] == train_y.shape[0]
    assert test_x.shape[0] == test_y.shape[0]
    assert set(train_y.tolist() + test_y.tolist()) == set(labels.tolist())


def test_train_test_split_rejects_mismatched_samples() -> None:
    with pytest.raises(ValueError, match="matching sample counts"):
        train_test_split(np.ones((3, 2)), np.ones(2, dtype=np.int64))


def test_mnist_module_exposes_idx_magic() -> None:
    assert LABEL_MAGIC == 2049
