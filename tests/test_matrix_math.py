import numpy as np
import pytest

from src.math.matrix import Matrix


def test_matrix_stores_shape() -> None:
    matrix = Matrix(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
    )

    assert matrix.rows == 2
    assert matrix.columns == 3
    assert matrix.shape == (2, 3)


def test_matrix_element_access() -> None:
    matrix = Matrix(
        np.array(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ]
        )
    )

    assert matrix.element(0, 0) == 10.0
    assert matrix.element(1, 1) == 40.0


def test_matrix_supports_integer_values() -> None:
    matrix = Matrix(np.array([[1, 2], [3, 4]], dtype=np.int64))

    assert matrix.shape == (2, 2)
    assert matrix.element(1, 0) == 3.0


def test_matrix_rejects_non_matrix_array() -> None:
    with pytest.raises(ValueError, match="2D"):
        Matrix(np.zeros(3))


def test_matrix_rejects_empty_array() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        Matrix(np.empty((0, 2)))


def test_matrix_rejects_non_numeric_values() -> None:
    with pytest.raises(TypeError, match="numeric"):
        Matrix(np.array([["a", "b"]]))


@pytest.mark.parametrize(
    ("row", "column"),
    [
        (-1, 0),
        (2, 0),
        (0, -1),
        (0, 2),
    ],
)
def test_matrix_element_rejects_out_of_bounds(row: int, column: int) -> None:
    matrix = Matrix(np.zeros((2, 2)))

    with pytest.raises(IndexError, match="out of bounds"):
        matrix.element(row, column)