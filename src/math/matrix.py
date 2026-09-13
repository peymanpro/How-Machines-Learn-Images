from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Matrix:
    """A finite two-dimensional numerical matrix."""

    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Matrix values must be a 2D array.")

        if self.values.size == 0:
            raise ValueError("Matrix cannot be empty.")

        if not np.issubdtype(self.values.dtype, np.number):
            raise TypeError("Matrix values must be numeric.")

    @property
    def rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def columns(self) -> int:
        return int(self.values.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.columns)

    def element(self, row: int, column: int) -> float:
        """Return one matrix element using zero-based indices."""
        if row < 0 or row >= self.rows:
            raise IndexError("Matrix row is out of bounds.")

        if column < 0 or column >= self.columns:
            raise IndexError("Matrix column is out of bounds.")

        return float(self.values[row, column])

    def multiply(self, other: Matrix) -> Matrix:
        """Return the matrix product with another matrix."""
        if self.columns != other.rows:
            raise ValueError(
                "Matrix dimensions are incompatible for multiplication."
            )

        result = self.values @ other.values
        return Matrix(np.asarray(result))