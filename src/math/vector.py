from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Vector:
    """A finite-dimensional mathematical vector."""

    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("Vector values must be a 1D array.")

        if self.values.size == 0:
            raise ValueError("Vector cannot be empty.")

        if not np.issubdtype(self.values.dtype, np.number):
            raise TypeError("Vector values must be numeric.")

    @property
    def dimension(self) -> int:
        return int(self.values.shape[0])

    @property
    def components(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.values)

    @property
    def magnitude(self) -> float:
        return float(np.sqrt(np.sum(self.values * self.values)))

    def dot(self, other: Vector) -> float:
        """Return the dot product with another vector."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension.")

        return float(np.sum(self.values * other.values))