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
        return self.norm()

    def norm(self, order: float = 2.0) -> float:
        """Return the vector p-norm for a finite p >= 1."""
        if not np.isfinite(order) or order < 1:
            raise ValueError("Norm order must be finite and at least 1.")

        return float(np.sum(np.abs(self.values) ** order) ** (1.0 / order))

    def dot(self, other: Vector) -> float:
        """Return the dot product with another vector."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension.")

        return float(np.sum(self.values * other.values))

    def distance_to(self, other: Vector) -> float:
        """Return the Euclidean distance to another vector."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension.")

        difference = self.values - other.values
        return float(np.sqrt(np.sum(difference * difference)))