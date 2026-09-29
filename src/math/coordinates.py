from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Point2D:
    """A point in the two-dimensional Euclidean plane."""

    x: float
    y: float

    def translate(self, dx: float, dy: float) -> Point2D:
        """Return the point translated by a displacement vector."""
        return Point2D(self.x + dx, self.y + dy)

    def scale(self, factor: float) -> Point2D:
        """Scale the point relative to the origin."""
        return Point2D(self.x * factor, self.y * factor)


@dataclass(frozen=True)
class Vector2D:
    """A two-dimensional displacement vector."""

    x: float
    y: float

    def magnitude(self) -> float:
        return hypot(self.x, self.y)

    def add_to(self, point: Point2D) -> Point2D:
        """Translate a point by this vector."""
        return point.translate(self.x, self.y)


def displacement(first: Point2D, second: Point2D) -> Vector2D:
    """Return the displacement from first to second."""
    return Vector2D(second.x - first.x, second.y - first.y)
