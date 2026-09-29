import pytest

from src.math.coordinates import Point2D, Vector2D, displacement


def test_displacement_from_first_point_to_second() -> None:
    first = Point2D(1.0, 2.0)
    second = Point2D(4.0, 6.0)

    assert displacement(first, second) == Vector2D(3.0, 4.0)


def test_vector_magnitude() -> None:
    assert Vector2D(3.0, 4.0).magnitude() == 5.0


def test_vector_translates_point() -> None:
    point = Point2D(1.0, 2.0)
    vector = Vector2D(3.0, -1.0)

    assert vector.add_to(point) == Point2D(4.0, 1.0)


def test_point_scale() -> None:
    assert Point2D(2.0, -3.0).scale(2.0) == Point2D(4.0, -6.0)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (Point2D(0.0, 0.0), Point2D(1.0, 1.0), Vector2D(1.0, 1.0)),
        (Point2D(-2.0, 3.0), Point2D(4.0, -1.0), Vector2D(6.0, -4.0)),
    ],
)
def test_displacement_is_second_minus_first(
    first: Point2D, second: Point2D, expected: Vector2D
) -> None:
    assert displacement(first, second) == expected
