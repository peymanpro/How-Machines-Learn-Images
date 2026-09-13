import numpy as np
import pytest

from src.math.vector import Vector


def test_vector_stores_components() -> None:
    vector = Vector(np.array([1.0, 2.0, 3.0]))

    assert vector.dimension == 3
    assert vector.components == (1.0, 2.0, 3.0)


def test_vector_magnitude() -> None:
    vector = Vector(np.array([3.0, 4.0]))

    assert vector.magnitude == 5.0


def test_vector_norm_default_is_euclidean() -> None:
    vector = Vector(np.array([3.0, 4.0]))

    assert vector.norm() == 5.0


def test_vector_l1_norm() -> None:
    vector = Vector(np.array([-3.0, 4.0]))

    assert vector.norm(order=1) == 7.0


def test_vector_l3_norm() -> None:
    vector = Vector(np.array([2.0, 2.0]))

    assert vector.norm(order=3) == pytest.approx(16 ** (1 / 3))


def test_vector_magnitude_matches_euclidean_norm() -> None:
    vector = Vector(np.array([1.0, -2.0, 2.0]))

    assert vector.magnitude == vector.norm(order=2)


def test_vector_distance() -> None:
    first = Vector(np.array([1.0, 2.0]))
    second = Vector(np.array([4.0, 6.0]))

    assert first.distance_to(second) == 5.0


def test_vector_distance_is_symmetric() -> None:
    first = Vector(np.array([1.0, -2.0, 3.0]))
    second = Vector(np.array([4.0, 5.0, -6.0]))

    assert first.distance_to(second) == second.distance_to(first)


def test_vector_dot_product() -> None:
    first = Vector(np.array([1.0, 2.0, 3.0]))
    second = Vector(np.array([4.0, 5.0, 6.0]))

    assert first.dot(second) == 32.0


def test_vector_dot_product_is_symmetric() -> None:
    first = Vector(np.array([1.0, -2.0, 3.0]))
    second = Vector(np.array([4.0, 5.0, -6.0]))

    assert first.dot(second) == second.dot(first)


def test_vector_dot_product_rejects_different_dimensions() -> None:
    first = Vector(np.array([1.0, 2.0]))
    second = Vector(np.array([3.0, 4.0, 5.0]))

    with pytest.raises(ValueError, match="same dimension"):
        first.dot(second)


def test_vector_norm_rejects_invalid_order() -> None:
    vector = Vector(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="at least 1"):
        vector.norm(order=0.5)


def test_vector_norm_rejects_non_finite_order() -> None:
    vector = Vector(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="finite"):
        vector.norm(order=np.inf)


def test_vector_distance_rejects_different_dimensions() -> None:
    first = Vector(np.array([1.0, 2.0]))
    second = Vector(np.array([3.0, 4.0, 5.0]))

    with pytest.raises(ValueError, match="same dimension"):
        first.distance_to(second)


def test_vector_supports_integer_values() -> None:
    vector = Vector(np.array([1, 2, 3], dtype=np.int64))

    assert vector.dimension == 3
    assert vector.components == (1.0, 2.0, 3.0)


def test_vector_rejects_non_vector_array() -> None:
    with pytest.raises(ValueError, match="1D"):
        Vector(np.zeros((2, 2)))


def test_vector_rejects_empty_array() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        Vector(np.array([], dtype=np.float64))


def test_vector_rejects_non_numeric_values() -> None:
    with pytest.raises(TypeError, match="numeric"):
        Vector(np.array(["a", "b"]))


def test_vector_dimension_matches_number_of_components() -> None:
    vector = Vector(np.array([7.0, 8.0, 9.0, 10.0]))

    assert vector.dimension == 4