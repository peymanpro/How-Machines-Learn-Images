import numpy as np
import pytest

from src.math.matrix import Matrix
from src.math.vector import Vector


def test_matrix_transforms_vector() -> None:
    transformation = Matrix(
        np.array(
            [
                [2.0, 0.0],
                [0.0, 3.0],
            ]
        )
    )
    vector = Vector(np.array([1.0, 2.0]))

    result = transformation.transform(vector)

    assert result.components == (2.0, 6.0)


def test_matrix_transform_preserves_dimension_rule() -> None:
    transformation = Matrix(np.eye(3))
    vector = Vector(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="match vector dimension"):
        transformation.transform(vector)


def test_matrix_transform_supports_reduction() -> None:
    transformation = Matrix(np.array([[1.0, 2.0, 3.0]]))
    vector = Vector(np.array([1.0, 0.0, 2.0]))

    result = transformation.transform(vector)

    assert result.components == (7.0,)
