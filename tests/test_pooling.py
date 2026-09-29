import numpy as np
import pytest

from src.vision.pooling import max_pool2d


def test_max_pooling_selects_local_maxima() -> None:
    image = np.array(
        [
            [1.0, 4.0, 2.0, 0.0],
            [3.0, 5.0, 1.0, 2.0],
            [0.0, 2.0, 8.0, 6.0],
            [1.0, 3.0, 4.0, 7.0],
        ]
    )

    result = max_pool2d(image)

    assert np.array_equal(result, np.array([[5.0, 2.0], [3.0, 8.0]]))


def test_max_pooling_supports_overlap() -> None:
    image = np.arange(1.0, 10.0).reshape(3, 3)

    result = max_pool2d(image, pool_size=2, stride=1)

    assert result.shape == (2, 2)
    assert result[0, 0] == 5.0
    assert result[1, 1] == 9.0


@pytest.mark.parametrize(
    ("pool_size", "stride"),
    [(0, 1), (2, 0)],
)
def test_pooling_rejects_invalid_geometry(pool_size: int, stride: int) -> None:
    with pytest.raises(ValueError):
        max_pool2d(np.ones((4, 4)), pool_size=pool_size, stride=stride)
