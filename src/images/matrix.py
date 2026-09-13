from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageMatrix:
    data: np.ndarray

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("ImageMatrix requires a 2D array.")
        if self.data.size == 0:
            raise ValueError("ImageMatrix cannot be empty.")

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape
