from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageTensor:
    data: np.ndarray

    def __post_init__(self) -> None:
        if self.data.ndim not in (2, 3):
            raise ValueError("ImageTensor requires a 2D or 3D array.")
        if self.data.size == 0:
            raise ValueError("ImageTensor cannot be empty.")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dimension) for dimension in self.data.shape)

    @property
    def dimensions(self) -> int:
        return int(self.data.ndim)
