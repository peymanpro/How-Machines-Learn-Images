from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pixel:
    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value <= 255:
            raise ValueError("Pixel value must be between 0 and 255.")