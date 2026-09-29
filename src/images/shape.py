from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageShape:
    height: int
    width: int
    channels: int = 1

    def __post_init__(self) -> None:
        if self.height <= 0:
            raise ValueError("Image height must be positive.")
        if self.width <= 0:
            raise ValueError("Image width must be positive.")
        if self.channels <= 0:
            raise ValueError("Image channels must be positive.")

    @property
    def dimensions(self) -> tuple[int, ...]:
        if self.channels == 1:
            return (self.height, self.width)
        return (self.height, self.width, self.channels)

    @property
    def size(self) -> int:
        return self.height * self.width * self.channels
