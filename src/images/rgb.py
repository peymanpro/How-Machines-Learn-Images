from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RGBPixel:
    red: int
    green: int
    blue: int

    def __post_init__(self) -> None:
        for channel_name, value in (
            ("red", self.red),
            ("green", self.green),
            ("blue", self.blue),
        ):
            if not 0 <= value <= 255:
                raise ValueError(f"{channel_name} channel must be between 0 and 255.")

    @property
    def normalized(self) -> tuple[float, float, float]:
        return (
            self.red / 255.0,
            self.green / 255.0,
            self.blue / 255.0,
        )
