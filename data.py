"""The record a detector returns. Used by every part of the project."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    label: str
    confidence: float
    pixel_position: tuple[float, float]
    # (x1, y1, x2, y2) in image pixels; None for detectors that only give a point.
    box: Optional[tuple[float, float, float, float]] = None
