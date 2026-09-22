"""Input boundary only. No camera is opened and no desktop events emitted."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Observation:
    timestamp: float
    x: float = 0.5
    y: float = 0.5
    valid: bool = True
    left_closed: bool = False
    right_closed: bool = False
    mouth_open: bool = False

    def usable(self):
        values = (self.timestamp, self.x, self.y)
        flags = (self.valid, self.left_closed, self.right_closed, self.mouth_open)
        return (all(type(v) in (int, float) and math.isfinite(v) for v in values)
                and all(type(v) is bool for v in flags)
                and self.timestamp >= 0 and 0 <= self.x <= 1
                and 0 <= self.y <= 1 and self.valid)
