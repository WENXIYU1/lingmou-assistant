"""Validated parameters. Defaults are experimental, not measured optima."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Settings:
    gain_x: float = 1.0
    gain_y: float = 1.0
    deadzone_x: float = 0.008
    deadzone_y: float = 0.008
    smoothing_seconds: float = 0.10
    hold_seconds: float = 0.65
    max_frame_gap: float = 0.25

    def __post_init__(self):
        limits = {
            'gain_x': (0.1, 4.0), 'gain_y': (0.1, 4.0),
            'deadzone_x': (0, 0.1), 'deadzone_y': (0, 0.1),
            'smoothing_seconds': (0, 1), 'hold_seconds': (0.3, 3),
            'max_frame_gap': (0.05, 0.5),
        }
        for key, (low, high) in limits.items():
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{key}: 必须为有限数值')
            if not low <= value <= high:
                raise ValueError(f'{key}: 范围 {low}–{high}')

    def to_dict(self):
        return asdict(self)
