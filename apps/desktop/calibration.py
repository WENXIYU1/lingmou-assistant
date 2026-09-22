"""Honest first-step status: calibration is NOT implemented or certified."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptationStatus:
    status: str = 'not_calibrated'
    message: str = '尚未进行五点校准和独立验证；当前仅可安全练习。'

    @property
    def system_control_allowed(self):
        # No UI switch or loaded settings may bypass the unfinished calibration.
        return False
