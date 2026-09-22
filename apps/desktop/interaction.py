"""Deterministic safety/motion/gesture core; normalized practice coordinates."""
import math
from .config import Settings
from .vision import Observation


class MotionFilter:
    def __init__(self, settings, absolute=False):
        self.settings = settings
        self.absolute = absolute
        self.reset()

    def reset(self):
        self.origin = None
        self.position = (0.5, 0.5)
        self.last_time = None

    def update(self, observation):
        if self.origin is None:
            self.origin = (.5, .5) if self.absolute else (observation.x, observation.y)
            self.last_time = observation.timestamp
            if self.absolute:
                self.position = (observation.x, observation.y)
            return self.position
        dt = observation.timestamp - self.last_time
        self.last_time = observation.timestamp
        tau = self.settings.smoothing_seconds
        alpha = 1 if tau == 0 else -math.expm1(-dt / tau)
        output = []
        for raw, origin, previous, gain, deadzone in zip(
            (observation.x, observation.y), self.origin, self.position,
            (self.settings.gain_x, self.settings.gain_y),
            (self.settings.deadzone_x, self.settings.deadzone_y),
        ):
            target = min(1.0, max(0.0, 0.5 + (raw - origin) * gain))
            output.append(previous if abs(target - previous) <= deadzone
                          else previous + alpha * (target - previous))
        self.position = tuple(output)
        return self.position


class HoldGesture:
    """One action per hold; neutral release required even after invalidation."""
    def __init__(self, seconds):
        self.seconds = seconds
        self.reset()

    def reset(self):
        self.armed = False
        self.candidate = None
        self.since = None

    def update(self, gesture, now):
        if gesture is None:
            self.armed = True
            self.candidate = self.since = None
            return None
        if not self.armed:
            return None
        if gesture != self.candidate:
            if self.candidate is not None:
                self.reset()
                return None
            self.candidate, self.since = gesture, now
        if now - self.since >= self.seconds:
            self.reset()
            return gesture
        return None


class Controller:
    """Explicit resume; loss, stale frames, pause, fault all invalidate actions."""
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.motion = MotionFilter(self.settings)
        self.gesture = HoldGesture(self.settings.hold_seconds)
        self.paused = True
        self.tracking = False
        self.closed = False
        self.error = None
        self.recording = False
        self.last_time = None
        self.epoch = 0

    @property
    def can_practice(self):
        return not self.closed and not self.error and not self.paused and self.tracking

    def pause(self):
        self.paused = True
        self.epoch += 1
        self.motion.reset()
        self.gesture.reset()

    def resume(self):
        if self.closed or self.error or not self.tracking:
            return False
        self.pause()  # invalidate all old proposed actions and require release
        self.paused = False
        return True

    def lose_tracking(self):
        self.tracking = False
        self.pause()

    def fail(self, message):
        self.error = str(message)
        self.lose_tracking()

    def close(self):
        self.closed = True
        self.lose_tracking()

    def configure(self, settings):
        if not isinstance(settings, Settings):
            raise ValueError('需要有效Settings')
        self.pause()
        self.settings = settings
        self.motion = MotionFilter(settings)
        self.gesture = HoldGesture(settings.hold_seconds)

    def set_recording(self, enabled):
        self.recording = bool(enabled)
        self.pause()

    def watchdog(self, now):
        if self.last_time is not None and now - self.last_time > self.settings.max_frame_gap:
            if self.tracking:
                self.lose_tracking()

    def observe(self, obs):
        if self.closed:
            return None
        if not obs.usable():
            self.lose_tracking()
            return None
        if self.last_time is not None:
            dt = obs.timestamp - self.last_time
            if dt <= 0:
                self.lose_tracking()
                return None
            if dt > self.settings.max_frame_gap:
                self.lose_tracking()
        self.last_time = obs.timestamp
        self.tracking = True
        if obs.mouth_open and not self.recording:
            self.pause()  # closing the mouth NEVER resumes
        if not self.can_practice:
            return None
        if obs.left_closed and obs.right_closed:
            self.gesture.reset()  # never map bilateral blink to double-click
            return self.motion.position, None
        gesture = ('left' if obs.left_closed else 'right' if obs.right_closed else None)
        # Closed eyes must not distort motion. Hold gesture is optional practice.
        position = self.motion.position if gesture else self.motion.update(obs)
        action = self.gesture.update(gesture, obs.timestamp)
        return position, action
