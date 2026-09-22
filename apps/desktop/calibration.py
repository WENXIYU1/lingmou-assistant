"""Local five-point affine fitting and held-out validation for a practice canvas."""
from dataclasses import dataclass
import math
from statistics import median
from .eye_features import FeatureFrame, head_matches

TRAIN_TARGETS = ((.5, .5), (.15, .2), (.85, .2), (.85, .8), (.15, .8))
CHECK_TARGETS = ((.3, .5), (.7, .5), (.5, .3))
POLICY_VERSION = 'experimental-canvas-v1'
MAPPING_VERSION = 'affine-standardized-v1'
# Fractions of canvas diagonal. These are trial gates, NOT measured accuracy claims.
MEDIAN_LIMIT = .10
MAX_LIMIT = .18


@dataclass(frozen=True)
class Mapping:
    center: tuple
    scale: tuple
    coefficients: tuple
    head: tuple

    def predict(self, features, settings=None):
        if len(features) != 2 or not all(math.isfinite(x) for x in features):
            raise ValueError('无效特征')
        row = ((features[0]-self.center[0])/self.scale[0],
               (features[1]-self.center[1])/self.scale[1], 1)
        xy = tuple(sum(row[i] * self.coefficients[i][j] for i in range(3)) for j in range(2))
        if settings:
            xy = (.5+(xy[0]-.5)*settings.gain_x, .5+(xy[1]-.5)*settings.gain_y)
        return xy


def fit_mapping(groups):
    import numpy as np
    if len(groups) != 5 or any(len(group) < 18 for group in groups):
        raise ValueError('五点采样不完整')
    if any(not sample.usable() for group in groups for sample in group):
        raise ValueError('采样包含无效帧')
    anchor = tuple(median(s.head[i] for s in groups[0]) for i in range(5))
    if any(not head_matches(anchor, s.head) for group in groups for s in group):
        raise ValueError('采样期间姿态变化过大，请重新校准')
    points = []
    for group in groups:
        values = np.array([s.features for s in group])
        center = np.median(values, axis=0)
        distances = np.linalg.norm(values-center, axis=1)
        keep = values[distances <= max(.004, float(np.median(distances))*3)]
        if len(keep) < 18 or float(np.max(np.std(keep, axis=0))) > .035:
            raise ValueError('采样不稳定，请重采')
        points.append(np.median(keep, axis=0))
    features = np.array(points)
    center, scale = features.mean(axis=0), features.std(axis=0)
    if np.any(scale < .002):
        raise ValueError('视线变化不足，无法可靠拟合')
    design = np.column_stack(((features-center)/scale, np.ones(5)))
    if np.linalg.matrix_rank(design) < 3 or np.linalg.cond(design) > 100:
        raise ValueError('采样退化，无法拟合二维映射')
    coefficients, _, _, _ = np.linalg.lstsq(design, np.array(TRAIN_TARGETS), rcond=None)
    if not np.all(np.isfinite(coefficients)) or np.max(np.abs(coefficients)) > 10:
        raise ValueError('映射系数异常')
    return Mapping(tuple(float(v) for v in center), tuple(float(v) for v in scale),
                   tuple(tuple(float(v) for v in row) for row in coefficients), anchor)


def validate_mapping(mapping, groups, settings, canvas_size):
    import numpy as np
    if len(groups) != len(CHECK_TARGETS) or any(len(g) < 18 for g in groups):
        raise ValueError('独立验证采样不完整')
    width, height = canvas_size
    if width <= 0 or height <= 0:
        raise ValueError('无效画布尺寸')
    errors = []
    for target, group in zip(CHECK_TARGETS, groups):
        for sample in group:
            if not sample.usable() or not head_matches(mapping.head, sample.head):
                raise ValueError('验证期间姿态或跟踪质量无效')
            x, y = mapping.predict(sample.features, settings)
            # Never clip validation output; clipping could hide a bad fit.
            errors.append(math.hypot((x-target[0])*width, (y-target[1])*height)/math.hypot(width, height))
    metrics = {'median_error': float(np.median(errors)), 'p90_error': float(np.quantile(errors, .9)),
               'max_error': float(max(errors)), 'sample_count': len(errors),
               'policy_version': POLICY_VERSION}
    metrics['passed'] = metrics['median_error'] <= MEDIAN_LIMIT and metrics['max_error'] <= MAX_LIMIT
    return metrics


class CalibrationSession:
    """UI-controlled progression with automatic sampling; no eye-click prerequisite."""
    def __init__(self, settings, canvas_size, mapping=None):
        self.settings, self.canvas_size = settings, canvas_size
        self.mapping = mapping
        self.phase = 'verify' if mapping else 'train'
        self.groups = []
        self.buffer = []
        self.started = None
        self.last_time = None
        self.index = 0
        self.error = ''
        self.metrics = None
        self.anchor = mapping.head if mapping else None

    @property
    def target(self):
        return (TRAIN_TARGETS if self.phase == 'train' else CHECK_TARGETS)[self.index]

    def begin_point(self, now):
        if self.phase not in ('train', 'verify'):
            return
        self.started, self.last_time = now, None
        self.buffer = []

    def pause(self):
        self.started = None
        self.buffer = []
        self.last_time = None

    def feed(self, frame):
        if self.started is None or self.phase not in ('train', 'verify'):
            return False
        if frame.timestamp - self.started > 20:
            self.pause()
            self.error = '本点采样超时；请休息或调整位置后重采'
            return False
        gap = self.last_time is not None and (frame.timestamp <= self.last_time
                                              or frame.timestamp-self.last_time > .25)
        self.last_time = frame.timestamp
        if not frame.usable() or gap or (self.anchor and not head_matches(self.anchor, frame.head)):
            self.buffer = []
            return False
        if frame.timestamp-self.started < 1.0:
            return False  # settle after target movement
        self.buffer.append(frame)
        if len(self.buffer) > 240:
            self.buffer = self.buffer[-240:]
        if len(self.buffer) < 18 or self.buffer[-1].timestamp-self.buffer[0].timestamp < 1.2:
            return False
        for axis in (0, 1):
            values = [s.features[axis] for s in self.buffer]
            mid = median(values)
            if median(abs(v-mid) for v in values) > .025:
                self.buffer = []
                return False
        if self.anchor is None:
            self.anchor = tuple(median(s.head[i] for s in self.buffer) for i in range(5))
        self.groups.append(self.buffer[:])
        self.index += 1
        self.pause()
        count = 5 if self.phase == 'train' else 3
        if self.index == count:
            try:
                if self.phase == 'train':
                    self.mapping = fit_mapping(self.groups)
                    self.phase, self.groups, self.index = 'verify', [], 0
                else:
                    self.metrics = validate_mapping(self.mapping, self.groups, self.settings, self.canvas_size)
                    self.phase = 'trial' if self.metrics['passed'] else 'failed'
                    if self.phase == 'failed':
                        self.error = '独立验证未通过；不能保存有效档案，请重新校准'
            except ValueError as exc:
                self.error, self.phase = str(exc), 'failed'
        return True


@dataclass(frozen=True)
class AdaptationStatus:
    status: str = 'not_calibrated'
    message: str = '当前未验证；摄像头工作区可进行五点校准。本阶段只开放画布练习。'

    @property
    def system_control_allowed(self):
        # Even successful practice calibration never enables OS injection in v2.
        return False
