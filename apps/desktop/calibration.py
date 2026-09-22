"""Local five-point affine fitting and held-out validation for a practice canvas."""
from dataclasses import dataclass
import math
from statistics import median
from .eye_features import FeatureFrame, head_matches
from .mapping_diagnostics import MappingDiagnostics

TRAIN_TARGETS = ((.5, .5), (.15, .2), (.85, .2), (.85, .8), (.15, .8))
CHECK_TARGETS = ((.3, .5), (.7, .5), (.5, .3))
POLICY_VERSION = 'experimental-canvas-v1'
MAPPING_VERSION = 'hybrid-candidate-two-check-v3'
# Fractions of canvas diagonal. These are trial gates, NOT measured accuracy claims.
MEDIAN_LIMIT = .10
MAX_LIMIT = .18
SETTLE_SECONDS = 1.0
SAMPLE_SECONDS = 1.2
MIN_SAMPLES = 18


@dataclass(frozen=True)
class Mapping:
    center: tuple
    scale: tuple
    coefficients: tuple
    head: tuple
    feature_mode: str = 'average'
    window_seconds: float = 0.0

    def predict(self, features, settings=None):
        if len(features) != 2 or not all(math.isfinite(x) for x in features):
            raise ValueError('无效特征')
        row = ((features[0]-self.center[0])/self.scale[0],
               (features[1]-self.center[1])/self.scale[1], 1)
        xy = tuple(sum(row[i] * self.coefficients[i][j] for i in range(3)) for j in range(2))
        if settings:
            xy = (.5+(xy[0]-.5)*settings.gain_x, .5+(xy[1]-.5)*settings.gain_y)
        return xy


class MappingFeatureFilter:
    """Causal feature preparation shared by validation and live practice."""
    def __init__(self, mapping):
        self.mapping = mapping
        self.history = []

    def reset(self):
        self.history = []

    def update(self, frame):
        if self.mapping.feature_mode == 'average':
            return frame.features
        if self.mapping.feature_mode != 'average-x-left-y' or len(frame.left_eye) != 2:
            raise ValueError('所选混合特征当前不可用')
        raw = (frame.features[0], frame.left_eye[1])
        if self.history and (frame.timestamp <= self.history[-1][0]
                             or frame.timestamp-self.history[-1][0] > .25):
            self.reset()
        self.history.append((frame.timestamp, raw))
        cutoff = frame.timestamp-self.mapping.window_seconds
        self.history = [(t, v) for t, v in self.history if t >= cutoff]
        return tuple(float(median(v[axis] for _, v in self.history)) for axis in (0, 1))


def prepared_groups(groups, mapping):
    prepared=[]
    for group in groups:
        feature_filter=MappingFeatureFilter(mapping)
        prepared.append([FeatureFrame(s.timestamp,feature_filter.update(s),s.head,s.valid,s.reason,
                                      s.capture_size,s.left_eye,s.right_eye,s.left_lid,s.right_lid,
                                      s.average_lid,s.eye_widths_px)
                         for s in group])
    return prepared


def fit_mapping(groups, reference=None, audit=None, *, stabilize=True,
                feature_mode='average', window_seconds=0.0):
    import numpy as np
    if len(groups) != 5 or any(len(group) < 18 for group in groups):
        raise ValueError('五点采样不完整')
    template=Mapping((0.,0.),(1.,1.),((0.,0.),(0.,0.),(0.,0.)),
                     reference or groups[0][0].head,feature_mode,window_seconds)
    groups=prepared_groups(groups,template)
    if any(not sample.usable() for group in groups for sample in group):
        raise ValueError('采样包含无效帧')
    anchor = reference if reference is not None else tuple(median(s.head[i] for s in groups[0]) for i in range(5))
    if any(not head_matches(anchor, s.head) for group in groups for s in group):
        raise ValueError('采样期间姿态变化过大，请重新校准')
    points = []
    retained=[]
    failures=[]
    for index,group in enumerate(groups,1):
        values = np.array([s.features for s in group])
        center = np.median(values, axis=0)
        distances = np.linalg.norm(values-center, axis=1)
        keep = values[distances <= max(.004, float(np.median(distances))*3)]
        spread=np.std(keep, axis=0)
        if audit is not None:
            audit.append({'point':index,'input':len(values),'kept':len(keep),
                          'removed':len(values)-len(keep),'std':tuple(float(v) for v in spread)})
        if len(keep) < 18:
            failures.append(f'点{index}：剔除后样本不足，保留{len(keep)}/{len(values)}帧，需≥18')
        if float(np.max(spread)) > .035:
            failures.append(f'点{index}：保留样本波动超限，X/Y标准差{spread[0]:.5f}/{spread[1]:.5f}，限≤0.035')
        points.append(np.median(keep, axis=0))
        retained.append(keep)
    if failures:
        raise ValueError('；'.join(failures))
    features = np.array(points)
    center, scale = features.mean(axis=0), features.std(axis=0)
    if np.any(scale < .002):
        raise ValueError('视线变化不足，无法可靠拟合')
    design = np.column_stack(((features-center)/scale, np.ones(5)))
    if np.linalg.matrix_rank(design) < 3 or np.linalg.cond(design) > 100:
        raise ValueError('采样退化，无法拟合二维映射')
    if stabilize:
        # Penalize output variance measured within training points. Each target
        # has equal weight, irrespective of how many frames it contributed.
        noise=np.zeros((3,3))
        for point,keep in zip(points,retained):
            residual=(keep-point)/scale
            noise[:2,:2]+=residual.T@residual/len(keep)
        coefficients=np.linalg.solve(design.T@design+noise,design.T@np.array(TRAIN_TARGETS))
    else:
        # Diagnostic baseline only; never selected through validation results.
        coefficients, _, _, _ = np.linalg.lstsq(design, np.array(TRAIN_TARGETS), rcond=None)
    if not np.all(np.isfinite(coefficients)) or np.max(np.abs(coefficients)) > 10:
        raise ValueError('映射系数异常')
    return Mapping(tuple(float(v) for v in center), tuple(float(v) for v in scale),
                   tuple(tuple(float(v) for v in row) for row in coefficients), anchor,
                   feature_mode,float(window_seconds))


def validate_mapping(mapping, groups, settings, canvas_size, diagnostics=None):
    import numpy as np
    if len(groups) != len(CHECK_TARGETS) or any(len(g) < 18 for g in groups):
        raise ValueError('独立验证采样不完整')
    groups=prepared_groups(groups,mapping)
    width, height = canvas_size
    if width <= 0 or height <= 0:
        raise ValueError('无效画布尺寸')
    errors = []
    point_details = []
    for target, group in zip(CHECK_TARGETS, groups):
        offsets = []
        for sample in group:
            if not sample.usable() or not head_matches(mapping.head, sample.head):
                raise ValueError('验证期间姿态或跟踪质量无效')
            x, y = mapping.predict(sample.features, settings)
            # Never clip validation output; clipping could hide a bad fit.
            offsets.append(((x-target[0])*width, (y-target[1])*height))
        offsets = np.array(offsets)
        diagonal = math.hypot(width, height)
        point_errors = np.linalg.norm(offsets, axis=1)/diagonal
        bias = np.median(offsets, axis=0)
        jitter = np.linalg.norm(offsets-bias, axis=1)/diagonal
        errors.extend(point_errors.tolist())
        point_details.append({'point': len(point_details)+1,
                              'median_error': float(np.median(point_errors)),
                              'max_error': float(max(point_errors)),
                              'bias_px': tuple(float(v) for v in bias),
                              'jitter_p90': float(np.quantile(jitter, .9))})
    metrics = {'median_error': float(np.median(errors)), 'p90_error': float(np.quantile(errors, .9)),
               'max_error': float(max(errors)), 'sample_count': len(errors),
               'policy_version': POLICY_VERSION}
    metrics['passed'] = metrics['median_error'] <= MEDIAN_LIMIT and metrics['max_error'] <= MAX_LIMIT
    if diagnostics is not None:
        diagnostics[:] = point_details
    return metrics


def validation_report(metrics, diagnostics):
    """In-memory display only; diagnostics do not change profile schema or gates."""
    lines = ['独立验证' + ('通过（仅画布练习）' if metrics['passed'] else '未通过（系统控制锁定）'),
             f'中位误差 {metrics["median_error"]:.1%} / 门槛 ≤{MEDIAN_LIMIT:.0%}',
             f'最大误差 {metrics["max_error"]:.1%} / 门槛 ≤{MAX_LIMIT:.0%}',
             '误差与波动百分比均相对画布对角线；偏移为画布像素。']
    if diagnostics:
        worst = max(diagnostics, key=lambda p: p['max_error'])['point']
        lines.append(f'最大偏差出现在验证点 {worst}；以下方向是预测相对目标的偏移：')
    for point in diagnostics:
        dx, dy = point['bias_px']
        horizontal = f'{"右" if dx >= 0 else "左"}{abs(dx):.0f}'
        vertical = f'{"下" if dy >= 0 else "上"}{abs(dy):.0f}'
        lines.append(f'点{point["point"]}：中位 {point["median_error"]:.1%} / 最大 {point["max_error"]:.1%}'
                     f'\n  偏移 {horizontal}、{vertical} px；波动P90 {point["jitter_p90"]:.1%}')
    lines.append('波动指围绕本点预测中位位置的距离，不等于定位误差或病因。')
    if not metrics['passed']:
        lines.append('请截图保留结果，再开始新的五点校准；不会自动降低门槛。')
    return '\n'.join(lines)


def training_medians(mapping, groups, settings, canvas_size):
    """Descriptive in-sample gate; never substitutes for either check round."""
    import numpy as np
    width,height=canvas_size
    diagonal=math.hypot(width,height)
    result=[]
    for target,group in zip(TRAIN_TARGETS,prepared_groups(groups,mapping)):
        errors=[]
        for sample in group:
            x,y=mapping.predict(sample.features,settings)
            errors.append(math.hypot((x-target[0])*width,(y-target[1])*height)/diagonal)
        result.append(float(np.median(errors)))
    return result


def build_candidates(groups, reference, settings, canvas_size):
    """Fit only on five points. Check data is never used to construct candidates."""
    specs=[('双眼平均·原处理','average',0.0),
           ('混合·左眼纵向·100毫秒','average-x-left-y',.10),
           ('混合·左眼纵向·150毫秒','average-x-left-y',.15),
           ('混合·左眼纵向·200毫秒','average-x-left-y',.20)]
    candidates={}
    rejected=[]
    for name,mode,window in specs:
        try:
            mapping=fit_mapping(groups,reference,feature_mode=mode,window_seconds=window)
            medians=training_medians(mapping,groups,settings,canvas_size)
            if max(medians) > MEDIAN_LIMIT:
                rejected.append(f'{name}：训练最差点中位{max(medians):.2%} > {MEDIAN_LIMIT:.0%}')
            else:
                candidates[name]=(mapping,medians)
        except ValueError as exc:
            rejected.append(f'{name}：{exc}')
    if not candidates:
        raise ValueError('所有候选训练质量均不可靠；'+'；'.join(rejected))
    return candidates,rejected


class CalibrationSession:
    """UI-controlled progression with automatic sampling; no eye-click prerequisite."""
    def __init__(self, settings, canvas_size, mapping=None, reference=None):
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
        self.diagnostics = []
        self.analysis = MappingDiagnostics()
        self.anchor = mapping.head if mapping else reference
        self.candidates = {}
        self.selection = []
        self.selection_report = ''
        self.loaded_mapping = mapping is not None

    @property
    def target(self):
        return (TRAIN_TARGETS if self.phase == 'train' else CHECK_TARGETS)[self.index]

    def begin_point(self, now):
        if self.phase not in ('train', 'select', 'verify'):
            return
        self.started, self.last_time = now, None
        self.buffer = []

    def pause(self):
        self.started = None
        self.buffer = []
        self.last_time = None

    def progress_text(self, now):
        if self.started is None:
            return self.error or '已暂停；点击继续 / 重采当前点'
        remaining = max(0., SETTLE_SECONDS-(now-self.started))
        if remaining > 0:
            return f'准备 {remaining:.1f} 秒；请看圆点，尚未采样'
        duration = self.buffer[-1].timestamp-self.buffer[0].timestamp if self.buffer else 0.
        return (f'采样：稳定片段 {duration:.1f}/{SAMPLE_SECONDS:.1f} 秒'
                f' · {len(self.buffer)}/{MIN_SAMPLES} 帧（至少）\n'
                '继续看圆点；无效或不稳定片段会重新计数')

    def feed(self, frame):
        if self.started is None or self.phase not in ('train', 'select', 'verify'):
            return False
        if frame.timestamp - self.started > 20:
            self.pause()
            self.error = '本点采样超时；请休息或调整位置后重采'
            return False
        gap = self.last_time is not None and (frame.timestamp <= self.last_time
                                              or frame.timestamp-self.last_time > .25)
        self.last_time = frame.timestamp
        if frame.usable() and self.anchor and not head_matches(self.anchor, frame.head):
            self.pause()
            self.error = '头部偏离基准；对齐后点击继续 / 重采当前点'
            return False
        if not frame.usable() or gap:
            self.buffer = []
            return False
        if frame.timestamp-self.started < SETTLE_SECONDS:
            return False  # settle after target movement
        self.buffer.append(frame)
        if len(self.buffer) > 240:
            self.buffer = self.buffer[-240:]
        if len(self.buffer) < MIN_SAMPLES or self.buffer[-1].timestamp-self.buffer[0].timestamp < SAMPLE_SECONDS:
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
                    self.analysis.fit(self.groups, self.anchor, self.settings, self.canvas_size)
                    self.candidates,rejected=build_candidates(self.groups,self.anchor,self.settings,self.canvas_size)
                    lines=['候选仅使用五点训练；第一组三点只选型，第二组全新三点才作最终验收。']
                    for name,(_,medians) in self.candidates.items():
                        lines.append(f'{name}：训练最差点中位{max(medians):.2%}')
                    lines.extend('训练淘汰：'+item for item in rejected)
                    self.selection_report='\n'.join(lines)
                    self.mapping = None
                    self.phase, self.groups, self.index = 'select', [], 0
                elif self.phase == 'select':
                    ranked=[]
                    self.selection=[]
                    for name,(candidate,_) in self.candidates.items():
                        details=[]
                        metrics=validate_mapping(candidate,self.groups,self.settings,self.canvas_size,details)
                        self.selection.append((name,metrics,details))
                        ranked.append((metrics['median_error'],metrics['max_error'],name,candidate))
                    _,_,name,self.mapping=min(ranked)
                    self.selection_report+='\n选型三点（不作为最终通过证据）：'
                    for candidate_name,metrics,_ in self.selection:
                        self.selection_report+=(f'\n{candidate_name}：中位{metrics["median_error"]:.2%}，'
                                                f'最大{metrics["max_error"]:.2%}')
                    self.selection_report+=f'\n已选择：{name}；现在必须采集一组全新的三点验收。'
                    self.phase,self.groups,self.index='verify',[],0
                else:
                    self.analysis.evaluate(self.groups, self.settings, self.canvas_size)
                    self.metrics = validate_mapping(self.mapping, self.groups, self.settings, self.canvas_size,
                                                    self.diagnostics)
                    self.phase = 'trial' if self.metrics['passed'] else 'failed'
                    if self.phase == 'failed':
                        self.error = '独立验证未通过；不能保存有效档案，请重新校准'
            except ValueError as exc:
                self.error, self.phase = str(exc), 'failed'
        return True

    def diagnostic_report(self):
        base=self.analysis.report()
        return base+('\n\n'+self.selection_report if self.selection_report else '')


@dataclass(frozen=True)
class AdaptationStatus:
    status: str = 'not_calibrated'
    message: str = '当前未验证；摄像头工作区可进行五点校准。本阶段只开放画布练习。'

    @property
    def system_control_allowed(self):
        # Even successful practice calibration never enables OS injection in v2.
        return False
