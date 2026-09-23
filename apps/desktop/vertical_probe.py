"""Session-only vertical signal experiment; never a calibration or controller."""
import math
import numpy as np
from .eye_features import head_matches

TARGETS = ((.5, .2), (.5, .5), (.5, .8), (.5, .5))
NAMES = ('上', '中', '下', '中复测')
CHANNELS = (('features', '双眼平均'), ('left_eye', '左眼'), ('right_eye', '右眼'))


def paired_samples(group, channel):
    """Return raw/causal-filtered pairs at the same output timestamps."""
    raw, filtered, spans = [], [], []
    for i, (stamp, values) in enumerate(group):
        if i < 2:
            continue
        window = [(t, v[channel]) for t, v in group[:i+1] if stamp-t <= .20]
        if any(v is None for _, v in window):
            continue
        raw.append(values[channel])
        filtered.append(float(np.median([v for _, v in window])))
        spans.append(stamp-window[0][0])
    return raw, filtered, spans


class VerticalProbe:
    def __init__(self, reference):
        self.reference = reference
        self.groups = []
        self.buffer = []
        self.started = None
        self.last = None
        self.error = ''

    @property
    def done(self):
        return len(self.groups) == 4

    @property
    def target(self):
        return None if self.done else TARGETS[len(self.groups)]

    def begin(self, now):
        if not self.done:
            self.buffer = []
            self.started = now
            self.last = None
            self.error = ''

    def pause(self, reason='已暂停，请明确继续 / 重采当前点'):
        self.started = None
        self.buffer = []
        self.last = None
        self.error = reason

    def feed(self, frame):
        if self.started is None or self.done:
            return False
        if not frame.usable() or not head_matches(self.reference, frame.head):
            self.pause('跟踪中断或头部偏离；回到参考范围后明确继续')
            return False
        if self.last is not None and frame.timestamp <= self.last:
            self.pause('采样时间倒退或重复，请明确重采')
            return False
        # Inference can finish after begin() for an image captured before it.
        # Discard that image without changing the point timer or sample history.
        if frame.timestamp < self.started:
            return False
        if self.last is not None and frame.timestamp-self.last > .25:
            self.pause('采样间隔超过0.25秒，请明确重采')
            return False
        self.last = frame.timestamp
        elapsed = frame.timestamp-self.started
        if elapsed > 12:
            self.pause('本点采样超时（超过12秒），请明确重采')
            return False
        if elapsed < 1:
            return False
        values = self.sample_values(frame)
        # Keep scalar features only, never FeatureFrame / image bytes.
        self.buffer.append((frame.timestamp, tuple(values)))
        self.buffer = self.buffer[-240:]
        if len(self.buffer) >= 24 and self.buffer[-1][0]-self.buffer[0][0] >= 2:
            self.groups.append(self.buffer)
            self.pause('')
            return True
        return False

    def sample_values(self, frame):
        values = []
        for key, _ in CHANNELS:
            pair = getattr(frame, key)
            values.append(float(pair[1]) if len(pair) == 2 and all(math.isfinite(v) for v in pair) else None)
        return values

    def status(self):
        if self.done:
            return '上下短测完成；请点击“查看上下短测报告”。这不是校准通过。'
        return self.error or f'上下短测 {len(self.groups)+1}/4：看{NAMES[len(self.groups)]}方圆点；准备1秒，采样约2秒（{len(self.buffer)}帧）'

    def report(self):
        lines = ['上下短测（仅本次会话；不保存、不上传）',
                 f'完成 {len(self.groups)}/4 点；{self.error}',
                 '顺序：上—中—下—中复测，水平位置固定。',
                 '处理：每点先预热3个连续有效帧，再取过去0.20秒内样本的中位数。',
                 '原始/处理后使用相同时间点。窗口跨度不是实际眼动响应延迟；真人延迟未测。',
                 '只描述信号，不代表屏幕定位精度、通过门槛或自动选择算法。']
        if not self.done:
            return '\n'.join(lines)
        for channel, (_, name) in enumerate(CHANNELS):
            raw, filtered, spans = [], [], []
            for group in self.groups:
                a, b, history = paired_samples(group, channel)
                spans.extend(history)
                raw.append(a)
                filtered.append(b)
            lines.append(f'\n【{name}】')
            if any(len(a) < 18 for a in raw):
                lines.append('配对样本不足，不能比较。')
                continue
            lines.append('各点配对帧数：'+', '.join(str(len(a)) for a in raw))
            lines.append(f'历史窗口跨度中位：{np.median(spans)*1000:.0f}毫秒；不据此宣称无延迟。')
            for label, groups in [('原始', raw), ('短时中位数', filtered)]:
                centers = [float(np.median(a)) for a in groups]
                widths = [float(np.percentile(a,90)-np.percentile(a,10)) for a in groups]
                gap = centers[2]-centers[0]
                ordered = all(min(centers[0],centers[2]) < centers[i] < max(centers[0],centers[2]) for i in (1,3))
                lines.extend([label+'：', '  四点中位：'+', '.join(f'{v:+.5f}' for v in centers),
                              '  各点P90-P10：'+', '.join(f'{v:.5f}' for v in widths),
                              f'  下减上：{gap:+.5f}；两次中点差：{centers[3]-centers[1]:+.5f}',
                              f'  两个中点均在上下中位之间：{"是" if ordered else "否"}（非通过判据）'])
        lines.append('\n波动减小不等于定位更准；本结果不会替换映射、保存档案或解锁控制。')
        return '\n'.join(lines)
