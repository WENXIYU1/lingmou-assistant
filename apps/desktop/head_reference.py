"""Local, bounded posture history; no images or automatic profile writes."""
from collections import deque
from statistics import median
from .eye_features import head_matches


class HeadReference:
    def __init__(self):
        self.frames = deque(maxlen=90)

    def clear(self):
        self.frames.clear()

    def feed(self, frame):
        if not frame.usable():
            self.clear()
            return
        if self.frames and not 0 < frame.timestamp-self.frames[-1].timestamp <= .25:
            self.clear()
        self.frames.append(frame)
        while self.frames and frame.timestamp-self.frames[0].timestamp > 2.0:
            self.frames.popleft()

    def candidate(self, now):
        if (len(self.frames) < 18 or not 0 <= now-self.frames[-1].timestamp <= .25
                or self.frames[-1].timestamp-self.frames[0].timestamp < 1.2):
            return None
        head = tuple(median(f.head[i] for f in self.frames) for i in range(5))
        # Stability for establishing a reference, not a stricter runtime gate.
        limits = (.025, .025, head[2]*.08, .08, .06)
        if any(abs(f.head[i]-head[i]) > limits[i] for f in self.frames for i in range(5)):
            return None
        return head


def guidance(reference, current):
    if head_matches(reference, current):
        return '位置在参考范围内；无需完全不动'
    hints = []
    # Display is explicitly camera-coordinate, avoiding ambiguous physical left/right.
    if abs(current[0]-reference[0]) > .08:
        hints.append('蓝轮廓向' + ('左' if current[0]>reference[0] else '右') + '对齐')
    if abs(current[1]-reference[1]) > .08:
        hints.append('适当抬高位置' if current[1]>reference[1] else '适当降低位置')
    ratio = current[2]/reference[2]
    if abs(ratio-1) > .20:
        hints.append('稍远离摄像头' if ratio>1 else '稍靠近摄像头')
    if abs(current[3]-reference[3]) > .20:
        hints.append('恢复记录时的倾斜角度')
    if abs(current[4]-reference[4]) > .15:
        hints.append('恢复记录时的面向角度')
    return '；'.join(hints)
