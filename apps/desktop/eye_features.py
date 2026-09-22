"""Geometric features, not a pre-trained screen-gaze estimator.

Iris ring indices verified against MediaPipe face_mesh_connections.py.
All quality limits are conservative experimental defaults requiring real trials.
"""
from dataclasses import dataclass
import math
from statistics import mean

FEATURE_VERSION = 'binocular-eye-local-v1'


@dataclass(frozen=True)
class FeatureFrame:
    timestamp: float
    features: tuple = ()
    head: tuple = ()
    valid: bool = False
    reason: str = ''
    capture_size: tuple = (0, 0)
    left_eye: tuple = ()
    right_eye: tuple = ()
    left_lid: tuple = ()
    right_lid: tuple = ()
    average_lid: tuple = ()
    eye_widths_px: tuple = ()  # left, right; measured only when both eyes are usable
    preview_ppm: bytes = b''  # opt-in transient crop; never stored in profiles

    def usable(self):
        values = (self.timestamp, *self.features, *self.head)
        return (self.valid is True and len(self.features) == 2 and len(self.head) == 5
                and all(type(v) in (int, float) and math.isfinite(v) for v in values)
                and self.timestamp >= 0 and all(abs(v) <= 2 for v in self.features)
                and self.head[2] > 0)


def head_matches(reference, current):
    if len(reference) != 5 or len(current) != 5 or reference[2] <= 0:
        return False
    return (abs(current[0]-reference[0]) <= .08
            and abs(current[1]-reference[1]) <= .08
            and abs(current[2]/reference[2]-1) <= .20
            and abs(current[3]-reference[3]) <= .20
            and abs(current[4]-reference[4]) <= .15)


def extract_features(landmarks, timestamp, size):
    """Both eyes required in v1; unavailable eye must not silently change mapping."""
    def invalid(reason):
        return FeatureFrame(timestamp, reason=reason, capture_size=size)
    if len(landmarks) != 478 or size[0] <= 0 or size[1] <= 0:
        return invalid('关键点或画面尺寸不完整')

    def point(i):
        p = landmarks[i]
        if not all(math.isfinite(v) for v in (p.x, p.y)) or not (0 <= p.x <= 1 and 0 <= p.y <= 1):
            raise ValueError('关键点越界')
        return (p.x * size[0], p.y * size[1])

    def eye(a, b, top, bottom, ring):
        pa, pb, pt, pd = point(a), point(b), point(top), point(bottom)
        dx, dy = pb[0]-pa[0], pb[1]-pa[1]
        width = math.hypot(dx, dy)
        if width < 12:
            raise ValueError('距离过远或眼部过小')
        ux, uy = dx/width, dy/width
        opening = abs((pd[0]-pt[0]) * -uy + (pd[1]-pt[1]) * ux) / width
        if opening < .12 or opening > .65:
            raise ValueError('闭眼、遮挡或眼部质量不足')
        iris = [point(i) for i in ring]
        ix, iy = mean(p[0] for p in iris), mean(p[1] for p in iris)
        fx = ((ix-pa[0])*ux + (iy-pa[1])*uy) / width
        fy = ((ix-pa[0])*-uy + (iy-pa[1])*ux) / width
        if not (.05 < fx < .95 and abs(fy) < .4):
            raise ValueError('虹膜位置异常')
        lid_height=(pd[0]-pt[0])*-uy+(pd[1]-pt[1])*ux
        lid_y=((ix-pt[0])*-uy+(iy-pt[1])*ux)/lid_height
        # Optional shadow feature: never clip an out-of-range value into validity.
        lid=(fx,lid_y) if math.isfinite(lid_y) and 0 <= lid_y <= 1 else ()
        return (fx, fy), lid, width

    try:
        right, right_lid, right_width = eye(33, 133, 159, 145, (469, 470, 471, 472))
        left, left_lid, left_width = eye(362, 263, 386, 374, (474, 475, 476, 477))
        p, q, nose = point(33), point(263), point(1)
        face_width = math.dist(p, q)
        if face_width < 50:
            raise ValueError('请靠近摄像头')
        center = ((p[0]+q[0])/2, (p[1]+q[1])/2)
        head = (center[0]/size[0], center[1]/size[1], face_width/size[0],
                math.atan2(q[1]-p[1], q[0]-p[0]), (nose[0]-center[0])/face_width)
        if abs(head[3]) > .35 or abs(head[4]) > .35:
            raise ValueError('请面向摄像头并减少侧转')
        return FeatureFrame(timestamp, ((left[0]+right[0])/2, (left[1]+right[1])/2),
                            head, True, '双眼可用', size, left, right, left_lid, right_lid,
                            tuple((a+b)/2 for a,b in zip(left_lid,right_lid))
                            if left_lid and right_lid else (), (left_width,right_width))
    except (ValueError, IndexError, TypeError, AttributeError) as exc:
        return invalid(str(exc))
