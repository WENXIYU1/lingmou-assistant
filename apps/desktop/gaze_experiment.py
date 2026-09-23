"""Isolated local gaze-model input boundary; never controls the desktop.

No camera, network, model download or profile writes occur here. The caller must
provide genuine head-pose angles, not the five-value head proxy in FeatureFrame.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GazeVector:
    timestamp: float
    xyz: tuple


def _eye_crop(bgr, landmarks, corners, roll=0.):
    import cv2
    height, width = bgr.shape[:2]
    coords = []
    for index in corners:
        point = landmarks[index]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (point.x, point.y)):
            raise ValueError('眼角关键点无效')
        coords.append((point.x * width, point.y * height))
    (x0, y0), (x1, y1) = coords
    span = math.hypot(x1 - x0, y1 - y0)
    if span < 12:
        raise ValueError('眼部像素不足')
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(24, round(span * 1.8))
    left, top = round(cx - side / 2), round(cy - side / 2)
    right, bottom = left + side, top + side
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError('眼部裁剪越界')
    crop = bgr[top:bottom, left:right]
    # Match OMZ's default roll alignment before resizing, not after inference.
    if roll:
        rotation = cv2.getRotationMatrix2D((side // 2, side // 2), roll, 1.)
        crop = cv2.warpAffine(crop, rotation, (side, side),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    crop = cv2.resize(crop, (60, 60), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(crop.transpose(2, 0, 1)[None, ...])


def prepare_inputs(bgr, landmarks, head_angles_deg):
    """Produce BGR BCHW eye inputs and yaw/pitch/roll degrees for OMZ model.

    Eye indices are MediaPipe's anatomical left/right, not the mirrored preview.
    Crop scale and roll alignment follow OMZ; landmarks still come from MediaPipe.
    """
    if (not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3
            or bgr.dtype != np.uint8 or len(landmarks) != 478):
        raise ValueError('需要一帧BGR图像和478个关键点')
    if (len(head_angles_deg) != 3 or not all(type(v) in (int, float)
            and math.isfinite(v) and abs(v) <= 90 for v in head_angles_deg)):
        raise ValueError('需要真实的yaw/pitch/roll角度（度）')
    return {
        'left_eye_image': _eye_crop(bgr, landmarks, (362, 263), head_angles_deg[2]),
        'right_eye_image': _eye_crop(bgr, landmarks, (33, 133), head_angles_deg[2]),
        'head_pose_angles': np.asarray([[head_angles_deg[0], head_angles_deg[1], 0.]], dtype=np.float32),
    }


def estimate_local(bgr, landmarks, head_angles_deg, timestamp, infer):
    """Run an injected local-only inference callable; result is not a screen point."""
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError('无效时间戳')
    inputs = prepare_inputs(bgr, landmarks, head_angles_deg)
    output = np.asarray(infer(inputs), dtype=float)
    if output.shape not in ((3,), (1, 3)) or not np.all(np.isfinite(output)):
        raise ValueError('模型输出不是有效三维向量')
    xyz = tuple(float(v) for v in output.reshape(3))
    if math.sqrt(sum(v * v for v in xyz)) < 1e-6:
        raise ValueError('模型输出零向量')
    angle = math.radians(head_angles_deg[2])
    cs, sn = math.cos(angle), math.sin(angle)
    x, y, z = xyz
    xyz = (x * cs + y * sn, -x * sn + y * cs, z)
    return GazeVector(float(timestamp), xyz)
