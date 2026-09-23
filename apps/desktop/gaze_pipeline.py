"""CPU gaze + head-pose inference, invoked only in the camera worker."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .gaze_experiment import estimate_local

ROOT = Path(__file__).resolve().parents[2]
MODEL_VERSION = 'omz-gaze-head-fp16-roll-aligned-xy-v2'


def load_models():
    import openvino as ov
    if ov.__version__.split('-')[0] != '2026.4.0':
        raise RuntimeError('实验入口需要固定版本OpenVINO 2026.4.0')
    manifest = json.loads((ROOT / 'models/manifest/openvino_gaze.json').read_text(encoding='utf-8'))
    blobs = {}
    for item in manifest['files']:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to((ROOT / 'models/weights').resolve()):
            raise ValueError('模型路径越界')
        if not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError('实验模型缺失，请运行 scripts.setup_gaze_models')
        data = path.read_bytes()
        if hashlib.sha384(data).hexdigest() != item['sha384']:
            raise ValueError('实验模型SHA384不匹配')
        blobs[(item['model'], path.suffix)] = data
    core = ov.Core()
    compiled = {}
    for name in ('gaze-estimation-adas-0002', 'head-pose-estimation-adas-0001'):
        model = core.read_model(model=blobs[(name, '.xml')], weights=blobs[(name, '.bin')])
        compiled[name] = core.compile_model(model, 'CPU', {'INFERENCE_NUM_THREADS': 4})
    return compiled['gaze-estimation-adas-0002'], compiled['head-pose-estimation-adas-0001']


def face_tensor(bgr, landmarks):
    import cv2
    h, w = bgr.shape[:2]
    coords = np.array([(p.x * w, p.y * h) for p in landmarks[:468]])
    if coords.shape != (468, 2) or not np.all(np.isfinite(coords)):
        raise ValueError('人脸关键点无效')
    low, high = coords.min(axis=0), coords.max(axis=0)
    center = (low + high) / 2
    side = math.ceil(float(max(high - low)) * 1.10)
    x, y = (int(round(v - side / 2)) for v in center)
    if side < 60 or x < 0 or y < 0 or x + side > w or y + side > h:
        raise ValueError('请使整张脸进入画面，并留出少量边缘空间')
    crop = cv2.resize(bgr[y:y+side, x:x+side], (60, 60))
    return np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32)


class GazePipeline:
    def __init__(self):
        self.gaze, self.pose = load_models()
        for name, shape in {'left_eye_image': (1,3,60,60), 'right_eye_image': (1,3,60,60),
                            'head_pose_angles': (1,3)}.items():
            if tuple(self.gaze.input(name).shape) != shape:
                raise ValueError('视线模型输入尺寸不符')
        if tuple(self.pose.input('data').shape) != (1,3,60,60):
            raise ValueError('头姿模型输入尺寸不符')

    def infer(self, bgr, landmarks, observation):
        if not observation.usable():
            return observation
        result = self.pose({'data': face_tensor(bgr, landmarks)})
        angles = tuple(float(np.asarray(result[self.pose.output(name)]).item())
                       for name in ('angle_y_fc', 'angle_p_fc', 'angle_r_fc'))
        if any(not math.isfinite(v) or abs(v) > limit for v, limit in zip(angles, (90,70,70))):
            raise ValueError('头部角度超出模型支持范围')
        def infer_gaze(inputs):
            inputs = {name: np.asarray(value, dtype=np.float32) for name, value in inputs.items()}
            output = self.gaze(inputs)
            return output[self.gaze.output('gaze_vector')]
        vector = estimate_local(bgr, landmarks, angles, observation.timestamp, infer_gaze).xyz
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm < 1e-6:
            raise ValueError('视线模型向量长度无效')
        direction = tuple(v/norm for v in vector)
        # OMZ does not require positive Z. Its display projects normalized X/Y;
        # a negative Z is not a confidence or tracking-failure signal.
        if not all(math.isfinite(v) for v in direction):
            raise ValueError('视线模型方向数值无效')
        return replace(observation, features=direction[:2], left_eye=(), right_eye=(),
                       left_lid=(), right_lid=(), average_lid=(), reason='新视线模型可用')
