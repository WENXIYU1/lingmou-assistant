"""Local MobileGaze ResNet18 adapter. No device access or network downloads."""
from dataclasses import replace
from pathlib import Path
import hashlib
import math
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models/weights/resnet18_gaze.onnx'
SHA256 = '404fec1efd07ff49f981e47f461c20c2627119e465ec441bbd1c067d3f16e657'
SIZE = 45066134
URL = 'https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet18_gaze.onnx'


def prepare_face(bgr, landmarks, *, with_geometry=False):
    import cv2
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('需要BGR uint8图像')
    h,w = bgr.shape[:2]
    xy = np.array([(p.x*w,p.y*h) for p in landmarks[:468]])
    if xy.shape != (468,2) or not np.isfinite(xy).all():
        raise ValueError('人脸关键点无效')
    lo,hi = np.floor(xy.min(0)).astype(int), np.ceil(xy.max(0)).astype(int)
    if np.any(lo<0) or hi[0]>w or hi[1]>h or np.any(hi-lo<60):
        raise ValueError('请使完整人脸进入画面')
    # Local landmark bounding box, not the upstream RetinaFace detector.
    rgb=cv2.cvtColor(bgr[lo[1]:hi[1],lo[0]:hi[0]],cv2.COLOR_BGR2RGB)
    rgb=cv2.resize(rgb,(448,448)).astype(np.float32)/255.
    rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    tensor = np.ascontiguousarray(rgb.transpose(2,0,1)[None])
    geometry = (float((lo[0]+hi[0])/(2*w)), float((lo[1]+hi[1])/(2*h)),
                float((hi[0]-lo[0])/w), float((hi[1]-lo[1])/h))
    return (tensor, geometry) if with_geometry else tensor


def prepare_aligned_face(bgr, landmarks, reference=None):
    """Normalize translation/scale/roll to a session reference, in memory only."""
    import cv2
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('需要BGR uint8图像')
    h,w=bgr.shape[:2]
    xy=np.array([(p.x*w,p.y*h) for p in landmarks[:468]],dtype=np.float64)
    if xy.shape!=(468,2) or not np.isfinite(xy).all():
        raise ValueError('人脸关键点无效')
    lo,hi=np.floor(xy.min(0)).astype(int),np.ceil(xy.max(0)).astype(int)
    if np.any(lo<0) or hi[0]>w or hi[1]>h or np.any(hi-lo<60):
        raise ValueError('请使完整人脸进入画面')
    left,right=xy[33],xy[263]
    distance=float(np.linalg.norm(right-left))
    if not math.isfinite(distance) or distance<40:
        raise ValueError('双眼间距不足，无法稳定裁剪')
    current=(left.copy(),right.copy(),lo.copy(),hi.copy())
    if reference is None:
        reference=current
    ref_left,ref_right,ref_lo,ref_hi=(np.asarray(v,dtype=np.float64) for v in reference)
    ref_distance=float(np.linalg.norm(ref_right-ref_left))
    if (ref_distance<40 or np.any(ref_lo<0) or ref_hi[0]>w or ref_hi[1]>h
            or np.any(ref_hi-ref_lo<60)):
        raise ValueError('稳定裁剪参考无效')
    source_vector=right-left
    target_vector=ref_right-ref_left
    scale=ref_distance/distance
    angle=math.atan2(target_vector[1],target_vector[0])-math.atan2(source_vector[1],source_vector[0])
    cosine,sine=math.cos(angle)*scale,math.sin(angle)*scale
    matrix=np.array(((cosine,-sine,0.),(sine,cosine,0.)),dtype=np.float64)
    source_mid=(left+right)/2
    target_mid=(ref_left+ref_right)/2
    matrix[:,2]=target_mid-matrix[:,:2]@source_mid
    aligned=cv2.warpAffine(bgr,matrix,(w,h),flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    rlo=np.floor(ref_lo).astype(int); rhi=np.ceil(ref_hi).astype(int)
    rgb=cv2.cvtColor(aligned[rlo[1]:rhi[1],rlo[0]:rhi[0]],cv2.COLOR_BGR2RGB)
    rgb=cv2.resize(rgb,(448,448)).astype(np.float32)/255.
    rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return np.ascontiguousarray(rgb.transpose(2,0,1)[None]), reference


def decode(logits):
    values=np.asarray(logits,dtype=np.float64)
    if values.shape != (1,90) or not np.isfinite(values).all():
        raise ValueError('视线模型输出形状或数值无效')
    prob=np.exp(values-values.max(axis=1,keepdims=True))
    prob/=prob.sum(axis=1,keepdims=True)
    return float((prob @ np.arange(90))[0]*4-180)


class MobileGazePipeline:
    def __init__(self):
        import openvino as ov
        if ov.__version__.split('-')[0] != '2026.4.0':
            raise ValueError('需要OpenVINO 2026.4.0')
        if not MODEL_PATH.is_file():
            raise ValueError('请先运行 python -m scripts.setup_mobile_gaze')
        data=MODEL_PATH.read_bytes()
        if len(data)!=SIZE or hashlib.sha256(data).hexdigest()!=SHA256:
            raise ValueError('MobileGaze权重校验失败')
        core=ov.Core()
        self.model=core.compile_model(core.read_model(data), 'CPU', {'INFERENCE_NUM_THREADS':4})
        if tuple(self.model.input().shape)!=(1,3,448,448) or len(self.model.outputs)!=2:
            raise ValueError('MobileGaze模型接口不匹配')
        for name in ('yaw','pitch'):
            if tuple(self.model.output(name).shape)!=(1,90):
                raise ValueError('MobileGaze角度输出接口不匹配')
        self.alignment_reference=None

    def _angles(self,tensor):
        outputs=self.model([tensor])
        return tuple(decode(outputs[self.model.output(name)]) for name in ('yaw','pitch'))

    def reset_alignment(self):
        self.alignment_reference=None

    def infer(self,bgr,landmarks,observation):
        if not observation.usable():
            return observation
        tensor, geometry = prepare_face(bgr,landmarks,with_geometry=True)
        angles=self._angles(tensor)
        shadow=()
        # Normalized angles keep FeatureFrame's finite [-2,2] contract.
        return replace(observation, features=tuple(v/180. for v in angles),
                       face_crop=geometry, shadow_features=tuple(v/180. for v in shadow),
                       left_eye=(),right_eye=(),left_lid=(),right_lid=(),average_lid=(),
                       reason='MobileGaze整脸方向可用（仅短测）')
