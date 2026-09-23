"""Optional CPU-only OpenVINO runner for the isolated gaze experiment.

The official IR files and their reviewed hashes must be supplied explicitly.
This module does not download files, open a camera, or expose desktop actions.
"""
import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = (ROOT / 'models' / 'weights').resolve()
INPUTS = {'left_eye_image', 'right_eye_image', 'head_pose_angles'}


def _verified_file(path, expected_sha256, suffix, max_bytes):
    if not isinstance(path, (str, Path)) or not isinstance(expected_sha256, str):
        raise ValueError('模型路径或校验值缺失')
    if len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('需要已核实的SHA-256校验值')
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(WEIGHTS) or resolved.suffix.lower() != suffix:
        raise ValueError('模型文件必须位于本地models/weights目录')
    if not resolved.is_file() or not 0 < resolved.stat().st_size <= max_bytes:
        raise ValueError('模型文件缺失或大小异常')
    digest = hashlib.sha256()
    with resolved.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('模型文件校验不匹配')
    return resolved


class OpenVinoGazeRunner:
    def __init__(self, xml_path, bin_path, xml_sha256, bin_sha256, *, core=None):
        xml = _verified_file(xml_path, xml_sha256, '.xml', 2 * 1024 * 1024)
        weights = _verified_file(bin_path, bin_sha256, '.bin', 32 * 1024 * 1024)
        if core is None:
            try:
                import openvino as ov
            except ImportError as exc:
                raise RuntimeError('未安装可选OpenVINO运行时；原眼控流程不受影响') from exc
            core = ov.Core()
        model = core.read_model(model=str(xml), weights=str(weights))
        self.compiled = core.compile_model(model, 'CPU')
        if {port.get_any_name() for port in self.compiled.inputs} != INPUTS:
            raise ValueError('模型输入与已审查接口不符')
        if {port.get_any_name() for port in self.compiled.outputs} != {'gaze_vector'}:
            raise ValueError('模型输出与已审查接口不符')
        self.output_port = self.compiled.output('gaze_vector')

    def __call__(self, inputs):
        if set(inputs) != INPUTS:
            raise ValueError('视线模型输入不完整')
        result = self.compiled(inputs)
        return np.asarray(result[self.output_port]).copy()
