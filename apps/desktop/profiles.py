"""Versioned individual profiles; loading always requires a new quick check."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from .calibration import Mapping, MAPPING_VERSION, POLICY_VERSION, MEDIAN_LIMIT, MAX_LIMIT
from .config import Settings
from .eye_features import FEATURE_VERSION
from .storage import _atomic_write


def profile_path(directory, name):
    if not isinstance(name, str) or not re.fullmatch(r'[\w\-\u4e00-\u9fff]{1,32}', name):
        raise ValueError('档案名限1–32个字母、数字、汉字、下划线或短横线')
    # Prefix avoids Windows reserved device names and never treats names as paths.
    return Path(directory) / f'profile-{name}.json'


def validate_profile(data):
    expected = {'schema_version', 'name', 'feature_version', 'mapping_version', 'policy_version',
                'environment', 'mapping', 'settings', 'metrics', 'trial_hits', 'updated_at'}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError('档案字段不兼容')
    if type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ValueError('档案版本不兼容')
    if (data['feature_version'], data['mapping_version'], data['policy_version']) != (
            FEATURE_VERSION, MAPPING_VERSION, POLICY_VERSION):
        raise ValueError('特征、映射或验证规则已改变，请重新校准')
    profile_path('.', data['name'])
    if not isinstance(data['updated_at'], str):
        raise ValueError('日期字段无效')
    environment = data['environment']
    if not isinstance(environment, dict) or not environment or len(json.dumps(environment)) > 4096:
        raise ValueError('设备配置无效')
    settings = Settings(**data['settings'])
    if set(data['settings']) != set(settings.to_dict()):
        raise ValueError('设置不完整')
    raw = data['mapping']
    if not isinstance(raw, dict) or set(raw) != {'center', 'scale', 'coefficients', 'head'}:
        raise ValueError('映射结构无效')
    if (len(raw['center']) != 2 or len(raw['scale']) != 2 or len(raw['head']) != 5
            or len(raw['coefficients']) != 3 or any(len(row) != 2 for row in raw['coefficients'])):
        raise ValueError('映射尺寸无效')
    numbers = [*raw['center'], *raw['scale'], *raw['head'], *(v for row in raw['coefficients'] for v in row)]
    if any(type(x) not in (float, int) or not math.isfinite(x) or abs(x) > 10 for x in numbers):
        raise ValueError('映射含无效数值')
    if any(x < .002 for x in raw['scale']) or raw['head'][2] <= 0:
        raise ValueError('映射尺度无效')
    metrics = data['metrics']
    if not isinstance(metrics, dict) or set(metrics) != {
        'median_error', 'p90_error', 'max_error', 'sample_count', 'policy_version', 'passed'}:
        raise ValueError('验证指标无效')
    for key in ('median_error', 'p90_error', 'max_error'):
        x = metrics[key]
        if type(x) not in (float, int) or not math.isfinite(x) or x < 0:
            raise ValueError('验证指标无效')
    if (metrics['passed'] is not True or metrics['policy_version'] != POLICY_VERSION
            or metrics['median_error'] > MEDIAN_LIMIT or metrics['max_error'] > MAX_LIMIT
            or not metrics['median_error'] <= metrics['p90_error'] <= metrics['max_error']
            or type(metrics['sample_count']) is not int or metrics['sample_count'] < 54
            or type(data['trial_hits']) is not int or data['trial_hits'] < 3):
        raise ValueError('档案缺少有效验证或试用结果')
    return Mapping(tuple(raw['center']), tuple(raw['scale']),
                   tuple(tuple(row) for row in raw['coefficients']), tuple(raw['head'])), settings


def load_profile(directory, name, environment):
    path = profile_path(directory, name)
    if path.stat().st_size > 65536:
        raise ValueError('档案过大')
    data = json.loads(path.read_text(encoding='utf-8'))
    mapping, settings = validate_profile(data)
    if data['name'] != name or data['environment'] != environment:
        raise ValueError('档案用户或设备/窗口配置不匹配，请重新校准')
    return data, mapping, settings  # caller must start independent verification


def save_profile(directory, name, environment, mapping, settings, metrics, hits):
    path = profile_path(directory, name)
    data = {'schema_version': 1, 'name': name, 'feature_version': FEATURE_VERSION,
            'mapping_version': MAPPING_VERSION, 'policy_version': POLICY_VERSION,
            'environment': environment, 'mapping': asdict(mapping), 'settings': settings.to_dict(),
            'metrics': metrics, 'trial_hits': hits, 'updated_at': datetime.now(timezone.utc).isoformat()}
    validate_profile(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        validate_profile(old)  # damaged original is preserved for user decision
        _atomic_write(path.with_suffix('.json.bak'), json.dumps(old, ensure_ascii=False, indent=2))
    _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    return path
