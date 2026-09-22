"""Practice settings only, never a calibration certificate. Explicit path API."""
import json
import os
from pathlib import Path
import tempfile
from .config import Settings


def default_path():
    base = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local' / 'share'))
    return base / 'LingmouAssistant' / 'practice-settings.json'


def load_settings(path):
    path = Path(path)
    if not path.exists():
        return Settings()
    if path.stat().st_size > 16384:
        raise ValueError('配置过大')
    data = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(data, dict) or set(data) != {'schema_version', 'settings'}
            or type(data['schema_version']) is not int or data['schema_version'] != 1
            or not isinstance(data['settings'], dict)
            or set(data['settings']) != set(Settings().to_dict())):
        raise ValueError('配置版本或字段不兼容，未加载')
    return Settings(**data['settings'])


def _atomic_write(path, payload):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_settings(path, settings):
    path = Path(path)
    if not isinstance(settings, Settings):
        raise ValueError('需要有效Settings')
    # Do not overwrite a broken user's file silently.
    if path.exists():
        load_settings(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _atomic_write(path.with_suffix('.json.bak'), path.read_text(encoding='utf-8'))
    payload = json.dumps({'schema_version': 1, 'settings': settings.to_dict()}, indent=2)
    _atomic_write(path, payload)
