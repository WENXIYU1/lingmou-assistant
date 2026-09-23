"""Explicit developer download only; runtime never downloads models."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'models/manifest/openvino_gaze.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        target = (ROOT / item['path']).resolve()
        if not target.is_relative_to((ROOT / 'models/weights').resolve()):
            raise ValueError('模型路径越界')
        if target.is_file():
            data = target.read_bytes()
            if len(data) == item['size'] and hashlib.sha384(data).hexdigest() == item['sha384']:
                print(f'Verified existing: {target.name}')
                continue
            raise ValueError(f'已有模型校验失败，请检查文件：{target.name}')
        if not item['url'].startswith('https://storage.openvinotoolkit.org/'):
            raise ValueError('非登记的模型来源')
        with urllib.request.urlopen(item['url'], timeout=60) as response:
            data = response.read(item['size'] + 1)
        if len(data) != item['size'] or hashlib.sha384(data).hexdigest() != item['sha384']:
            raise ValueError('官方模型大小或SHA384校验失败')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(data)
        print(f'Downloaded and verified: {target.name}')


if __name__ == '__main__':
    main()
