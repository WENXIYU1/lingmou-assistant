"""Explicit official resource download; never invoked at app startup."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
from apps.desktop.camera import MODEL_MANIFEST, ROOT, verified_model


def main():
    manifest=json.loads(MODEL_MANIFEST.read_text(encoding='utf-8'))
    url=manifest['url']
    if url!='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task':
        raise ValueError('只允许经过核对的固定官方地址')
    target=ROOT/manifest['path']
    if target.exists():
        verified_model()
        print('Model already present and verified.')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=target.parent,suffix='.download')
    try:
        digest=hashlib.sha256()
        total=0
        with os.fdopen(fd,'wb') as stream, urllib.request.urlopen(url,timeout=30) as response:
            while chunk:=response.read(1024*1024):
                total+=len(chunk)
                if total>30*1024*1024:
                    raise ValueError('模型下载大小异常')
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest()!=manifest['sha256']:
            raise ValueError('模型校验失败，未安装')
        os.replace(temporary,target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print('Official model downloaded and verified. No camera accessed.')


if __name__=='__main__':
    main()
