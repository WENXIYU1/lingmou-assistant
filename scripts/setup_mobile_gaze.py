"""Explicit developer download; never called by the camera runtime."""
import hashlib
import urllib.request
from apps.desktop.mobile_gaze import MODEL_PATH, SHA256, SIZE, URL

def main():
    if MODEL_PATH.exists():
        data=MODEL_PATH.read_bytes()
    else:
        with urllib.request.urlopen(URL,timeout=60) as response:
            data=response.read(SIZE+1)
    if len(data)!=SIZE or hashlib.sha256(data).hexdigest()!=SHA256:
        raise ValueError('官方权重大小/SHA256不匹配；未写入')
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
        with MODEL_PATH.open('xb') as stream:
            stream.write(data)
    print('MobileGaze ResNet18 official release verified; runtime stays offline.')

if __name__=='__main__':
    main()
