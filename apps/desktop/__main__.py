import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='灵动视眸：本地安全练习')
    parser.add_argument('--check', action='store_true', help='无UI依赖/资源检查，不打开设备')
    parser.add_argument('--camera', action='store_true', help='打开摄像头适配界面；仍需点击确认才能采集')
    parser.add_argument('--gaze-experiment', action='store_true', help='本地OpenVINO新视线算法实验画布')
    parser.add_argument('--mobilegaze-probe', action='store_true', help='MobileGaze整脸方向短测，不开放控制')
    args = parser.parse_args()
    if args.check:
        print(json.dumps({
            'mode': 'canvas-only; camera opt-in', 'system_control': False,
            'calibration': 'implemented; human validation pending',
            'dependencies': {name: importlib.util.find_spec(name) is not None
                             for name in ('customtkinter', 'cv2', 'mediapipe')},
            'legacy_model_present': (Path(__file__).resolve().parents[2] /
                                     'source_review/eyemouse-main/face_landmarker.task').exists(),
            'camera_model_present': (Path(__file__).resolve().parents[2] /
                                     'models/weights/face_landmarker.task').exists(),
        }, ensure_ascii=False, indent=2))
        return
    try:
        if args.camera or args.gaze_experiment or args.mobilegaze_probe:
            from .camera_ui import CameraApp as Application
        else:
            from .ui import PracticeApp as Application
    except ModuleNotFoundError as exc:
        raise SystemExit(f'界面依赖缺失：{exc.name}。请安装 requirements-ui.txt。') from exc
    if args.mobilegaze_probe:
        Application(mobilegaze=True).mainloop()
    elif args.gaze_experiment:
        Application(experimental=True).mainloop()
    else:
        Application().mainloop()


if __name__ == '__main__':
    main()
