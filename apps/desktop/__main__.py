import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='灵动视眸：模拟输入安全练习')
    parser.add_argument('--check', action='store_true', help='无UI依赖/资源检查，不打开设备')
    args = parser.parse_args()
    if args.check:
        print(json.dumps({
            'mode': 'synthetic-practice-only', 'system_control': False,
            'calibration': 'not_implemented',
            'dependencies': {name: importlib.util.find_spec(name) is not None
                             for name in ('customtkinter', 'cv2', 'mediapipe')},
            'legacy_model_present': (Path(__file__).resolve().parents[2] /
                                     'source_review/eyemouse-main/face_landmarker.task').exists(),
        }, ensure_ascii=False, indent=2))
        return
    try:
        from .ui import PracticeApp
    except ModuleNotFoundError as exc:
        raise SystemExit(f'界面依赖缺失：{exc.name}。请安装 requirements-ui.txt。') from exc
    PracticeApp().mainloop()


if __name__ == '__main__':
    main()
