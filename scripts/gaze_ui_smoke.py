"""Experimental UI wiring checks; no camera access."""
from pathlib import Path
import tempfile
from unittest.mock import patch
from apps.desktop.camera_ui import CameraApp
from apps.desktop.eye_features import FeatureFrame
from scripts.camera_ui_smoke import FakeCamera


def main():
    with tempfile.TemporaryDirectory() as directory:
        app = CameraApp(FakeCamera(), Path(directory), experimental=True)
        try:
            app.withdraw()
            app.update_idletasks()
            assert app.experimental and not app.active
            assert app.resolution.get() == '1280x720'
            app.latest = FeatureFrame(1., (.1,.1), (.5,.4,.3,0.,0.), True, capture_size=(1280,720))
            assert 'omz-gaze' in app.environment()['model']
            app.load()
            app.save()
            assert not list(Path(directory).iterdir())
            assert app.session is None and not app.checked
            app.pause()
            assert not app.active
            for i in range(30):
                stamp=10.+i*.06
                frame=FeatureFrame(stamp,head=(.5,.4,.3,0.,0.),head_valid=True,
                                   capture_size=(1280,720),reason='synthetic gaze rejection')
                app.service.poll=lambda f=frame: (True,('frame',f))
                with patch('apps.desktop.camera_ui.time.monotonic',return_value=stamp):
                    app.tick()
            with patch('apps.desktop.camera_ui.time.monotonic',return_value=stamp):
                app.redraw_head()
                assert app.head_canvas.find_all(), 'Posture outline must survive gaze rejection'
                app.confirm_head()
                assert app.head_reference is not None
                app.start_calibration()
                assert app.session is None, 'Posture alone must not authorize gaze calibration'
            print('Experimental UI smoke passed; no camera access.')
        finally:
            app.destroy()


if __name__ == '__main__':
    main()
