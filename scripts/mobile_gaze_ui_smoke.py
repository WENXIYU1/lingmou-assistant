from pathlib import Path
import tempfile
import time
from apps.desktop.camera_ui import CameraApp
from apps.desktop.eye_features import FeatureFrame
from apps.desktop.direction_probe import HeadCompProbe
from scripts.camera_ui_smoke import FakeCamera

def main():
    with tempfile.TemporaryDirectory() as folder:
        app=CameraApp(FakeCamera(),Path(folder),mobilegaze=True)
        try:
            app.withdraw()
            app.update_idletasks()
            app.latest=FeatureFrame(time.monotonic(),(.1,.1),(.5,.4,.3,0.,0.),True,capture_size=(1280,720))
            app.head_reference=app.latest.head
            app.start_calibration()
            assert app.session is None
            app.start_probe()
            assert isinstance(app.probe,HeadCompProbe)
            app.pause()
            assert app.probe.started is None and not app.active
            app.load(); app.save(); app.resume()
            assert not app.active and not app.checked and not list(Path(folder).iterdir())
            print('MobileGaze UI smoke passed; no camera access.')
        finally:
            app.destroy()

if __name__=='__main__': main()
