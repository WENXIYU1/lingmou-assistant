"""Opt-in UI smoke using fake camera + synthetic observations; no device access."""
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
from apps.desktop.camera_ui import CameraApp
from apps.desktop.calibration import TRAIN_TARGETS, CHECK_TARGETS
from apps.desktop.eye_features import FeatureFrame


class FakeCamera:
    running=False
    def poll(self): return False,None
    def stop(self): self.running=False
    def start(self,index): raise AssertionError('Camera must not start in UI smoke')


def main():
    with tempfile.TemporaryDirectory() as directory:
        app=CameraApp(FakeCamera(),Path(directory))
        app.withdraw()
        app.update_idletasks()
        assert not app.service.running
        app.camera_index,app.camera_label=0,'synthetic'
        now=100.
        def frame(target):
            nonlocal now
            now+=.06
            f=FeatureFrame(now,(.35+.2*target[0],.02+.1*target[1]),(.5,.4,.3,0.,0.),True,'synthetic',(640,480))
            with patch('apps.desktop.camera_ui.time.monotonic',return_value=now):
                app.process_frame(f)
        frame((.5,.5))
        with patch('apps.desktop.camera_ui.time.monotonic',return_value=now):
            app.start_calibration()
        for _ in range(8):
            target=app.session.target
            old_phase,old_index=app.session.phase,app.session.index
            for _ in range(100):
                frame(target)
                if (app.session.phase,app.session.index)!=(old_phase,old_index):
                    break
        assert app.checked, app.message.get()
        assert app.session.phase=='trial'
        with patch('apps.desktop.camera_ui.time.monotonic',return_value=now):
            app.resume()
        assert app.active
        app.pause()
        frame((.5,.5))
        assert not app.active, 'Valid frame must not release manual pause'
        with patch('apps.desktop.camera_ui.time.monotonic',return_value=now):
            app.resume()
        for target in app.practice_targets:
            for _ in range(60): frame(target)
        assert app.trial_hits==3,app.message.get()
        with patch('apps.desktop.camera_ui.messagebox.askyesno',return_value=True):
            app.save()
        assert list(Path(directory).glob('*.json'))
        with patch('apps.desktop.camera_ui.time.monotonic',return_value=now):
            app.load()
        assert not app.checked
        assert app.session.phase=='verify'
        app.name.set('其他用户')
        assert app.mapping is None and not app.checked
        app.shutdown()
    print('Camera UI synthetic wizard smoke passed; no camera access or human validation.')


if __name__=='__main__':
    main()
