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
    preview=False
    def set_preview(self,enabled): self.preview=enabled
    def poll(self): return False,None
    def stop(self): self.running=False
    def start(self,index): raise AssertionError('Camera must not start in UI smoke')


def main():
    with tempfile.TemporaryDirectory() as directory:
        app=CameraApp(FakeCamera(),Path(directory))
        app.withdraw()
        app.update_idletasks()
        assert not app.service.running
        app.service.running=True
        with patch('apps.desktop.camera_ui.messagebox.askyesno',return_value=False):
            app.open_preview()
        assert app.preview_window is None and not app.service.preview
        with patch('apps.desktop.camera_ui.messagebox.askyesno',return_value=True):
            app.open_preview()
        assert app.preview_window is not None and app.service.preview
        preview_time=time.monotonic()
        preview_frame=FeatureFrame(preview_time,(.45,.07),(.5,.4,.3,0,0),True,'synthetic',(640,480),
                                   preview_ppm=b'P6\n1 1\n255\n\x00\x00\x00')
        app.process_frame(preview_frame)
        assert app.preview_photo is not None
        assert not app.latest.preview_ppm
        assert all(not f.preview_ppm for f in app.head_history.frames)
        app.close_preview()
        assert not app.service.preview and app.preview_photo is None
        app.service.running=False
        app.latest=None
        app.head_history.clear()
        audit_window=app.show_capture_audit()
        assert app.session is None and not app.active
        audit_window.destroy()
        app.camera_index,app.camera_label=0,'synthetic'
        now=100.
        def frame(target):
            nonlocal now
            now+=.06
            f=FeatureFrame(now,(.35+.2*target[0],.02+.1*target[1]),(.5,.4,.3,0.,0.),True,'synthetic',(640,480))
            with patch('apps.desktop.camera_ui.time.monotonic',return_value=now):
                app.process_frame(f)
        for _ in range(30): frame((.5,.5))
        with patch('apps.desktop.camera_ui.time.monotonic',return_value=now):
            app.start_calibration()
            assert app.session is None, 'Reference must be confirmed first'
            app.confirm_head()
            assert app.head_reference is not None
            app.start_calibration()
            app.redraw()
            target=app.session.target
            circles=[item for item in app.canvas.find_all() if app.canvas.type(item)=='oval']
            coords=app.canvas.coords(circles[0])
            assert abs((coords[0]+coords[2])/2-target[0]*app.canvas.winfo_width())<.001
            assert abs((coords[1]+coords[3])/2-target[1]*app.canvas.winfo_height())<.001
        for _ in range(11):
            target=app.session.target
            old_phase,old_index=app.session.phase,app.session.index
            for _ in range(100):
                frame(target)
                if (app.session.phase,app.session.index)!=(old_phase,old_index):
                    break
        assert app.checked, app.message.get()
        assert app.session.phase=='trial'
        assert len(app.session.analysis.training['average'])==5
        diagnostic_window=app.show_diagnostics()
        assert not app.active
        diagnostic_window.destroy()
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
        with patch('apps.desktop.camera_ui.time.monotonic',return_value=now):
            app.continue_point()
            app.redraw()
        texts=[app.canvas.itemcget(item,'text') for item in app.canvas.find_all()
               if app.canvas.type(item)=='text']
        assert any('准备' in text for text in texts)
        for _ in range(3):
            old_index=app.session.index
            for _ in range(100):
                frame((.95,.95))
                if app.session.index!=old_index or app.session.phase=='failed':
                    break
        assert app.session.phase=='failed'
        assert not app.checked and not app.active
        app.redraw()
        texts=[app.canvas.itemcget(item,'text') for item in app.canvas.find_all()
               if app.canvas.type(item)=='text']
        assert any('最大偏差出现在验证点' in text and '波动P90' in text for text in texts)
        with patch('apps.desktop.camera_ui.messagebox.askyesno',side_effect=AssertionError('Must not save')):
            app.save()
        app.invalidate('模拟：窗口尺寸变化')
        assert app.session is None
        assert '窗口尺寸变化' in app.previous_diagnostic
        assert '独立验证未通过' in app.previous_diagnostic
        app.invalidate('模拟：后续关闭摄像头')
        assert '窗口尺寸变化' in app.previous_diagnostic
        report_window=app.show_diagnostics()
        report_window.destroy()
        for _ in range(30): frame((.5,.5))
        with patch('apps.desktop.camera_ui.time.monotonic',return_value=now):
            app.start_probe()
        assert app.probe is not None and app.session is None
        assert not app.checked and not app.active and app.mapping is None
        app.pause()
        frame((.5,.2))
        assert app.probe.started is None
        with patch('apps.desktop.camera_ui.time.monotonic',return_value=now):
            app.continue_point()
        for _ in range(4):
            target=app.probe.target
            count=len(app.probe.groups)
            for _ in range(90):
                frame(target)
                if len(app.probe.groups)!=count:
                    break
        assert app.probe.done, app.probe.status()
        assert not app.checked and not app.active and app.mapping is None
        app.redraw()
        app.show_probe().destroy()
        with patch('apps.desktop.camera_ui.messagebox.askyesno',side_effect=AssertionError('Probe must not save')):
            app.save()
        app.name.set('其他用户')
        assert app.probe is None
        assert app.mapping is None and not app.checked
        assert app.head_reference is None
        assert not app.previous_diagnostic
        app.shutdown()
    print('Camera UI synthetic wizard smoke passed; no camera access or human validation.')


if __name__=='__main__':
    main()
