import queue
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from apps.desktop.camera import camera_worker, CameraService, verified_model


class Flag:
    def __init__(self): self.value=False
    def set(self): self.value=True
    def clear(self): self.value=False
    def is_set(self): return self.value


class Counter:
    value=0
    def get_lock(self): return nullcontext()


class FakeProcess:
    alive=True
    terminated=False
    closed=False
    def is_alive(self): return self.alive
    def terminate(self): self.terminated=True; self.alive=False
    def join(self,timeout): pass
    def close(self): self.closed=True


class Channel(queue.Queue):
    def cancel_join_thread(self): pass
    def close(self): pass


class WorkerTests(unittest.TestCase):
    def test_missing_model_never_opens_device(self):
        # Modules are fakes; CI requires no vision binaries, model, or camera.
        opened=[]
        fake_cv=SimpleNamespace(VideoCapture=lambda *a: opened.append(a))
        frames=queue.Queue(maxsize=1)
        with patch.dict('sys.modules',{'cv2':fake_cv,'mediapipe':SimpleNamespace()}), \
             patch('apps.desktop.camera.verified_model',side_effect=FileNotFoundError('missing')):
            camera_worker(0,frames,Flag(),Counter())
        self.assertEqual(opened,[])
        self.assertEqual(frames.get()[0],'error')

    def test_read_failure_releases_camera_and_detector(self):
        class Capture:
            released=False
            settings=None
            def isOpened(self): return True
            def set(self,*args):
                if self.settings is None: self.settings=[]
                self.settings.append(args)
            def read(self): return False,None
            def release(self): self.released=True
        class Detector:
            closed=False
            def close(self): self.closed=True
        cap,det=Capture(),Detector()
        cv=SimpleNamespace(VideoCapture=lambda *a:cap,CAP_DSHOW=1,
            CAP_PROP_FRAME_WIDTH=2,CAP_PROP_FRAME_HEIGHT=3,CAP_PROP_FPS=4)
        frames=queue.Queue(maxsize=1)
        with patch.dict('sys.modules',{'cv2':cv,'mediapipe':SimpleNamespace()}), \
             patch('apps.desktop.camera.verified_model',return_value=b'model'), \
             patch('apps.desktop.camera.create_detector',return_value=det):
            camera_worker(0,frames,Flag(),Counter(),(1280,720))
        self.assertTrue(cap.released)
        self.assertIn((2,1280),cap.settings)
        self.assertIn((3,720),cap.settings)
        self.assertTrue(det.closed)
        self.assertEqual(frames.get()[0],'error')

    def test_service_creation_does_not_open_camera(self):
        self.assertFalse(CameraService().running)

    def test_bad_index_rejected(self):
        with self.assertRaises(ValueError):
            CameraService().start(-1)

    def service(self):
        service=CameraService()
        service.process=FakeProcess()
        service.frames=Channel(maxsize=1)
        service.stop_event=Flag()
        service.loss=Counter()
        return service

    def test_stopped_session_discards_late_frame(self):
        service=self.service()
        service.frames.put(('frame','old'))
        with patch('apps.desktop.camera.time.monotonic',return_value=100):
            service.stop()
            self.assertIsNone(service.poll()[1])

    def test_stuck_driver_is_terminated_after_deadline(self):
        service=self.service()
        process=service.process
        with patch('apps.desktop.camera.time.monotonic',return_value=100):
            service.stop()
        with patch('apps.desktop.camera.time.monotonic',return_value=104):
            lost,item=service.poll()
        self.assertTrue(process.terminated)
        self.assertTrue(process.closed)
        self.assertFalse(service.running)
        self.assertTrue(lost)
        self.assertEqual(item[0],'error')

    def test_loss_counter_not_overwritten_by_valid_frame(self):
        service=self.service()
        service.loss.value=3
        service.frames.put(('frame','new'))
        lost,item=service.poll()
        self.assertTrue(lost)
        self.assertEqual(item,('frame','new'))
        self.assertFalse(service.poll()[0])


if __name__=='__main__':
    unittest.main()
