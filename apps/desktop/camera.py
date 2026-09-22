"""Opt-in camera process. Latest result only; no frames saved or sent online."""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
from dataclasses import replace
from .eye_features import FeatureFrame, extract_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST = ROOT / 'models/manifest/face_landmarker.json'


def verified_model():
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding='utf-8'))
    path = ROOT / manifest['path']
    if not path.is_file():
        raise FileNotFoundError('视觉模型缺失，请按README安装官方模型')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest['sha256']:
        raise ValueError('视觉模型校验不匹配，拒绝加载')
    return data


def create_detector(model):
    import mediapipe as mediapipe
    if mediapipe.__version__ != '0.10.21':
        raise RuntimeError('仅允许已审查的MediaPipe 0.10.21；请按固定依赖安装')
    from mediapipe.tasks.python import vision
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mediapipe.tasks.BaseOptions(model_asset_buffer=model),
        running_mode=vision.RunningMode.VIDEO, num_faces=2,
        min_face_detection_confidence=.6, min_face_presence_confidence=.6,
        min_tracking_confidence=.6))


def _latest(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


def _mark_loss(counter):
    with counter.get_lock():
        counter.value += 1


def camera_worker(index, frames, stop, loss, resolution=(640,480), preview_enabled=None):
    capture = detector = None
    try:
        import cv2
        import mediapipe
        model = verified_model()  # failure before opening a device
        detector = create_detector(model)
        if stop.is_set():
            return
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            raise RuntimeError('摄像头无法打开：检查占用、编号及Windows权限')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        capture.set(cv2.CAP_PROP_FPS, 30)
        last_ms = -1
        while not stop.is_set():
            captured_at = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('摄像头读取失败，已停止采集')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stamp = max(last_ms + 1, int(captured_at * 1000))
            last_ms = stamp
            result = detector.detect_for_video(mediapipe.Image(
                image_format=mediapipe.ImageFormat.SRGB, data=rgb), stamp)
            size = (frame.shape[1], frame.shape[0])
            if len(result.face_landmarks) != 1:
                observation = FeatureFrame(captured_at, reason='需要画面中恰好一张脸', capture_size=size)
            else:
                observation = extract_features(result.face_landmarks[0], captured_at, size)
                if preview_enabled is not None and preview_enabled.is_set():
                    from .eye_preview import eye_preview
                    crop=eye_preview(rgb,result.face_landmarks[0])
                    if crop is not None:
                        observation=replace(observation,preview_ppm=crop)
            if not observation.usable():
                _mark_loss(loss)  # monotonic counter cannot lose set/clear races
            if not stop.is_set():
                _latest(frames, ('frame', observation))
    except Exception as exc:
        _mark_loss(loss)
        _latest(frames, ('error', f'{type(exc).__name__}: {exc}'))
    finally:
        if capture is not None:
            capture.release()
        if detector is not None:
            detector.close()


class CameraService:
    def __init__(self):
        self.process = None
        self.frames = self.stop_event = self.loss = None
        self.stopping_at = None
        self.seen_losses = 0
        self.preview_enabled = None

    def set_preview(self, enabled):
        if self.preview_enabled is not None:
            self.preview_enabled.set() if enabled else self.preview_enabled.clear()

    @property
    def running(self):
        return self.process is not None

    def start(self, index, resolution=(640,480)):
        if self.running:
            raise RuntimeError('旧采集进程尚未退出')
        if type(index) is not int or not 0 <= index <= 9:
            raise ValueError('摄像头编号应为0–9')
        if resolution not in ((640,480),(1280,720)):
            raise ValueError('不支持的请求分辨率')
        context = mp.get_context('spawn')
        self.frames = context.Queue(maxsize=1)
        self.stop_event, self.loss = context.Event(), context.Value('L', 0)
        self.preview_enabled = context.Event()
        self.seen_losses = 0
        self.process = context.Process(target=camera_worker,
            args=(index, self.frames, self.stop_event, self.loss, resolution, self.preview_enabled), daemon=True)
        try:
            self.process.start()
        except Exception:
            self.frames.close()
            self.process = None
            raise

    def stop(self):
        self.set_preview(False)
        if self.running and self.stopping_at is None:
            self.stop_event.set()
            self.stopping_at = time.monotonic()

    def poll(self):
        if not self.running:
            return False, None
        with self.loss.get_lock():
            total_losses = self.loss.value
        lost = total_losses != self.seen_losses
        self.seen_losses = total_losses
        item = None
        try:
            item = self.frames.get_nowait()
        except queue.Empty:
            pass
        if self.stopping_at is not None:
            item = None  # cancellation discards queued frame
            if time.monotonic()-self.stopping_at > 3 and self.process.is_alive():
                self.process.terminate()  # bounded fallback for a stuck camera driver
                item = ('error', '驱动未及时退出，已强制结束采集进程')
        if not self.process.is_alive():
            self.process.join(timeout=0)
            self.process.close()
            self.process = None
            self.frames.cancel_join_thread()
            self.frames.close()
            if item is None:
                item = ('stopped', '采集已停止' if self.stopping_at is not None else '采集异常退出，请重新开启')
            self.stopping_at = None
            lost = True
        return lost, item
