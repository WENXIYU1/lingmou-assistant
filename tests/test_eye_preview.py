import unittest
import numpy as np
from apps.desktop.eye_preview import eye_preview
from apps.desktop.camera import CameraService
from test_calibration import landmarks


class PreviewTests(unittest.TestCase):
    def test_bounded_crop_and_no_source_mutation(self):
        rgb=np.zeros((480,640,3),dtype=np.uint8)
        packet=eye_preview(rgb,landmarks())
        header,width_height,maximum,data=packet.split(b'\n',3)
        width,height=map(int,width_height.split())
        self.assertEqual(header,b'P6')
        self.assertEqual(maximum,b'255')
        self.assertLessEqual(width,600)
        self.assertLessEqual(height,360)
        self.assertEqual(len(data),width*height*3)
        self.assertFalse(rgb.any())
        self.assertNotEqual(data[:len(data)//2],data[len(data)//2:])

    def test_missing_or_invalid_landmarks_no_image(self):
        rgb=np.zeros((480,640,3),dtype=np.uint8)
        self.assertIsNone(eye_preview(rgb,[]))
        points=landmarks()
        points[469].x=float('nan')
        self.assertIsNone(eye_preview(rgb,points))

    def test_disallowed_resolution_before_camera(self):
        with self.assertRaises(ValueError):
            CameraService().start(0,(1920,1080))

    def test_preview_default_disabled(self):
        service=CameraService()
        self.assertIsNone(service.preview_enabled)
        service.set_preview(True)
        self.assertFalse(service.running)
