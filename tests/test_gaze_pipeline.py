from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from apps.desktop.calibration import CalibrationSession, MAX_LIMIT
from apps.desktop.config import Settings
from apps.desktop.eye_features import FeatureFrame
from apps.desktop.gaze_pipeline import GazePipeline, face_tensor


class GazePipelineTests(unittest.TestCase):
    def test_direction_replaces_geometry_and_clears_incompatible_channels(self):
        class Pose:
            def __call__(self, inputs):
                return {key: np.array([[0.]]) for key in ('angle_y_fc','angle_p_fc','angle_r_fc')}
            def output(self, name):
                return name
        pipeline = object.__new__(GazePipeline)
        pipeline.pose = Pose()
        pipeline.gaze = None
        base = FeatureFrame(1., (.4,.1), (.5,.4,.3,0.,0.), True, left_eye=(.4,.1), right_eye=(.4,.1))
        with patch('apps.desktop.gaze_pipeline.face_tensor', return_value=np.zeros((1,3,60,60))), \
             patch('apps.desktop.gaze_pipeline.estimate_local', return_value=SimpleNamespace(xyz=(3.,4.,12.))):
            result = pipeline.infer(None, None, base)
        np.testing.assert_allclose(result.features, (3/13,4/13))
        self.assertEqual(result.left_eye, ())
        self.assertEqual(result.head, base.head)
        with patch('apps.desktop.gaze_pipeline.face_tensor', return_value=np.zeros((1,3,60,60))), \
             patch('apps.desktop.gaze_pipeline.estimate_local', return_value=SimpleNamespace(xyz=(3.,4.,-12.))):
            negative_z = pipeline.infer(None, None, base)
        self.assertTrue(negative_z.usable())
        np.testing.assert_allclose(negative_z.features, result.features)
        invalid = replace(base, valid=False)
        self.assertIs(pipeline.infer(None, None, invalid), invalid)

    def test_face_crop_rejects_outside_image(self):
        image = np.zeros((200,300,3), dtype=np.uint8)
        landmarks = [SimpleNamespace(x=.5,y=.5) for _ in range(478)]
        landmarks[0] = SimpleNamespace(x=-.5,y=.5)
        with self.assertRaises(ValueError):
            face_tensor(image, landmarks)

    def test_new_model_still_requires_fresh_final_validation(self):
        session = CalibrationSession(Settings(), (800,600), experimental=True)
        now = 0.
        def point(offset=0.):
            nonlocal now
            target = session.target
            session.begin_point(now)
            for _ in range(80):
                now += .06
                frame = FeatureFrame(now, (.1 + .2*(target[0]+offset), .02+.1*target[1]),
                                     (.5,.4,.3,0.,0.), True)
                if session.feed(frame):
                    return
            self.fail('synthetic point failed to complete')
        for _ in range(5):
            point()
        self.assertEqual(session.phase, 'train')
        self.assertEqual(session.target, (.5,.5))
        point(offset=.2)  # Drift observation must not enter the five-point fit.
        self.assertEqual(session.phase, 'select')
        self.assertEqual(len(session.candidates), 3)
        self.assertIn('不参与拟合', session.center_report)
        for _ in range(3):
            point()
        self.assertEqual(session.phase, 'verify')
        self.assertIsNone(session.metrics)
        for _ in range(3):
            point(offset=.4)
        self.assertEqual(session.phase, 'failed')
        self.assertFalse(session.metrics['passed'])
        self.assertGreater(session.metrics['max_error'], MAX_LIMIT)
