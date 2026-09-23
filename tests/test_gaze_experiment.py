import unittest
from pathlib import Path
from types import SimpleNamespace
import hashlib
import tempfile
from unittest.mock import patch

import numpy as np

from apps.desktop.gaze_experiment import estimate_local, prepare_inputs
from apps.desktop.openvino_gaze import OpenVinoGazeRunner


class GazeExperimentTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((200, 300, 3), dtype=np.uint8)
        self.points = [SimpleNamespace(x=.5, y=.5) for _ in range(478)]
        for index, x in ((362, .56), (263, .72), (33, .28), (133, .44)):
            self.points[index] = SimpleNamespace(x=x, y=.5)

    def test_prepares_separate_bgr_eye_inputs_and_true_angles(self):
        self.frame[:, :150, 0] = 12
        self.frame[:, 150:, 0] = 48
        result = prepare_inputs(self.frame, self.points, (1., -2., 3.))
        self.assertEqual(result['left_eye_image'].shape, (1, 3, 60, 60))
        self.assertEqual(result['right_eye_image'].shape, (1, 3, 60, 60))
        self.assertEqual(result['left_eye_image'][0, 0, 30, 30], 48)
        self.assertEqual(result['right_eye_image'][0, 0, 30, 30], 12)
        np.testing.assert_array_equal(result['head_pose_angles'], [[1., -2., 0.]])

    def test_roll_alignment_rotates_input_and_restores_output_direction(self):
        import cv2
        self.frame[:, :, 0] = np.arange(200, dtype=np.uint8)[:, None]
        from apps.desktop.gaze_experiment import _eye_crop
        original = _eye_crop(self.frame, self.points, (362, 263))[0].transpose(1, 2, 0)
        aligned = prepare_inputs(self.frame, self.points, (0., 0., 90.))['left_eye_image']
        # Rotation acts on the original 86px crop before the 60px resize.
        raw = self.frame[57:143, 149:235]
        rotation = cv2.getRotationMatrix2D((43, 43), 90., 1.)
        expected = cv2.resize(cv2.warpAffine(raw, rotation, (86,86),
                             borderMode=cv2.BORDER_REPLICATE), (60,60))
        np.testing.assert_array_equal(aligned[0].transpose(1,2,0), expected)
        self.assertFalse(np.array_equal(original, expected))
        for roll, expected_xy in ((90., (0.,-1.)), (-90., (0.,1.)), (0., (1.,0.))):
            vector = estimate_local(self.frame, self.points, (0.,0.,roll), 1.,
                                    lambda inputs: np.array([[1.,0.,2.]]))
            np.testing.assert_allclose(vector.xyz[:2], expected_xy, atol=1e-12)
            self.assertEqual(vector.xyz[2], 2.)

    def test_rejects_proxy_and_out_of_frame_crop(self):
        with self.assertRaises(ValueError):
            prepare_inputs(self.frame, self.points, (.5, .5, .3, 0., 0.))
        self.points[362] = SimpleNamespace(x=0., y=.5)
        self.points[263] = SimpleNamespace(x=.05, y=.5)
        with self.assertRaises(ValueError):
            prepare_inputs(self.frame, self.points, (0., 0., 0.))

    def test_inference_output_is_direction_only_and_invalid_output_rejected(self):
        result = estimate_local(self.frame, self.points, (0., 0., 0.), 1.,
                                lambda inputs: np.array([[.1, -.2, .9]]))
        self.assertEqual(result.xyz, (.1, -.2, .9))
        with self.assertRaises(ValueError):
            estimate_local(self.frame, self.points, (0., 0., 0.), 1.,
                           lambda inputs: np.array([float('nan'), 0., 1.]))
        with self.assertRaises(ValueError):
            estimate_local(self.frame, self.points, (0., 0., 0.), 1.,
                           lambda inputs: np.zeros(3))


class OpenVinoRunnerTests(unittest.TestCase):
    def test_verified_files_and_cpu_only_inference(self):
        class Port:
            def __init__(self, name):
                self.name = name

            def get_any_name(self):
                return self.name

        class Compiled:
            inputs = [Port(name) for name in sorted(('left_eye_image', 'right_eye_image',
                                                       'head_pose_angles'))]
            outputs = [Port('gaze_vector')]

            def output(self, name):
                return self.outputs[0]

            def __call__(self, inputs):
                return {self.outputs[0]: np.array([[.1, .2, .9]])}

        class Core:
            def read_model(self, *, model, weights):
                self.paths = (model, weights)
                return 'verified-model'

            def compile_model(self, model, device):
                self.assertions = (model, device)
                return Compiled()

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            xml, weights = folder / 'gaze.xml', folder / 'gaze.bin'
            xml.write_bytes(b'<net/>')
            weights.write_bytes(b'weights')
            xml_hash = hashlib.sha256(xml.read_bytes()).hexdigest()
            bin_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
            core = Core()
            with patch('apps.desktop.openvino_gaze.WEIGHTS', folder.resolve()):
                runner = OpenVinoGazeRunner(xml, weights, xml_hash, bin_hash, core=core)
                self.assertEqual(core.assertions, ('verified-model', 'CPU'))
                self.assertEqual(runner({name: 1 for name in ('left_eye_image',
                                 'right_eye_image', 'head_pose_angles')}).shape, (1, 3))
                with self.assertRaises(ValueError):
                    runner({'left_eye_image': 1})
                with self.assertRaises(ValueError):
                    OpenVinoGazeRunner(xml, weights, '0' * 64, bin_hash, core=Core())
                weights.write_bytes(b'modified')
                with self.assertRaises(ValueError):
                    OpenVinoGazeRunner(xml, weights, xml_hash, bin_hash, core=Core())

    def test_rejects_files_outside_allowed_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            xml = folder / 'gaze.xml'
            xml.write_bytes(b'<net/>')
            with patch('apps.desktop.openvino_gaze.WEIGHTS', folder / 'other'):
                with self.assertRaises(ValueError):
                    OpenVinoGazeRunner(xml, xml, hashlib.sha256(xml.read_bytes()).hexdigest(),
                                       '0' * 64, core=object())


if __name__ == '__main__':
    unittest.main()
