"""Synthetic tests: feature units must not masquerade as screen accuracy."""
from dataclasses import replace
import unittest
from apps.desktop.calibration import fit_mapping, TRAIN_TARGETS
from apps.desktop.config import Settings
from test_calibration import groups


class GazeQualityTests(unittest.TestCase):
    def samples(self, scale, amplitude):
        return [[replace(s, features=(scale*(target[0]+amplitude*((i%3)-1)),
                                      scale*target[1]))
                 for i,s in enumerate(group)]
                for target,group in zip(TRAIN_TARGETS,groups(TRAIN_TARGETS))]

    def test_screen_noise_gate_is_invariant_to_feature_units(self):
        for scale in (.2, 1.):
            mapping = fit_mapping(self.samples(scale,.06), stabilize=False,
                                  screen_quality=(Settings(),(800,600)))
            self.assertIsNotNone(mapping)
        # Old geometric path keeps its old numerical gate.
        with self.assertRaisesRegex(ValueError, '波动超限'):
            fit_mapping(self.samples(1.,.06))

    def test_screen_noise_rejects_unstable_all_frame_output(self):
        for scale in (.2, 1.):
            with self.assertRaisesRegex(ValueError, '屏幕波动RMS'):
                fit_mapping(self.samples(scale,.3), stabilize=False,
                            screen_quality=(Settings(),(800,600)))
