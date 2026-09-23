import unittest
from apps.desktop.interaction import StableMotionFilter
from apps.desktop.config import Settings
from apps.desktop.vision import Observation


class StableMotionTests(unittest.TestCase):
    def test_isolated_spike_is_suppressed_but_disables_dwell(self):
        f=StableMotionFilter(Settings(smoothing_seconds=.2),absolute=True)
        for i in range(8):
            f.update(Observation(i*.06,.4,.4))
        result=f.update(Observation(.48,.9,.1))
        self.assertEqual(result,(.4,.4))
        self.assertFalse(f.stable)

    def test_sustained_move_follows_without_permanent_lock(self):
        f=StableMotionFilter(Settings(smoothing_seconds=.2),absolute=True)
        for i in range(8):
            f.update(Observation(i*.06,.2,.3))
        for i in range(8,20):
            result=f.update(Observation(i*.06,.8,.3))
        self.assertGreater(result[0],.75)
        self.assertAlmostEqual(result[1],.3)
        self.assertTrue(f.stable)

    def test_gap_and_invalid_input_need_new_warmup(self):
        f=StableMotionFilter(Settings(),absolute=True)
        for i in range(5):
            f.update(Observation(i*.06,.5,.5))
        self.assertIsNone(f.update(Observation(1.,.8,.8)))
        self.assertIsNone(f.update(Observation(1.06,.8,.8,valid=False)))
        self.assertIsNone(f.update(Observation(1.12,.8,.8)))
