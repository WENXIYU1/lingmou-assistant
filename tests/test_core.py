import math
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from apps.desktop.calibration import AdaptationStatus
from apps.desktop.config import Settings
from apps.desktop.executor import PracticeAction, PracticeExecutor
from apps.desktop.interaction import Controller, HoldGesture
from apps.desktop.storage import load_settings, save_settings
from apps.desktop.vision import Observation


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.c = Controller(Settings(smoothing_seconds=0))
        self.t = 0.0

    def frame(self, **kwargs):
        self.t += .05
        return self.c.observe(Observation(self.t, **kwargs))

    def start(self):
        self.frame()
        self.assertTrue(self.c.resume())
        self.frame()

    def test_default_pause_and_uncalibrated_lock(self):
        self.assertIsNone(self.frame())
        self.assertFalse(AdaptationStatus().system_control_allowed)
        with self.assertRaises(PermissionError):
            PracticeExecutor(self.c).execute_system('click')

    def test_pause_never_released_by_closed_mouth(self):
        self.start()
        self.c.pause()
        for _ in range(20):
            self.assertIsNone(self.frame(mouth_open=False))

    def test_open_mouth_pauses_closed_does_not_resume(self):
        self.start()
        self.frame(mouth_open=True)
        self.frame(mouth_open=False)
        self.assertTrue(self.c.paused)

    def test_tracking_loss_requires_resume(self):
        self.start()
        self.frame(valid=False)
        self.frame()
        self.assertFalse(self.c.can_practice)
        self.assertTrue(self.c.resume())

    def test_stale_frame_pauses(self):
        self.start()
        self.t += 1
        self.assertIsNone(self.frame())
        self.assertTrue(self.c.paused)

    def test_watchdog_without_frames(self):
        self.start()
        self.c.watchdog(self.t + 1)
        self.assertFalse(self.c.can_practice)

    def test_invalid_or_out_of_order_frame(self):
        for kwargs in ({'x': math.nan}, {'y': 1.2}, {'valid': 'yes'}):
            self.start()
            self.assertIsNone(self.frame(**kwargs))
            self.assertTrue(self.c.paused)
        self.start()
        self.c.observe(Observation(self.t - 1))
        self.assertTrue(self.c.paused)

    def test_short_bilateral_blink_never_clicks(self):
        self.start()
        for _ in range(30):
            self.assertIsNone(self.frame(left_closed=True, right_closed=True)[1])

    def test_short_unilateral_blink_never_clicks(self):
        self.start()
        for _ in range(3):
            self.assertIsNone(self.frame(left_closed=True)[1])
        self.assertIsNone(self.frame()[1])

    def test_hold_once_then_release(self):
        self.start()
        actions = [self.frame(left_closed=True)[1] for _ in range(40)]
        self.assertEqual(actions.count('left'), 1)
        self.frame()
        actions = [self.frame(left_closed=True)[1] for _ in range(20)]
        self.assertEqual(actions.count('left'), 1)

    def test_pause_clears_pending_hold_until_release(self):
        self.start()
        for _ in range(8):
            self.frame(left_closed=True)
        self.c.pause()
        self.c.resume()
        for _ in range(30):
            self.assertIsNone(self.frame(left_closed=True)[1])

    def test_reset_reanchors_without_jump(self):
        self.start()
        self.frame(x=.8)
        self.c.pause()
        self.c.resume()
        self.assertEqual(self.frame(x=.8)[0], (.5, .5))

    def test_axes_deadzone_independent(self):
        self.start()
        x, y = self.frame(x=.7, y=.504)[0]
        self.assertAlmostEqual(x, .7)
        self.assertEqual(y, .5)

    def test_gains_independent_and_clamped(self):
        self.c.configure(replace(self.c.settings, gain_x=2, gain_y=.5))
        self.start()
        self.assertEqual(self.frame(x=.9, y=.9)[0], (1., .7))

    def test_smoothing_is_monotonic_and_bounded(self):
        self.c.configure(Settings(smoothing_seconds=.1, deadzone_x=0))
        self.start()
        positions = [self.frame(x=.9)[0][0] for _ in range(20)]
        self.assertTrue(all(.5 < x <= .9 for x in positions))
        self.assertEqual(sorted(positions), positions)

    def test_zero_time_does_not_fire(self):
        self.start()
        self.c.observe(Observation(self.t, left_closed=True))
        self.assertTrue(self.c.paused)

    def test_left_eye_hold_does_not_move_pointer(self):
        self.start()
        self.assertEqual(self.frame(x=.9, left_closed=True)[0], (.5, .5))

    def test_right_hold_once(self):
        self.start()
        actions = [self.frame(right_closed=True)[1] for _ in range(40)]
        self.assertEqual(actions.count('right'), 1)

    def test_recording_invalidates_and_ignores_mouth(self):
        self.start()
        self.c.set_recording(True)
        self.assertTrue(self.c.paused)
        self.c.resume()
        self.frame(mouth_open=True)
        self.assertTrue(self.c.can_practice)

    def test_fault_and_close_cannot_resume(self):
        self.start()
        self.c.fail('test fault')
        self.frame()
        self.assertFalse(self.c.resume())
        self.c.close()
        self.assertIsNone(self.frame())

    def test_configure_clears_and_pauses(self):
        self.start()
        self.frame(x=.7)
        self.c.configure(Settings())
        self.assertTrue(self.c.paused)
        self.assertEqual(self.c.motion.position, (.5, .5))

    def test_duplicate_and_late_actions_rejected(self):
        self.start()
        executor = PracticeExecutor(self.c)
        action = PracticeAction('one', self.c.epoch, 'left')
        self.assertTrue(executor.execute(action))
        self.assertFalse(executor.execute(action))
        self.c.pause()
        self.c.resume()
        self.assertFalse(executor.execute(action))
        self.assertFalse(executor.execute(PracticeAction('two', self.c.epoch, 'shell')))


class ConfigurationTests(unittest.TestCase):
    def test_reject_bad_values(self):
        for value in (math.nan, math.inf, -1, 99, True, '1'):
            with self.assertRaises(ValueError):
                Settings(gain_x=value)

    def test_save_load_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '练习.json'
            save_settings(path, Settings())
            save_settings(path, Settings(gain_x=2))
            self.assertEqual(load_settings(path).gain_x, 2)
            self.assertEqual(load_settings(path.with_suffix('.json.bak')), Settings())

    def test_broken_file_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text('{broken', encoding='utf-8')
            with self.assertRaises(ValueError):
                save_settings(path, Settings())
            self.assertEqual(path.read_text(encoding='utf-8'), '{broken')

    def test_incompatible_or_extra_fields_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            for data in ({'schema_version': 2, 'settings': Settings().to_dict()},
                         {'schema_version': 1, 'settings': {}, 'calibrated': True}):
                path.write_text(json.dumps(data), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_settings(path)

    def test_hold_change_requires_release(self):
        h = HoldGesture(.5)
        h.update(None, 0)
        h.update('left', .1)
        self.assertIsNone(h.update('right', .6))
        self.assertIsNone(h.update('right', 2))


if __name__ == '__main__':
    unittest.main()
