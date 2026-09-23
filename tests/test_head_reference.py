from dataclasses import replace
import unittest
from apps.desktop.head_reference import HeadReference, guidance
from apps.desktop.calibration import CalibrationSession, fit_mapping, validate_mapping, TRAIN_TARGETS, CHECK_TARGETS
from apps.desktop.config import Settings
from test_calibration import feature, groups, HEAD


class HeadReferenceTests(unittest.TestCase):
    def test_gaze_rejection_preserves_explicitly_valid_posture_only(self):
        history=HeadReference()
        for i in range(30):
            frame=replace(feature((.5,.5),i*.06), valid=False, features=(), head_valid=True)
            self.assertFalse(frame.usable())
            history.feed(frame)
        self.assertEqual(history.candidate(1.74),HEAD)
        self.assertIsNone(history.candidate(3.))
        history.feed(replace(frame,timestamp=1.8,head_valid=False))
        self.assertIsNone(history.candidate(1.8))

    def stable(self):
        history=HeadReference()
        for i in range(30):
            history.feed(feature((.5,.5),i*.06))
        return history

    def test_stable_reference_and_stale_rejection(self):
        history=self.stable()
        self.assertEqual(history.candidate(1.74),HEAD)
        self.assertIsNone(history.candidate(3))

    def test_invalid_and_gap_clear_history(self):
        history=self.stable()
        history.feed(replace(feature((.5,.5),1.8),valid=False))
        self.assertIsNone(history.candidate(1.8))
        history=self.stable()
        history.feed(feature((.5,.5),4))
        self.assertEqual(len(history.frames),1)

    def test_unstable_not_ready(self):
        history=HeadReference()
        for i in range(30):
            history.feed(replace(feature((.5,.5),i*.06),head=(.4 if i%2 else .6,*HEAD[1:])))
        self.assertIsNone(history.candidate(1.74))

    def test_reference_persists_in_mapping(self):
        reference=(.51,.41,.3,.01,0.)
        mapping=fit_mapping(groups(TRAIN_TARGETS),reference)
        self.assertEqual(mapping.head,reference)

    def test_deviation_pauses_without_losing_completed_points(self):
        session=CalibrationSession(Settings(),(800,600),reference=HEAD)
        session.begin_point(0)
        for i in range(50):
            if session.feed(feature(session.target,i*.06)):
                break
        self.assertEqual(session.index,1)
        session.begin_point(4)
        session.feed(replace(feature(session.target,4.1),head=(.8,*HEAD[1:])))
        self.assertIsNone(session.started)
        session.feed(feature(session.target,4.2))
        self.assertIsNone(session.started)
        self.assertEqual(len(session.groups),1)
        self.assertEqual(session.buffer,[])

    def test_direction_and_distance_hints(self):
        self.assertIn('范围内',guidance(HEAD,HEAD))
        self.assertIn('向左',guidance(HEAD,(.7,*HEAD[1:])))
        self.assertIn('远离',guidance(HEAD,(.5,.4,.5,0,0)))

    def test_canvas_aspect_ratio_does_not_change_exact_mapping(self):
        mapping=fit_mapping(groups(TRAIN_TARGETS))
        for size in ((800,600),(1566,946),(400,900)):
            metrics=validate_mapping(mapping,groups(CHECK_TARGETS),Settings(),size)
            self.assertLess(metrics['max_error'],1e-12)


if __name__=='__main__':
    unittest.main()
