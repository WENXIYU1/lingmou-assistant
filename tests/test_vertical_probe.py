from dataclasses import replace
import unittest
from apps.desktop.vertical_probe import VerticalProbe, TARGETS, paired_samples
from test_calibration import HEAD, feature


class VerticalProbeTests(unittest.TestCase):
    def test_prestart_inflight_frame_ignored_on_begin_and_retry(self):
        from apps.desktop.direction_probe import DirectionProbe
        for kind in (VerticalProbe, DirectionProbe):
            probe=kind(HEAD)
            for start in (10.,20.):
                probe.begin(start)
                self.assertFalse(probe.feed(feature(TARGETS[0],start-.125)))
                self.assertEqual(probe.started,start)
                self.assertIsNone(probe.last)
                self.assertEqual(probe.buffer,[])
                self.assertEqual(probe.error,'')
                for i in range(45):
                    if probe.feed(feature(TARGETS[0],start+i*.09)):
                        break
                self.assertIsNone(probe.started)
            self.assertEqual(len(probe.groups),2)

    def test_reverse_timestamp_not_hidden_as_prestart_frame(self):
        probe=VerticalProbe(HEAD)
        probe.begin(10.)
        probe.feed(feature(TARGETS[0],10.1))
        probe.feed(feature(TARGETS[0],9.9))
        self.assertIsNone(probe.started)
        self.assertIn('时间倒退',probe.error)

    def test_prestart_invalid_frame_still_pauses(self):
        probe=VerticalProbe(HEAD)
        probe.begin(10.)
        probe.feed(replace(feature(TARGETS[0],9.9),valid=False))
        self.assertIsNone(probe.started)

    def test_causal_window_future_cannot_change_past(self):
        group=[(i*.04,(float(i>=10),None,None)) for i in range(20)]
        a,b,spans=paired_samples(group,0)
        past=paired_samples(group[:12],0)
        self.assertEqual(past[1],b[:len(past[1])])
        self.assertTrue(all(0<=s<=.20 for s in spans))
        self.assertEqual(a[8],1.)
        self.assertLess(b[8],1.)  # Step response is delayed, not magically instantaneous.
        self.assertEqual(b[-1],1.)

    def test_four_points_report_and_scalar_only_storage(self):
        probe=VerticalProbe(HEAD)
        for point, target in enumerate(TARGETS):
            start=point*5.
            probe.begin(start)
            for i in range(70):
                f=feature(target,start+i*.05)
                f=replace(f,left_eye=f.features,right_eye=f.features,preview_ppm=b'private-image')
                if probe.feed(f):
                    break
            self.assertEqual(len(probe.groups),point+1)
        self.assertTrue(probe.done)
        report=probe.report()
        self.assertIn('短时中位数',report)
        self.assertIn('均在上下中位之间：是',report)
        self.assertNotIn('private-image',repr(probe.groups))
        self.assertIsNone(probe.target)

    def test_pause_requires_explicit_restart(self):
        probe=VerticalProbe(HEAD)
        probe.begin(0)
        probe.pause()
        for i in range(100):
            probe.feed(feature(TARGETS[0],i*.05))
        self.assertEqual(probe.groups,[])
        self.assertEqual(probe.buffer,[])

    def test_gap_and_head_drift_pause(self):
        for bad in (feature(TARGETS[0],1), replace(feature(TARGETS[0],.1),head=(.9,.4,.3,0,0))):
            probe=VerticalProbe(HEAD)
            probe.begin(0)
            probe.feed(feature(TARGETS[0],0))
            probe.feed(bad)
            self.assertIsNone(probe.started)
            self.assertTrue(probe.error)

    def test_incomplete_and_missing_channels(self):
        probe=VerticalProbe(HEAD)
        self.assertIn('完成 0/4',probe.report())
        probe.groups=[[(i*.05,(.1,None,None)) for i in range(40)] for _ in range(4)]
        self.assertIn('配对样本不足',probe.report())
        self.assertIn('均在上下中位之间：否',probe.report())

    def test_timeout(self):
        probe=VerticalProbe(HEAD)
        probe.begin(0)
        probe.feed(feature(TARGETS[0],13))
        self.assertIn('超时',probe.error)


if __name__=='__main__':
    unittest.main()
