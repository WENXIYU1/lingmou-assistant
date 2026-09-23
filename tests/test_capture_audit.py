from dataclasses import replace
import unittest
from apps.desktop.capture_audit import CaptureAudit
from apps.desktop.calibration import fit_mapping, TRAIN_TARGETS
from apps.desktop.mapping_diagnostics import MappingDiagnostics
from apps.desktop.config import Settings
from apps.desktop.eye_features import extract_features
from test_calibration import feature, groups, landmarks, HEAD


class CaptureAuditTests(unittest.TestCase):
    def test_no_data_and_expiry(self):
        audit=CaptureAudit()
        self.assertIn('尚无最近数据',audit.report(0))
        audit.feed(feature((.5,.5),0),.1)
        self.assertIn('尚无最近数据',audit.report(11))

    def test_rates_widths_and_stale(self):
        audit=CaptureAudit()
        for i in range(21):
            audit.feed(replace(feature((.5,.5),i*.1),eye_widths_px=(40,42)),i*.1+.05)
        report=audit.report(2.05)
        self.assertIn('10.0/秒',report)
        self.assertIn('左40.0px / 右42.0px',report)
        self.assertIn('640×480',report)
        audit.feed(feature((.5,.5),2),3)
        self.assertIn('过期/时间异常',audit.report(3))

    def test_bounded_and_reset(self):
        audit=CaptureAudit()
        for i in range(400):
            audit.feed(feature((.5,.5),i*.01),i*.01)
        self.assertEqual(len(audit.samples),300)
        audit.clear()
        self.assertEqual(len(audit.samples),0)

    def test_pixel_width_extraction(self):
        frame=extract_features(landmarks(),0,(640,480))
        self.assertAlmostEqual(frame.eye_widths_px[0],64)
        self.assertAlmostEqual(frame.eye_widths_px[1],64)

    def test_retained_shortage_is_explicit(self):
        data=groups(TRAIN_TARGETS)
        data[0]=data[0][:18]
        data[0][-1]=replace(data[0][-1],features=(.9,.3))
        audit=[]
        with self.assertRaisesRegex(ValueError,'剔除后样本不足'):
            fit_mapping(data,audit=audit)
        self.assertEqual(len(audit),5)
        self.assertEqual(audit[0]['kept'],17)
        self.assertEqual(audit[0]['removed'],1)

    def test_spread_failure_is_explicit(self):
        data=groups(TRAIN_TARGETS)
        data[0]=[replace(f,features=(.4+(.08 if i%2 else -.08),.05)) for i,f in enumerate(data[0][:24])]
        with self.assertRaisesRegex(ValueError,'波动超限'):
            fit_mapping(data)

    def test_affine_amplification_matches_finite_difference(self):
        d=MappingDiagnostics()
        d.fit(groups(TRAIN_TARGETS),HEAD,Settings(),(800,600))
        mapping=d.candidates['average']
        a=mapping.predict((.45,.07))
        b=mapping.predict((.45,.071))
        expected=d.amplification['average'][1]
        self.assertAlmostEqual(expected[0],(b[0]-a[0])*800)
        self.assertAlmostEqual(expected[1],(b[1]-a[1])*600)
