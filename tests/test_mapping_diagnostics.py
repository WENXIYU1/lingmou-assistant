from dataclasses import replace
import unittest
from apps.desktop.mapping_diagnostics import MappingDiagnostics, training_quality, vertical_signal
from apps.desktop.calibration import TRAIN_TARGETS, CHECK_TARGETS, CalibrationSession
from apps.desktop.config import Settings
from apps.desktop.eye_features import extract_features
from test_calibration import groups, HEAD, landmarks, feature


def binocular(targets):
    return [[replace(f,left_eye=f.features,right_eye=f.features) for f in group]
            for group in groups(targets)]


class DiagnosticTests(unittest.TestCase):
    def test_lid_center_and_vertical_movement(self):
        points=landmarks()
        before=extract_features(points,1,(640,480))
        self.assertAlmostEqual(before.average_lid[1],.5)
        for index in (469,470,471,472,474,475,476,477):
            points[index].y+=.005
        after=extract_features(points,2,(640,480))
        self.assertAlmostEqual(after.average_lid[1],.625)
        self.assertAlmostEqual(after.average_lid[0],before.average_lid[0])

    def test_closed_eye_no_lid_candidate(self):
        points=landmarks()
        points[159].y=points[145].y
        result=extract_features(points,1,(640,480))
        self.assertFalse(result.usable())
        self.assertEqual(result.average_lid,())

    def test_candidate_comparison_no_validation_refit(self):
        def data(targets):
            return [[replace(f,average_lid=f.features,left_lid=f.features,right_lid=f.features)
                     for f in group] for group in binocular(targets)]
        d=MappingDiagnostics()
        d.fit(data(TRAIN_TARGETS),HEAD,Settings(),(800,600))
        before=d.candidates.copy()
        d.evaluate(data(CHECK_TARGETS),Settings(),(800,600))
        self.assertEqual(len(d.candidates),6)
        self.assertTrue(d.validation['average_lid'][0]['passed'])
        self.assertEqual(d.candidates,before)

    def test_training_gate_boundary_and_nonfinite(self):
        rows=[{'point':i+1,'median':.10} for i in range(5)]
        self.assertTrue(training_quality(rows)[0])
        rows[2]['median']=.10001
        self.assertFalse(training_quality(rows)[0])
        rows[2]['median']=float('nan')
        self.assertFalse(training_quality(rows)[0])

    def test_bad_training_stops_before_independent_validation(self):
        session=CalibrationSession(Settings(),(800,600),reference=HEAD)
        now=0.
        for index in range(5):
            session.begin_point(now)
            target=(.9,.8) if index==0 else session.target
            for _ in range(100):
                now+=.06
                if session.feed(feature(target,now)):
                    break
        self.assertEqual(session.phase,'failed')
        self.assertIsNone(session.mapping)
        self.assertIsNone(session.metrics)
        self.assertIn('训练质量',session.error)
        self.assertEqual(session.analysis.validation,{})
        self.assertTrue(session.analysis.training)

    def test_vertical_signal_reports_range_and_noise(self):
        summary=vertical_signal(groups(TRAIN_TARGETS))
        self.assertAlmostEqual(summary['down_minus_up'],.06)
        self.assertEqual(summary['spreads'],[0.]*5)

    def fitted(self):
        d=MappingDiagnostics()
        d.fit(binocular(TRAIN_TARGETS),HEAD,Settings(),(800,600))
        return d

    def test_extraction_keeps_individual_eyes_and_average(self):
        f=extract_features(landmarks(),1,(640,480))
        self.assertEqual(len(f.left_eye),2)
        self.assertEqual(len(f.right_eye),2)
        for i in range(2):
            self.assertAlmostEqual(f.features[i],(f.left_eye[i]+f.right_eye[i])/2)

    def test_training_fit_and_heldout_report(self):
        d=self.fitted()
        d.evaluate(binocular(CHECK_TARGETS),Settings(),(800,600))
        self.assertEqual(len(d.training['average']),5)
        self.assertEqual(len(d.validation['left_eye'][1]),3)
        self.assertLess(d.validation['right_eye'][0]['max_error'],1e-12)

    def test_bad_validation_never_refits_candidates(self):
        d=self.fitted()
        before=d.candidates.copy()
        d.evaluate(binocular([(.95,.95)]*3),Settings(),(800,600))
        self.assertEqual(before,d.candidates)
        self.assertFalse(d.validation['average'][0]['passed'])
        self.assertLess(d.training['average'][0]['max'],1e-12)

    def test_one_eye_degrades_and_head_change_recorded(self):
        d=self.fitted()
        data=binocular(CHECK_TARGETS)
        data=[[replace(f,left_eye=(f.left_eye[0],f.left_eye[1]+.04),
                       head=(.51,.4,.315,.01,0)) for f in group] for group in data]
        d.evaluate(data,Settings(),(800,600))
        self.assertFalse(d.validation['left_eye'][0]['passed'])
        self.assertTrue(d.validation['right_eye'][0]['passed'])
        row=d.validation['left_eye'][1][0]
        self.assertAlmostEqual(row['axis_abs_px'][1],240)
        self.assertAlmostEqual(row['head_delta'][2],.05)

    def test_missing_individual_data_is_not_fabricated(self):
        d=MappingDiagnostics()
        d.fit(groups(TRAIN_TARGETS),HEAD,Settings(),(800,600))
        self.assertNotIn('left_eye',d.candidates)
        self.assertIn('缺少',d.errors['left_eye'])

    def test_training_mismatch_is_visible(self):
        data=binocular(TRAIN_TARGETS)
        data[0]=binocular([(.65,.55)])[0]
        d=MappingDiagnostics()
        d.fit(data,HEAD,Settings(),(800,600))
        self.assertGreater(d.training['average'][0]['median'],.01)

    def test_empty_report_explains_old_profiles(self):
        self.assertIn('加载旧档案',MappingDiagnostics().report())
