from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from apps.desktop.calibration import (CalibrationSession, TRAIN_TARGETS, CHECK_TARGETS,
    fit_mapping, validate_mapping, validation_report)
from apps.desktop.config import Settings
from apps.desktop.eye_features import FeatureFrame, extract_features, head_matches
from apps.desktop.interaction import MotionFilter
from apps.desktop.profiles import save_profile, load_profile, profile_path
from apps.desktop.vision import Observation

HEAD=(.5,.4,.3,0.,0.)


def feature(target, timestamp):
    # Synthetic invertible affine relation, NOT recorded eye data.
    return FeatureFrame(timestamp,(.35+.2*target[0],.02+.1*target[1]),HEAD,True,'synthetic',(640,480))


def groups(targets):
    return [[feature(target,i*.06) for i in range(25)] for target in targets]


class MappingTests(unittest.TestCase):
    def test_reports_isolated_and_sustained_exceedances_without_weakening_gate(self):
        mapping=fit_mapping(groups(TRAIN_TARGETS))
        for count in (1,5):
            data=groups(CHECK_TARGETS)
            for i in range(count):
                data[0][i]=feature((.9,.5),i*.06)
            details=[]
            metrics=validate_mapping(mapping,data,Settings(),(800,600),details)
            self.assertFalse(metrics['passed'])
            self.assertEqual(details[0]['over_limit_count'],count)
            self.assertEqual(details[0]['longest_over_limit_frames'],count)
            self.assertAlmostEqual(details[0]['longest_over_limit_span'],(count-1)*.06)
            self.assertEqual(len(details[0]['timeline']),25)
            session=CalibrationSession(Settings(),(800,600),experimental=True)
            session.metrics,session.diagnostics=metrics,details
            self.assertIn('【最终独立验收】',session.diagnostic_report())
            self.assertIn('误差序列',session.diagnostic_report())

    def test_diagnostics_bias_without_jitter_and_unchanged_schema(self):
        mapping=fit_mapping(groups(TRAIN_TARGETS))
        shifted=groups([(x+.3,y-.2) for x,y in CHECK_TARGETS])
        details=[]
        metrics=validate_mapping(mapping,shifted,Settings(),(800,600),details)
        self.assertFalse(metrics['passed'])
        self.assertEqual(len(details),3)
        self.assertAlmostEqual(details[0]['bias_px'][0],240)
        self.assertAlmostEqual(details[0]['bias_px'][1],-120)
        self.assertLess(details[0]['jitter_p90'],1e-12)
        self.assertNotIn('diagnostics',metrics)
        report=validation_report(metrics,details)
        self.assertIn('右240、上120',report)
        self.assertIn('门槛 ≤18%',report)

    def test_one_outlier_still_fails_maximum_gate(self):
        mapping=fit_mapping(groups(TRAIN_TARGETS))
        data=groups(CHECK_TARGETS)
        data[1][0]=feature((1.,1.),0)
        details=[]
        metrics=validate_mapping(mapping,data,Settings(),(800,600),details)
        self.assertFalse(metrics['passed'])
        self.assertLess(metrics['median_error'],1e-12)
        self.assertIn('验证点 2',validation_report(metrics,details))

    def test_jitter_is_distinct_from_bias(self):
        mapping=fit_mapping(groups(TRAIN_TARGETS))
        data=[[feature((x+(.1 if i%2 else -.1),y),i*.06)
               for i in range(26)] for x,y in CHECK_TARGETS]
        details=[]
        validate_mapping(mapping,data,Settings(),(800,600),details)
        self.assertAlmostEqual(details[0]['bias_px'][0],0)
        self.assertAlmostEqual(details[0]['jitter_p90'],.08)

    def test_affine_fit_and_independent_validation(self):
        mapping=fit_mapping(groups(TRAIN_TARGETS))
        metrics=validate_mapping(mapping,groups(CHECK_TARGETS),Settings(),(800,600))
        self.assertTrue(metrics['passed'])
        self.assertLess(metrics['max_error'],1e-12)

    def test_insufficient_points_rejected(self):
        with self.assertRaises(ValueError):
            fit_mapping(groups(TRAIN_TARGETS)[:4])

    def test_degenerate_samples_rejected(self):
        with self.assertRaises(ValueError):
            fit_mapping(groups([(.5,.5)]*5))

    def test_invalid_sample_rejected(self):
        data=groups(TRAIN_TARGETS)
        data[0][0]=replace(data[0][0],valid=False)
        with self.assertRaises(ValueError):
            fit_mapping(data)

    def test_head_drift_rejected(self):
        data=groups(TRAIN_TARGETS)
        data[3]=[replace(s,head=(.8,.4,.3,0,0)) for s in data[3]]
        with self.assertRaises(ValueError):
            fit_mapping(data)

    def test_bad_validation_not_clipped_or_passed(self):
        mapping=fit_mapping(groups(TRAIN_TARGETS))
        metrics=validate_mapping(mapping,groups([(.95,.95)]*3),Settings(),(800,600))
        self.assertFalse(metrics['passed'])

    def test_gain_changes_need_validation(self):
        mapping=fit_mapping(groups(TRAIN_TARGETS))
        metrics=validate_mapping(mapping,groups(CHECK_TARGETS),Settings(gain_x=4,gain_y=4),(800,600))
        self.assertFalse(metrics['passed'])

    def test_absolute_filter_does_not_recenter_calibrated_gaze(self):
        f=MotionFilter(Settings(smoothing_seconds=0),absolute=True)
        self.assertEqual(f.update(Observation(1,.2,.3)),(.2,.3))
        f.reset()
        self.assertEqual(f.update(Observation(2,.8,.7)),(.8,.7))

    def test_head_quality_limits(self):
        self.assertTrue(head_matches(HEAD,HEAD))
        self.assertFalse(head_matches(HEAD,(.5,.4,.6,0,0)))
        self.assertFalse(head_matches(HEAD,(.5,.4,.3,.5,0)))


class SessionTests(unittest.TestCase):
    def test_countdown_progress_and_pause(self):
        s=CalibrationSession(Settings(),(800,600))
        s.begin_point(0)
        self.assertIn('准备 0.5 秒',s.progress_text(.5))
        s.feed(feature(s.target,.5))
        self.assertEqual(s.buffer,[])
        s.feed(feature(s.target,.75))
        s.feed(feature(s.target,1.))
        self.assertIn('1/18',s.progress_text(1.))
        s.pause()
        self.assertIn('已暂停',s.progress_text(2.))

    def fill_point(self,s,now):
        s.begin_point(now)
        target=s.target
        for _ in range(80):
            now+=.06
            if s.feed(feature(target,now)):
                break
        return now

    def test_complete_training_then_separate_check(self):
        s=CalibrationSession(Settings(),(800,600))
        now=0
        for _ in range(5):
            now=self.fill_point(s,now)
        self.assertEqual(s.phase,'select')
        self.assertIsNone(s.metrics)
        for _ in range(3):
            now=self.fill_point(s,now)
        self.assertEqual(s.phase,'verify')
        self.assertIsNone(s.metrics)
        self.assertIn('不作为最终通过证据',s.selection_report)
        for _ in range(3):
            now=self.fill_point(s,now)
        self.assertEqual(s.phase,'trial')
        self.assertTrue(s.metrics['passed'])

    def test_loaded_mapping_needs_only_fresh_final_check(self):
        mapping=fit_mapping(groups(TRAIN_TARGETS))
        s=CalibrationSession(Settings(),(800,600),mapping)
        now=0
        for _ in range(3):
            now=self.fill_point(s,now)
        self.assertEqual(s.phase,'trial')
        self.assertEqual(s.mapping,mapping)

    def test_paused_does_not_resume_on_valid_frames(self):
        s=CalibrationSession(Settings(),(800,600))
        s.begin_point(0)
        s.feed(feature(s.target,1))
        s.pause()
        for i in range(100):
            s.feed(feature(s.target,2+i*.05))
        self.assertEqual(s.index,0)
        self.assertEqual(s.buffer,[])

    def test_bad_frame_clears_sample_segment(self):
        s=CalibrationSession(Settings(),(800,600))
        s.begin_point(0)
        for i in range(10):
            s.feed(feature(s.target,1+i*.06))
        self.assertGreater(len(s.buffer),0)
        s.feed(FeatureFrame(1.6,valid=False))
        self.assertEqual(s.buffer,[])

    def test_sampling_timeout_allows_retry(self):
        s=CalibrationSession(Settings(),(800,600))
        s.begin_point(0)
        s.feed(feature(s.target,21))
        self.assertIsNone(s.started)
        self.assertTrue(s.error)
        s.begin_point(22)
        self.assertEqual(s.started,22)

    def test_quick_check_starts_unverified(self):
        mapping=fit_mapping(groups(TRAIN_TARGETS))
        s=CalibrationSession(Settings(),(800,600),mapping)
        self.assertEqual(s.phase,'verify')
        self.assertIsNone(s.metrics)
        self.assertIsNone(s.started)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.mapping=fit_mapping(groups(TRAIN_TARGETS))
        self.settings=Settings()
        self.metrics=validate_mapping(self.mapping,groups(CHECK_TARGETS),self.settings,(800,600))
        self.env={'camera':0,'screen':[1920,1080],'canvas':[0,0,800,600]}

    def write(self,directory,**kwargs):
        return save_profile(directory,'测试用户',self.env,self.mapping,self.settings,self.metrics,kwargs.get('hits',3))

    def test_roundtrip_and_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.write(directory)
            self.write(directory)
            data,mapping,settings=load_profile(directory,'测试用户',self.env)
            self.assertTrue(path.with_suffix('.json.bak').exists())
            self.assertEqual(mapping,self.mapping)
            self.assertEqual(settings,self.settings)
            self.assertNotIn('images',data)

    def test_device_or_display_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write(directory)
            with self.assertRaises(ValueError):
                load_profile(directory,'测试用户',{'camera':1})

    def test_unsafe_names_rejected(self):
        for name in ('../other','', 'C:\\foo', 'a/b', 'x'*33):
            with self.assertRaises(ValueError):
                profile_path('.',name)

    def test_incomplete_trial_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.write(directory,hits=0)

    def test_corrupted_profile_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=profile_path(directory,'测试用户')
            path.write_text('broken',encoding='utf-8')
            with self.assertRaises(ValueError):
                self.write(directory)
            self.assertEqual(path.read_text(encoding='utf-8'),'broken')

    def test_nonfinite_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.write(directory)
            data=json.loads(path.read_text(encoding='utf-8'))
            data['mapping']['coefficients'][0][0]=float('nan')
            path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(ValueError):
                load_profile(directory,'测试用户',self.env)


def landmarks():
    points=[SimpleNamespace(x=.5,y=.5) for _ in range(478)]
    for a,b,top,bottom,ring,start,end in [
        (33,133,159,145,(469,470,471,472),.3,.4),
        (362,263,386,374,(474,475,476,477),.6,.7)]:
        points[a]=SimpleNamespace(x=start,y=.4)
        points[b]=SimpleNamespace(x=end,y=.4)
        points[top]=SimpleNamespace(x=(start+end)/2,y=.38)
        points[bottom]=SimpleNamespace(x=(start+end)/2,y=.42)
        for i in ring:
            points[i]=SimpleNamespace(x=(start+end)/2,y=.4)
    return points


class FeatureTests(unittest.TestCase):
    def test_valid_geometry(self):
        frame=extract_features(landmarks(),1,(640,480))
        self.assertTrue(frame.usable(),frame.reason)
        self.assertAlmostEqual(frame.features[0],.5)

    def test_closed_eye_rejected(self):
        points=landmarks()
        points[159].y=points[145].y
        self.assertFalse(extract_features(points,1,(640,480)).usable())

    def test_missing_or_bad_landmarks_rejected(self):
        self.assertFalse(extract_features([],1,(640,480)).usable())
        points=landmarks()
        points[474].x=float('nan')
        self.assertFalse(extract_features(points,1,(640,480)).usable())

    def test_translation_invariance_of_eye_local_features(self):
        first=extract_features(landmarks(),1,(640,480))
        shifted=landmarks()
        for p in shifted:
            p.x+=.05
        second=extract_features(shifted,2,(640,480))
        for a,b in zip(first.features,second.features):
            self.assertAlmostEqual(a,b)


if __name__=='__main__':
    unittest.main()
