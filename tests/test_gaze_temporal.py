from dataclasses import replace
import unittest
from apps.desktop.calibration import MappingFeatureFilter, fit_mapping, prepared_groups, TRAIN_TARGETS
from test_calibration import groups


class GazeTemporalTests(unittest.TestCase):
    def setUp(self):
        self.mapping = replace(fit_mapping(groups(TRAIN_TARGETS)), feature_mode='gaze-median', window_seconds=.2)
        self.frame = groups(TRAIN_TARGETS)[0][0]

    def test_causal_spike_rejection_and_invalid_reset(self):
        filt = MappingFeatureFilter(self.mapping)
        values = [filt.update(replace(self.frame, timestamp=i*.06, features=(x,.1)))
                  for i,x in enumerate((.1,.1,.9,.1,.1))]
        self.assertEqual(values, [None,None,(.1,.1),(.1,.1),(.1,.1)])
        self.assertIsNone(filt.update(replace(self.frame, timestamp=.3, valid=False)))
        self.assertIsNone(filt.update(replace(self.frame, timestamp=.36)))

    def test_gap_reverse_time_and_sparse_window_warmup(self):
        filt = MappingFeatureFilter(self.mapping)
        for t in (0.,.06,.12):
            filt.update(replace(self.frame,timestamp=t))
        for t in (1., .5, .74, .98):
            self.assertIsNone(filt.update(replace(self.frame,timestamp=t)))

    def test_prepared_training_matches_live_exactly(self):
        data=groups(TRAIN_TARGETS)
        prepared=prepared_groups(data,self.mapping)
        for raw, expected in zip(data,prepared):
            filt=MappingFeatureFilter(self.mapping)
            actual=[value for sample in raw if (value:=filt.update(sample)) is not None]
            self.assertEqual(actual,[sample.features for sample in expected])

    def test_sustained_step_follows_not_frozen(self):
        filt=MappingFeatureFilter(self.mapping)
        result=None
        for i in range(12):
            result=filt.update(replace(self.frame,timestamp=i*.06,features=(.1 if i<5 else .8,.2)))
        self.assertEqual(result,(.8,.2))
