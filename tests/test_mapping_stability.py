from dataclasses import replace
import unittest
import numpy as np
from apps.desktop.calibration import (fit_mapping, build_candidates, MappingFeatureFilter,
                                      TRAIN_TARGETS, CHECK_TARGETS)
from apps.desktop.config import Settings
from test_calibration import groups


class StabilityTests(unittest.TestCase):
    def test_hybrid_candidates_use_left_vertical_and_versioned_windows(self):
        data=[]
        for target,group in zip(TRAIN_TARGETS,groups(TRAIN_TARGETS)):
            data.append([replace(sample,left_eye=(sample.features[0],.02+.1*target[1]))
                         for sample in group])
        candidates,rejected=build_candidates(data,None,Settings(),(800,600))
        hybrid=[mapping for mapping,_ in candidates.values() if mapping.feature_mode=='average-x-left-y']
        self.assertEqual({m.window_seconds for m in hybrid},{.10,.15,.20})
        self.assertTrue(all(m.feature_mode in ('average','average-x-left-y') for m,_ in candidates.values()))

    def test_live_hybrid_filter_is_causal_and_resets_on_gap(self):
        mapping=fit_mapping(groups(TRAIN_TARGETS))
        mapping=replace(mapping,feature_mode='average-x-left-y',window_seconds=.20)
        filt=MappingFeatureFilter(mapping)
        base=groups(TRAIN_TARGETS)[0][0]
        values=[]
        for i,y in enumerate((0.,0.,1.,1.)):
            values.append(filt.update(replace(base,timestamp=i*.05,left_eye=(.4,y))))
        self.assertEqual(values[0][1],0.)
        self.assertEqual(values[2][1],0.)  # no future samples can repair this past output
        after_gap=filt.update(replace(base,timestamp=1.,left_eye=(.4,1.)))
        self.assertEqual(after_gap[1],1.)

    def test_noiseless_mapping_is_unchanged(self):
        data=groups(TRAIN_TARGETS)
        old=fit_mapping(data,stabilize=False)
        new=fit_mapping(data)
        for group in groups(CHECK_TARGETS):
            self.assertTrue(np.allclose(old.predict(group[0].features),new.predict(group[0].features)))

    def test_weak_noisy_vertical_signal_less_amplified_on_independent_samples(self):
        rng=np.random.default_rng(42)
        def samples(targets):
            return [[replace(group[0],features=(.35+.2*x+rng.normal(0,.001),
                                               .02+.02*y+rng.normal(0,.006))) for _ in range(120)]
                    for (x,y),group in zip(targets,groups(targets))]
        data=samples(TRAIN_TARGETS)
        old,new=fit_mapping(data,stabilize=False),fit_mapping(data)
        check=samples(CHECK_TARGETS)
        def mse(mapping):
            return np.mean([(mapping.predict(f.features)[1]-target[1])**2
                            for target,group in zip(CHECK_TARGETS,check) for f in group])
        self.assertLess(abs(new.coefficients[1][1]/new.scale[1]),abs(old.coefficients[1][1]/old.scale[1]))
        self.assertLess(mse(new),mse(old))
        # Same training data always produces same model; no check data is fitted.
        self.assertEqual(new,fit_mapping(data))
