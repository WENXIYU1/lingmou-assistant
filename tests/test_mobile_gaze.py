import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from apps.desktop.mobile_gaze import MobileGazePipeline,prepare_face,prepare_aligned_face,decode
from apps.desktop.direction_probe import DirectionProbe, DriftProbe, HeadCompProbe
from apps.desktop.eye_features import FeatureFrame

class MobileGazeTests(unittest.TestCase):
    def test_head_compensation_uses_five_groups_then_nine_evaluation_groups(self):
        reference=(.5,.5,.3,0.,0.)
        probe=HeadCompProbe(reference)
        train_offsets=((0.,0.),(-.02,0.),(.02,0.),(0.,-.02),(0.,.02))
        eval_targets=probe.eval_targets
        now=0.
        for group_index in range(14):
            if group_index<5:
                dx,dy=train_offsets[group_index]; true_x=true_y=0.
            else:
                target=eval_targets[group_index-5]
                dx,dy=(0.,(group_index-5)*.0015)
                true_x=(target[0]-.5)*20.; true_y=(target[1]-.5)*20.
            # Synthetic model head leakage: X += 80*dx, Y -= 120*dy degrees.
            yaw=true_x+80*dx
            pitch=true_y-120*dy
            head=(reference[0]+dx,reference[1]+dy,.3,0.,0.)
            probe.begin(now)
            for _ in range(80):
                now+=.08
                frame=FeatureFrame(now,(yaw/180.,pitch/180.),head,True,
                                   face_crop=(.5,.6,.2,.4))
                if probe.feed(frame): break
        self.assertTrue(probe.done)
        model,error=probe._fit()
        self.assertEqual(error,'')
        last_center=np.median([probe.corrected(v,model) for _,v in probe.groups[13]],axis=0)
        self.assertLess(abs(last_center[1]),.35)
        report=probe.report()
        for text in ('完成14/14','训练头位X/Y跨度','独立评估·中心','独立九点汇总',
                     '头动补偿因果中位数','训练内RMS不能证明补偿有效'):
            self.assertIn(text,report)
        self.assertNotIn('不可评估',report)

    def test_head_compensation_refuses_insufficient_head_coverage(self):
        reference=(.5,.5,.3,0.,0.)
        probe=HeadCompProbe(reference)
        now=0.
        for _ in range(5):
            probe.begin(now)
            for _ in range(80):
                now+=.08
                if probe.feed(FeatureFrame(now,(0.,0.),reference,True,face_crop=(.5,.6,.2,.4))):
                    break
        self.assertTrue(probe.done)
        self.assertIn('已停止',probe.status())
        self.assertIn('头位覆盖不足',probe.report())
        self.assertIn('不生成补偿',probe.report())

    def test_rejected_aligned_shadow_is_not_run_in_primary_pipeline(self):
        pipeline=MobileGazePipeline.__new__(MobileGazePipeline)
        pipeline.alignment_reference='kept'
        pipeline._angles=lambda tensor:(9.,-3.)
        frame=np.zeros((200,300,3),np.uint8)
        points=[SimpleNamespace(x=.3,y=.2),SimpleNamespace(x=.7,y=.8)]*234
        observation=FeatureFrame(1.,(.1,.1),(.5,.4,.3,0.,0.),True)
        with patch('apps.desktop.mobile_gaze.prepare_aligned_face') as aligned:
            result=pipeline.infer(frame,points,observation)
        aligned.assert_not_called()
        self.assertTrue(result.usable())
        self.assertEqual(result.shadow_features,())
        self.assertEqual(pipeline.alignment_reference,'kept')
        pipeline.reset_alignment()
        self.assertIsNone(pipeline.alignment_reference)

    def test_crop_geometry_matches_actual_tensor_crop(self):
        frame=np.zeros((200,300,3),np.uint8)
        points=[SimpleNamespace(x=.3,y=.2),SimpleNamespace(x=.7,y=.8)]*234
        tensor,crop=prepare_face(frame,points,with_geometry=True)
        np.testing.assert_allclose(crop,(.5,.5,.4,.6))
        np.testing.assert_array_equal(tensor,prepare_face(frame,points))

    def test_aligned_face_reference_and_translation_normalization(self):
        frame=np.zeros((240,320,3),np.uint8)
        frame[50:190,80:240]=127
        points=[SimpleNamespace(x=.3,y=.2),SimpleNamespace(x=.7,y=.8)]*234
        points[33]=SimpleNamespace(x=.35,y=.45)
        points[263]=SimpleNamespace(x=.65,y=.45)
        first,reference=prepare_aligned_face(frame,points)
        self.assertEqual(first.shape,(1,3,448,448))
        shifted=[SimpleNamespace(x=p.x+.02,y=p.y+.01) for p in points]
        shifted_frame=np.roll(np.roll(frame,round(.02*320),axis=1),round(.01*240),axis=0)
        second,same_reference=prepare_aligned_face(shifted_frame,shifted,reference)
        self.assertIs(same_reference,reference)
        self.assertLess(float(np.mean(np.abs(first-second))),.03)

    def test_drift_nine_points_scalar_metadata_and_report(self):
        head=(.5,.4,.3,0.,0.)
        probe=DriftProbe(head)
        now=0.
        for i in range(9):
            probe.begin(now)
            for j in range(100):
                now+=.08
                if probe.feed(FeatureFrame(now,(.01*i,.02*i),head,True,
                                          face_crop=(.5,.5,.4,.6),shadow_features=(.03*i,.04*i),
                                          preview_ppm=b'private')):
                    break
        self.assertTrue(probe.done)
        self.assertEqual(len(probe.groups),9)
        self.assertEqual(len(probe.groups[0][0][1]),14)
        self.assertTrue(all(x is None or isinstance(x,float) for g in probe.groups for _,v in g for x in v))
        report=probe.report()
        for text in ('完成9/9','中心复测4','裁剪框中位','头部代理中位','原始配对',
                     '因果中位数','稳定裁剪原始','稳定裁剪因果中位数','同帧汇总',
                     '中心复测最大绝对偏移'):
            self.assertIn(text,report)
        self.assertNotIn('private',report)
        self.assertFalse(hasattr(probe,'mapping'))

    def test_drift_filter_causal_and_requires_three_recent_frames(self):
        group=[(t,(y,x,None)) for t,x,y in [(0.,0.,0.),(.09,1.,1.),(.18,0.,0.),(.27,1.,1.)]]
        raw,filtered,spans=DriftProbe.paired(group)
        self.assertEqual(len(raw),2)
        np.testing.assert_allclose(filtered[0],(0.,0.))
        np.testing.assert_allclose(filtered[1],(90.,90.))
        self.assertTrue(all(v<=.3 for v in spans))
        self.assertEqual(len(DriftProbe.paired([(0.,(0.,0.)),(.21,(0.,0.)),(.42,(0.,0.))])[0]),0)

    def test_drift_pairing_survives_shadow_inference_result_rate(self):
        group=[(i*.11,(.01*i,.02*i,None)) for i in range(24)]
        raw,filtered,spans=DriftProbe.paired(group)
        self.assertEqual(len(raw),22)
        self.assertEqual(len(filtered),22)
        self.assertTrue(all(v<=.3 for v in spans))

    def test_drift_pause_retry_drops_incomplete_metadata(self):
        head=(.5,.4,.3,0.,0.)
        probe=DriftProbe(head)
        probe.begin(0.)
        probe.feed(FeatureFrame(1.,(.1,.1),head,True))
        self.assertEqual(len(probe.buffer),1)
        probe.pause()
        probe.begin(2.)
        self.assertEqual(probe.buffer,[])
        probe.feed(FeatureFrame(1.9,(.1,.1),head,True))
        self.assertEqual(probe.buffer,[])
        self.assertIsNone(probe.last)

    def test_rgb_normalization_and_invalid_crop(self):
        frame=np.zeros((200,300,3),np.uint8)
        frame[:,:,2]=255
        points=[SimpleNamespace(x=.3,y=.2),SimpleNamespace(x=.7,y=.8)]*234
        output=prepare_face(frame,points)
        self.assertEqual(output.shape,(1,3,448,448))
        self.assertAlmostEqual(float(output[0,0,0,0]),(1-.485)/.229,places=5)
        points[0]=SimpleNamespace(x=-1,y=.2)
        with self.assertRaises(ValueError): prepare_face(frame,points)

    def test_decoder_bins_and_invalid_values(self):
        logits=np.full((1,90),-1000.)
        logits[0,45]=1000
        self.assertAlmostEqual(decode(logits),0.)
        with self.assertRaises(ValueError): decode(np.zeros((1,89)))
        with self.assertRaises(ValueError): decode(np.full((1,90),np.nan))

    def test_probe_seven_points_and_no_mapping(self):
        head=(.5,.4,.3,0.,0.)
        probe=DirectionProbe(head)
        now=0.
        for i in range(7):
            probe.begin(now)
            for j in range(80):
                now+=.06
                if probe.feed(FeatureFrame(now,(i*.01,i*.02),head,True)): break
        self.assertTrue(probe.done)
        self.assertIn('中心复测2减首次中心',probe.report())
        self.assertFalse(hasattr(probe,'mapping'))
        probe=DirectionProbe(head)
        probe.begin(now)
        probe.feed(FeatureFrame(now+.1,valid=False))
        self.assertIsNone(probe.started)
