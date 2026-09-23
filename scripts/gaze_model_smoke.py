"""Real CPU models, synthetic tensors only; no camera or personal images."""
import json
import statistics
import time
from types import SimpleNamespace
import numpy as np
from apps.desktop.gaze_pipeline import GazePipeline
from apps.desktop.eye_features import FeatureFrame


def main():
    start = time.perf_counter()
    pipeline = GazePipeline()
    loaded = time.perf_counter() - start
    points = [SimpleNamespace(x=.5,y=.5) for _ in range(478)]
    for index,x,y in ((0,.35,.3),(1,.65,.7),(33,.40,.45),(133,.46,.45),
                      (362,.54,.45),(263,.60,.45)):
        points[index] = SimpleNamespace(x=x,y=y)
    observation = FeatureFrame(1.,(.4,.1),(.5,.4,.3,0.,0.),True,capture_size=(1280,720))
    full = pipeline.infer(np.full((720,1280,3),127,dtype=np.uint8),points,observation)
    assert full.usable(), 'Real model output must reach the calibration input boundary'
    image = np.full((1,3,60,60), 127, dtype=np.float32)
    timings = []
    for i in range(25):
        start = time.perf_counter()
        result = pipeline.pose({'data': image})
        angles = np.array([[float(result[pipeline.pose.output(key)].item())
                            for key in ('angle_y_fc','angle_p_fc','angle_r_fc')]], dtype=np.float32)
        output = pipeline.gaze({'left_eye_image':image,'right_eye_image':image,'head_pose_angles':angles})
        vector = output[pipeline.gaze.output('gaze_vector')]
        assert vector.shape == (1,3) and np.all(np.isfinite(vector))
        if i >= 5:
            timings.append((time.perf_counter()-start)*1000)
    print(json.dumps({'kind':'real-model / synthetic-input / CPU', 'load_seconds':round(loaded,3),
                      'two_models_median_ms':round(statistics.median(timings),2),
                      'two_models_max_ms':round(max(timings),2),
                      'full_pipeline_synthetic_passed':True,
                      'human_accuracy_tested':False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
