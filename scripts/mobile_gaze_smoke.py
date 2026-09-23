"""Real weights with synthetic input; no webcam or human accuracy test."""
from types import SimpleNamespace
import numpy as np
from apps.desktop.mobile_gaze import MobileGazePipeline
from apps.desktop.eye_features import FeatureFrame

def main():
    pipeline=MobileGazePipeline()
    points=[SimpleNamespace(x=.3,y=.2),SimpleNamespace(x=.7,y=.8)]*234
    points[33]=SimpleNamespace(x=.35,y=.45)
    points[263]=SimpleNamespace(x=.65,y=.45)
    frame=np.full((480,640,3),128,np.uint8)
    result=pipeline.infer(frame,points,FeatureFrame(1.,(.1,.1),(.5,.4,.3,0.,0.),True))
    assert result.usable() and result.shadow_features==()
    print('Real MobileGaze model / synthetic input passed; human accuracy NOT tested.')

if __name__=='__main__': main()
