"""In-memory eye crops only. No file or network access."""
import math
import numpy as np


def eye_preview(rgb, landmarks):
    if len(landmarks)!=478:
        return None
    h,w=rgb.shape[:2]
    indices=(33,133,159,145,362,263,386,374,469,470,471,472,474,475,476,477)
    points=[]
    for i in indices:
        p=landmarks[i]
        if not all(math.isfinite(v) and 0<=v<=1 for v in (p.x,p.y)):
            return None
        points.append((int(p.x*(w-1)),int(p.y*(h-1))))
    x0,x1=min(x for x,y in points),max(x for x,y in points)
    y0,y1=min(y for x,y in points),max(y for x,y in points)
    margin=max(4,int((x1-x0)*.08))
    x0,x1=max(0,x0-margin),min(w,x1+margin+1)
    y0,y1=max(0,y0-margin),min(h,y1+margin+1)
    if x1<=x0 or y1<=y0:
        return None
    # Fixed upper bound; magnification is for inspection, not added detail.
    scale=min(3.,600/(x1-x0),180/(y1-y0))
    ow,oh=max(1,int((x1-x0)*scale)),max(1,int((y1-y0)*scale))
    xs=np.minimum(x1-1,x0+(np.arange(ow)/scale).astype(int))
    ys=np.minimum(y1-1,y0+(np.arange(oh)/scale).astype(int))
    plain=rgb[ys[:,None],xs].copy()
    overlay=plain.copy()
    for i,(x,y) in zip(indices,points):
        px,py=int((x-x0)*scale),int((y-y0)*scale)
        overlay[max(0,py-1):min(oh,py+2),max(0,px-1):min(ow,px+2)]=(0,255,100) if i>=469 else (255,210,0)
    result=np.concatenate((plain,overlay),axis=0)
    return f'P6\n{ow} {oh*2}\n255\n'.encode()+result.tobytes()
