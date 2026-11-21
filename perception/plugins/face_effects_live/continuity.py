"""Transient association appearance and conservative shot discontinuities."""
import cv2
import numpy as np


def appearance(image, face):
    x,y,w,h = face.box
    height,width = image.shape[:2]
    regions = [(x+.12*w,y+.12*h,.76*w,.72*h), (x-.25*w,y+1.05*h,1.5*w,1.1*h)]
    pieces = []
    for rx,ry,rw,rh in regions:
        x1,y1=max(0,int(rx)),max(0,int(ry))
        x2,y2=min(width,int(rx+rw)),min(height,int(ry+rh))
        if x2-x1 < 4 or y2-y1 < 4:
            pieces.append(np.zeros(32,np.float64))
            continue
        patch=cv2.resize(image[y1:y2,x1:x2],(24,24))
        hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
        histogram=cv2.calcHist([hsv],[0,1],None,[8,4],[0,180,0,256]).ravel().astype(np.float64)
        histogram=np.sqrt(histogram/max(histogram.sum(),1.))
        pieces.append(histogram)
    vector=np.concatenate(pieces)
    return vector/max(np.linalg.norm(vector),1e-9)


class SceneContinuity:
    def __init__(self):
        self.previous=None

    def reset(self):
        self.previous=None

    def update(self,image):
        gray=cv2.cvtColor(cv2.resize(image,(64,36)),cv2.COLOR_BGR2GRAY).astype(np.float32)
        old,self.previous=self.previous,gray
        if old is None:return False
        mean_difference=float(np.mean(np.abs(gray-old)))
        a,b=gray-gray.mean(),old-old.mean()
        denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
        correlation=float(np.sum(a*b)/denominator) if denominator>1e-6 else 1.
        # A global brightness offset retains correlation. Normal motion is much
        # smaller at this scale; no face appearance is stored across a reset.
        return mean_difference>12. and correlation<.8
