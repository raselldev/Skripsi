from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2

def Preprocess(img, imgSize, dataAug=False):
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    if dataAug:
        stretch = (random.random()-0.5)
        #random width setidaknya 1
        wStretch = max(int(img.shape[1]*(1+stretch)), 1)
        #horizontal stretch dari size awal
        img = cv2.resize(img, (wStretch, img.shape[0]))

    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w/wt
    fy = h/ht
    f = max(fx, fy)

    #size baru setelah di scale
    newSize = (max(min(wt, int(w/f)), 1), max(min(ht, int(h/f)), 1))
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt])*255
    img = target[0:newSize[1], 0:newSize[0]]

    #transpose
    img = cv2.transpose(target)

    #normalisasi
    (i, j) = cv2.meanStdDev(img)
    i = m[0][0]
    j = m[0][0]
    img = img-i
    img = img/j if i>0 else img
    return img