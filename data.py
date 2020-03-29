from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from preprocess import Preprocess as pp

class dataSample:
    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath

class batch:
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

class dataLoader:
    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        assert filePath[-1] == '/'
        self.dataAug = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        f = open(filePath+'words.txt')
        chars = set()
        #bad_samples = []
        #bad_samples_ref = []
        for line in f:
            if not line or line[0]=='#':
                continue
            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9
            
