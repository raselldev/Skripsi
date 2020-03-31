from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import os
import backend

class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2

class Model:
    batchSize = 50
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False, dump=False):
        self.dump = dump
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        self.is_train = backend.placeholder(backend.bool, name='is_train')