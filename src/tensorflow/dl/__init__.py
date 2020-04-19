from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.dl import nn
from tensorflow.dl import layers
from tensorflow.dl import train

from tensorflow.python.util.lazy_loader import LazyLoader
contrib = LazyLoader('contrib', globals(), 'tensorflow.contrib')