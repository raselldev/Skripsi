from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.dl import nn
from tensorflow.dl import layers
from tensorflow.dl import train


from tensorflow.python.ops import rnn_cell_impl as rnn
from tensorflow.python import placeholder
from tensorflow.python import expand_dims
from tensorflow.python import VariableV1 as Variable
from tensorflow.python import truncated_normal
from tensorflow.python import squeeze
from tensorflow.python import concat
from tensorflow.python import transpose
from tensorflow.python import SparseTensor
from tensorflow.python import reduce_mean
from tensorflow.python import get_collection
from tensorflow.python import GraphKeys
from tensorflow.python import control_dependencies
from tensorflow.python import Session

from tensorflow.python.framework.dtypes import bool
from tensorflow.python.framework.dtypes import float32
from tensorflow.python.framework.dtypes import int32
from tensorflow.python.framework.dtypes import int64
