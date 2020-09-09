from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.ops import nn
from tensorflow.python.ops import rnn
nn.bidirectional_dynamic_rnn = rnn.bidirectional_dynamic_rnn

#nn
from tensorflow.python.ops.nn import conv2d
from tensorflow.python.ops.nn import relu
from tensorflow.python.ops.nn import max_pool
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn
from tensorflow.python.ops.nn import atrous_conv2d
from tensorflow.python.ops.ctc_ops import ctc_loss
from tensorflow.python.ops.ctc_ops import ctc_greedy_decoder

#layers
from tensorflow.python.normalization import batch_normalization

#train
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.training.saver import Saver
from tensorflow.python.training.checkpoint_management import latest_checkpoint

from tensorflow.python.ops import rnn_cell_impl as rnn
from tensorflow.python.ops.array_ops import placeholder
from tensorflow.python.ops.array_ops import expand_dims
from tensorflow.python.ops.variables import VariableV1 as Variable
from tensorflow.python.ops.random_ops import truncated_normal
from tensorflow.python.ops.array_ops import squeeze
from tensorflow.python.ops.array_ops import concat
from tensorflow.python.ops.array_ops import transpose
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.ops.math_ops import reduce_mean
from tensorflow.python.framework.ops import get_collection
from tensorflow.python.framework.ops import GraphKeys
from tensorflow.python.framework.ops import control_dependencies
from tensorflow.python.session import Session

from tensorflow.python.framework.dtypes import bool
from tensorflow.python.framework.dtypes import float32
from tensorflow.python.framework.dtypes import int32
from tensorflow.python.framework.dtypes import int64