from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from backend.python.ops import nn
from backend.python.ops import rnn
nn.bidirectional_dynamic_rnn = rnn.bidirectional_dynamic_rnn

#nn
from backend.python.ops.nn import conv2d
from backend.python.ops.nn import relu
from backend.python.ops.nn import max_pool
from backend.python.ops.nn import bidirectional_dynamic_rnn
from backend.python.ops.nn import atrous_conv2d
from backend.python.ops.ctc_ops import ctc_loss
from backend.python.ops.ctc_ops import ctc_greedy_decoder

#layers
from backend.normalization import batch_normalization

#train
from backend.python.training.optimizer import RMSPropOptimizer
from backend.python.training.saver import Saver
from backend.python.training.checkpoint_management import latest_checkpoint

from backend.python.ops import rnn_cell_impl as rnn
from backend.python.ops.array_ops import placeholder
from backend.python.ops.array_ops import expand_dims
from backend.python.ops.variables import VariableV1 as Variable
from backend.python.ops.random_ops import truncated_normal
from backend.python.ops.array_ops import squeeze
from backend.python.ops.array_ops import concat
from backend.python.ops.array_ops import transpose
from backend.python.framework.sparse_tensor import SparseTensor
from backend.python.ops.math_ops import reduce_mean
from backend.python.framework.ops import get_collection
from backend.python.framework.ops import GraphKeys
from backend.python.framework.ops import control_dependencies
from backend.session import BaseSession

from backend.python.framework.dtypes import bool
from backend.python.framework.dtypes import float32
from backend.python.framework.dtypes import int32
from backend.python.framework.dtypes import int64