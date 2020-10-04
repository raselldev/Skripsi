from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from backend.ops import nn
from backend.ops import rnn
nn.bidirectional_dynamic_rnn = rnn.bidirectional_dynamic_rnn

#nn
from backend.ops.nn import conv2d
from backend.ops.nn import relu
from backend.ops.nn import max_pool
from backend.ops.nn import bidirectional_dynamic_rnn
from backend.ops.nn import atrous_conv2d
from backend.ops.ctc_ops import ctc_loss
from backend.ops.ctc_ops import ctc_greedy_decoder

#layers
from backend.normalization import batch_normalization

#train
from backend.training.optimizer import RMSPropOptimizer
from backend.training.saver import Saver
from backend.training.checkpoint_management import latest_checkpoint

#from backend.ops import rnn_cell_impl as rnn
from backend.ops.array_ops import placeholder
from backend.ops.array_ops import expand_dims
from backend.ops.array_ops import squeeze
from backend.ops.array_ops import concat
from backend.ops.array_ops import transpose
from backend.ops.variables import VariableV1 as Variable
from backend.ops.random_ops import truncated_normal
from backend.framework.sparse_tensor import SparseTensor
from backend.framework.ops import get_collection
from backend.framework.ops import GraphKeys
from backend.framework.ops import control_dependencies
from backend.session import BaseSession
from backend.ops.math_ops import reduce_mean

from backend.framework.dtypes import bool
from backend.framework.dtypes import float32
from backend.framework.dtypes import int32
from backend.framework.dtypes import int64