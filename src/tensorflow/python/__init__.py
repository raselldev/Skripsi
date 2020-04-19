import ctypes
import importlib
import sys
import traceback

import numpy as np

from tensorflow.python import pywrap_tensorflow

from tensorflow.core.protobuf.tensorflow_server_pb2 import *
from tensorflow.python.framework.framework_lib import *
from tensorflow.python.framework.versions import *
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util

# Session
from tensorflow.python.client.client_lib import *

# Ops
from tensorflow.python.ops.standard_ops import *

from tensorflow.python.ops import nn
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
nn.bidirectional_dynamic_rnn = rnn.bidirectional_dynamic_rnn
nn.dynamic_rnn = rnn.dynamic_rnn
nn.raw_rnn = rnn.raw_rnn
nn.static_rnn = rnn.static_rnn
nn.static_state_saving_rnn = rnn.static_state_saving_rnn
nn.rnn_cell = rnn_cell