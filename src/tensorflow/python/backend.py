from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import json
import os
import weakref

import numpy as np

from tensorflow.python.ops import tensor_array_grad 
from tensorflow.python import context
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util import tf_contextlib

py_all = all
py_sum = sum

_SESSION = None
_GRAPH_LEARNING_PHASES = weakref.WeakKeyDictionary() 
_MANUAL_VAR_INIT = False
_FLOATX = 'float32'
_EPSILON = 1e-7
_IMAGE_DATA_FORMAT = 'channels_last'
_LOCAL_DEVICES = None
_GRAPH_VARIABLES = weakref.WeakKeyDictionary()
_GRAPH_TF_OPTIMIZERS = weakref.WeakKeyDictionary()
PER_GRAPH_LAYER_NAME_UIDS = weakref.WeakKeyDictionary()

class _DummyEagerGraph(object):
  pass
_DUMMY_EAGER_GRAPH = _DummyEagerGraph()

def track_variable(v):
  if context.executing_eagerly():
    return
  graph = v.graph if hasattr(v, 'graph') else ops.get_default_graph()
  if graph not in _GRAPH_VARIABLES:
    _GRAPH_VARIABLES[graph] = weakref.WeakSet()
  _GRAPH_VARIABLES[graph].add(v)

  