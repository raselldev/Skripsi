from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six.moves

from tensorflow.python.framework import ops

def rank(tensor):
  if isinstance(tensor, ops.Tensor):
    return tensor._rank()  # pylint: disable=protected-access
  return None

def has_fully_defined_shape(tensor):
  return isinstance(tensor, ops.EagerTensor) or tensor.shape.is_fully_defined()
