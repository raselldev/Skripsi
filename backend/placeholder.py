from __future__ import division
from __future__ import print_function

import numpy as np

def placeholder(dtype, shape=None, name=None):
  if context.executing_eagerly():
    raise RuntimeError("error")

  return placeholder(dtype=dtype, shape=shape, name=name)
