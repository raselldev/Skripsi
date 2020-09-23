from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from backend.python.framework import constant_op
#from backend.python.framework import function
from backend.python.framework import ops
from backend.python.framework import dtypes
#from backend.python.util.deprecation import deprecated_args
from backend.python.ops import nn

def fused_batch_norm(
    x,
    scale,
    offset,  # pylint: disable=invalid-name
    mean=None,
    variance=None,
    epsilon=0.001,
    data_format="NHWC",
    is_training=True,
    name=None):
  x = ops.convert_to_tensor(x, name="input")
  scale = ops.convert_to_tensor(scale, name="scale")
  offset = ops.convert_to_tensor(offset, name="offset")
  if is_training:
    if (mean is not None) or (variance is not None):
      raise ValueError("Both 'mean' and 'variance' must be None "
                       "if is_training is True.")
  if mean is None:
    mean = constant_op.constant([])
  if variance is None:
    variance = constant_op.constant([])
  min_epsilon = 1.001e-5
  epsilon = epsilon if epsilon > min_epsilon else min_epsilon
  if x.dtype == dtypes.float16 or x.dtype == dtypes.bfloat16:
    fused_batch_norm_func = nn.fused_batch_norm_v2
  else:
    fused_batch_norm_func = nn._fused_batch_norm
  y, batch_mean, batch_var, _, _ = fused_batch_norm_func(
      x,
      scale,
      offset,
      mean,
      variance,
      epsilon=epsilon,
      data_format=data_format,
      is_training=is_training,
      name=name)
  return y, batch_mean, batch_var
