from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from backend.python.framework import random_seed
from backend.python.framework import ops
from backend.python.framework import dtypes
from backend.python.ops import math_ops
from backend.python.ops import gen_random_ops

def truncated_normal(shape,
                     mean=0.0,
                     stddev=1.0,
                     dtype=dtypes.float32,
                     seed=None,
                     name=None):
  with ops.name_scope(name, "truncated_normal", [shape, mean, stddev]) as name:
    shape_tensor = _ShapeTensor(shape)
    mean_tensor = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
    stddev_tensor = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    seed1, seed2 = random_seed.get_seed(seed)
    rnd = gen_random_ops.truncated_normal(
        shape_tensor, dtype, seed=seed1, seed2=seed2)
    mul = rnd * stddev_tensor
    value = math_ops.add(mul, mean_tensor, name=name)
    return value

def _ShapeTensor(shape):
  """Convert to an int32 or int64 tensor, defaulting to int32 if empty."""
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int32
  else:
    dtype = None
  return ops.convert_to_tensor(shape, dtype=dtype, name="shape")

def random_uniform(shape,
                   minval=0,
                   maxval=None,
                   dtype=dtypes.float32,
                   seed=None,
                   name=None):
  
  dtype = dtypes.as_dtype(dtype)
  if dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32,
                   dtypes.float64, dtypes.int32, dtypes.int64):
    raise ValueError("Invalid dtype %r" % dtype)
  if maxval is None:
    if dtype.is_integer:
      raise ValueError("Must specify maxval for integer dtype %r" % dtype)
    maxval = 1
  with ops.name_scope(name, "random_uniform", [shape, minval, maxval]) as name:
    shape = _ShapeTensor(shape)
    minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
    maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
    seed1, seed2 = random_seed.get_seed(seed)
    if dtype.is_integer:
      return gen_random_ops.random_uniform_int(
          shape, minval, maxval, seed=seed1, seed2=seed2, name=name)
    else:
      rnd = gen_random_ops.random_uniform(shape, dtype, seed=seed1, seed2=seed2)
      return math_ops.add(rnd * (maxval - minval), minval, name=name)
