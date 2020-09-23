from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from backend.python.ops import array_ops
from backend.python.ops import random_ops
from backend.python.framework import dtypes



class Initializer(object):
  def __call__(self, shape, dtype=None, partition_info=None):
    raise NotImplementedError

  def get_config(self):
    return {}

  @classmethod
  def from_config(cls, config):
    return cls(**config)

class Zeros(Initializer):
  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return array_ops.zeros(shape, dtype)

  def get_config(self):
    return {"dtype": self.dtype.name}

class Ones(Initializer):
  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return array_ops.ones(shape, dtype)

  def get_config(self):
    return {"dtype": self.dtype.name}

class VarianceScaling(Initializer):
  def __init__(self,
               scale=1.0,
               mode="fan_in",
               distribution="truncated_normal",
               seed=None,
               dtype=dtypes.float32):
    if scale <= 0.:
      raise ValueError("`scale` must be positive float.")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
      raise ValueError("Invalid `mode` argument:", mode)
    distribution = distribution.lower()
    if distribution not in {"normal", "uniform",
                            "truncated_normal", "untruncated_normal"}:
      raise ValueError("Invalid `distribution` argument:", distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed
    self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    scale = self.scale
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape
    fan_in, fan_out = _compute_fans(scale_shape)
    if self.mode == "fan_in":
      scale /= max(1., fan_in)
    elif self.mode == "fan_out":
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == "normal" or self.distribution == "truncated_normal":
      stddev = math.sqrt(scale) / .87962566103423978
      return random_ops.truncated_normal(
          shape, 0.0, stddev, dtype, seed=self.seed)
    elif self.distribution == "untruncated_normal":
      stddev = math.sqrt(scale)
      return random_ops.random_normal(
          shape, 0.0, stddev, dtype, seed=self.seed)
    else:
      limit = math.sqrt(3.0 * scale)
      return random_ops.random_uniform(
          shape, -limit, limit, dtype, seed=self.seed)

  def get_config(self):
    return {
        "scale": self.scale,
        "mode": self.mode,
        "distribution": self.distribution,
        "seed": self.seed,
        "dtype": self.dtype.name
    }

class GlorotUniform(VarianceScaling):
  def __init__(self,
               seed=None,
               dtype=dtypes.float32):
    super(GlorotUniform, self).__init__(
        scale=1.0,
        mode="fan_avg",
        distribution="uniform",
        seed=seed,
        dtype=dtype)

  def get_config(self):
    return {
        "seed": self.seed,
        "dtype": self.dtype.name
    }

def _assert_float_dtype(dtype):
  if not dtype.is_floating:
    raise ValueError("Expected floating point type, got %s." % dtype)
  return dtype

def _compute_fans(shape):
  if len(shape) < 1:
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    receptive_field_size = 1.
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out

