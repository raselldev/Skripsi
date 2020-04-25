from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys as _sys



import numbers
import math
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op




#from tensorflow.python.ops.nn_ops import *
#from tensorflow.python.ops.nn_impl import *

class _NonAtrousConvolution(object):

  def __init__(
      self,
      input_shape,
      filter_shape,  # pylint: disable=redefined-builtin
      padding,
      data_format=None,
      strides=None,
      name=None):
    filter_shape = filter_shape.with_rank(input_shape.ndims)
    self.padding = padding
    self.name = name
    input_shape = input_shape.with_rank(filter_shape.ndims)
    if input_shape.ndims is None:
      raise ValueError("Rank of convolution must be known")
    if input_shape.ndims < 3 or input_shape.ndims > 5:
      raise ValueError(
          "`input` and `filter` must have rank at least 3 and at most 5")
    conv_dims = input_shape.ndims - 2
    if strides is None:
      strides = [1] * conv_dims
    elif len(strides) != conv_dims:
      raise ValueError("len(strides)=%d, but should be %d" % (len(strides),
                                                              conv_dims))
    if conv_dims == 1:
      # conv1d uses the 2-d data format names
      if data_format is None:
        data_format = "NWC"
      elif data_format not in {"NCW", "NWC", "NCHW", "NHWC"}:
        raise ValueError("data_format must be \"NWC\" or \"NCW\".")
      self.strides = strides[0]
      self.data_format = data_format
      self.conv_op = self._conv1d
    elif conv_dims == 2:
      if data_format is None or data_format == "NHWC":
        data_format = "NHWC"
        strides = [1] + list(strides) + [1]
      elif data_format == "NCHW":
        strides = [1, 1] + list(strides)
      else:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
      self.strides = strides
      self.data_format = data_format
      self.conv_op = gen_nn_ops.conv2d
    elif conv_dims == 3:
      if data_format is None or data_format == "NDHWC":
        strides = [1] + list(strides) + [1]
      elif data_format == "NCDHW":
        strides = [1, 1] + list(strides)
      else:
        raise ValueError("data_format must be \"NDHWC\" or \"NCDHW\". Have: %s"
                         % data_format)
      self.strides = strides
      self.data_format = data_format
      self.conv_op = gen_nn_ops.conv3d

  def _conv1d(self, input, filter, strides, padding, data_format, name):
    return conv1d(
        value=input,
        filters=filter,
        stride=strides,
        padding=padding,
        data_format=data_format,
        name=name)

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    return self.conv_op(
        input=inp,
        filter=filter,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        name=self.name)

class _WithSpaceToBatch(object):


  def __init__(self,
               input_shape,
               dilation_rate,
               padding,
               build_op,
               filter_shape=None,
               spatial_dims=None,
               data_format=None):
    """Helper class for _with_space_to_batch."""
    dilation_rate = ops.convert_to_tensor(
        dilation_rate, dtypes.int32, name="dilation_rate")
    try:
      rate_shape = dilation_rate.get_shape().with_rank(1)
    except ValueError:
      raise ValueError("rate must be rank 1")

    if not dilation_rate.get_shape().is_fully_defined():
      raise ValueError("rate must have known shape")

    num_spatial_dims = rate_shape[0].value

    if data_format is not None and data_format.startswith("NC"):
      starting_spatial_dim = 2
    else:
      starting_spatial_dim = 1

    if spatial_dims is None:
      spatial_dims = range(starting_spatial_dim,
                           num_spatial_dims + starting_spatial_dim)
    orig_spatial_dims = list(spatial_dims)
    spatial_dims = sorted(set(int(x) for x in orig_spatial_dims))
    if spatial_dims != orig_spatial_dims or any(x < 1 for x in spatial_dims):
      raise ValueError(
          "spatial_dims must be a montonically increasing sequence of positive "
          "integers")  # pylint: disable=line-too-long

    if data_format is not None and data_format.startswith("NC"):
      expected_input_rank = spatial_dims[-1]
    else:
      expected_input_rank = spatial_dims[-1] + 1

    try:
      input_shape.with_rank_at_least(expected_input_rank)
    except ValueError:
      ValueError("input tensor must have rank %d at least" %
                 (expected_input_rank))

    const_rate = tensor_util.constant_value(dilation_rate)
    rate_or_const_rate = dilation_rate
    if const_rate is not None:
      rate_or_const_rate = const_rate
      if np.any(const_rate < 1):
        raise ValueError("dilation_rate must be positive")
      if np.all(const_rate == 1):
        self.call = build_op(num_spatial_dims, padding)
        return

    # We have two padding contributions. The first is used for converting "SAME"
    # to "VALID". The second is required so that the height and width of the
    # zero-padded value tensor are multiples of rate.

    # Padding required to reduce to "VALID" convolution
    if padding == "SAME":
      if filter_shape is None:
        raise ValueError("filter_shape must be specified for SAME padding")
      filter_shape = ops.convert_to_tensor(filter_shape, name="filter_shape")
      const_filter_shape = tensor_util.constant_value(filter_shape)
      if const_filter_shape is not None:
        filter_shape = const_filter_shape
        self.base_paddings = _with_space_to_batch_base_paddings(
            const_filter_shape, num_spatial_dims, rate_or_const_rate)
      else:
        self.num_spatial_dims = num_spatial_dims
        self.rate_or_const_rate = rate_or_const_rate
        self.base_paddings = None
    elif padding == "VALID":
      self.base_paddings = np.zeros([num_spatial_dims, 2], np.int32)
    else:
      raise ValueError("Invalid padding method %r" % padding)

    self.input_shape = input_shape
    self.spatial_dims = spatial_dims
    self.dilation_rate = dilation_rate
    self.data_format = data_format
    self.op = build_op(num_spatial_dims, "VALID")
    self.call = self._with_space_to_batch_call

  def _with_space_to_batch_call(self, inp, filter):  # pylint: disable=redefined-builtin
    """Call functionality for with_space_to_batch."""
    # Handle input whose shape is unknown during graph creation.
    input_spatial_shape = None
    input_shape = self.input_shape
    spatial_dims = self.spatial_dims
    if input_shape.ndims is not None:
      input_shape_list = input_shape.as_list()
      input_spatial_shape = [input_shape_list[i] for i in spatial_dims]
    if input_spatial_shape is None or None in input_spatial_shape:
      input_shape_tensor = array_ops.shape(inp)
      input_spatial_shape = array_ops.stack(
          [input_shape_tensor[i] for i in spatial_dims])

    base_paddings = self.base_paddings
    if base_paddings is None:
      # base_paddings could not be computed at build time since static filter
      # shape was not fully defined.
      filter_shape = array_ops.shape(filter)
      base_paddings = _with_space_to_batch_base_paddings(
          filter_shape, self.num_spatial_dims, self.rate_or_const_rate)
    paddings, crops = array_ops.required_space_to_batch_paddings(
        input_shape=input_spatial_shape,
        base_paddings=base_paddings,
        block_shape=self.dilation_rate)

    dilation_rate = _with_space_to_batch_adjust(self.dilation_rate, 1,
                                                spatial_dims)
    paddings = _with_space_to_batch_adjust(paddings, 0, spatial_dims)
    crops = _with_space_to_batch_adjust(crops, 0, spatial_dims)
    input_converted = array_ops.space_to_batch_nd(
        input=inp, block_shape=dilation_rate, paddings=paddings)

    result = self.op(input_converted, filter)

    result_converted = array_ops.batch_to_space_nd(
        input=result, block_shape=dilation_rate, crops=crops)

    # Recover channel information for output shape if channels are not last.
    if self.data_format is not None and self.data_format.startswith("NC"):
      if not result_converted.shape[1].value and filter is not None:
        output_shape = result_converted.shape.as_list()
        output_shape[1] = filter.shape[-1]
        result_converted.set_shape(output_shape)

    return result_converted

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    return self.call(inp, filter)


def _get_strides_and_dilation_rate(num_spatial_dims, strides, dilation_rate):
  if dilation_rate is None:
    dilation_rate = [1] * num_spatial_dims
  elif len(dilation_rate) != num_spatial_dims:
    raise ValueError("len(dilation_rate)=%d but should be %d" %
                     (len(dilation_rate), num_spatial_dims))
  dilation_rate = np.array(dilation_rate, dtype=np.int32)
  if np.any(dilation_rate < 1):
    raise ValueError("all values of dilation_rate must be positive")

  if strides is None:
    strides = [1] * num_spatial_dims
  elif len(strides) != num_spatial_dims:
    raise ValueError("len(strides)=%d but should be %d" % (len(strides),
                                                           num_spatial_dims))
  strides = np.array(strides, dtype=np.int32)
  if np.any(strides < 1):
    raise ValueError("all values of strides must be positive")

  if np.any(strides > 1) and np.any(dilation_rate > 1):
    raise ValueError(
        "strides > 1 not supported in conjunction with dilation_rate > 1")
  return strides, dilation_rate

@tf_export("nn.convolution")
def convolution(
    input,  # pylint: disable=redefined-builtin
    filter,  # pylint: disable=redefined-builtin
    padding,
    strides=None,
    dilation_rate=None,
    name=None,
    data_format=None):
  # pylint: disable=line-too-long
  
  # pylint: enable=line-too-long
  with ops.name_scope(name, "convolution", [input, filter]) as name:
    input = ops.convert_to_tensor(input, name="input")  # pylint: disable=redefined-builtin
    input_shape = input.get_shape()
    filter = ops.convert_to_tensor(filter, name="filter")  # pylint: disable=redefined-builtin
    filter_shape = filter.get_shape()
    op = Convolution(
        input_shape,
        filter_shape,
        padding,
        strides=strides,
        dilation_rate=dilation_rate,
        name=name,
        data_format=data_format)
    return op(input, filter)

class Convolution(object):
  
  def __init__(self,
               input_shape,
               filter_shape,
               padding,
               strides=None,
               dilation_rate=None,
               name=None,
               data_format=None):
    """Helper function for convolution."""
    num_total_dims = filter_shape.ndims
    if num_total_dims is None:
      num_total_dims = input_shape.ndims
    if num_total_dims is None:
      raise ValueError("rank of input or filter must be known")

    num_spatial_dims = num_total_dims - 2

    try:
      input_shape.with_rank(num_spatial_dims + 2)
    except ValueError:
      ValueError("input tensor must have rank %d" % (num_spatial_dims + 2))

    try:
      filter_shape.with_rank(num_spatial_dims + 2)
    except ValueError:
      ValueError("filter tensor must have rank %d" % (num_spatial_dims + 2))

    if data_format is None or not data_format.startswith("NC"):
      input_channels_dim = input_shape[num_spatial_dims + 1]
      spatial_dims = range(1, num_spatial_dims + 1)
    else:
      input_channels_dim = input_shape[1]
      spatial_dims = range(2, num_spatial_dims + 2)

    if not input_channels_dim.is_compatible_with(
        filter_shape[num_spatial_dims]):
      raise ValueError(
          "number of input channels does not match corresponding dimension of "
          "filter, {} != {}".format(input_channels_dim,
                                    filter_shape[num_spatial_dims]))

    strides, dilation_rate = _get_strides_and_dilation_rate(
        num_spatial_dims, strides, dilation_rate)

    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.data_format = data_format
    self.strides = strides
    self.name = name
    self.conv_op = _WithSpaceToBatch(
        input_shape,
        dilation_rate=dilation_rate,
        padding=padding,
        build_op=self._build_op,
        filter_shape=filter_shape,
        spatial_dims=spatial_dims,
        data_format=data_format)

  def _build_op(self, _, padding):
    return _NonAtrousConvolution(
        self.input_shape,
        filter_shape=self.filter_shape,
        padding=padding,
        data_format=self.data_format,
        strides=self.strides,
        name=self.name)

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    return self.conv_op(inp, filter)

@tf_export("nn.atrous_conv2d")
def atrous_conv2d(value, filters, rate, padding, name=None):
  
  return convolution(
      input=value,
      filter=filters,
      padding=padding,
      dilation_rate=np.broadcast_to(rate, (2,)),
      name=name)

@tf_export("nn.relu6")
def relu6(features, name=None):
  
  with ops.name_scope(name, "Relu6", [features]) as name:
    features = ops.convert_to_tensor(features, name="features")
    return gen_nn_ops.relu6(features, name=name)

@tf_export("nn.max_pool")
def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):

  
  with ops.name_scope(name, "MaxPool", [value]) as name:
    value = ops.convert_to_tensor(value, name="input")
    return gen_nn_ops.max_pool(
        value,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)


@tf_export("nn.batch_normalization")
def batch_normalization(x,
                        mean,
                        variance,
                        offset,
                        scale,
                        variance_epsilon,
                        name=None):
  
  with ops.name_scope(name, "batchnorm", [x, mean, variance, scale, offset]):
    inv = math_ops.rsqrt(variance + variance_epsilon)
    if scale is not None:
      inv *= scale
    # Note: tensorflow/contrib/quantize/python/fold_batch_norms.py depends on
    # the precise order of ops that are generated by the expression below.
    return x * math_ops.cast(inv, x.dtype) + math_ops.cast(
        offset - mean * inv if offset is not None else -mean * inv, x.dtype)


@tf_export("nn.fused_batch_norm")
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
  # Set a minimum epsilon to 1.001e-5, which is a requirement by CUDNN to
  # prevent exception (see cudnn.h).
  min_epsilon = 1.001e-5
  epsilon = epsilon if epsilon > min_epsilon else min_epsilon
  # TODO(reedwm): In a few weeks, switch to using the V2 version exclusively. We
  # currently only use the V2 version for float16 inputs, which is not supported
  # by the V1 version.
  if x.dtype == dtypes.float16 or x.dtype == dtypes.bfloat16:
    fused_batch_norm_func = gen_nn_ops.fused_batch_norm_v2
  else:
    fused_batch_norm_func = gen_nn_ops._fused_batch_norm  # pylint: disable=protected-access
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

