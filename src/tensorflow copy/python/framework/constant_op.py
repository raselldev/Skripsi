from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python import context
from tensorflow.python.framework import ops

def constant(value, dtype=None, shape=None, name="Const", verify_shape=False):
  ctx = context.context()
  if ctx.executing_eagerly():
    t = convert_to_eager_tensor(value, ctx, dtype)
    if shape is None:
      return t
    shape = tensor_shape.as_shape(shape)
    if shape == t.shape:
      return t
    if verify_shape:
      raise TypeError("Expected Tensor's shape: %s, got %s." % (tuple(shape),
                                                                tuple(t.shape)))
    num_t = t.shape.num_elements()
    # TODO(josh11b): Implement shape -> eager tensor conversion.
    if num_t == shape.num_elements():
      return _eager_reshape(t, shape.as_list(), ctx)
    if num_t == 1:
      if t.dtype == dtypes.bool:
        # We don't have a Fill kernel for bool dtype on GPU. So we first run
        # Fill on CPU and then copy to GPU if needed.
        with ops.device("/device:CPU:0"):
          x = _eager_fill(shape.as_list(), t.cpu(), ctx)
        return _eager_identity(x, ctx)
      else:
        return _eager_fill(shape.as_list(), t, ctx)
    raise TypeError("Eager execution of tf.constant with unsupported shape "
                    "(value has %d elements, shape is %s with %d elements)." %
                    (num_t, shape, shape.num_elements()))
  g = ops.get_default_graph()
  tensor_value = attr_value_pb2.AttrValue()
  tensor_value.tensor.CopyFrom(
      tensor_util.make_tensor_proto(
          value, dtype=dtype, shape=shape, verify_shape=verify_shape))
  dtype_value = attr_value_pb2.AttrValue(type=tensor_value.tensor.dtype)
  const_tensor = g.create_op(
      "Const", [], [dtype_value.type],
      attrs={"value": tensor_value,
             "dtype": dtype_value},
      name=name).outputs[0]
  return const_tensor

def _constant_tensor_conversion_function(v, dtype=None, name=None,
                                         as_ref=False):
  _ = as_ref
  return constant(v, dtype=dtype, name=name)

ops.register_tensor_conversion_function(
    object, _constant_tensor_conversion_function, 200)

def _tensor_shape_tensor_conversion_function(s,
                                             dtype=None,
                                             name=None,
                                             as_ref=False):
  """Function to convert TensorShape to Tensor."""
  _ = as_ref
  if not s.is_fully_defined():
    raise ValueError(
        "Cannot convert a partially known TensorShape to a Tensor: %s" % s)
  s_list = s.as_list()
  int64_value = 0
  for dim in s_list:
    if dim >= 2**31:
      int64_value = dim
      break

  if dtype is not None:
    if dtype not in (dtypes.int32, dtypes.int64):
      raise TypeError("Cannot convert a TensorShape to dtype: %s" % dtype)
    if dtype == dtypes.int32 and int64_value:
      raise ValueError("Cannot convert a TensorShape to dtype int32; "
                       "a dimension is too large (%s)" % int64_value)
  else:
    dtype = dtypes.int64 if int64_value else dtypes.int32
  if name is None:
    name = "shape_as_tensor"
  return constant(s_list, dtype=dtype, name=name)

def rank(tensor):
  if isinstance(tensor, ops.Tensor):
    return tensor._rank()  # pylint: disable=protected-access
  return None

def has_fully_defined_shape(tensor):
  return isinstance(tensor, ops.EagerTensor) or tensor.shape.is_fully_defined()
