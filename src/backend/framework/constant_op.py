from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from backend.protobuf import attr_value_pb2
from backend import context
from backend.framework import dtypes
from backend.framework import ops
from backend.framework import tensor_shape
from backend.framework import tensor_util

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


#ops.register_tensor_conversion_function( (list, tuple), _constant_tensor_conversion_function, 100)
#ops.register_tensor_conversion_function( np.ndarray, _constant_tensor_conversion_function, 100)
#ops.register_tensor_conversion_function(    np.generic, _constant_tensor_conversion_function, 100)
ops.register_tensor_conversion_function(object, _constant_tensor_conversion_function, 200)







