import six

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.util import compat


def record_gradient(unused_op_name, unused_inputs, unused_attrs, unused_results,
                    unused_name):
  """Import backprop if you want gradients recorded."""
  pass

def make_bool(v, arg_name):
  if not isinstance(v, bool):
    raise TypeError("Expected bool for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return v

def make_float(v, arg_name):
  if not isinstance(v, compat.real_types):
    raise TypeError("Expected float for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return float(v)

def make_str(v, arg_name):
  if not isinstance(v, compat.bytes_or_text_types):
    raise TypeError("Expected string for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return compat.as_bytes(v)  # Convert unicode strings to bytes.

def make_int(v, arg_name):
  if isinstance(v, six.string_types):
    raise TypeError("Expected int for argument '%s' not %s." %
                    (arg_name, repr(v)))
  try:
    return int(v)
  except (ValueError, TypeError):
    raise TypeError("Expected int for argument '%s' not %s." %
                    (arg_name, repr(v)))

def make_tensor(v, arg_name):
  """Ensure v is a TensorProto."""
  if isinstance(v, tensor_pb2.TensorProto):
    return v
  elif isinstance(v, six.string_types):
    pb = tensor_pb2.TensorProto()
    text_format.Merge(v, pb)
    return pb
  raise TypeError(
      "Don't know how to convert %s to a TensorProto for argument '%s'." %
      (repr(v), arg_name))

def make_type(v, arg_name):
  try:
    v = dtypes.as_dtype(v).base_dtype
  except TypeError:
    raise TypeError("Expected DataType for argument '%s' not %s." %
                    (arg_name, repr(v)))
  i = v.as_datatype_enum
  return i

def make_shape(v, arg_name):
  """Convert v into a list."""
  try:
    shape = tensor_shape.as_shape(v)
  except TypeError as e:
    raise TypeError("Error converting %s to a TensorShape: %s." % (arg_name, e))
  except ValueError as e:
    raise ValueError("Error converting %s to a TensorShape: %s." % (arg_name,
                                                                    e))
  if shape.ndims is None:
    return None
  else:
    return shape.as_list()