from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from backend.protobuf import tensor_pb2
from backend.protobuf import tensor_shape_pb2
from backend.framework import ops
from backend.framework import tensor_shape
from backend.util import compat
from backend.framework import dtypes
from backend.framework import ops


def SlowAppendFloat32ArrayToTensorProto(tensor_proto, proto_values):
  tensor_proto.float_val.extend([np.asscalar(x) for x in proto_values])

def SlowAppendIntArrayToTensorProto(tensor_proto, proto_values):
  tensor_proto.int_val.extend([np.asscalar(x) for x in proto_values])

_NP_TO_APPEND_FN = {
    np.float32: SlowAppendFloat32ArrayToTensorProto,
    np.int32: SlowAppendIntArrayToTensorProto,
}

def constant_value(tensor, partial=False):  
  if isinstance(tensor, ops.EagerTensor):
    return tensor.numpy()
  ret = _ConstantValue(tensor, partial)
  if ret is not None:
    tensor.graph.prevent_feeding(tensor)
  return ret

def _ConstantValue(tensor, partial):
  if not isinstance(tensor, ops.Tensor):
    raise TypeError("tensor is not a Tensor")
  if tensor.op.type == "Const":
    return MakeNdarray(tensor.op.get_attr("value"))
  elif tensor.op.type == "Shape":
    input_shape = tensor.op.inputs[0].get_shape()
    if input_shape.is_fully_defined():
      return np.array(
          [dim.value for dim in input_shape.dims],
          dtype=tensor.dtype.as_numpy_dtype)
    else:
      return None
  elif tensor.op.type == "Size":
    input_shape = tensor.op.inputs[0].get_shape()
    if input_shape.is_fully_defined():
      return np.prod([dim.value for dim in input_shape.dims], dtype=np.int32)
    else:
      return None
  elif tensor.op.type == "Rank":
    input_shape = tensor.op.inputs[0].get_shape()
    if input_shape.ndims is not None:
      return np.ndarray(
          shape=(),
          buffer=np.array([input_shape.ndims], dtype=np.int32),
          dtype=np.int32)
    else:
      return None
  elif tensor.op.type == "Range":
    start = constant_value(tensor.op.inputs[0])
    if start is None:
      return None
    limit = constant_value(tensor.op.inputs[1])
    if limit is None:
      return None
    delta = constant_value(tensor.op.inputs[2])
    if delta is None:
      return None
    return np.arange(start, limit, delta, dtype=tensor.dtype.as_numpy_dtype)
  elif tensor.op.type == "Cast":
    pre_cast = constant_value(tensor.op.inputs[0])
    if pre_cast is None:
      return None
    cast_dtype = dtypes.as_dtype(tensor.op.get_attr("DstT"))
    return pre_cast.astype(cast_dtype.as_numpy_dtype)
  elif tensor.op.type == "Concat":
    dim = constant_value(tensor.op.inputs[0])
    if dim is None:
      return None
    values = []
    for x in tensor.op.inputs[1:]:
      value = constant_value(x)
      if value is None:
        return None
      values.append(value)
    return np.concatenate(values, axis=dim)
  elif tensor.op.type == "ConcatV2":
    dim = constant_value(tensor.op.inputs[-1])
    if dim is None:
      return None
    values = []
    for x in tensor.op.inputs[:-1]:
      value = constant_value(x)
      if value is None:
        return None
      values.append(value)
    return np.concatenate(values, axis=dim)
  elif tensor.op.type == "Pack":
    values = []
    if not tensor.op.inputs:
      return None
    if tensor.op.get_attr("axis") != 0:
      return None
    for x in tensor.op.inputs:
      value = constant_value(x, partial)
      if value is None and not partial:
        return None
      values.append(value)
    return np.array(values)
  elif tensor.op.type == "Fill":
    fill_shape = tensor.shape
    fill_value = constant_value(tensor.op.inputs[1])
    if fill_shape.is_fully_defined() and fill_value is not None:
      return np.full(fill_shape.as_list(), fill_value, dtype=fill_value.dtype)
    else:
      return None
  elif tensor.op.type == "Equal":
    value1 = constant_value(tensor.op.inputs[0])
    if value1 is None:
      return None
    value2 = constant_value(tensor.op.inputs[1])
    if value2 is None:
      return None
    return np.equal(value1, value2)
  elif tensor.op.type == "NotEqual":
    value1 = constant_value(tensor.op.inputs[0])
    if value1 is None:
      return None
    value2 = constant_value(tensor.op.inputs[1])
    if value2 is None:
      return None
    return np.not_equal(value1, value2)
  else:
    return None

def make_tensor_proto(values, dtype=None, shape=None, verify_shape=False):
  if isinstance(values, tensor_pb2.TensorProto):
    return values

  if dtype:
    dtype = dtypes.as_dtype(dtype)

  is_quantized = (
      dtype in [
          dtypes.qint8, dtypes.quint8, dtypes.qint16, dtypes.quint16,
          dtypes.qint32
      ])


  if isinstance(values, (np.ndarray, np.generic)):
    if dtype:
      nparray = values.astype(dtype.as_numpy_dtype)
    else:
      nparray = values
  elif callable(getattr(values, "__array__", None)) or isinstance(
      getattr(values, "__array_interface__", None), dict):

    nparray = np.asarray(values, dtype=dtype)


    values = nparray
  else:
    if dtype and dtype.is_numpy_compatible:
      np_dt = dtype.as_numpy_dtype
    else:
      np_dt = None
    if shape is not None and np.prod(shape, dtype=np.int64) == 0:
      nparray = np.empty(shape, dtype=np_dt)
    else:
      #_AssertCompatible(values, dtype)
      nparray = np.array(values, dtype=np_dt)
      if (list(nparray.shape) != _GetDenseDimensions(values) and
          not is_quantized):
        raise ValueError("""Argument must be a dense tensor: %s"""
                         """ - got shape %s, but wanted %s.""" %
                         (values, list(nparray.shape),
                          _GetDenseDimensions(values)))

    if (nparray.dtype == np.float64) and dtype is None:
      nparray = nparray.astype(np.float32)
    elif (nparray.dtype == np.int64) and dtype is None:
      downcasted_array = nparray.astype(np.int32)
      if np.array_equal(downcasted_array, nparray):
        nparray = downcasted_array


  numpy_dtype = dtypes.as_dtype(nparray.dtype)
  if numpy_dtype is None:
    raise TypeError("Unrecognized data type: %s" % nparray.dtype)


  if is_quantized:
    numpy_dtype = dtype

  if dtype is not None and (not hasattr(dtype, "base_dtype") or
                            dtype.base_dtype != numpy_dtype.base_dtype):
    raise TypeError("Incompatible types: %s vs. %s. Value is %s" %
                    (dtype, nparray.dtype, values))

  if shape is None:
    shape = nparray.shape
    is_same_size = True
    shape_size = nparray.size
  else:
    shape = [int(dim) for dim in shape]
    shape_size = np.prod(shape, dtype=np.int64)
    is_same_size = shape_size == nparray.size

    if verify_shape:
      if not nparray.shape == tuple(shape):
        raise TypeError("Expected Tensor's shape: %s, got %s." %
                        (tuple(shape), nparray.shape))

    if nparray.size > shape_size:
      raise ValueError(
          "Too many elements provided. Needed at most %d, but received %d" %
          (shape_size, nparray.size))

  tensor_proto = tensor_pb2.TensorProto(
      dtype=numpy_dtype.as_datatype_enum,
      tensor_shape=tensor_shape.as_shape(shape).as_proto())


  if numpy_dtype == dtypes.string and not isinstance(values, np.ndarray):
    proto_values = _FlattenToStrings(values)
    str_values = [compat.as_bytes(x) for x in proto_values]
    tensor_proto.string_val.extend(str_values)
    return tensor_proto

  proto_values = nparray.ravel()

  append_fn = GetNumpyAppendFn(proto_values.dtype)
  if append_fn is None:
    raise TypeError(
        "Element type not supported in TensorProto: %s" % numpy_dtype.name)
  append_fn(tensor_proto, proto_values)

  return tensor_proto

def _GetDenseDimensions(list_of_lists):
  if not isinstance(list_of_lists, (list, tuple)):
    return []
  elif not list_of_lists:
    return [0]
  else:
    return [len(list_of_lists)] + _GetDenseDimensions(list_of_lists[0])

def MakeNdarray(tensor):
  shape = [d.size for d in tensor.tensor_shape.dim]
  num_elements = np.prod(shape, dtype=np.int64)
  tensor_dtype = dtypes.as_dtype(tensor.dtype)
  dtype = tensor_dtype.as_numpy_dtype

  if tensor.tensor_content:
    return (np.frombuffer(tensor.tensor_content, dtype=dtype).copy()
            .reshape(shape))
  elif tensor_dtype == dtypes.float16 or tensor_dtype == dtypes.bfloat16:
    if len(tensor.half_val) == 1:
      tmp = np.array(tensor.half_val[0], dtype=np.uint16)
      tmp.dtype = tensor_dtype.as_numpy_dtype
      return np.repeat(tmp, num_elements).reshape(shape)
    else:
      tmp = np.fromiter(tensor.half_val, dtype=np.uint16)
      tmp.dtype = tensor_dtype.as_numpy_dtype
      return tmp.reshape(shape)
  elif tensor_dtype == dtypes.float32:
    if len(tensor.float_val) == 1:
      return np.repeat(
          np.array(tensor.float_val[0], dtype=dtype),
          num_elements).reshape(shape)
    else:
      return np.fromiter(tensor.float_val, dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.float64:
    if len(tensor.double_val) == 1:
      return np.repeat(
          np.array(tensor.double_val[0], dtype=dtype),
          num_elements).reshape(shape)
    else:
      return np.fromiter(tensor.double_val, dtype=dtype).reshape(shape)
  elif tensor_dtype in [
      dtypes.int32, dtypes.uint8, dtypes.uint16, dtypes.int16, dtypes.int8,
      dtypes.qint32, dtypes.quint8, dtypes.qint8, dtypes.qint16, dtypes.quint16
  ]:
    if len(tensor.int_val) == 1:
      return np.repeat(np.array(tensor.int_val[0], dtype=dtype),
                       num_elements).reshape(shape)
    else:
      return np.fromiter(tensor.int_val, dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.int64:
    if len(tensor.int64_val) == 1:
      return np.repeat(
          np.array(tensor.int64_val[0], dtype=dtype),
          num_elements).reshape(shape)
    else:
      return np.fromiter(tensor.int64_val, dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.string:
    if len(tensor.string_val) == 1:
      return np.repeat(
          np.array(tensor.string_val[0], dtype=dtype),
          num_elements).reshape(shape)
    else:
      return np.array(
          [x for x in tensor.string_val], dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.complex64:
    it = iter(tensor.scomplex_val)
    if len(tensor.scomplex_val) == 2:
      return np.repeat(
          np.array(
              complex(tensor.scomplex_val[0], tensor.scomplex_val[1]),
              dtype=dtype), num_elements).reshape(shape)
    else:
      return np.array(
          [complex(x[0], x[1]) for x in zip(it, it)],
          dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.complex128:
    it = iter(tensor.dcomplex_val)
    if len(tensor.dcomplex_val) == 2:
      return np.repeat(
          np.array(
              complex(tensor.dcomplex_val[0], tensor.dcomplex_val[1]),
              dtype=dtype), num_elements).reshape(shape)
    else:
      return np.array(
          [complex(x[0], x[1]) for x in zip(it, it)],
          dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.bool:
    if len(tensor.bool_val) == 1:
      return np.repeat(np.array(tensor.bool_val[0], dtype=dtype),
                       num_elements).reshape(shape)
    else:
      return np.fromiter(tensor.bool_val, dtype=dtype).reshape(shape)
  else:
    raise TypeError("Unsupported tensor type: %s" % tensor.dtype)

def _FlattenToStrings(nested_strings):
  if isinstance(nested_strings, (list, tuple)):
    for inner in nested_strings:
      for flattened_string in _FlattenToStrings(inner):
        yield flattened_string
  else:
    yield nested_strings


def GetNumpyAppendFn(dtype):
  if dtype.type == np.string_ or dtype.type == np.unicode_:
    return SlowAppendObjectArrayToTensorProto
  return GetFromNumpyDTypeDict(_NP_TO_APPEND_FN, dtype)

def GetFromNumpyDTypeDict(dtype_dict, dtype):
  for key, val in six.iteritems(dtype_dict):
    if key == dtype:
      return val
  return None

