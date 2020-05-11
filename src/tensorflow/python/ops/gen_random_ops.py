import collections as _collections
import six as _six

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python import context as _context


def truncated_normal(shape, dtype, seed=0, seed2=0, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    dtype = make_type(dtype, "dtype")
    if seed is None:
      seed = 0
    seed = make_int(seed, "seed")
    if seed2 is None:
      seed2 = 0
    seed2 = make_int(seed2, "seed2")
    _, _, _op = _op_def_lib._apply_op_helper(
        "TruncatedNormal", shape=shape, dtype=dtype, seed=seed, seed2=seed2,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("seed", _op.get_attr("seed"), "seed2", _op.get_attr("seed2"),
              "dtype", _op.get_attr("dtype"), "T", _op.get_attr("T"))
    record_gradient(
      "TruncatedNormal", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "TruncatedNormal", name, _ctx._post_execution_callbacks, shape,
        "seed", seed, "seed2", seed2, "dtype", dtype)
      return _result
    except _core._FallbackException:
      return truncated_normal_eager_fallback(
          shape, seed=seed, seed2=seed2, dtype=dtype, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def make_type(v, arg_name):
  try:
    v = _dtypes.as_dtype(v).base_dtype
  except TypeError:
    raise TypeError("Expected DataType for argument '%s' not %s." %
                    (arg_name, repr(v)))
  i = v.as_datatype_enum
  return i

def make_int(v, arg_name):
  if isinstance(v, _six.string_types):
    raise TypeError("Expected int for argument '%s' not %s." %
                    (arg_name, repr(v)))
  try:
    return int(v)
  except (ValueError, TypeError):
    raise TypeError("Expected int for argument '%s' not %s." %
                    (arg_name, repr(v)))

def record_gradient(unused_op_name, unused_inputs, unused_attrs, unused_results,
                    unused_name):
  
  pass

def random_uniform(shape, dtype, seed=0, seed2=0, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    dtype = make_type(dtype, "dtype")
    if seed is None:
      seed = 0
    seed = make_int(seed, "seed")
    if seed2 is None:
      seed2 = 0
    seed2 = make_int(seed2, "seed2")
    _, _, _op = _op_def_lib._apply_op_helper(
        "RandomUniform", shape=shape, dtype=dtype, seed=seed, seed2=seed2,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("seed", _op.get_attr("seed"), "seed2", _op.get_attr("seed2"),
              "dtype", _op.get_attr("dtype"), "T", _op.get_attr("T"))
    record_gradient(
      "RandomUniform", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "RandomUniform", name, _ctx._post_execution_callbacks, shape, "seed",
        seed, "seed2", seed2, "dtype", dtype)
      return _result
    except _core._FallbackException:
      return random_uniform_eager_fallback(
          shape, seed=seed, seed2=seed2, dtype=dtype, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib

_op_def_lib = _InitOpDefLibrary(b"\n\250\001\n\013Multinomial\022\013\n\006logits\"\001T\022\017\n\013num_samples\030\003\032\026\n\006output\"\014output_dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\" \n\014output_dtype\022\004type\032\0020\t:\006\n\0042\002\003\t\210\001\001\n\322\001\n\034ParameterizedTruncatedNormal\022\n\n\005shape\"\001T\022\016\n\005means\"\005dtype\022\017\n\006stdevs\"\005dtype\022\020\n\007minvals\"\005dtype\022\020\n\007maxvals\"\005dtype\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\027\n\005dtype\022\004type:\010\n\0062\004\023\016\001\002\"\021\n\001T\022\004type:\006\n\0042\002\003\t\210\001\001\n\177\n\013RandomGamma\022\n\n\005shape\"\001S\022\n\n\005alpha\"\001T\032\013\n\006output\"\001T\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\021\n\001S\022\004type:\006\n\0042\002\003\t\"\022\n\001T\022\004type:\007\n\0052\003\023\001\002\210\001\001\nJ\n\017RandomGammaGrad\022\n\n\005alpha\"\001T\022\013\n\006sample\"\001T\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\n\255\001\n\rRandomPoisson\022\n\n\005shape\"\001S\022\r\n\004rate\"\005dtype\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\021\n\001S\022\004type:\006\n\0042\002\003\t\"\026\n\005dtype\022\004type:\007\n\0052\003\023\001\002B\037\010\031\022\033Replaced by RandomPoissonV2\210\001\001\n\252\001\n\017RandomPoissonV2\022\n\n\005shape\"\001S\022\t\n\004rate\"\001R\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\021\n\001S\022\004type:\006\n\0042\002\003\t\"\030\n\001R\022\004type\032\0020\002:\t\n\0072\005\023\001\002\003\t\"\034\n\005dtype\022\004type\032\0020\t:\t\n\0072\005\023\001\002\003\t\210\001\001\nY\n\rRandomShuffle\022\n\n\005value\"\001T\032\013\n\006output\"\001T\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\t\n\001T\022\004type\210\001\001\n\205\001\n\024RandomStandardNormal\022\n\n\005shape\"\001T\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\027\n\005dtype\022\004type:\010\n\0062\004\023\016\001\002\"\021\n\001T\022\004type:\006\n\0042\002\003\t\210\001\001\n~\n\rRandomUniform\022\n\n\005shape\"\001T\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\027\n\005dtype\022\004type:\010\n\0062\004\023\016\001\002\"\021\n\001T\022\004type:\006\n\0042\002\003\t\210\001\001\n\235\001\n\020RandomUniformInt\022\n\n\005shape\"\001T\022\016\n\006minval\"\004Tout\022\016\n\006maxval\"\004Tout\032\016\n\006output\"\004Tout\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\024\n\004Tout\022\004type:\006\n\0042\002\003\t\"\021\n\001T\022\004type:\006\n\0042\002\003\t\210\001\001\n\200\001\n\017TruncatedNormal\022\n\n\005shape\"\001T\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\027\n\005dtype\022\004type:\010\n\0062\004\023\016\001\002\"\021\n\001T\022\004type:\006\n\0042\002\003\t\210\001\001")
