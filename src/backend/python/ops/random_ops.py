from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from backend.python.framework import dtypes
from backend.python.framework import ops
from backend.python.ops import math_ops
from backend import context as _context
from backend import execute as _execute
from backend.core import op_def_pb2 as _op_def_pb2
from backend.python.framework import dtypes as _dtypes
#from backend.python.framework import op_def_registry as _op_def_registry
from backend.python.framework import op_def_library as _op_def_library


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
    seed1, seed2 = get_seed(seed)
    rnd = truncated_normals(
        shape_tensor, dtype, seed=seed1, seed2=seed2)
    mul = rnd * stddev_tensor
    value = math_ops.add(mul, mean_tensor, name=name)
    return value

def _ShapeTensor(shape):
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int32
  else:
    dtype = None
  return ops.convert_to_tensor(shape, dtype=dtype, name="shape")

def get_seed(op_seed):
 
  global_seed = ops.get_default_graph().seed

  if global_seed is not None:
    if op_seed is None:
      # pylint: disable=protected-access
      if eager:
        op_seed = context.internal_operation_seed()
      else:
        op_seed = ops.get_default_graph()._last_id

    seeds = _truncate_seed(global_seed), _truncate_seed(op_seed)
  else:
    if op_seed is not None:
      seeds = DEFAULT_GRAPH_SEED, _truncate_seed(op_seed)
    else:
      seeds = None, None
  # Avoid (0, 0) as the C++ ops interpret it as nondeterminism, which would
  # be unexpected since Python docs say nondeterminism is (None, None).
  if seeds == (0, 0):
    return (0, _MAXINT32)
  return seeds

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
    seed1, seed2 = get_seed(seed)
    if dtype.is_integer:
      return gen_random_ops.random_uniform_int(
          shape, minval, maxval, seed=seed1, seed2=seed2, name=name)
    else:
      rnd = random_uniforms(shape, dtype, seed=seed1, seed2=seed2)
      return math_ops.add(rnd * (maxval - minval), minval, name=name)

def truncated_normals(shape, dtype, seed=0, seed2=0, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    dtype = _execute.make_type(dtype, "dtype")
    if seed is None:
      seed = 0
    seed = _execute.make_int(seed, "seed")
    if seed2 is None:
      seed2 = 0
    seed2 = _execute.make_int(seed2, "seed2")
    _, _, _op = _op_def_lib._apply_op_helper(
        "TruncatedNormal", shape=shape, dtype=dtype, seed=seed, seed2=seed2,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("seed", _op.get_attr("seed"), "seed2", _op.get_attr("seed2"),
              "dtype", _op.get_attr("dtype"), "T", _op.get_attr("T"))
    _execute.record_gradient(
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

def random_uniforms(shape, dtype, seed=0, seed2=0, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    dtype = _execute.make_type(dtype, "dtype")
    if seed is None:
      seed = 0
    seed = _execute.make_int(seed, "seed")
    if seed2 is None:
      seed2 = 0
    seed2 = _execute.make_int(seed2, "seed2")
    _, _, _op = _op_def_lib._apply_op_helper(
        "RandomUniform", shape=shape, dtype=dtype, seed=seed, seed2=seed2,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("seed", _op.get_attr("seed"), "seed2", _op.get_attr("seed2"),
              "dtype", _op.get_attr("dtype"), "T", _op.get_attr("T"))
    _execute.record_gradient(
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
#  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib

_op_def_lib = _InitOpDefLibrary(b"\n\250\001\n\013Multinomial\022\013\n\006logits\"\001T\022\017\n\013num_samples\030\003\032\026\n\006output\"\014output_dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\" \n\014output_dtype\022\004type\032\0020\t:\006\n\0042\002\003\t\210\001\001\n\322\001\n\034ParameterizedTruncatedNormal\022\n\n\005shape\"\001T\022\016\n\005means\"\005dtype\022\017\n\006stdevs\"\005dtype\022\020\n\007minvals\"\005dtype\022\020\n\007maxvals\"\005dtype\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\027\n\005dtype\022\004type:\010\n\0062\004\023\016\001\002\"\021\n\001T\022\004type:\006\n\0042\002\003\t\210\001\001\n\177\n\013RandomGamma\022\n\n\005shape\"\001S\022\n\n\005alpha\"\001T\032\013\n\006output\"\001T\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\021\n\001S\022\004type:\006\n\0042\002\003\t\"\022\n\001T\022\004type:\007\n\0052\003\023\001\002\210\001\001\nJ\n\017RandomGammaGrad\022\n\n\005alpha\"\001T\022\013\n\006sample\"\001T\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\n\255\001\n\rRandomPoisson\022\n\n\005shape\"\001S\022\r\n\004rate\"\005dtype\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\021\n\001S\022\004type:\006\n\0042\002\003\t\"\026\n\005dtype\022\004type:\007\n\0052\003\023\001\002B\037\010\031\022\033Replaced by RandomPoissonV2\210\001\001\n\252\001\n\017RandomPoissonV2\022\n\n\005shape\"\001S\022\t\n\004rate\"\001R\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\021\n\001S\022\004type:\006\n\0042\002\003\t\"\030\n\001R\022\004type\032\0020\002:\t\n\0072\005\023\001\002\003\t\"\034\n\005dtype\022\004type\032\0020\t:\t\n\0072\005\023\001\002\003\t\210\001\001\nY\n\rRandomShuffle\022\n\n\005value\"\001T\032\013\n\006output\"\001T\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\t\n\001T\022\004type\210\001\001\n\205\001\n\024RandomStandardNormal\022\n\n\005shape\"\001T\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\027\n\005dtype\022\004type:\010\n\0062\004\023\016\001\002\"\021\n\001T\022\004type:\006\n\0042\002\003\t\210\001\001\n~\n\rRandomUniform\022\n\n\005shape\"\001T\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\027\n\005dtype\022\004type:\010\n\0062\004\023\016\001\002\"\021\n\001T\022\004type:\006\n\0042\002\003\t\210\001\001\n\235\001\n\020RandomUniformInt\022\n\n\005shape\"\001T\022\016\n\006minval\"\004Tout\022\016\n\006maxval\"\004Tout\032\016\n\006output\"\004Tout\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\024\n\004Tout\022\004type:\006\n\0042\002\003\t\"\021\n\001T\022\004type:\006\n\0042\002\003\t\210\001\001\n\200\001\n\017TruncatedNormal\022\n\n\005shape\"\001T\032\017\n\006output\"\005dtype\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\027\n\005dtype\022\004type:\010\n\0062\004\023\016\001\002\"\021\n\001T\022\004type:\006\n\0042\002\003\t\210\001\001")
