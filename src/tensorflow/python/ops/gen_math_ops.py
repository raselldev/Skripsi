import collections as _collections
import six as _six


from tensorflow.python import context as _context
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.util.tf_export import tf_export

def make_bool(v, arg_name):
  if not isinstance(v, bool):
    raise TypeError("Expected bool for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return v

def mean(input, axis, keep_dims=False, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if keep_dims is None:
      keep_dims = False
    keep_dims = make_bool(keep_dims, "keep_dims")
    _, _, _op = _op_def_lib._apply_op_helper(
        "Mean", input=input, reduction_indices=axis, keep_dims=keep_dims,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("keep_dims", _op.get_attr("keep_dims"), "T", _op.get_attr("T"),
              "Tidx", _op.get_attr("Tidx"))
    record_gradient(
      "Mean", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Mean", name,
        _ctx._post_execution_callbacks, input, axis, "keep_dims", keep_dims)
      return _result
    except _core._FallbackException:
      return mean_eager_fallback(
          input, axis, keep_dims=keep_dims, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def floor_mod(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "FloorMod", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "FloorMod", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "FloorMod",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return floor_mod_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def real_div(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "RealDiv", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "RealDiv", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "RealDiv",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return real_div_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def truncate_div(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "TruncateDiv", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "TruncateDiv", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "TruncateDiv",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return truncate_div_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def floor_div(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "FloorDiv", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "FloorDiv", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "FloorDiv",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return floor_div_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def truncate_mod(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "TruncateMod", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "TruncateMod", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "TruncateMod",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return truncate_mod_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def add(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Add", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "Add", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Add", name,
        _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return add_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def sub(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Sub", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "Sub", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Sub", name,
        _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return sub_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def logical_and(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "LogicalAnd", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
    record_gradient(
      "LogicalAnd", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "LogicalAnd",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return logical_and_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def logical_or(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "LogicalOr", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
    record_gradient(
      "LogicalOr", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "LogicalOr",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return logical_or_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def less(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Less", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "Less", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Less", name,
        _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return less_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def less_equal(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "LessEqual", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "LessEqual", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "LessEqual",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return less_equal_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def greater(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Greater", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "Greater", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Greater",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return greater_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def greater_equal(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "GreaterEqual", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "GreaterEqual", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "GreaterEqual",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return greater_equal_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def mul(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Mul", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "Mul", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Mul", name,
        _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return mul_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def record_gradient(unused_op_name, unused_inputs, unused_attrs, unused_results,
                    unused_name):
  
  pass

def tanh(x, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Tanh", x=x, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "Tanh", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Tanh", name,
        _ctx._post_execution_callbacks, x)
      return _result
    except _core._FallbackException:
      return tanh_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def _range(start, limit, delta, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Range", start=start, limit=limit, delta=delta, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("Tidx", _op.get_attr("Tidx"))
    record_gradient(
      "Range", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Range", name,
        _ctx._post_execution_callbacks, start, limit, delta)
      return _result
    except _core._FallbackException:
      return _range_eager_fallback(
          start, limit, delta, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def minimum(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Minimum", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "Minimum", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Minimum",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return minimum_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def maximum(x, y, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Maximum", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "Maximum", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Maximum",
        name, _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return maximum_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def sigmoid(x, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Sigmoid", x=x, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    record_gradient(
      "Sigmoid", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Sigmoid",
        name, _ctx._post_execution_callbacks, x)
      return _result
    except _core._FallbackException:
      return sigmoid_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def mat_mul(a, b, transpose_a=False, transpose_b=False, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if transpose_a is None:
      transpose_a = False
    transpose_a = make_bool(transpose_a, "transpose_a")
    if transpose_b is None:
      transpose_b = False
    transpose_b = make_bool(transpose_b, "transpose_b")
    _, _, _op = _op_def_lib._apply_op_helper(
        "MatMul", a=a, b=b, transpose_a=transpose_a, transpose_b=transpose_b,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("transpose_a", _op.get_attr("transpose_a"), "transpose_b",
              _op.get_attr("transpose_b"), "T", _op.get_attr("T"))
    record_gradient(
      "MatMul", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "MatMul", name,
        _ctx._post_execution_callbacks, a, b, "transpose_a", transpose_a,
        "transpose_b", transpose_b)
      return _result
    except _core._FallbackException:
      return mat_mul_eager_fallback(
          a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name,
          ctx=_ctx)
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

_op_def_lib = _InitOpDefLibrary(b"\n,\n\003Abs\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\003\t\no\n\rAccumulateNV2\022\016\n\006inputs\"\001T*\001N\032\010\n\003sum\"\001T\"\014\n\001N\022\003int(\0010\001\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\016\n\005shape\022\005shape\200\001\001\220\001\001\n/\n\004Acos\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n.\n\005Acosh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n:\n\003Add\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\005\003\t\010\022\007\nW\n\004AddN\022\016\n\006inputs\"\001T*\001N\032\010\n\003sum\"\001T\"\014\n\001N\022\003int(\0010\001\"!\n\001T\022\004type:\026\n\0242\022\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\025\200\001\001\220\001\001\nA\n\005AddV2\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\032\n\001T\022\004type:\017\n\r2\013\016\023\001\002\004\006\005\003\t\010\022\200\001\001\220\001\001\nh\n\003All\022\t\n\005input\030\n\022\031\n\021reduction_indices\"\004Tidx\032\n\n\006output\030\n\"\025\n\tkeep_dims\022\004bool\032\002(\000\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\nT\n\005Angle\022\n\n\005input\"\001T\032\016\n\006output\"\004Tout\"\025\n\001T\022\004type\032\0020\010:\006\n\0042\002\010\022\"\030\n\004Tout\022\004type\032\0020\001:\006\n\0042\002\001\002\nh\n\003Any\022\t\n\005input\030\n\022\031\n\021reduction_indices\"\004Tidx\032\n\n\006output\030\n\"\025\n\tkeep_dims\022\004bool\032\002(\000\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\ni\n\020ApproximateEqual\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\031\n\ttolerance\022\005float\032\005%\254\305\'7\220\001\001\n\233\001\n\006ArgMax\022\n\n\005input\"\001T\022\021\n\tdimension\"\004Tidx\032\025\n\006output\"\013output_type\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\"\037\n\013output_type\022\004type\032\0020\t:\006\n\0042\002\003\t\n\233\001\n\006ArgMin\022\n\n\005input\"\001T\022\021\n\tdimension\"\004Tidx\032\025\n\006output\"\013output_type\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\"\037\n\013output_type\022\004type\032\0020\t:\006\n\0042\002\003\t\n/\n\004Asin\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n.\n\005Asinh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n/\n\004Atan\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n4\n\005Atan2\022\006\n\001y\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n.\n\005Atanh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\nh\n\013BatchMatMul\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\013\n\006output\"\001T\"\026\n\001T\022\004type:\013\n\t2\007\016\023\001\002\003\010\022\"\021\n\005adj_x\022\004bool\032\002(\000\"\021\n\005adj_y\022\004bool\032\002(\000\n0\n\tBesselI0e\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n0\n\tBesselI1e\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n<\n\007Betainc\022\006\n\001a\"\001T\022\006\n\001b\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\nK\n\010Bincount\022\007\n\003arr\030\003\022\010\n\004size\030\003\022\014\n\007weights\"\001T\032\t\n\004bins\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\003\t\001\002\nS\n\tBucketize\022\n\n\005input\"\001T\032\n\n\006output\030\003\"\023\n\001T\022\004type:\010\n\0062\004\003\t\001\002\"\031\n\nboundaries\022\013list(float)\nN\n\004Cast\022\t\n\001x\"\004SrcT\032\t\n\001y\"\004DstT\"\014\n\004SrcT\022\004type\"\014\n\004DstT\022\004type\"\024\n\010Truncate\022\004bool\032\002(\000\n+\n\004Ceil\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\nn\n\013ClipByValue\022\006\n\001t\"\001T\022\023\n\016clip_value_min\"\001T\022\023\n\016clip_value_max\"\001T\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\nT\n\021CompareAndBitpack\022\n\n\005input\"\001T\022\016\n\tthreshold\"\001T\032\n\n\006output\030\004\"\027\n\001T\022\004type:\014\n\n2\010\n\023\001\002\006\005\003\t\n]\n\007Complex\022\t\n\004real\"\001T\022\t\n\004imag\"\001T\032\013\n\003out\"\004Tout\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\030\n\004Tout\022\004type\032\0020\010:\006\n\0042\002\010\022\nP\n\nComplexAbs\022\006\n\001x\"\001T\032\t\n\001y\"\004Tout\"\025\n\001T\022\004type\032\0020\010:\006\n\0042\002\010\022\"\030\n\004Tout\022\004type\032\0020\001:\006\n\0042\002\001\002\n7\n\004Conj\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\026\n\001T\022\004type\032\0020\010:\007\n\0052\003\010\022\025\n,\n\003Cos\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n-\n\004Cosh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\nB\n\005Cross\022\006\n\001a\"\001T\022\006\n\001b\"\001T\032\014\n\007product\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\221\001\n\007Cumprod\022\006\n\001x\"\001T\022\014\n\004axis\"\004Tidx\032\010\n\003out\"\001T\"\025\n\texclusive\022\004bool\032\002(\000\"\023\n\007reverse\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\220\001\n\006Cumsum\022\006\n\001x\"\001T\022\014\n\004axis\"\004Tidx\032\010\n\003out\"\001T\"\025\n\texclusive\022\004bool\032\002(\000\"\023\n\007reverse\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n.\n\007Digamma\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n:\n\003Div\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\n5\n\010DivNoNan\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\nB\n\005Equal\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\037\n\001T\022\004type:\024\n\0222\020\016\023\001\002\004\006\005\003\t\010\014\013\r\007\n\022\220\001\001\n*\n\003Erf\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n+\n\004Erfc\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n,\n\003Exp\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n.\n\005Expm1\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n,\n\005Floor\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n?\n\010FloorDiv\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\n9\n\010FloorMod\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\003\t\016\023\001\002\n=\n\007Greater\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\nB\n\014GreaterEqual\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n}\n\023HistogramFixedWidth\022\013\n\006values\"\001T\022\020\n\013value_range\"\001T\022\t\n\005nbins\030\003\032\014\n\003out\"\005dtype\"\023\n\001T\022\004type:\010\n\0062\004\003\t\001\002\"\031\n\005dtype\022\004type\032\0020\003:\006\n\0042\002\003\t\n3\n\006Igamma\022\006\n\001a\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\n8\n\013IgammaGradA\022\006\n\001a\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\n4\n\007Igammac\022\006\n\001a\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\nS\n\004Imag\022\n\n\005input\"\001T\032\016\n\006output\"\004Tout\"\025\n\001T\022\004type\032\0020\010:\006\n\0042\002\010\022\"\030\n\004Tout\022\004type\032\0020\001:\006\n\0042\002\001\002\n.\n\003Inv\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n9\n\007InvGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n.\n\010IsFinite\022\006\n\001x\"\001T\032\005\n\001y\030\n\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n+\n\005IsInf\022\006\n\001x\"\001T\032\005\n\001y\030\n\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n+\n\005IsNan\022\006\n\001x\"\001T\032\005\n\001y\030\n\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n:\n\004Less\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n?\n\tLessEqual\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n-\n\006Lgamma\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\ni\n\010LinSpace\022\n\n\005start\"\001T\022\t\n\004stop\"\001T\022\013\n\003num\"\004Tidx\032\013\n\006output\"\001T\"\022\n\001T\022\004type:\007\n\0052\003\016\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n,\n\003Log\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n.\n\005Log1p\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n$\n\nLogicalAnd\022\005\n\001x\030\n\022\005\n\001y\030\n\032\005\n\001z\030\n\220\001\001\n\032\n\nLogicalNot\022\005\n\001x\030\n\032\005\n\001y\030\n\n#\n\tLogicalOr\022\005\n\001x\030\n\022\005\n\001y\030\n\032\005\n\001z\030\n\220\001\001\np\n\006MatMul\022\006\n\001a\"\001T\022\006\n\001b\"\001T\032\014\n\007product\"\001T\"\027\n\013transpose_a\022\004bool\032\002(\000\"\027\n\013transpose_b\022\004bool\032\002(\000\"\026\n\001T\022\004type:\013\n\t2\007\016\023\001\002\003\010\022\n\214\001\n\003Max\022\n\n\005input\"\001T\022\031\n\021reduction_indices\"\004Tidx\032\013\n\006output\"\001T\"\025\n\tkeep_dims\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n;\n\007Maximum\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\003\t\220\001\001\n\215\001\n\004Mean\022\n\n\005input\"\001T\022\031\n\021reduction_indices\"\004Tidx\032\013\n\006output\"\001T\"\025\n\tkeep_dims\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\214\001\n\003Min\022\n\n\005input\"\001T\022\031\n\021reduction_indices\"\004Tidx\032\013\n\006output\"\001T\"\025\n\tkeep_dims\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n;\n\007Minimum\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\003\t\220\001\001\n5\n\003Mod\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\026\n\001T\022\004type:\013\n\t2\007\003\t\023\023\016\001\002\n=\n\003Mul\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\220\001\001\n.\n\003Neg\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\nE\n\010NotEqual\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\037\n\001T\022\004type:\024\n\0222\020\016\023\001\002\004\006\005\003\t\010\014\013\r\007\n\022\220\001\001\n6\n\tPolygamma\022\006\n\001a\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\n6\n\003Pow\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\001\023\002\003\t\010\022\n\215\001\n\004Prod\022\n\n\005input\"\001T\022\031\n\021reduction_indices\"\004Tidx\032\013\n\006output\"\001T\"\025\n\tkeep_dims\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\267\001\n\032QuantizeDownAndShrinkRange\022\017\n\005input\"\006Tinput\022\r\n\tinput_min\030\001\022\r\n\tinput_max\030\001\032\022\n\006output\"\010out_type\032\016\n\noutput_min\030\001\032\016\n\noutput_max\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\033\n\010out_type\022\004type:\t\n\0072\005\013\014\r\017\020\n\301\001\n\014QuantizedAdd\022\007\n\001x\"\002T1\022\007\n\001y\"\002T2\022\t\n\005min_x\030\001\022\t\n\005max_x\030\001\022\t\n\005min_y\030\001\022\t\n\005max_y\030\001\032\014\n\001z\"\007Toutput\032\t\n\005min_z\030\001\032\t\n\005max_z\030\001\"\025\n\002T1\022\004type:\t\n\0072\005\013\014\r\017\020\"\025\n\002T2\022\004type:\t\n\0072\005\013\014\r\017\020\"\036\n\007Toutput\022\004type\032\0020\r:\t\n\0072\005\013\014\r\017\020\220\001\001\n\235\002\n\017QuantizedMatMul\022\007\n\001a\"\002T1\022\007\n\001b\"\002T2\022\t\n\005min_a\030\001\022\t\n\005max_a\030\001\022\t\n\005min_b\030\001\022\t\n\005max_b\030\001\032\016\n\003out\"\007Toutput\032\013\n\007min_out\030\001\032\013\n\007max_out\030\001\"\025\n\002T1\022\004type:\t\n\0072\005\013\014\r\017\020\"\025\n\002T2\022\004type:\t\n\0072\005\013\014\r\017\020\"\036\n\007Toutput\022\004type\032\0020\r:\t\n\0072\005\013\014\r\017\020\"\027\n\013transpose_a\022\004bool\032\002(\000\"\027\n\013transpose_b\022\004bool\032\002(\000\"\"\n\013Tactivation\022\004type\032\0020\014:\t\n\0072\005\013\014\r\017\020\n\301\001\n\014QuantizedMul\022\007\n\001x\"\002T1\022\007\n\001y\"\002T2\022\t\n\005min_x\030\001\022\t\n\005max_x\030\001\022\t\n\005min_y\030\001\022\t\n\005max_y\030\001\032\014\n\001z\"\007Toutput\032\t\n\005min_z\030\001\032\t\n\005max_z\030\001\"\025\n\002T1\022\004type:\t\n\0072\005\013\014\r\017\020\"\025\n\002T2\022\004type:\t\n\0072\005\013\014\r\017\020\"\036\n\007Toutput\022\004type\032\0020\r:\t\n\0072\005\013\014\r\017\020\220\001\001\na\n\005Range\022\r\n\005start\"\004Tidx\022\r\n\005limit\"\004Tidx\022\r\n\005delta\"\004Tidx\032\016\n\006output\"\004Tidx\"\033\n\004Tidx\022\004type\032\0020\003:\t\n\0072\005\016\001\002\003\t\nS\n\004Real\022\n\n\005input\"\001T\032\016\n\006output\"\004Tout\"\025\n\001T\022\004type\032\0020\010:\006\n\0042\002\010\022\"\030\n\004Tout\022\004type\032\0020\001:\006\n\0042\002\001\002\n>\n\007RealDiv\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\n5\n\nReciprocal\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n@\n\016ReciprocalGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n\177\n\023RequantizationRange\022\017\n\005input\"\006Tinput\022\r\n\tinput_min\030\001\022\r\n\tinput_max\030\001\032\016\n\noutput_min\030\001\032\016\n\noutput_max\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\n\333\001\n\nRequantize\022\017\n\005input\"\006Tinput\022\r\n\tinput_min\030\001\022\r\n\tinput_max\030\001\022\030\n\024requested_output_min\030\001\022\030\n\024requested_output_max\030\001\032\022\n\006output\"\010out_type\032\016\n\noutput_min\030\001\032\016\n\noutput_max\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\033\n\010out_type\022\004type:\t\n\0072\005\013\014\r\017\020\n+\n\004Rint\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n0\n\005Round\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n.\n\005Rsqrt\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n;\n\tRsqrtGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\nt\n\nSegmentMax\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\nz\n\013SegmentMean\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\nt\n\nSegmentMin\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\nz\n\013SegmentProd\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\ny\n\nSegmentSum\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\n?\n\006Select\022\r\n\tcondition\030\n\022\006\n\001t\"\001T\022\006\n\001e\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n0\n\007Sigmoid\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n=\n\013SigmoidGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n/\n\004Sign\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n,\n\003Sin\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n-\n\004Sinh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n\301\001\n\014SparseMatMul\022\007\n\001a\"\002Ta\022\007\n\001b\"\002Tb\032\013\n\007product\030\001\"\027\n\013transpose_a\022\004bool\032\002(\000\"\027\n\013transpose_b\022\004bool\032\002(\000\"\027\n\013a_is_sparse\022\004bool\032\002(\000\"\027\n\013b_is_sparse\022\004bool\032\002(\000\"\026\n\002Ta\022\004type\032\0020\001:\006\n\0042\002\001\016\"\026\n\002Tb\022\004type\032\0020\001:\006\n\0042\002\001\016\nz\n\021SparseSegmentMean\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\217\001\n\025SparseSegmentMeanGrad\022\t\n\004grad\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\022\017\n\013output_dim0\030\003\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\311\001\n SparseSegmentMeanWithNumSegments\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n{\n\022SparseSegmentSqrtN\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\220\001\n\026SparseSegmentSqrtNGrad\022\t\n\004grad\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\022\017\n\013output_dim0\030\003\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\312\001\n!SparseSegmentSqrtNWithNumSegments\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n\203\001\n\020SparseSegmentSum\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\322\001\n\037SparseSegmentSumWithNumSegments\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n-\n\004Sqrt\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n:\n\010SqrtGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n1\n\006Square\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\nG\n\021SquaredDifference\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\220\001\001\n:\n\003Sub\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\n\214\001\n\003Sum\022\n\n\005input\"\001T\022\031\n\021reduction_indices\"\004Tidx\032\013\n\006output\"\001T\"\025\n\tkeep_dims\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n.\n\003Tan\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n-\n\004Tanh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n:\n\010TanhGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\nB\n\013TruncateDiv\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\n<\n\013TruncateMod\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\003\t\016\023\001\002\n\274\001\n\022UnsortedSegmentMax\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n\274\001\n\022UnsortedSegmentMin\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n\302\001\n\023UnsortedSegmentProd\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n\301\001\n\022UnsortedSegmentSum\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n5\n\005Xdivy\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\024\n\001T\022\004type:\t\n\0072\005\023\001\002\010\022\n5\n\005Xlogy\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\024\n\001T\022\004type:\t\n\0072\005\023\001\002\010\022\n1\n\004Zeta\022\006\n\001x\"\001T\022\006\n\001q\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002")
