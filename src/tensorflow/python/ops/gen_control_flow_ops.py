import collections as _collections
import six as _six


from tensorflow.python import execute as _execute
from tensorflow.python import context as _context
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.core import op_def_pb2 as _op_def_pb2

def switch(data, pred, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Switch", data=data, pred=pred, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "Switch", _inputs_flat, _attrs, _result, name)
    _result = _SwitchOutput._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Switch", name,
        _ctx._post_execution_callbacks, data, pred)
      _result = _SwitchOutput._make(_result)
      return _result
    except _core._FallbackException:
      return switch_eager_fallback(
          data, pred, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

_switch_outputs = ["output_false", "output_true"]
_SwitchOutput = _collections.namedtuple(
    "Switch", _switch_outputs)

def merge(inputs, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(inputs, (list, tuple)):
      raise TypeError(
          "Expected list for 'inputs' argument to "
          "'merge' Op, not %r." % inputs)
    _attr_N = len(inputs)
    _, _, _op = _op_def_lib._apply_op_helper(
        "Merge", inputs=inputs, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "N", _op.get_attr("N"))
    _execute.record_gradient(
      "Merge", _inputs_flat, _attrs, _result, name)
    _result = _MergeOutput._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Merge", name,
        _ctx._post_execution_callbacks, inputs)
      _result = _MergeOutput._make(_result)
      return _result
    except _core._FallbackException:
      return merge_eager_fallback(
          inputs, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

_merge_outputs = ["output", "value_index"]
_MergeOutput = _collections.namedtuple(
    "Merge", _merge_outputs)

def enter(data, frame_name, is_constant=False, parallel_iterations=10, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    frame_name = _execute.make_str(frame_name, "frame_name")
    if is_constant is None:
      is_constant = False
    is_constant = _execute.make_bool(is_constant, "is_constant")
    if parallel_iterations is None:
      parallel_iterations = 10
    parallel_iterations = _execute.make_int(parallel_iterations, "parallel_iterations")
    _, _, _op = _op_def_lib._apply_op_helper(
        "Enter", data=data, frame_name=frame_name, is_constant=is_constant,
        parallel_iterations=parallel_iterations, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "frame_name",
              _op.get_attr("frame_name"), "is_constant",
              _op.get_attr("is_constant"), "parallel_iterations",
              _op.get_attr("parallel_iterations"))
    _execute.record_gradient(
      "Enter", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Enter", name,
        _ctx._post_execution_callbacks, data, "frame_name", frame_name,
        "is_constant", is_constant, "parallel_iterations",
        parallel_iterations)
      return _result
    except _core._FallbackException:
      return enter_eager_fallback(
          data, frame_name=frame_name, is_constant=is_constant,
          parallel_iterations=parallel_iterations, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def loop_cond(input, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "LoopCond", input=input, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
    _execute.record_gradient(
      "LoopCond", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "LoopCond",
        name, _ctx._post_execution_callbacks, input)
      return _result
    except _core._FallbackException:
      return loop_cond_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def next_iteration(data, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "NextIteration", data=data, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "NextIteration", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "NextIteration", name, _ctx._post_execution_callbacks, data)
      return _result
    except _core._FallbackException:
      return next_iteration_eager_fallback(
          data, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def _exit(data, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Exit", data=data, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "Exit", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Exit", name,
        _ctx._post_execution_callbacks, data)
      return _result
    except _core._FallbackException:
      return _exit_eager_fallback(
          data, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def no_op(name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "NoOp", name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "NoOp", name,
        _ctx._post_execution_callbacks)
      return _result
    except _core._FallbackException:
      return no_op_eager_fallback(
          name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def control_trigger(name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "ControlTrigger", name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "ControlTrigger", name, _ctx._post_execution_callbacks)
      return _result
    except _core._FallbackException:
      return control_trigger_eager_fallback(
          name=name, ctx=_ctx)
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

_op_def_lib = _InitOpDefLibrary(b"\n@\n\005Abort\"\027\n\terror_msg\022\006string\032\002\022\000\"\036\n\022exit_without_error\022\004bool\032\002(\000\n\020\n\016ControlTrigger\ny\n\005Enter\022\t\n\004data\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\024\n\nframe_name\022\006string\"\027\n\013is_constant\022\004bool\032\002(\000\"\036\n\023parallel_iterations\022\003int\032\002\030\n\n)\n\004Exit\022\t\n\004data\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n!\n\010LoopCond\022\t\n\005input\030\n\032\n\n\006output\030\n\nN\n\005Merge\022\016\n\006inputs\"\001T*\001N\032\013\n\006output\"\001T\032\017\n\013value_index\030\003\"\t\n\001T\022\004type\"\014\n\001N\022\003int(\0010\001\n2\n\rNextIteration\022\t\n\004data\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n\006\n\004NoOp\n\202\001\n\010RefEnter\022\014\n\004data\"\001T\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\"\024\n\nframe_name\022\006string\"\027\n\013is_constant\022\004bool\032\002(\000\"\036\n\023parallel_iterations\022\003int\032\002\030\n\n2\n\007RefExit\022\014\n\004data\"\001T\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\nW\n\010RefMerge\022\021\n\006inputs\"\001T*\001N\200\001\001\032\016\n\006output\"\001T\200\001\001\032\017\n\013value_index\030\003\"\t\n\001T\022\004type\"\014\n\001N\022\003int(\0010\001\n;\n\020RefNextIteration\022\014\n\004data\"\001T\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\nR\n\tRefSelect\022\t\n\005index\030\003\022\021\n\006inputs\"\001T*\001N\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\"\014\n\001N\022\003int(\0010\001\n\\\n\tRefSwitch\022\014\n\004data\"\001T\200\001\001\022\010\n\004pred\030\n\032\024\n\014output_false\"\001T\200\001\001\032\023\n\013output_true\"\001T\200\001\001\"\t\n\001T\022\004type\230\001\001\nM\n\006Switch\022\t\n\004data\"\001T\022\010\n\004pred\030\n\032\021\n\014output_false\"\001T\032\020\n\013output_true\"\001T\"\t\n\001T\022\004type")
