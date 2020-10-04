import collections as _collections
import six as _six


from backend import execute as _execute
from backend import context as _context
from backend.framework import op_def_library as _op_def_library
#from backend.framework import op_def_registry as _op_def_registry
#from backend.python.util.tf_export import tf_export
from backend.core import op_def_pb2 as _op_def_pb2


_tensor_array_v3_outputs = ["handle", "flow"]
_TensorArrayV3Output = _collections.namedtuple(
    "TensorArrayV3", _tensor_array_v3_outputs)

def tensor_array_v3(size, dtype, element_shape=None, dynamic_size=False, clear_after_read=True, identical_element_shapes=False, tensor_array_name="", name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    dtype = _execute.make_type(dtype, "dtype")
    if element_shape is None:
      element_shape = None
    element_shape = _execute.make_shape(element_shape, "element_shape")
    if dynamic_size is None:
      dynamic_size = False
    dynamic_size = _execute.make_bool(dynamic_size, "dynamic_size")
    if clear_after_read is None:
      clear_after_read = True
    clear_after_read = _execute.make_bool(clear_after_read, "clear_after_read")
    if identical_element_shapes is None:
      identical_element_shapes = False
    identical_element_shapes = _execute.make_bool(identical_element_shapes, "identical_element_shapes")
    if tensor_array_name is None:
      tensor_array_name = ""
    #tensor_array_name = _execute.make_str(tensor_array_name, "tensor_array_name")
    _, _, _op = _op_def_lib._apply_op_helper(
        "TensorArrayV3", size=size, dtype=dtype, element_shape=element_shape,
        dynamic_size=dynamic_size, clear_after_read=clear_after_read,
        identical_element_shapes=identical_element_shapes,
        tensor_array_name=tensor_array_name, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dtype", _op.get_attr("dtype"), "element_shape",
              _op.get_attr("element_shape"), "dynamic_size",
              _op.get_attr("dynamic_size"), "clear_after_read",
              _op.get_attr("clear_after_read"), "identical_element_shapes",
              _op.get_attr("identical_element_shapes"), "tensor_array_name",
              _op.get_attr("tensor_array_name"))
    _execute.record_gradient(
      "TensorArrayV3", _inputs_flat, _attrs, _result, name)
    _result = _TensorArrayV3Output._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "TensorArrayV3", name, _ctx._post_execution_callbacks, size, "dtype",
        dtype, "element_shape", element_shape, "dynamic_size", dynamic_size,
        "clear_after_read", clear_after_read, "identical_element_shapes",
        identical_element_shapes, "tensor_array_name", tensor_array_name)
      _result = _TensorArrayV3Output._make(_result)
      return _result
    except _core._FallbackException:
      return tensor_array_v3_eager_fallback(
          size, dtype=dtype, element_shape=element_shape,
          dynamic_size=dynamic_size, clear_after_read=clear_after_read,
          identical_element_shapes=identical_element_shapes,
          tensor_array_name=tensor_array_name, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def tensor_array_scatter_v3(handle, indices, value, flow_in, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "TensorArrayScatterV3", handle=handle, indices=indices, value=value,
        flow_in=flow_in, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "TensorArrayScatterV3", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "TensorArrayScatterV3", name, _ctx._post_execution_callbacks, handle,
        indices, value, flow_in)
      return _result
    except _core._FallbackException:
      return tensor_array_scatter_v3_eager_fallback(
          handle, indices, value, flow_in, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def tensor_array_read_v3(handle, index, flow_in, dtype, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    dtype = _execute.make_type(dtype, "dtype")
    _, _, _op = _op_def_lib._apply_op_helper(
        "TensorArrayReadV3", handle=handle, index=index, flow_in=flow_in,
        dtype=dtype, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dtype", _op.get_attr("dtype"))
    _execute.record_gradient(
      "TensorArrayReadV3", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "TensorArrayReadV3", name, _ctx._post_execution_callbacks, handle,
        index, flow_in, "dtype", dtype)
      return _result
    except _core._FallbackException:
      return tensor_array_read_v3_eager_fallback(
          handle, index, flow_in, dtype=dtype, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def tensor_array_write_v3(handle, index, value, flow_in, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "TensorArrayWriteV3", handle=handle, index=index, value=value,
        flow_in=flow_in, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "TensorArrayWriteV3", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "TensorArrayWriteV3", name, _ctx._post_execution_callbacks, handle,
        index, value, flow_in)
      return _result
    except _core._FallbackException:
      return tensor_array_write_v3_eager_fallback(
          handle, index, value, flow_in, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def tensor_array_size_v3(handle, flow_in, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "TensorArraySizeV3", handle=handle, flow_in=flow_in, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
    _execute.record_gradient(
      "TensorArraySizeV3", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "TensorArraySizeV3", name, _ctx._post_execution_callbacks, handle,
        flow_in)
      return _result
    except _core._FallbackException:
      return tensor_array_size_v3_eager_fallback(
          handle, flow_in, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def tensor_array_gather_v3(handle, indices, flow_in, dtype, element_shape=None, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    dtype = _execute.make_type(dtype, "dtype")
    if element_shape is None:
      element_shape = None
    element_shape = _execute.make_shape(element_shape, "element_shape")
    _, _, _op = _op_def_lib._apply_op_helper(
        "TensorArrayGatherV3", handle=handle, indices=indices,
        flow_in=flow_in, dtype=dtype, element_shape=element_shape, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dtype", _op.get_attr("dtype"), "element_shape",
              _op.get_attr("element_shape"))
    _execute.record_gradient(
      "TensorArrayGatherV3", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "TensorArrayGatherV3", name, _ctx._post_execution_callbacks, handle,
        indices, flow_in, "dtype", dtype, "element_shape", element_shape)
      return _result
    except _core._FallbackException:
      return tensor_array_gather_v3_eager_fallback(
          handle, indices, flow_in, dtype=dtype, element_shape=element_shape,
          name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


_tensor_array_grad_v3_outputs = ["grad_handle", "flow_out"]
_TensorArrayGradV3Output = _collections.namedtuple(
    "TensorArrayGradV3", _tensor_array_grad_v3_outputs)

def tensor_array_grad_v3(handle, flow_in, source, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    #source = _execute.make_str(source, "source")
    _, _, _op = _op_def_lib._apply_op_helper(
        "TensorArrayGradV3", handle=handle, flow_in=flow_in, source=source,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("source", _op.get_attr("source"))
    _execute.record_gradient(
      "TensorArrayGradV3", _inputs_flat, _attrs, _result, name)
    _result = _TensorArrayGradV3Output._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "TensorArrayGradV3", name, _ctx._post_execution_callbacks, handle,
        flow_in, "source", source)
      _result = _TensorArrayGradV3Output._make(_result)
      return _result
    except _core._FallbackException:
      return tensor_array_grad_v3_eager_fallback(
          handle, flow_in, source=source, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def stack_v2(max_size, elem_type, stack_name="", name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    elem_type = _execute.make_type(elem_type, "elem_type")
    if stack_name is None:
      stack_name = ""
    #stack_name = _execute.make_str(stack_name, "stack_name")
    _, _, _op = _op_def_lib._apply_op_helper(
        "StackV2", max_size=max_size, elem_type=elem_type,
        stack_name=stack_name, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("elem_type", _op.get_attr("elem_type"), "stack_name",
              _op.get_attr("stack_name"))
    _execute.record_gradient(
      "StackV2", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "StackV2",
        name, _ctx._post_execution_callbacks, max_size, "elem_type",
        elem_type, "stack_name", stack_name)
      return _result
    except _core._FallbackException:
      return stack_v2_eager_fallback(
          max_size, elem_type=elem_type, stack_name=stack_name, name=name,
          ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def stack_push_v2(handle, elem, swap_memory=False, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if swap_memory is None:
      swap_memory = False
    swap_memory = _execute.make_bool(swap_memory, "swap_memory")
    _, _, _op = _op_def_lib._apply_op_helper(
        "StackPushV2", handle=handle, elem=elem, swap_memory=swap_memory,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "swap_memory",
              _op.get_attr("swap_memory"))
    _execute.record_gradient(
      "StackPushV2", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "StackPushV2",
        name, _ctx._post_execution_callbacks, handle, elem, "swap_memory",
        swap_memory)
      return _result
    except _core._FallbackException:
      return stack_push_v2_eager_fallback(
          handle, elem, swap_memory=swap_memory, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def stack_pop_v2(handle, elem_type, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    elem_type = _execute.make_type(elem_type, "elem_type")
    _, _, _op = _op_def_lib._apply_op_helper(
        "StackPopV2", handle=handle, elem_type=elem_type, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("elem_type", _op.get_attr("elem_type"))
    _execute.record_gradient(
      "StackPopV2", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "StackPopV2",
        name, _ctx._post_execution_callbacks, handle, "elem_type", elem_type)
      return _result
    except _core._FallbackException:
      return stack_pop_v2_eager_fallback(
          handle, elem_type=elem_type, name=name, ctx=_ctx)
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

_op_def_lib = _InitOpDefLibrary(b"\nr\n\030AccumulatorApplyGradient\022\r\n\006handle\030\007\200\001\001\022\016\n\nlocal_step\030\t\022\021\n\010gradient\"\005dtype\"$\n\005dtype\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\n?\n\031AccumulatorNumAccumulated\022\r\n\006handle\030\007\200\001\001\032\023\n\017num_accumulated\030\003\n>\n\030AccumulatorSetGlobalStep\022\r\n\006handle\030\007\200\001\001\022\023\n\017new_global_step\030\t\nr\n\027AccumulatorTakeGradient\022\r\n\006handle\030\007\200\001\001\022\020\n\014num_required\030\003\032\020\n\007average\"\005dtype\"$\n\005dtype\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\n\255\001\n\007Barrier\032\r\n\006handle\030\007\200\001\001\"!\n\017component_types\022\nlist(type)(\0010\001\"\033\n\006shapes\022\013list(shape)\032\002\n\000(\001\"\034\n\010capacity\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\nB\n\014BarrierClose\022\r\n\006handle\030\007\200\001\001\"#\n\027cancel_pending_enqueues\022\004bool\032\002(\000\n0\n\025BarrierIncompleteSize\022\r\n\006handle\030\007\200\001\001\032\010\n\004size\030\003\n\\\n\021BarrierInsertMany\022\r\n\006handle\030\007\200\001\001\022\010\n\004keys\030\007\022\013\n\006values\"\001T\"\t\n\001T\022\004type\"\026\n\017component_index\022\003int\n+\n\020BarrierReadySize\022\r\n\006handle\030\007\200\001\001\032\010\n\004size\030\003\n\347\001\n\017BarrierTakeMany\022\r\n\006handle\030\007\200\001\001\022\020\n\014num_elements\030\003\032\013\n\007indices\030\t\032\010\n\004keys\030\007\032\031\n\006values2\017component_types\"!\n\017component_types\022\nlist(type)(\0010\001\"\035\n\021allow_small_batch\022\004bool\032\002(\000\"\037\n\023wait_for_incomplete\022\004bool\032\002(\000\"\036\n\ntimeout_ms\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\n\305\001\n\026ConditionalAccumulator\032\r\n\006handle\030\007\200\001\001\"$\n\005dtype\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\016\n\005shape\022\005shape\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\"/\n\016reduction_type\022\006string\032\006\022\004MEAN:\r\n\013\022\004MEAN\022\003SUM\210\001\001\n$\n\023DeleteSessionTensor\022\n\n\006handle\030\007\210\001\001\nq\n\020DynamicPartition\022\t\n\004data\"\001T\022\016\n\npartitions\030\003\032\034\n\007outputs\"\001T*\016num_partitions\"\031\n\016num_partitions\022\003int(\0010\001\"\t\n\001T\022\004type\nS\n\rDynamicStitch\022\016\n\007indices\030\003*\001N\022\014\n\004data\"\001T*\001N\032\013\n\006merged\"\001T\"\014\n\001N\022\003int(\0010\001\"\t\n\001T\022\004type\n\257\001\n\tFIFOQueue\032\r\n\006handle\030\007\200\001\001\"!\n\017component_types\022\nlist(type)(\0010\001\"\033\n\006shapes\022\013list(shape)\032\002\n\000(\001\"\034\n\010capacity\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\256\001\n\013FIFOQueueV2\032\n\n\006handle\030\024\"!\n\017component_types\022\nlist(type)(\0010\001\"\033\n\006shapes\022\013list(shape)\032\002\n\000(\001\"\034\n\010capacity\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n+\n\tFakeQueue\022\014\n\010resource\030\024\032\r\n\006handle\030\007\200\001\001\210\001\001\n8\n\020GetSessionHandle\022\n\n\005value\"\001T\032\n\n\006handle\030\007\"\t\n\001T\022\004type\210\001\001\n:\n\022GetSessionHandleV2\022\n\n\005value\"\001T\032\n\n\006handle\030\024\"\t\n\001T\022\004type\210\001\001\n@\n\020GetSessionTensor\022\n\n\006handle\030\007\032\016\n\005value\"\005dtype\"\r\n\005dtype\022\004type\210\001\001\n\211\001\n\010MapClear\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\024\n\006dtypes\022\nlist(type)\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\234\001\n\021MapIncompleteSize\032\010\n\004size\030\003\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\024\n\006dtypes\022\nlist(type)\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\264\001\n\007MapPeek\022\007\n\003key\030\t\022\013\n\007indices\030\003\032\020\n\006values2\006dtypes\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\222\001\n\007MapSize\032\010\n\004size\030\003\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\024\n\006dtypes\022\nlist(type)\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\325\001\n\010MapStage\022\007\n\003key\030\t\022\013\n\007indices\030\003\022\025\n\006values2\013fake_dtypes\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\024\n\006dtypes\022\nlist(type)\"\035\n\013fake_dtypes\022\nlist(type)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\267\001\n\nMapUnstage\022\007\n\003key\030\t\022\013\n\007indices\030\003\032\020\n\006values2\006dtypes\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\274\001\n\017MapUnstageNoKey\022\013\n\007indices\030\003\032\007\n\003key\030\t\032\020\n\006values2\006dtypes\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\220\001\n\017OrderedMapClear\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\024\n\006dtypes\022\nlist(type)\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\243\001\n\030OrderedMapIncompleteSize\032\010\n\004size\030\003\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\024\n\006dtypes\022\nlist(type)\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\273\001\n\016OrderedMapPeek\022\007\n\003key\030\t\022\013\n\007indices\030\003\032\020\n\006values2\006dtypes\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\231\001\n\016OrderedMapSize\032\010\n\004size\030\003\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\024\n\006dtypes\022\nlist(type)\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\334\001\n\017OrderedMapStage\022\007\n\003key\030\t\022\013\n\007indices\030\003\022\025\n\006values2\013fake_dtypes\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\024\n\006dtypes\022\nlist(type)\"\035\n\013fake_dtypes\022\nlist(type)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\276\001\n\021OrderedMapUnstage\022\007\n\003key\030\t\022\013\n\007indices\030\003\032\020\n\006values2\006dtypes\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\303\001\n\026OrderedMapUnstageNoKey\022\013\n\007indices\030\003\032\007\n\003key\030\t\032\020\n\006values2\006dtypes\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\266\001\n\020PaddingFIFOQueue\032\r\n\006handle\030\007\200\001\001\"!\n\017component_types\022\nlist(type)(\0010\001\"\033\n\006shapes\022\013list(shape)\032\002\n\000(\001\"\034\n\010capacity\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\265\001\n\022PaddingFIFOQueueV2\032\n\n\006handle\030\024\"!\n\017component_types\022\nlist(type)(\0010\001\"\033\n\006shapes\022\013list(shape)\032\002\n\000(\001\"\034\n\010capacity\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n[\n\025ParallelDynamicStitch\022\016\n\007indices\030\003*\001N\022\014\n\004data\"\001T*\001N\032\013\n\006merged\"\001T\"\014\n\001N\022\003int(\0010\001\"\t\n\001T\022\004type\n\261\001\n\rPriorityQueue\032\r\n\006handle\030\007\200\001\001\"#\n\017component_types\022\nlist(type)\032\002\n\000(\001\"\027\n\006shapes\022\013list(shape)(\001\"\034\n\010capacity\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\260\001\n\017PriorityQueueV2\032\n\n\006handle\030\024\"#\n\017component_types\022\nlist(type)\032\002\n\000(\001\"\027\n\006shapes\022\013list(shape)(\001\"\034\n\010capacity\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n@\n\nQueueClose\022\r\n\006handle\030\007\200\001\001\"#\n\027cancel_pending_enqueues\022\004bool\032\002(\000\nB\n\014QueueCloseV2\022\n\n\006handle\030\024\"#\n\027cancel_pending_enqueues\022\004bool\032\002(\000\210\001\001\n\177\n\014QueueDequeue\022\r\n\006handle\030\007\200\001\001\032\035\n\ncomponents2\017component_types\"!\n\017component_types\022\nlist(type)(\0010\001\"\036\n\ntimeout_ms\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\n\212\001\n\020QueueDequeueMany\022\r\n\006handle\030\007\200\001\001\022\005\n\001n\030\003\032\035\n\ncomponents2\017component_types\"!\n\017component_types\022\nlist(type)(\0010\001\"\036\n\ntimeout_ms\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\n\214\001\n\022QueueDequeueManyV2\022\n\n\006handle\030\024\022\005\n\001n\030\003\032\035\n\ncomponents2\017component_types\"!\n\017component_types\022\nlist(type)(\0010\001\"\036\n\ntimeout_ms\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\n\212\001\n\020QueueDequeueUpTo\022\r\n\006handle\030\007\200\001\001\022\005\n\001n\030\003\032\035\n\ncomponents2\017component_types\"!\n\017component_types\022\nlist(type)(\0010\001\"\036\n\ntimeout_ms\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\n\214\001\n\022QueueDequeueUpToV2\022\n\n\006handle\030\024\022\005\n\001n\030\003\032\035\n\ncomponents2\017component_types\"!\n\017component_types\022\nlist(type)(\0010\001\"\036\n\ntimeout_ms\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\n\201\001\n\016QueueDequeueV2\022\n\n\006handle\030\024\032\035\n\ncomponents2\017component_types\"!\n\017component_types\022\nlist(type)(\0010\001\"\036\n\ntimeout_ms\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\nw\n\014QueueEnqueue\022\r\n\006handle\030\007\200\001\001\022\031\n\ncomponents2\013Tcomponents\"\035\n\013Tcomponents\022\nlist(type)(\0010\001\"\036\n\ntimeout_ms\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\n{\n\020QueueEnqueueMany\022\r\n\006handle\030\007\200\001\001\022\031\n\ncomponents2\013Tcomponents\"\035\n\013Tcomponents\022\nlist(type)(\0010\001\"\036\n\ntimeout_ms\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\n}\n\022QueueEnqueueManyV2\022\n\n\006handle\030\024\022\031\n\ncomponents2\013Tcomponents\"\035\n\013Tcomponents\022\nlist(type)(\0010\001\"\036\n\ntimeout_ms\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\ny\n\016QueueEnqueueV2\022\n\n\006handle\030\024\022\031\n\ncomponents2\013Tcomponents\"\035\n\013Tcomponents\022\nlist(type)(\0010\001\"\036\n\ntimeout_ms\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\n-\n\rQueueIsClosed\022\r\n\006handle\030\007\200\001\001\032\r\n\tis_closed\030\n\n/\n\017QueueIsClosedV2\022\n\n\006handle\030\024\032\r\n\tis_closed\030\n\210\001\001\n$\n\tQueueSize\022\r\n\006handle\030\007\200\001\001\032\010\n\004size\030\003\n&\n\013QueueSizeV2\022\n\n\006handle\030\024\032\010\n\004size\030\003\210\001\001\n\371\001\n\022RandomShuffleQueue\032\r\n\006handle\030\007\200\001\001\"!\n\017component_types\022\nlist(type)(\0010\001\"\033\n\006shapes\022\013list(shape)\032\002\n\000(\001\"\034\n\010capacity\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\034\n\021min_after_dequeue\022\003int\032\002\030\000\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\370\001\n\024RandomShuffleQueueV2\032\n\n\006handle\030\024\"!\n\017component_types\022\nlist(type)(\0010\001\"\033\n\006shapes\022\013list(shape)\032\002\n\000(\001\"\034\n\010capacity\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\034\n\021min_after_dequeue\022\003int\032\002\030\000\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\357\001\n\013RecordInput\032\013\n\007records\030\007\"\026\n\014file_pattern\022\006string\"\034\n\020file_random_seed\022\003int\032\003\030\255\002\"(\n\030file_shuffle_shift_ratio\022\005float\032\005%\000\000\000\000\"\034\n\020file_buffer_size\022\003int\032\003\030\220N\"\033\n\020file_parallelism\022\003int\032\002\030\020\"\025\n\nbatch_size\022\003int\032\002\030 \"\036\n\020compression_type\022\006string\032\002\022\000\210\001\001\n\302\001\n\036SparseAccumulatorApplyGradient\022\r\n\006handle\030\007\200\001\001\022\016\n\nlocal_step\030\t\022\024\n\020gradient_indices\030\t\022\030\n\017gradient_values\"\005dtype\022\022\n\016gradient_shape\030\t\"$\n\005dtype\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\027\n\017has_known_shape\022\004bool\n\217\001\n\035SparseAccumulatorTakeGradient\022\r\n\006handle\030\007\200\001\001\022\020\n\014num_required\030\003\032\013\n\007indices\030\t\032\017\n\006values\"\005dtype\032\t\n\005shape\030\t\"$\n\005dtype\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\n\313\001\n\034SparseConditionalAccumulator\032\r\n\006handle\030\007\200\001\001\"$\n\005dtype\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\016\n\005shape\022\005shape\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\"/\n\016reduction_type\022\006string\032\006\022\004MEAN:\r\n\013\022\004MEAN\022\003SUM\210\001\001\nF\n\005Stack\032\r\n\006handle\030\007\200\001\001\"\021\n\telem_type\022\004type\"\030\n\nstack_name\022\006string\032\002\022\000\210\001\001\n\033\n\nStackClose\022\r\n\006handle\030\007\200\001\001\n\035\n\014StackCloseV2\022\n\n\006handle\030\024\210\001\001\n?\n\010StackPop\022\r\n\006handle\030\007\200\001\001\032\021\n\004elem\"\telem_type\"\021\n\telem_type\022\004type\nA\n\nStackPopV2\022\n\n\006handle\030\024\032\021\n\004elem\"\telem_type\"\021\n\telem_type\022\004type\210\001\001\nV\n\tStackPush\022\r\n\006handle\030\007\200\001\001\022\t\n\004elem\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\027\n\013swap_memory\022\004bool\032\002(\000\nX\n\013StackPushV2\022\n\n\006handle\030\024\022\t\n\004elem\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\027\n\013swap_memory\022\004bool\032\002(\000\210\001\001\nS\n\007StackV2\022\014\n\010max_size\030\003\032\n\n\006handle\030\024\"\021\n\telem_type\022\004type\"\030\n\nstack_name\022\006string\032\002\022\000\210\001\001\n\234\001\n\005Stage\022\020\n\006values2\006dtypes\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\213\001\n\nStageClear\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\024\n\006dtypes\022\nlist(type)\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\253\001\n\tStagePeek\022\t\n\005index\030\003\032\020\n\006values2\006dtypes\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\224\001\n\tStageSize\032\010\n\004size\030\003\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\024\n\006dtypes\022\nlist(type)\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\306\001\n\013TensorArray\022\010\n\004size\030\003\032\r\n\006handle\030\007\200\001\001\"\r\n\005dtype\022\004type\"\030\n\014dynamic_size\022\004bool\032\002(\000\"\034\n\020clear_after_read\022\004bool\032\002(\001\"\037\n\021tensor_array_name\022\006string\032\002\022\000\"\034\n\relement_shape\022\005shape\032\004:\002\030\001B\025\010\020\022\021Use TensorArrayV3\210\001\001\n=\n\020TensorArrayClose\022\r\n\006handle\030\007\200\001\001B\032\010\020\022\026Use TensorArrayCloseV3\n<\n\022TensorArrayCloseV2\022\n\n\006handle\030\007B\032\010\032\022\026Use TensorArrayCloseV3\n#\n\022TensorArrayCloseV3\022\n\n\006handle\030\024\210\001\001\n\234\001\n\021TensorArrayConcat\022\r\n\006handle\030\007\200\001\001\022\013\n\007flow_in\030\001\032\016\n\005value\"\005dtype\032\013\n\007lengths\030\t\"\r\n\005dtype\022\004type\"$\n\025element_shape_except0\022\005shape\032\004:\002\030\001B\031\010\020\022\025Use TensorArrayGradV3\n\200\001\n\023TensorArrayConcatV2\022\n\n\006handle\030\007\022\013\n\007flow_in\030\001\032\016\n\005value\"\005dtype\032\013\n\007lengths\030\t\"\r\n\005dtype\022\004type\"$\n\025element_shape_except0\022\005shape\032\004:\002\030\001\n\203\001\n\023TensorArrayConcatV3\022\n\n\006handle\030\024\022\013\n\007flow_in\030\001\032\016\n\005value\"\005dtype\032\013\n\007lengths\030\t\"\r\n\005dtype\022\004type\"$\n\025element_shape_except0\022\005shape\032\004:\002\030\001\210\001\001\n\226\001\n\021TensorArrayGather\022\r\n\006handle\030\007\200\001\001\022\013\n\007indices\030\003\022\013\n\007flow_in\030\001\032\016\n\005value\"\005dtype\"\r\n\005dtype\022\004type\"\034\n\relement_shape\022\005shape\032\004:\002\030\001B\033\010\020\022\027Use TensorArrayGatherV3\n\225\001\n\023TensorArrayGatherV2\022\n\n\006handle\030\007\022\013\n\007indices\030\003\022\013\n\007flow_in\030\001\032\016\n\005value\"\005dtype\"\r\n\005dtype\022\004type\"\034\n\relement_shape\022\005shape\032\004:\002\030\001B\033\010\032\022\027Use TensorArrayGatherV3\n{\n\023TensorArrayGatherV3\022\n\n\006handle\030\024\022\013\n\007indices\030\003\022\013\n\007flow_in\030\001\032\016\n\005value\"\005dtype\"\r\n\005dtype\022\004type\"\034\n\relement_shape\022\005shape\032\004:\002\030\001\210\001\001\nn\n\017TensorArrayGrad\022\n\n\006handle\030\007\022\013\n\007flow_in\030\001\032\022\n\013grad_handle\030\007\200\001\001\"\020\n\006source\022\006stringB\031\010\020\022\025Use TensorArrayGradV3\210\001\001\nm\n\021TensorArrayGradV2\022\n\n\006handle\030\007\022\013\n\007flow_in\030\001\032\017\n\013grad_handle\030\007\"\020\n\006source\022\006stringB\031\010\032\022\025Use TensorArrayGradV3\210\001\001\n`\n\021TensorArrayGradV3\022\n\n\006handle\030\024\022\013\n\007flow_in\030\001\032\017\n\013grad_handle\030\024\032\014\n\010flow_out\030\001\"\020\n\006source\022\006string\210\001\001\n}\n\030TensorArrayGradWithShape\022\n\n\006handle\030\024\022\013\n\007flow_in\030\001\022\024\n\020shape_to_prepend\030\003\032\017\n\013grad_handle\030\024\032\014\n\010flow_out\030\001\"\020\n\006source\022\006string\210\001\001\n\224\001\n\017TensorArrayPack\022\r\n\006handle\030\007\200\001\001\022\013\n\007flow_in\030\001\032\016\n\005value\"\005dtype\"\r\n\005dtype\022\004type\"\034\n\relement_shape\022\005shape\032\004:\002\030\001B(\010\020\022$Use TensorArrayGatherV3 with RangeOp\nr\n\017TensorArrayRead\022\r\n\006handle\030\007\200\001\001\022\t\n\005index\030\003\022\013\n\007flow_in\030\001\032\016\n\005value\"\005dtype\"\r\n\005dtype\022\004typeB\031\010\020\022\025Use TensorArrayReadV3\nq\n\021TensorArrayReadV2\022\n\n\006handle\030\007\022\t\n\005index\030\003\022\013\n\007flow_in\030\001\032\016\n\005value\"\005dtype\"\r\n\005dtype\022\004typeB\031\010\032\022\025Use TensorArrayReadV3\nY\n\021TensorArrayReadV3\022\n\n\006handle\030\024\022\t\n\005index\030\003\022\013\n\007flow_in\030\001\032\016\n\005value\"\005dtype\"\r\n\005dtype\022\004type\210\001\001\n}\n\022TensorArrayScatter\022\r\n\006handle\030\007\200\001\001\022\013\n\007indices\030\003\022\n\n\005value\"\001T\022\013\n\007flow_in\030\001\032\014\n\010flow_out\030\001\"\t\n\001T\022\004typeB\031\010\023\022\025Use TensorArrayGradV3\n\177\n\024TensorArrayScatterV2\022\n\n\006handle\030\007\022\013\n\007indices\030\003\022\n\n\005value\"\001T\022\013\n\007flow_in\030\001\032\014\n\010flow_out\030\001\"\t\n\001T\022\004typeB\034\010\032\022\030Use TensorArrayScatterV3\nd\n\024TensorArrayScatterV3\022\n\n\006handle\030\024\022\013\n\007indices\030\003\022\n\n\005value\"\001T\022\013\n\007flow_in\030\001\032\014\n\010flow_out\030\001\"\t\n\001T\022\004type\210\001\001\nR\n\017TensorArraySize\022\r\n\006handle\030\007\200\001\001\022\013\n\007flow_in\030\001\032\010\n\004size\030\003B\031\010\020\022\025Use TensorArraySizeV3\nQ\n\021TensorArraySizeV2\022\n\n\006handle\030\007\022\013\n\007flow_in\030\001\032\010\n\004size\030\003B\031\010\032\022\025Use TensorArraySizeV3\n9\n\021TensorArraySizeV3\022\n\n\006handle\030\024\022\013\n\007flow_in\030\001\032\010\n\004size\030\003\210\001\001\n|\n\020TensorArraySplit\022\r\n\006handle\030\007\200\001\001\022\n\n\005value\"\001T\022\013\n\007lengths\030\t\022\013\n\007flow_in\030\001\032\014\n\010flow_out\030\001\"\t\n\001T\022\004typeB\032\010\020\022\026Use TensorArraySplitV3\n{\n\022TensorArraySplitV2\022\n\n\006handle\030\007\022\n\n\005value\"\001T\022\013\n\007lengths\030\t\022\013\n\007flow_in\030\001\032\014\n\010flow_out\030\001\"\t\n\001T\022\004typeB\032\010\032\022\026Use TensorArraySplitV3\nb\n\022TensorArraySplitV3\022\n\n\006handle\030\024\022\n\n\005value\"\001T\022\013\n\007lengths\030\t\022\013\n\007flow_in\030\001\032\014\n\010flow_out\030\001\"\t\n\001T\022\004type\210\001\001\n\177\n\021TensorArrayUnpack\022\r\n\006handle\030\007\200\001\001\022\n\n\005value\"\001T\022\013\n\007flow_in\030\001\032\014\n\010flow_out\030\001\"\t\n\001T\022\004typeB)\010\024\022%Use TensorArrayScatterV3 with RangeOp\n\305\001\n\rTensorArrayV2\022\010\n\004size\030\003\032\n\n\006handle\030\007\"\r\n\005dtype\022\004type\"\034\n\relement_shape\022\005shape\032\004:\002\030\001\"\030\n\014dynamic_size\022\004bool\032\002(\000\"\034\n\020clear_after_read\022\004bool\032\002(\001\"\037\n\021tensor_array_name\022\006string\032\002\022\000B\025\010\032\022\021Use TensorArrayV3\210\001\001\n\336\001\n\rTensorArrayV3\022\010\n\004size\030\003\032\n\n\006handle\030\024\032\010\n\004flow\030\001\"\r\n\005dtype\022\004type\"\034\n\relement_shape\022\005shape\032\004:\002\030\001\"\030\n\014dynamic_size\022\004bool\032\002(\000\"\034\n\020clear_after_read\022\004bool\032\002(\001\"$\n\030identical_element_shapes\022\004bool\032\002(\000\"\037\n\021tensor_array_name\022\006string\032\002\022\000\210\001\001\nz\n\020TensorArrayWrite\022\r\n\006handle\030\007\200\001\001\022\t\n\005index\030\003\022\n\n\005value\"\001T\022\013\n\007flow_in\030\001\032\014\n\010flow_out\030\001\"\t\n\001T\022\004typeB\032\010\020\022\026Use TensorArrayWriteV3\ny\n\022TensorArrayWriteV2\022\n\n\006handle\030\007\022\t\n\005index\030\003\022\n\n\005value\"\001T\022\013\n\007flow_in\030\001\032\014\n\010flow_out\030\001\"\t\n\001T\022\004typeB\032\010\032\022\026Use TensorArrayWriteV3\n`\n\022TensorArrayWriteV3\022\n\n\006handle\030\024\022\t\n\005index\030\003\022\n\n\005value\"\001T\022\013\n\007flow_in\030\001\032\014\n\010flow_out\030\001\"\t\n\001T\022\004type\210\001\001\n\236\001\n\007Unstage\032\020\n\006values2\006dtypes\"\025\n\010capacity\022\003int\032\002\030\000(\001\"\031\n\014memory_limit\022\003int\032\002\030\000(\001\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001")
