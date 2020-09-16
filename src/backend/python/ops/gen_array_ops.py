import collections as _collections
import six as _six

from backend.python import pywrap_backend as _pywrap_tensorflow
from backend.python import context as _context
from backend.python import execute as _execute

from backend.core import op_def_pb2 as _op_def_pb2
from backend.python.framework import op_def_library as _op_def_library
from backend.python.framework import op_def_registry as _op_def_registry
from backend.python.framework import dtypes as _dtypes

def reverse_v2(tensor, axis, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "ReverseV2", tensor=tensor, axis=axis, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("Tidx", _op.get_attr("Tidx"), "T", _op.get_attr("T"))
    _execute.record_gradient(
      "ReverseV2", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "ReverseV2",
        name, _ctx._post_execution_callbacks, tensor, axis)
      return _result
    except _core._FallbackException:
      return reverse_v2_eager_fallback(
          tensor, axis, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def placeholder(dtype, shape=None, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    dtype = _execute.make_type(dtype, "dtype")
    if shape is None:
      shape = None
    shape = _execute.make_shape(shape, "shape")
    _, _, _op = _op_def_lib._apply_op_helper(
        "Placeholder", dtype=dtype, shape=shape, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dtype", _op.get_attr("dtype"), "shape", _op.get_attr("shape"))
    _execute.record_gradient(
      "Placeholder", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Placeholder",
        name, _ctx._post_execution_callbacks, "dtype", dtype, "shape", shape)
      return _result
    except _core._FallbackException:
      return placeholder_eager_fallback(
          dtype=dtype, shape=shape, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def expand_dims(input, axis, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "ExpandDims", input=input, dim=axis, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "Tdim", _op.get_attr("Tdim"))
    _execute.record_gradient(
      "ExpandDims", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "ExpandDims",
        name, _ctx._post_execution_callbacks, input, axis)
      return _result
    except _core._FallbackException:
      return expand_dims_eager_fallback(
          input, axis, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def squeeze(input, axis=[], name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if axis is None:
      axis = []
    if not isinstance(axis, (list, tuple)):
      raise TypeError(
          "Expected list for 'axis' argument to "
          "'squeeze' Op, not %r." % axis)
    axis = [_execute.make_int(_i, "axis") for _i in axis]
    _, _, _op = _op_def_lib._apply_op_helper(
        "Squeeze", input=input, squeeze_dims=axis, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "squeeze_dims",
              _op.get_attr("squeeze_dims"))
    _execute.record_gradient(
      "Squeeze", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Squeeze",
        name, _ctx._post_execution_callbacks, input, "squeeze_dims", axis)
      return _result
    except _core._FallbackException:
      return squeeze_eager_fallback(
          input, axis=axis, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def transpose(x, perm, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Transpose", x=x, perm=perm, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "Tperm", _op.get_attr("Tperm"))
    _execute.record_gradient(
      "Transpose", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Transpose",
        name, _ctx._post_execution_callbacks, x, perm)
      return _result
    except _core._FallbackException:
      return transpose_eager_fallback(
          x, perm, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def identity(input, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Identity", input=input, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "Identity", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Identity",
        name, _ctx._post_execution_callbacks, input)
      return _result
    except _core._FallbackException:
      return identity_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def rank(input, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Rank", input=input, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "Rank", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Rank", name,
        _ctx._post_execution_callbacks, input)
      return _result
    except _core._FallbackException:
      return rank_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def concat_v2(values, axis, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(values, (list, tuple)):
      raise TypeError(
          "Expected list for 'values' argument to "
          "'concat_v2' Op, not %r." % values)
    _attr_N = len(values)
    _, _, _op = _op_def_lib._apply_op_helper(
        "ConcatV2", values=values, axis=axis, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("N", _op.get_attr("N"), "T", _op.get_attr("T"), "Tidx",
              _op.get_attr("Tidx"))
    _execute.record_gradient(
      "ConcatV2", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "ConcatV2",
        name, _ctx._post_execution_callbacks, values, axis)
      return _result
    except _core._FallbackException:
      return concat_v2_eager_fallback(
          values, axis, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def shape(input, out_type=_dtypes.int32, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if out_type is None:
      out_type = _dtypes.int32
    out_type = _execute.make_type(out_type, "out_type")
    _, _, _op = _op_def_lib._apply_op_helper(
        "Shape", input=input, out_type=out_type, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "out_type", _op.get_attr("out_type"))
    _execute.record_gradient(
      "Shape", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Shape", name,
        _ctx._post_execution_callbacks, input, "out_type", out_type)
      return _result
    except _core._FallbackException:
      return shape_eager_fallback(
          input, out_type=out_type, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def strided_slice(input, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if begin_mask is None:
      begin_mask = 0
    begin_mask = _execute.make_int(begin_mask, "begin_mask")
    if end_mask is None:
      end_mask = 0
    end_mask = _execute.make_int(end_mask, "end_mask")
    if ellipsis_mask is None:
      ellipsis_mask = 0
    ellipsis_mask = _execute.make_int(ellipsis_mask, "ellipsis_mask")
    if new_axis_mask is None:
      new_axis_mask = 0
    new_axis_mask = _execute.make_int(new_axis_mask, "new_axis_mask")
    if shrink_axis_mask is None:
      shrink_axis_mask = 0
    shrink_axis_mask = _execute.make_int(shrink_axis_mask, "shrink_axis_mask")
    _, _, _op = _op_def_lib._apply_op_helper(
        "StridedSlice", input=input, begin=begin, end=end, strides=strides,
        begin_mask=begin_mask, end_mask=end_mask, ellipsis_mask=ellipsis_mask,
        new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "Index", _op.get_attr("Index"),
              "begin_mask", _op.get_attr("begin_mask"), "end_mask",
              _op.get_attr("end_mask"), "ellipsis_mask",
              _op.get_attr("ellipsis_mask"), "new_axis_mask",
              _op.get_attr("new_axis_mask"), "shrink_axis_mask",
              _op.get_attr("shrink_axis_mask"))
    _execute.record_gradient(
      "StridedSlice", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "StridedSlice",
        name, _ctx._post_execution_callbacks, input, begin, end, strides,
        "begin_mask", begin_mask, "end_mask", end_mask, "ellipsis_mask",
        ellipsis_mask, "new_axis_mask", new_axis_mask, "shrink_axis_mask",
        shrink_axis_mask)
      return _result
    except _core._FallbackException:
      return strided_slice_eager_fallback(
          input, begin, end, strides, begin_mask=begin_mask,
          end_mask=end_mask, ellipsis_mask=ellipsis_mask,
          new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask,
          name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def fill(dims, value, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Fill", dims=dims, value=value, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "index_type",
              _op.get_attr("index_type"))
    _execute.record_gradient(
      "Fill", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Fill", name,
        _ctx._post_execution_callbacks, dims, value)
      return _result
    except _core._FallbackException:
      return fill_eager_fallback(
          dims, value, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def split(axis, value, num_split, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    num_split = _execute.make_int(num_split, "num_split")
    _, _, _op = _op_def_lib._apply_op_helper(
        "Split", split_dim=axis, value=value, num_split=num_split, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("num_split", _op.get_attr("num_split"), "T", _op.get_attr("T"))
    _execute.record_gradient(
      "Split", _inputs_flat, _attrs, _result, name)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Split", name,
        _ctx._post_execution_callbacks, axis, value, "num_split", num_split)
      return _result
    except _core._FallbackException:
      return split_eager_fallback(
          axis, value, num_split=num_split, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def reshape(tensor, shape, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Reshape", tensor=tensor, shape=shape, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "Tshape", _op.get_attr("Tshape"))
    _execute.record_gradient(
      "Reshape", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Reshape",
        name, _ctx._post_execution_callbacks, tensor, shape)
      return _result
    except _core._FallbackException:
      return reshape_eager_fallback(
          tensor, shape, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def tile(input, multiples, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Tile", input=input, multiples=multiples, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "Tmultiples",
              _op.get_attr("Tmultiples"))
    _execute.record_gradient(
      "Tile", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Tile", name,
        _ctx._post_execution_callbacks, input, multiples)
      return _result
    except _core._FallbackException:
      return tile_eager_fallback(
          input, multiples, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def zeros_like(x, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "ZerosLike", x=x, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "ZerosLike", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "ZerosLike",
        name, _ctx._post_execution_callbacks, x)
      return _result
    except _core._FallbackException:
      return zeros_like_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def prevent_gradient(input, message="", name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if message is None:
      message = ""
    message = _execute.make_str(message, "message")
    _, _, _op = _op_def_lib._apply_op_helper(
        "PreventGradient", input=input, message=message, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "message", _op.get_attr("message"))
    _execute.record_gradient(
      "PreventGradient", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "PreventGradient", name, _ctx._post_execution_callbacks, input,
        "message", message)
      return _result
    except _core._FallbackException:
      return prevent_gradient_eager_fallback(
          input, message=message, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def invert_permutation(x, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "InvertPermutation", x=x, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "InvertPermutation", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "InvertPermutation", name, _ctx._post_execution_callbacks, x)
      return _result
    except _core._FallbackException:
      return invert_permutation_eager_fallback(
          x, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def shape_n(input, out_type=_dtypes.int32, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(input, (list, tuple)):
      raise TypeError(
          "Expected list for 'input' argument to "
          "'shape_n' Op, not %r." % input)
    _attr_N = len(input)
    if out_type is None:
      out_type = _dtypes.int32
    out_type = _execute.make_type(out_type, "out_type")
    _, _, _op = _op_def_lib._apply_op_helper(
        "ShapeN", input=input, out_type=out_type, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("N", _op.get_attr("N"), "T", _op.get_attr("T"), "out_type",
              _op.get_attr("out_type"))
    _execute.record_gradient(
      "ShapeN", _inputs_flat, _attrs, _result, name)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "ShapeN", name,
        _ctx._post_execution_callbacks, input, "out_type", out_type)
      return _result
    except _core._FallbackException:
      return shape_n_eager_fallback(
          input, out_type=out_type, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def concat_offset(concat_dim, shape, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(shape, (list, tuple)):
      raise TypeError(
          "Expected list for 'shape' argument to "
          "'concat_offset' Op, not %r." % shape)
    _attr_N = len(shape)
    _, _, _op = _op_def_lib._apply_op_helper(
        "ConcatOffset", concat_dim=concat_dim, shape=shape, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("N", _op.get_attr("N"))
    _execute.record_gradient(
      "ConcatOffset", _inputs_flat, _attrs, _result, name)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "ConcatOffset",
        name, _ctx._post_execution_callbacks, concat_dim, shape)
      return _result
    except _core._FallbackException:
      return concat_offset_eager_fallback(
          concat_dim, shape, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def _slice(input, begin, size, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Slice", input=input, begin=begin, size=size, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "Index", _op.get_attr("Index"))
    _execute.record_gradient(
      "Slice", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Slice", name,
        _ctx._post_execution_callbacks, input, begin, size)
      return _result
    except _core._FallbackException:
      return _slice_eager_fallback(
          input, begin, size, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def broadcast_gradient_args(s0, s1, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "BroadcastGradientArgs", s0=s0, s1=s1, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "BroadcastGradientArgs", _inputs_flat, _attrs, _result, name)
    _result = _BroadcastGradientArgsOutput._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "BroadcastGradientArgs", name, _ctx._post_execution_callbacks, s0, s1)
      _result = _BroadcastGradientArgsOutput._make(_result)
      return _result
    except _core._FallbackException:
      return broadcast_gradient_args_eager_fallback(
          s0, s1, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

_broadcast_gradient_args_outputs = ["r0", "r1"]
_BroadcastGradientArgsOutput = _collections.namedtuple(
    "BroadcastGradientArgs", _broadcast_gradient_args_outputs)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib

_op_def_lib = _InitOpDefLibrary(b"\nm\n\023BatchMatrixBandPart\022\n\n\005input\"\001T\022\r\n\tnum_lower\030\t\022\r\n\tnum_upper\030\t\032\t\n\004band\"\001T\"\t\n\001T\022\004typeB\026\010\016\022\022Use MatrixBandPart\nL\n\017BatchMatrixDiag\022\r\n\010diagonal\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004typeB\022\010\016\022\016Use MatrixDiag\nS\n\023BatchMatrixDiagPart\022\n\n\005input\"\001T\032\r\n\010diagonal\"\001T\"\t\n\001T\022\004typeB\026\010\016\022\022Use MatrixDiagPart\n^\n\022BatchMatrixSetDiag\022\n\n\005input\"\001T\022\r\n\010diagonal\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004typeB\025\010\016\022\021Use MatrixSetDiag\nr\n\014BatchToSpace\022\n\n\005input\"\001T\022\r\n\005crops\"\004Tidx\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\025\n\nblock_size\022\003int(\0010\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\240\001\n\016BatchToSpaceND\022\n\n\005input\"\001T\022\033\n\013block_shape\"\014Tblock_shape\022\017\n\005crops\"\006Tcrops\032\013\n\006output\"\001T\"\t\n\001T\022\004type\" \n\014Tblock_shape\022\004type\032\0020\003:\006\n\0042\002\003\t\"\032\n\006Tcrops\022\004type\032\0020\003:\006\n\0042\002\003\t\np\n\007Bitcast\022\n\n\005input\"\001T\032\016\n\006output\"\004type\"\"\n\001T\022\004type:\027\n\0252\023\016\023\001\002\t\003\004\021\026\027\006\005\010\022\013\014\017\020\r\"%\n\004type\022\004type:\027\n\0252\023\016\023\001\002\t\003\004\021\026\027\006\005\010\022\013\014\017\020\r\nA\n\rBroadcastArgs\022\007\n\002s0\"\001T\022\007\n\002s1\"\001T\032\007\n\002r0\"\001T\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\nR\n\025BroadcastGradientArgs\022\007\n\002s0\"\001T\022\007\n\002s1\"\001T\032\007\n\002r0\"\001T\032\007\n\002r1\"\001T\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\nZ\n\013BroadcastTo\022\n\n\005input\"\001T\022\r\n\005shape\"\004Tidx\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\nQ\n\rCheckNumerics\022\013\n\006tensor\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\"\021\n\007message\022\006string\nN\n\006Concat\022\016\n\nconcat_dim\030\003\022\016\n\006values\"\001T*\001N\032\013\n\006output\"\001T\"\014\n\001N\022\003int(\0010\002\"\t\n\001T\022\004type\nI\n\014ConcatOffset\022\016\n\nconcat_dim\030\003\022\014\n\005shape\030\003*\001N\032\r\n\006offset\030\003*\001N\"\014\n\001N\022\003int(\0010\002\nh\n\010ConcatV2\022\016\n\006values\"\001T*\001N\022\014\n\004axis\"\004Tidx\032\013\n\006output\"\001T\"\014\n\001N\022\003int(\0010\002\"\t\n\001T\022\004type\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\nY\n\022ConjugateTranspose\022\006\n\001x\"\001T\022\r\n\004perm\"\005Tperm\032\006\n\001y\"\001T\"\t\n\001T\022\004type\"\031\n\005Tperm\022\004type\032\0020\003:\006\n\0042\002\003\t\n8\n\005Const\032\017\n\006output\"\005dtype\"\017\n\005value\022\006tensor\"\r\n\005dtype\022\004type\n>\n\025DebugGradientIdentity\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\230\001\001\nG\n\030DebugGradientRefIdentity\022\r\n\005input\"\001T\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\230\001\001\n(\n\010DeepCopy\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\t\n\001T\022\004type\210\001\001\n\205\001\n\014DepthToSpace\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\025\n\nblock_size\022\003int(\0010\002\":\n\013data_format\022\006string\032\006\022\004NHWC:\033\n\031\022\004NHWC\022\004NCHW\022\013NCHW_VECT_C\n\235\001\n\nDequantize\022\n\n\005input\"\001T\022\r\n\tmin_range\030\001\022\r\n\tmax_range\030\001\032\n\n\006output\030\001\"\024\n\001T\022\004type:\t\n\0072\005\013\014\r\017\020\"C\n\004mode\022\006string\032\016\022\014MIN_COMBINED:#\n!\022\014MIN_COMBINED\022\tMIN_FIRST\022\006SCALED\n;\n\004Diag\022\r\n\010diagonal\"\001T\032\013\n\006output\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n>\n\010DiagPart\022\n\n\005input\"\001T\032\r\n\010diagonal\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n\271\001\n\014EditDistance\022\026\n\022hypothesis_indices\030\t\022\026\n\021hypothesis_values\"\001T\022\024\n\020hypothesis_shape\030\t\022\021\n\rtruth_indices\030\t\022\021\n\014truth_values\"\001T\022\017\n\013truth_shape\030\t\032\n\n\006output\030\001\"\025\n\tnormalize\022\004bool\032\002(\001\"\t\n\001T\022\004type\nG\n\005Empty\022\t\n\005shape\030\003\032\017\n\006output\"\005dtype\"\r\n\005dtype\022\004type\"\020\n\004init\022\004bool\032\002(\000\210\001\001\nA\n\013EnsureShape\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\016\n\005shape\022\005shape\"\t\n\001T\022\004type\nW\n\nExpandDims\022\n\n\005input\"\001T\022\013\n\003dim\"\004Tdim\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\030\n\004Tdim\022\004type\032\0020\003:\006\n\0042\002\003\t\n\274\001\n\023ExtractImagePatches\022\013\n\006images\"\001T\032\014\n\007patches\"\001T\"\027\n\006ksizes\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\026\n\005rates\022\tlist(int)(\0010\004\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\244\001\n\024ExtractVolumePatches\022\n\n\005input\"\001T\032\014\n\007patches\"\001T\"\027\n\006ksizes\022\tlist(int)(\0010\005\"\030\n\007strides\022\tlist(int)(\0010\005\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\213\001\n\027FakeQuantWithMinMaxArgs\022\n\n\006inputs\030\001\032\013\n\007outputs\030\001\"\023\n\003min\022\005float\032\005%\000\000\300\300\"\023\n\003max\022\005float\032\005%\000\000\300@\"\023\n\010num_bits\022\003int\032\002\030\010\"\030\n\014narrow_range\022\004bool\032\002(\000\n\244\001\n\037FakeQuantWithMinMaxArgsGradient\022\r\n\tgradients\030\001\022\n\n\006inputs\030\001\032\r\n\tbackprops\030\001\"\023\n\003min\022\005float\032\005%\000\000\300\300\"\023\n\003max\022\005float\032\005%\000\000\300@\"\023\n\010num_bits\022\003int\032\002\030\010\"\030\n\014narrow_range\022\004bool\032\002(\000\ns\n\027FakeQuantWithMinMaxVars\022\n\n\006inputs\030\001\022\007\n\003min\030\001\022\007\n\003max\030\001\032\013\n\007outputs\030\001\"\023\n\010num_bits\022\003int\032\002\030\010\"\030\n\014narrow_range\022\004bool\032\002(\000\n\302\001\n\037FakeQuantWithMinMaxVarsGradient\022\r\n\tgradients\030\001\022\n\n\006inputs\030\001\022\007\n\003min\030\001\022\007\n\003max\030\001\032\027\n\023backprops_wrt_input\030\001\032\024\n\020backprop_wrt_min\030\001\032\024\n\020backprop_wrt_max\030\001\"\023\n\010num_bits\022\003int\032\002\030\010\"\030\n\014narrow_range\022\004bool\032\002(\000\n}\n!FakeQuantWithMinMaxVarsPerChannel\022\n\n\006inputs\030\001\022\007\n\003min\030\001\022\007\n\003max\030\001\032\013\n\007outputs\030\001\"\023\n\010num_bits\022\003int\032\002\030\010\"\030\n\014narrow_range\022\004bool\032\002(\000\n\314\001\n)FakeQuantWithMinMaxVarsPerChannelGradient\022\r\n\tgradients\030\001\022\n\n\006inputs\030\001\022\007\n\003min\030\001\022\007\n\003max\030\001\032\027\n\023backprops_wrt_input\030\001\032\024\n\020backprop_wrt_min\030\001\032\024\n\020backprop_wrt_max\030\001\"\023\n\010num_bits\022\003int\032\002\030\010\"\030\n\014narrow_range\022\004bool\032\002(\000\n^\n\004Fill\022\022\n\004dims\"\nindex_type\022\n\n\005value\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\036\n\nindex_type\022\004type\032\0020\003:\006\n\0042\002\003\t\n\214\001\n\006Gather\022\021\n\006params\"\007Tparams\022\023\n\007indices\"\010Tindices\032\021\n\006output\"\007Tparams\"\034\n\020validate_indices\022\004bool\032\002(\001\"\017\n\007Tparams\022\004type\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\np\n\010GatherNd\022\021\n\006params\"\007Tparams\022\023\n\007indices\"\010Tindices\032\021\n\006output\"\007Tparams\"\017\n\007Tparams\022\004type\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\n\226\001\n\010GatherV2\022\021\n\006params\"\007Tparams\022\023\n\007indices\"\010Tindices\022\r\n\004axis\"\005Taxis\032\021\n\006output\"\007Tparams\"\017\n\007Tparams\022\004type\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\025\n\005Taxis\022\004type:\006\n\0042\002\003\t\n7\n\016GuaranteeConst\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\210\001\001\n.\n\010Identity\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n9\n\tIdentityN\022\n\n\005input2\001T\032\013\n\006output2\001T\"\023\n\001T\022\nlist(type)(\0010\001\n^\n\016ImmutableConst\032\017\n\006tensor\"\005dtype\"\r\n\005dtype\022\004type\"\016\n\005shape\022\005shape\"\034\n\022memory_region_name\022\006string\n6\n\nInplaceAdd\022\006\n\001x\"\001T\022\005\n\001i\030\003\022\006\n\001v\"\001T\032\006\n\001y\"\001T\"\t\n\001T\022\004type\n6\n\nInplaceSub\022\006\n\001x\"\001T\022\005\n\001i\030\003\022\006\n\001v\"\001T\032\006\n\001y\"\001T\"\t\n\001T\022\004type\n9\n\rInplaceUpdate\022\006\n\001x\"\001T\022\005\n\001i\030\003\022\006\n\001v\"\001T\032\006\n\001y\"\001T\"\t\n\001T\022\004type\n:\n\021InvertPermutation\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\n\\\n\010ListDiff\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\010\n\003out\"\001T\032\016\n\003idx\"\007out_idx\"\t\n\001T\022\004type\"\033\n\007out_idx\022\004type\032\0020\003:\006\n\0042\002\003\t\nj\n\nLowerBound\022\022\n\rsorted_inputs\"\001T\022\013\n\006values\"\001T\032\022\n\006output\"\010out_type\"\t\n\001T\022\004type\"\034\n\010out_type\022\004type\032\0020\003:\006\n\0042\002\003\t\nx\n\016MatrixBandPart\022\n\n\005input\"\001T\022\023\n\tnum_lower\"\006Tindex\022\023\n\tnum_upper\"\006Tindex\032\t\n\004band\"\001T\"\t\n\001T\022\004type\"\032\n\006Tindex\022\004type\032\0020\t:\006\n\0042\002\003\t\n3\n\nMatrixDiag\022\r\n\010diagonal\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n6\n\016MatrixDiagPart\022\n\n\005input\"\001T\032\r\n\010diagonal\"\001T\"\t\n\001T\022\004type\nB\n\rMatrixSetDiag\022\n\n\005input\"\001T\022\r\n\010diagonal\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n\215\001\n\tMirrorPad\022\n\n\005input\"\001T\022\025\n\010paddings\"\tTpaddings\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\035\n\tTpaddings\022\004type\032\0020\003:\006\n\0042\002\003\t\"&\n\004mode\022\006string:\026\n\024\022\007REFLECT\022\tSYMMETRIC\n\221\001\n\rMirrorPadGrad\022\n\n\005input\"\001T\022\025\n\010paddings\"\tTpaddings\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\035\n\tTpaddings\022\004type\032\0020\003:\006\n\0042\002\003\t\"&\n\004mode\022\006string:\026\n\024\022\007REFLECT\022\tSYMMETRIC\n\214\001\n\006OneHot\022\r\n\007indices\"\002TI\022\t\n\005depth\030\003\022\r\n\010on_value\"\001T\022\016\n\toff_value\"\001T\032\013\n\006output\"\001T\"\030\n\004axis\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\t\n\001T\022\004type\"\027\n\002TI\022\004type\032\0020\t:\007\n\0052\003\004\003\t\n8\n\010OnesLike\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\034\n\001T\022\004type:\021\n\0172\r\016\023\001\002\006\004\005\021\003\t\010\022\n\nM\n\004Pack\022\016\n\006values\"\001T*\001N\032\013\n\006output\"\001T\"\014\n\001N\022\003int(\0010\001\"\t\n\001T\022\004type\"\017\n\004axis\022\003int\032\002\030\000\n_\n\003Pad\022\n\n\005input\"\001T\022\025\n\010paddings\"\tTpaddings\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\035\n\tTpaddings\022\004type\032\0020\003:\006\n\0042\002\003\t\nw\n\005PadV2\022\n\n\005input\"\001T\022\025\n\010paddings\"\tTpaddings\022\024\n\017constant_values\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\035\n\tTpaddings\022\004type\032\0020\003:\006\n\0042\002\003\t\nV\n\016ParallelConcat\022\016\n\006values\"\001T*\001N\032\013\n\006output\"\001T\"\014\n\001N\022\003int(\0010\001\"\t\n\001T\022\004type\"\016\n\005shape\022\005shape\nC\n\013Placeholder\032\017\n\006output\"\005dtype\"\r\n\005dtype\022\004type\"\024\n\005shape\022\005shape\032\004:\002\030\001\nw\n\rPlaceholderV2\032\017\n\006output\"\005dtype\"\r\n\005dtype\022\004type\"\016\n\005shape\022\005shapeB6\010\027\0222Placeholder now behaves the same as PlaceholderV2.\nX\n\026PlaceholderWithDefault\022\016\n\005input\"\005dtype\032\017\n\006output\"\005dtype\"\r\n\005dtype\022\004type\"\016\n\005shape\022\005shape\nL\n\017PreventGradient\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\025\n\007message\022\006string\032\002\022\000\n\354\001\n\025QuantizeAndDequantize\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\030\n\014signed_input\022\004bool\032\002(\001\"\023\n\010num_bits\022\003int\032\002\030\010\"\027\n\013range_given\022\004bool\032\002(\000\"\031\n\tinput_min\022\005float\032\005%\000\000\000\000\"\031\n\tinput_max\022\005float\032\005%\000\000\000\000\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002B\'\010\026\022#Replaced by QuantizeAndDequantizeV2\n\257\001\n\027QuantizeAndDequantizeV2\022\n\n\005input\"\001T\022\016\n\tinput_min\"\001T\022\016\n\tinput_max\"\001T\032\013\n\006output\"\001T\"\030\n\014signed_input\022\004bool\032\002(\001\"\023\n\010num_bits\022\003int\032\002\030\010\"\027\n\013range_given\022\004bool\032\002(\000\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n\250\001\n\027QuantizeAndDequantizeV3\022\n\n\005input\"\001T\022\016\n\tinput_min\"\001T\022\016\n\tinput_max\"\001T\022\014\n\010num_bits\030\003\032\013\n\006output\"\001T\"\030\n\014signed_input\022\004bool\032\002(\001\"\027\n\013range_given\022\004bool\032\002(\001\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n\221\002\n\nQuantizeV2\022\t\n\005input\030\001\022\r\n\tmin_range\030\001\022\r\n\tmax_range\030\001\032\013\n\006output\"\001T\032\016\n\noutput_min\030\001\032\016\n\noutput_max\030\001\"\024\n\001T\022\004type:\t\n\0072\005\013\014\r\017\020\"C\n\004mode\022\006string\032\016\022\014MIN_COMBINED:#\n!\022\014MIN_COMBINED\022\tMIN_FIRST\022\006SCALED\"R\n\nround_mode\022\006string\032\025\022\023HALF_AWAY_FROM_ZERO:%\n#\022\023HALF_AWAY_FROM_ZERO\022\014HALF_TO_EVEN\n\236\001\n\017QuantizedConcat\022\016\n\nconcat_dim\030\003\022\016\n\006values\"\001T*\001N\022\021\n\ninput_mins\030\001*\001N\022\022\n\013input_maxes\030\001*\001N\032\013\n\006output\"\001T\032\016\n\noutput_min\030\001\032\016\n\noutput_max\030\001\"\014\n\001N\022\003int(\0010\002\"\t\n\001T\022\004type\n\205\002\n\025QuantizedInstanceNorm\022\006\n\001x\"\001T\022\t\n\005x_min\030\001\022\t\n\005x_max\030\001\032\006\n\001y\"\001T\032\t\n\005y_min\030\001\032\t\n\005y_max\030\001\"\024\n\001T\022\004type:\t\n\0072\005\013\014\r\017\020\"\036\n\022output_range_given\022\004bool\032\002(\000\"\033\n\013given_y_min\022\005float\032\005%\000\000\000\000\"\033\n\013given_y_max\022\005float\032\005%\000\000\000\000\" \n\020variance_epsilon\022\005float\032\005%\254\305\'7\"\036\n\016min_separation\022\005float\032\005%o\022\203:\n\242\001\n\020QuantizedReshape\022\013\n\006tensor\"\001T\022\017\n\005shape\"\006Tshape\022\r\n\tinput_min\030\001\022\r\n\tinput_max\030\001\032\013\n\006output\"\001T\032\016\n\noutput_min\030\001\032\016\n\noutput_max\030\001\"\t\n\001T\022\004type\"\032\n\006Tshape\022\004type\032\0020\003:\006\n\0042\002\003\t\n)\n\004Rank\022\n\n\005input\"\001T\032\n\n\006output\030\003\"\t\n\001T\022\004type\n:\n\013RefIdentity\022\r\n\005input\"\001T\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\230\001\001\n[\n\007Reshape\022\013\n\006tensor\"\001T\022\017\n\005shape\"\006Tshape\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\032\n\006Tshape\022\004type\032\0020\003:\006\n\0042\002\003\t\n\203\002\n\032ResourceStridedSliceAssign\022\007\n\003ref\030\024\022\016\n\005begin\"\005Index\022\014\n\003end\"\005Index\022\020\n\007strides\"\005Index\022\n\n\005value\"\001T\"\t\n\001T\022\004type\"\025\n\005Index\022\004type:\006\n\0042\002\003\t\"\025\n\nbegin_mask\022\003int\032\002\030\000\"\023\n\010end_mask\022\003int\032\002\030\000\"\030\n\rellipsis_mask\022\003int\032\002\030\000\"\030\n\rnew_axis_mask\022\003int\032\002\030\000\"\033\n\020shrink_axis_mask\022\003int\032\002\030\000\210\001\001\nK\n\007Reverse\022\013\n\006tensor\"\001T\022\010\n\004dims\030\n\032\013\n\006output\"\001T\"\034\n\001T\022\004type:\021\n\0172\r\004\006\021\005\003\t\n\023\001\002\010\022\007\n\212\001\n\017ReverseSequence\022\n\n\005input\"\001T\022\023\n\013seq_lengths\"\004Tlen\032\013\n\006output\"\001T\"\016\n\007seq_dim\022\003int\"\024\n\tbatch_dim\022\003int\032\002\030\000\"\t\n\001T\022\004type\"\030\n\004Tlen\022\004type\032\0020\t:\006\n\0042\002\003\t\nl\n\tReverseV2\022\013\n\006tensor\"\001T\022\014\n\004axis\"\004Tidx\032\013\n\006output\"\001T\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\"\035\n\001T\022\004type:\022\n\0202\016\004\006\021\005\003\t\n\016\023\001\002\010\022\007\ns\n\tScatterNd\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\022\021\n\005shape\"\010Tindices\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\n\222\001\n\027ScatterNdNonAliasingAdd\022\n\n\005input\"\001T\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\032\013\n\006output\"\001T\"!\n\001T\022\004type:\026\n\0242\022\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\n\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\nP\n\005Shape\022\n\n\005input\"\001T\032\022\n\006output\"\010out_type\"\t\n\001T\022\004type\"\034\n\010out_type\022\004type\032\0020\003:\006\n\0042\002\003\t\ne\n\006ShapeN\022\r\n\005input\"\001T*\001N\032\025\n\006output\"\010out_type*\001N\"\014\n\001N\022\003int(\0010\001\"\t\n\001T\022\004type\"\034\n\010out_type\022\004type\032\0020\003:\006\n\0042\002\003\t\nO\n\004Size\022\n\n\005input\"\001T\032\022\n\006output\"\010out_type\"\t\n\001T\022\004type\"\034\n\010out_type\022\004type\032\0020\003:\006\n\0042\002\003\t\na\n\005Slice\022\n\n\005input\"\001T\022\016\n\005begin\"\005Index\022\r\n\004size\"\005Index\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\025\n\005Index\022\004type:\006\n\0042\002\003\t\n.\n\010Snapshot\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n\177\n\014SpaceToBatch\022\n\n\005input\"\001T\022\025\n\010paddings\"\tTpaddings\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\035\n\tTpaddings\022\004type\032\0020\003:\006\n\0042\002\003\t\"\025\n\nblock_size\022\003int(\0010\002\n\251\001\n\016SpaceToBatchND\022\n\n\005input\"\001T\022\033\n\013block_shape\"\014Tblock_shape\022\025\n\010paddings\"\tTpaddings\032\013\n\006output\"\001T\"\t\n\001T\022\004type\" \n\014Tblock_shape\022\004type\032\0020\003:\006\n\0042\002\003\t\"\035\n\tTpaddings\022\004type\032\0020\003:\006\n\0042\002\003\t\n\205\001\n\014SpaceToDepth\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\025\n\nblock_size\022\003int(\0010\002\":\n\013data_format\022\006string\032\006\022\004NHWC:\033\n\031\022\004NHWC\022\004NCHW\022\013NCHW_VECT_C\n[\n\005Split\022\r\n\tsplit_dim\030\003\022\n\n\005value\"\001T\032\026\n\006output\"\001T*\tnum_split\"\024\n\tnum_split\022\003int(\0010\001\"\t\n\001T\022\004type\n\213\001\n\006SplitV\022\n\n\005value\"\001T\022\023\n\013size_splits\"\004Tlen\022\r\n\tsplit_dim\030\003\032\026\n\006output\"\001T*\tnum_split\"\024\n\tnum_split\022\003int(\0010\001\"\t\n\001T\022\004type\"\030\n\004Tlen\022\004type\032\0020\t:\006\n\0042\002\003\t\nN\n\007Squeeze\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\037\n\014squeeze_dims\022\tlist(int)\032\002\n\000(\001\n2\n\014StopGradient\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n\366\001\n\014StridedSlice\022\n\n\005input\"\001T\022\016\n\005begin\"\005Index\022\014\n\003end\"\005Index\022\020\n\007strides\"\005Index\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\025\n\005Index\022\004type:\006\n\0042\002\003\t\"\025\n\nbegin_mask\022\003int\032\002\030\000\"\023\n\010end_mask\022\003int\032\002\030\000\"\030\n\rellipsis_mask\022\003int\032\002\030\000\"\030\n\rnew_axis_mask\022\003int\032\002\030\000\"\033\n\020shrink_axis_mask\022\003int\032\002\030\000\n\220\002\n\022StridedSliceAssign\022\013\n\003ref\"\001T\200\001\001\022\016\n\005begin\"\005Index\022\014\n\003end\"\005Index\022\020\n\007strides\"\005Index\022\n\n\005value\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\"\t\n\001T\022\004type\"\025\n\005Index\022\004type:\006\n\0042\002\003\t\"\025\n\nbegin_mask\022\003int\032\002\030\000\"\023\n\010end_mask\022\003int\032\002\030\000\"\030\n\rellipsis_mask\022\003int\032\002\030\000\"\030\n\rnew_axis_mask\022\003int\032\002\030\000\"\033\n\020shrink_axis_mask\022\003int\032\002\030\000\n\207\002\n\020StridedSliceGrad\022\016\n\005shape\"\005Index\022\016\n\005begin\"\005Index\022\014\n\003end\"\005Index\022\020\n\007strides\"\005Index\022\007\n\002dy\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\025\n\005Index\022\004type:\006\n\0042\002\003\t\"\025\n\nbegin_mask\022\003int\032\002\030\000\"\023\n\010end_mask\022\003int\032\002\030\000\"\030\n\rellipsis_mask\022\003int\032\002\030\000\"\030\n\rnew_axis_mask\022\003int\032\002\030\000\"\033\n\020shrink_axis_mask\022\003int\032\002\030\000\nc\n\004Tile\022\n\n\005input\"\001T\022\027\n\tmultiples\"\nTmultiples\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\036\n\nTmultiples\022\004type\032\0020\003:\006\n\0042\002\003\t\nm\n\010TileGrad\022\n\n\005input\"\001T\022\r\n\tmultiples\030\003\032\013\n\006output\"\001T\"\t\n\001T\022\004typeB.\010\003\022*TileGrad has been replaced with reduce_sum\nP\n\tTranspose\022\006\n\001x\"\001T\022\r\n\004perm\"\005Tperm\032\006\n\001y\"\001T\"\t\n\001T\022\004type\"\031\n\005Tperm\022\004type\032\0020\003:\006\n\0042\002\003\t\nP\n\006Unique\022\006\n\001x\"\001T\032\006\n\001y\"\001T\032\016\n\003idx\"\007out_idx\"\t\n\001T\022\004type\"\033\n\007out_idx\022\004type\032\0020\003:\006\n\0042\002\003\t\n|\n\010UniqueV2\022\006\n\001x\"\001T\022\r\n\004axis\"\005Taxis\032\006\n\001y\"\001T\032\016\n\003idx\"\007out_idx\"\t\n\001T\022\004type\"\031\n\005Taxis\022\004type\032\0020\t:\006\n\0042\002\003\t\"\033\n\007out_idx\022\004type\032\0020\003:\006\n\0042\002\003\t\nl\n\020UniqueWithCounts\022\006\n\001x\"\001T\032\006\n\001y\"\001T\032\016\n\003idx\"\007out_idx\032\020\n\005count\"\007out_idx\"\t\n\001T\022\004type\"\033\n\007out_idx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\230\001\n\022UniqueWithCountsV2\022\006\n\001x\"\001T\022\r\n\004axis\"\005Taxis\032\006\n\001y\"\001T\032\016\n\003idx\"\007out_idx\032\020\n\005count\"\007out_idx\"\t\n\001T\022\004type\"\031\n\005Taxis\022\004type\032\0020\t:\006\n\0042\002\003\t\"\033\n\007out_idx\022\004type\032\0020\003:\006\n\0042\002\003\t\nP\n\006Unpack\022\n\n\005value\"\001T\032\020\n\006output\"\001T*\003num\"\014\n\003num\022\003int(\001\"\t\n\001T\022\004type\"\017\n\004axis\022\003int\032\002\030\000\nW\n\014UnravelIndex\022\017\n\007indices\"\004Tidx\022\014\n\004dims\"\004Tidx\032\016\n\006output\"\004Tidx\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\nj\n\nUpperBound\022\022\n\rsorted_inputs\"\001T\022\013\n\006values\"\001T\032\022\n\006output\"\010out_type\"\t\n\001T\022\004type\"\034\n\010out_type\022\004type\032\0020\003:\006\n\0042\002\003\t\nE\n\005Where\022\n\n\005input\"\001T\032\t\n\005index\030\t\"%\n\001T\022\004type\032\0020\n:\026\n\0242\022\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\n\n&\n\tZerosLike\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\t\n\001T\022\004type")
