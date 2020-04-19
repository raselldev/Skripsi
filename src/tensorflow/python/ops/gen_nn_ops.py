import collections as _collections
import six as _six

from tensorflow.python.eager import context as _context
from tensorflow.python.eager import execute as _execute
from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.tf_export import tf_export


def bias_add(value, bias, data_format="NHWC", name=None):
  r"""Adds `bias` to `value`.

  This is a special case of `tf.add` where `bias` is restricted to be 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Any number of dimensions.
    bias: A `Tensor`. Must have the same type as `value`.
      1-D with size the last dimension of `value`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to "in_channels", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if data_format is None:
      data_format = "NHWC"
    data_format = _execute.make_str(data_format, "data_format")
    _, _, _op = _op_def_lib._apply_op_helper(
        "BiasAdd", value=value, bias=bias, data_format=data_format, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "data_format",
              _op.get_attr("data_format"))
    _execute.record_gradient(
      "BiasAdd", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "BiasAdd",
        name, _ctx._post_execution_callbacks, value, bias, "data_format",
        data_format)
      return _result
    except _core._FallbackException:
      return bias_add_eager_fallback(
          value, bias, data_format=data_format, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def bias_add_grad(out_backprop, data_format="NHWC", name=None):
  r"""The backward operation for "BiasAdd" on the "bias" tensor.

  It accumulates all the values from out_backprop into the feature dimension.
  For NHWC data format, the feature dimension is the last. For NCHW data format,
  the feature dimension is the third-to-last.

  Args:
    out_backprop: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Any number of dimensions.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to "in_channels", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `out_backprop`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if data_format is None:
      data_format = "NHWC"
    data_format = _execute.make_str(data_format, "data_format")
    _, _, _op = _op_def_lib._apply_op_helper(
        "BiasAddGrad", out_backprop=out_backprop, data_format=data_format,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "data_format",
              _op.get_attr("data_format"))
    _execute.record_gradient(
      "BiasAddGrad", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "BiasAddGrad",
        name, _ctx._post_execution_callbacks, out_backprop, "data_format",
        data_format)
      return _result
    except _core._FallbackException:
      return bias_add_grad_eager_fallback(
          out_backprop, data_format=data_format, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

@tf_export('nn.conv2d')
def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", dilations=[1, 1, 1, 1], name=None):
  r"""Computes a 2-D convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, out_channels]`, this op
  performs the following:

  1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
  2. Extracts image patches from the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
  3. For each patch, right-multiplies the filter matrix and the image patch
     vector.

  In detail, with the default NHWC format,

      output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                          filter[di, dj, q, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      A 4-D tensor. The dimension order is interpreted according to the value
      of `data_format`, see below for details.
    filter: A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
    strides: A list of `ints`.
      1-D tensor of length 4.  The stride of the sliding window for each
      dimension of `input`. The dimension order is determined by the value of
      `data_format`, see below for details.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(strides, (list, tuple)):
      raise TypeError(
          "Expected list for 'strides' argument to "
          "'conv2d' Op, not %r." % strides)
    strides = [_execute.make_int(_i, "strides") for _i in strides]
    padding = _execute.make_str(padding, "padding")
    if use_cudnn_on_gpu is None:
      use_cudnn_on_gpu = True
    use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
    if data_format is None:
      data_format = "NHWC"
    data_format = _execute.make_str(data_format, "data_format")
    if dilations is None:
      dilations = [1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
      raise TypeError(
          "Expected list for 'dilations' argument to "
          "'conv2d' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
    _, _, _op = _op_def_lib._apply_op_helper(
        "Conv2D", input=input, filter=filter, strides=strides,
        padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
        data_format=data_format, dilations=dilations, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "strides", _op.get_attr("strides"),
              "use_cudnn_on_gpu", _op.get_attr("use_cudnn_on_gpu"), "padding",
              _op.get_attr("padding"), "data_format",
              _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _execute.record_gradient(
      "Conv2D", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Conv2D", name,
        _ctx._post_execution_callbacks, input, filter, "strides", strides,
        "use_cudnn_on_gpu", use_cudnn_on_gpu, "padding", padding,
        "data_format", data_format, "dilations", dilations)
      return _result
    except _core._FallbackException:
      return conv2d_eager_fallback(
          input, filter, strides=strides, use_cudnn_on_gpu=use_cudnn_on_gpu,
          padding=padding, data_format=data_format, dilations=dilations,
          name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

@tf_export('nn.conv2d_backprop_filter')
def conv2d_backprop_filter(input, filter_sizes, out_backprop, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", dilations=[1, 1, 1, 1], name=None):
  r"""Computes the gradients of convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, out_channels]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(strides, (list, tuple)):
      raise TypeError(
          "Expected list for 'strides' argument to "
          "'conv2d_backprop_filter' Op, not %r." % strides)
    strides = [_execute.make_int(_i, "strides") for _i in strides]
    padding = _execute.make_str(padding, "padding")
    if use_cudnn_on_gpu is None:
      use_cudnn_on_gpu = True
    use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
    if data_format is None:
      data_format = "NHWC"
    data_format = _execute.make_str(data_format, "data_format")
    if dilations is None:
      dilations = [1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
      raise TypeError(
          "Expected list for 'dilations' argument to "
          "'conv2d_backprop_filter' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
    _, _, _op = _op_def_lib._apply_op_helper(
        "Conv2DBackpropFilter", input=input, filter_sizes=filter_sizes,
        out_backprop=out_backprop, strides=strides, padding=padding,
        use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format,
        dilations=dilations, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "strides", _op.get_attr("strides"),
              "use_cudnn_on_gpu", _op.get_attr("use_cudnn_on_gpu"), "padding",
              _op.get_attr("padding"), "data_format",
              _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _execute.record_gradient(
      "Conv2DBackpropFilter", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "Conv2DBackpropFilter", name, _ctx._post_execution_callbacks, input,
        filter_sizes, out_backprop, "strides", strides, "use_cudnn_on_gpu",
        use_cudnn_on_gpu, "padding", padding, "data_format", data_format,
        "dilations", dilations)
      return _result
    except _core._FallbackException:
      return conv2d_backprop_filter_eager_fallback(
          input, filter_sizes, out_backprop, strides=strides,
          use_cudnn_on_gpu=use_cudnn_on_gpu, padding=padding,
          data_format=data_format, dilations=dilations, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

@tf_export('nn.conv2d_backprop_input')
def conv2d_backprop_input(input_sizes, filter, out_backprop, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", dilations=[1, 1, 1, 1], name=None):
  r"""Computes the gradients of convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`,
      where `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(strides, (list, tuple)):
      raise TypeError(
          "Expected list for 'strides' argument to "
          "'conv2d_backprop_input' Op, not %r." % strides)
    strides = [_execute.make_int(_i, "strides") for _i in strides]
    padding = _execute.make_str(padding, "padding")
    if use_cudnn_on_gpu is None:
      use_cudnn_on_gpu = True
    use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
    if data_format is None:
      data_format = "NHWC"
    data_format = _execute.make_str(data_format, "data_format")
    if dilations is None:
      dilations = [1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
      raise TypeError(
          "Expected list for 'dilations' argument to "
          "'conv2d_backprop_input' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
    _, _, _op = _op_def_lib._apply_op_helper(
        "Conv2DBackpropInput", input_sizes=input_sizes, filter=filter,
        out_backprop=out_backprop, strides=strides, padding=padding,
        use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format,
        dilations=dilations, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "strides", _op.get_attr("strides"),
              "use_cudnn_on_gpu", _op.get_attr("use_cudnn_on_gpu"), "padding",
              _op.get_attr("padding"), "data_format",
              _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _execute.record_gradient(
      "Conv2DBackpropInput", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "Conv2DBackpropInput", name, _ctx._post_execution_callbacks,
        input_sizes, filter, out_backprop, "strides", strides,
        "use_cudnn_on_gpu", use_cudnn_on_gpu, "padding", padding,
        "data_format", data_format, "dilations", dilations)
      return _result
    except _core._FallbackException:
      return conv2d_backprop_input_eager_fallback(
          input_sizes, filter, out_backprop, strides=strides,
          use_cudnn_on_gpu=use_cudnn_on_gpu, padding=padding,
          data_format=data_format, dilations=dilations, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

@tf_export('nn.elu')
def elu(features, name=None):
  r"""Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

  See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
  ](http://arxiv.org/abs/1511.07289)

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Elu", features=features, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "Elu", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Elu", name,
        _ctx._post_execution_callbacks, features)
      return _result
    except _core._FallbackException:
      return elu_eager_fallback(
          features, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

__fused_batch_norm_outputs = ["y", "batch_mean", "batch_variance",
                             "reserve_space_1", "reserve_space_2"]
_FusedBatchNormOutput = _collections.namedtuple(
    "FusedBatchNorm", __fused_batch_norm_outputs)

def _fused_batch_norm(x, scale, offset, mean, variance, epsilon=0.0001, data_format="NHWC", is_training=True, name=None):
  r"""Batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      The data format for x and y. Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `x`.
    batch_variance: A `Tensor`. Has the same type as `x`.
    reserve_space_1: A `Tensor`. Has the same type as `x`.
    reserve_space_2: A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if epsilon is None:
      epsilon = 0.0001
    epsilon = _execute.make_float(epsilon, "epsilon")
    if data_format is None:
      data_format = "NHWC"
    data_format = _execute.make_str(data_format, "data_format")
    if is_training is None:
      is_training = True
    is_training = _execute.make_bool(is_training, "is_training")
    _, _, _op = _op_def_lib._apply_op_helper(
        "FusedBatchNorm", x=x, scale=scale, offset=offset, mean=mean,
        variance=variance, epsilon=epsilon, data_format=data_format,
        is_training=is_training, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "epsilon", _op.get_attr("epsilon"),
              "data_format", _op.get_attr("data_format"), "is_training",
              _op.get_attr("is_training"))
    _execute.record_gradient(
      "FusedBatchNorm", _inputs_flat, _attrs, _result, name)
    _result = _FusedBatchNormOutput._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "FusedBatchNorm", name, _ctx._post_execution_callbacks, x, scale,
        offset, mean, variance, "epsilon", epsilon, "data_format",
        data_format, "is_training", is_training)
      _result = _FusedBatchNormOutput._make(_result)
      return _result
    except _core._FallbackException:
      return _fused_batch_norm_eager_fallback(
          x, scale, offset, mean, variance, epsilon=epsilon,
          data_format=data_format, is_training=is_training, name=name,
          ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

_fused_batch_norm_grad_outputs = ["x_backprop", "scale_backprop",
                                 "offset_backprop", "reserve_space_3",
                                 "reserve_space_4"]
_FusedBatchNormGradOutput = _collections.namedtuple(
    "FusedBatchNormGrad", _fused_batch_norm_grad_outputs)

def fused_batch_norm_grad(y_backprop, x, scale, reserve_space_1, reserve_space_2, epsilon=0.0001, data_format="NHWC", is_training=True, name=None):
  r"""Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `y_backprop`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      The data format for y_backprop, x, x_backprop.
      Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_3, reserve_space_4).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `y_backprop`.
    offset_backprop: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_3: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_4: A `Tensor`. Has the same type as `y_backprop`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if epsilon is None:
      epsilon = 0.0001
    epsilon = _execute.make_float(epsilon, "epsilon")
    if data_format is None:
      data_format = "NHWC"
    data_format = _execute.make_str(data_format, "data_format")
    if is_training is None:
      is_training = True
    is_training = _execute.make_bool(is_training, "is_training")
    _, _, _op = _op_def_lib._apply_op_helper(
        "FusedBatchNormGrad", y_backprop=y_backprop, x=x, scale=scale,
        reserve_space_1=reserve_space_1, reserve_space_2=reserve_space_2,
        epsilon=epsilon, data_format=data_format, is_training=is_training,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "epsilon", _op.get_attr("epsilon"),
              "data_format", _op.get_attr("data_format"), "is_training",
              _op.get_attr("is_training"))
    _execute.record_gradient(
      "FusedBatchNormGrad", _inputs_flat, _attrs, _result, name)
    _result = _FusedBatchNormGradOutput._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "FusedBatchNormGrad", name, _ctx._post_execution_callbacks,
        y_backprop, x, scale, reserve_space_1, reserve_space_2, "epsilon",
        epsilon, "data_format", data_format, "is_training", is_training)
      _result = _FusedBatchNormGradOutput._make(_result)
      return _result
    except _core._FallbackException:
      return fused_batch_norm_grad_eager_fallback(
          y_backprop, x, scale, reserve_space_1, reserve_space_2,
          epsilon=epsilon, data_format=data_format, is_training=is_training,
          name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def max_pool(input, ksize, strides, padding, data_format="NHWC", name=None):
  r"""Performs max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `qint8`.
      4-D input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW", "NCHW_VECT_C"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(ksize, (list, tuple)):
      raise TypeError(
          "Expected list for 'ksize' argument to "
          "'max_pool' Op, not %r." % ksize)
    ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
    if not isinstance(strides, (list, tuple)):
      raise TypeError(
          "Expected list for 'strides' argument to "
          "'max_pool' Op, not %r." % strides)
    strides = [_execute.make_int(_i, "strides") for _i in strides]
    padding = _execute.make_str(padding, "padding")
    if data_format is None:
      data_format = "NHWC"
    data_format = _execute.make_str(data_format, "data_format")
    _, _, _op = _op_def_lib._apply_op_helper(
        "MaxPool", input=input, ksize=ksize, strides=strides, padding=padding,
        data_format=data_format, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "ksize", _op.get_attr("ksize"),
              "strides", _op.get_attr("strides"), "padding",
              _op.get_attr("padding"), "data_format",
              _op.get_attr("data_format"))
    _execute.record_gradient(
      "MaxPool", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "MaxPool",
        name, _ctx._post_execution_callbacks, input, "ksize", ksize,
        "strides", strides, "padding", padding, "data_format", data_format)
      return _result
    except _core._FallbackException:
      return max_pool_eager_fallback(
          input, ksize=ksize, strides=strides, padding=padding,
          data_format=data_format, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def max_pool_grad(orig_input, orig_output, grad, ksize, strides, padding, data_format="NHWC", name=None):
  r"""Computes gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients w.r.t. the output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(ksize, (list, tuple)):
      raise TypeError(
          "Expected list for 'ksize' argument to "
          "'max_pool_grad' Op, not %r." % ksize)
    ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
    if not isinstance(strides, (list, tuple)):
      raise TypeError(
          "Expected list for 'strides' argument to "
          "'max_pool_grad' Op, not %r." % strides)
    strides = [_execute.make_int(_i, "strides") for _i in strides]
    padding = _execute.make_str(padding, "padding")
    if data_format is None:
      data_format = "NHWC"
    data_format = _execute.make_str(data_format, "data_format")
    _, _, _op = _op_def_lib._apply_op_helper(
        "MaxPoolGrad", orig_input=orig_input, orig_output=orig_output,
        grad=grad, ksize=ksize, strides=strides, padding=padding,
        data_format=data_format, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "T",
              _op.get_attr("T"))
    _execute.record_gradient(
      "MaxPoolGrad", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "MaxPoolGrad",
        name, _ctx._post_execution_callbacks, orig_input, orig_output, grad,
        "ksize", ksize, "strides", strides, "padding", padding, "data_format",
        data_format)
      return _result
    except _core._FallbackException:
      return max_pool_grad_eager_fallback(
          orig_input, orig_output, grad, ksize=ksize, strides=strides,
          padding=padding, data_format=data_format, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

@tf_export('nn.relu')
def relu(features, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Relu", features=features, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "Relu", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Relu", name,
        _ctx._post_execution_callbacks, features)
      return _result
    except _core._FallbackException:
      return relu_eager_fallback(
          features, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def relu_grad(gradients, features, name=None):
  r"""Computes rectified linear gradients for a Relu operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The backpropagated gradients to the corresponding Relu operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding Relu operation, OR
      the outputs of that operation (both work equivalently).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "ReluGrad", gradients=gradients, features=features, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "ReluGrad", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "ReluGrad",
        name, _ctx._post_execution_callbacks, gradients, features)
      return _result
    except _core._FallbackException:
      return relu_grad_eager_fallback(
          gradients, features, name=name, ctx=_ctx)
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
_op_def_lib = _InitOpDefLibrary(b"\n\274\001\n\007AvgPool\022\n\n\005value\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\301\001\n\tAvgPool3D\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\005\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\332\001\n\rAvgPool3DGrad\022\024\n\020orig_input_shape\030\003\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\005\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\325\001\n\013AvgPoolGrad\022\024\n\020orig_input_shape\030\003\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\343\001\n BatchNormWithGlobalNormalization\022\006\n\001t\"\001T\022\006\n\001m\"\001T\022\006\n\001v\"\001T\022\t\n\004beta\"\001T\022\n\n\005gamma\"\001T\032\013\n\006result\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\031\n\020variance_epsilon\022\005float\"!\n\031scale_after_normalization\022\004boolB#\010\t\022\037Use tf.nn.batch_normalization()\n\213\002\n$BatchNormWithGlobalNormalizationGrad\022\006\n\001t\"\001T\022\006\n\001m\"\001T\022\006\n\001v\"\001T\022\n\n\005gamma\"\001T\022\r\n\010backprop\"\001T\032\007\n\002dx\"\001T\032\007\n\002dm\"\001T\032\007\n\002dv\"\001T\032\007\n\002db\"\001T\032\007\n\002dg\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\031\n\020variance_epsilon\022\005float\"!\n\031scale_after_normalization\022\004boolB#\010\t\022\037Use tf.nn.batch_normalization()\n~\n\007BiasAdd\022\n\n\005value\"\001T\022\t\n\004bias\"\001T\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\n~\n\013BiasAddGrad\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\nQ\n\tBiasAddV1\022\n\n\005value\"\001T\022\t\n\004bias\"\001T\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\n\354\001\n\006Conv2D\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\034\n\020use_cudnn_on_gpu\022\004bool\032\002(\001\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\222\002\n\024Conv2DBackpropFilter\022\n\n\005input\"\001T\022\020\n\014filter_sizes\030\003\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\034\n\020use_cudnn_on_gpu\022\004bool\032\002(\001\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\221\002\n\023Conv2DBackpropInput\022\017\n\013input_sizes\030\003\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\034\n\020use_cudnn_on_gpu\022\004bool\032\002(\001\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\326\001\n\006Conv3D\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"!\n\tdilations\022\tlist(int)\032\t\n\007\032\005\001\001\001\001\001\n\344\001\n\024Conv3DBackpropFilter\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\022\n\001T\022\004type:\007\n\0052\003\023\001\002\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"!\n\tdilations\022\tlist(int)\032\t\n\007\032\005\001\001\001\001\001B\036\010\n\022\032Use Conv3DBackpropFilterV2\n\376\001\n\026Conv3DBackpropFilterV2\022\n\n\005input\"\001T\022\020\n\014filter_sizes\030\003\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"!\n\tdilations\022\tlist(int)\032\t\n\007\032\005\001\001\001\001\001\n\342\001\n\023Conv3DBackpropInput\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\022\n\001T\022\004type:\007\n\0052\003\023\001\002\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"!\n\tdilations\022\tlist(int)\032\t\n\007\032\005\001\001\001\001\001B\035\010\n\022\031Use Conv3DBackpropInputV2\n\237\002\n\025Conv3DBackpropInputV2\022\025\n\013input_sizes\"\006Tshape\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"!\n\tdilations\022\tlist(int)\032\t\n\007\032\005\001\001\001\001\001\"\032\n\006Tshape\022\004type\032\0020\003:\006\n\0042\002\003\t\nu\n\020DataFormatDimMap\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\"\034\n\nsrc_format\022\006string\032\006\022\004NHWC\"\034\n\ndst_format\022\006string\032\006\022\004NCHW\ny\n\024DataFormatVecPermute\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\"\034\n\nsrc_format\022\006string\032\006\022\004NHWC\"\034\n\ndst_format\022\006string\032\006\022\004NCHW\n\335\001\n\025DepthwiseConv2dNative\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\203\002\n#DepthwiseConv2dNativeBackpropFilter\022\n\n\005input\"\001T\022\020\n\014filter_sizes\030\003\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\202\002\n\"DepthwiseConv2dNativeBackpropInput\022\017\n\013input_sizes\030\003\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\245\001\n\nDilation2D\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\007strides\022\tlist(int)(\0010\004\"\026\n\005rates\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\317\001\n\030Dilation2DBackpropFilter\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\024\n\017filter_backprop\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\007strides\022\tlist(int)(\0010\004\"\026\n\005rates\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\312\001\n\027Dilation2DBackpropInput\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\020\n\013in_backprop\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\007strides\022\tlist(int)(\0010\004\"\026\n\005rates\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n;\n\003Elu\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\nL\n\007EluGrad\022\016\n\tgradients\"\001T\022\014\n\007outputs\"\001T\032\016\n\tbackprops\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\211\002\n\021FractionalAvgPool\022\n\n\005value\"\001T\032\013\n\006output\"\001T\032\030\n\024row_pooling_sequence\030\t\032\030\n\024col_pooling_sequence\030\t\" \n\rpooling_ratio\022\013list(float)(\0010\004\"\031\n\rpseudo_random\022\004bool\032\002(\000\"\027\n\013overlapping\022\004bool\032\002(\000\"\031\n\rdeterministic\022\004bool\032\002(\000\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\023\n\001T\022\004type:\010\n\0062\004\001\002\003\t\n\266\001\n\025FractionalAvgPoolGrad\022\033\n\027orig_input_tensor_shape\030\t\022\021\n\014out_backprop\"\001T\022\030\n\024row_pooling_sequence\030\t\022\030\n\024col_pooling_sequence\030\t\032\013\n\006output\"\001T\"\027\n\013overlapping\022\004bool\032\002(\000\"\023\n\001T\022\004type:\010\n\0062\004\001\002\003\t\n\211\002\n\021FractionalMaxPool\022\n\n\005value\"\001T\032\013\n\006output\"\001T\032\030\n\024row_pooling_sequence\030\t\032\030\n\024col_pooling_sequence\030\t\" \n\rpooling_ratio\022\013list(float)(\0010\004\"\031\n\rpseudo_random\022\004bool\032\002(\000\"\027\n\013overlapping\022\004bool\032\002(\000\"\031\n\rdeterministic\022\004bool\032\002(\000\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\023\n\001T\022\004type:\010\n\0062\004\001\002\003\t\n\274\001\n\025FractionalMaxPoolGrad\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\021\n\014out_backprop\"\001T\022\030\n\024row_pooling_sequence\030\t\022\030\n\024col_pooling_sequence\030\t\032\013\n\006output\"\001T\"\027\n\013overlapping\022\004bool\032\002(\000\"\023\n\001T\022\004type:\010\n\0062\004\001\002\003\t\n\230\002\n\016FusedBatchNorm\022\006\n\001x\"\001T\022\n\n\005scale\"\001T\022\013\n\006offset\"\001T\022\t\n\004mean\"\001T\022\r\n\010variance\"\001T\032\006\n\001y\"\001T\032\017\n\nbatch_mean\"\001T\032\023\n\016batch_variance\"\001T\032\024\n\017reserve_space_1\"\001T\032\024\n\017reserve_space_2\"\001T\"\020\n\001T\022\004type:\005\n\0032\001\001\"\027\n\007epsilon\022\005float\032\005%\027\267\3218\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\027\n\013is_training\022\004bool\032\002(\001\n\300\002\n\022FusedBatchNormGrad\022\017\n\ny_backprop\"\001T\022\006\n\001x\"\001T\022\n\n\005scale\"\001T\022\024\n\017reserve_space_1\"\001T\022\024\n\017reserve_space_2\"\001T\032\017\n\nx_backprop\"\001T\032\023\n\016scale_backprop\"\001T\032\024\n\017offset_backprop\"\001T\032\024\n\017reserve_space_3\"\001T\032\024\n\017reserve_space_4\"\001T\"\020\n\001T\022\004type:\005\n\0032\001\001\"\027\n\007epsilon\022\005float\032\005%\027\267\3218\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\027\n\013is_training\022\004bool\032\002(\001\n\325\002\n\024FusedBatchNormGradV2\022\017\n\ny_backprop\"\001T\022\006\n\001x\"\001T\022\t\n\005scale\030\001\022\024\n\017reserve_space_1\"\001U\022\024\n\017reserve_space_2\"\001U\032\017\n\nx_backprop\"\001T\032\023\n\016scale_backprop\"\001U\032\024\n\017offset_backprop\"\001U\032\024\n\017reserve_space_3\"\001U\032\024\n\017reserve_space_4\"\001U\"\022\n\001T\022\004type:\007\n\0052\003\023\016\001\"\020\n\001U\022\004type:\005\n\0032\001\001\"\027\n\007epsilon\022\005float\032\005%\027\267\3218\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\027\n\013is_training\022\004bool\032\002(\001\n\256\002\n\020FusedBatchNormV2\022\006\n\001x\"\001T\022\n\n\005scale\"\001U\022\013\n\006offset\"\001U\022\t\n\004mean\"\001U\022\r\n\010variance\"\001U\032\006\n\001y\"\001T\032\017\n\nbatch_mean\"\001U\032\023\n\016batch_variance\"\001U\032\024\n\017reserve_space_1\"\001U\032\024\n\017reserve_space_2\"\001U\"\022\n\001T\022\004type:\007\n\0052\003\023\016\001\"\020\n\001U\022\004type:\005\n\0032\001\001\"\027\n\007epsilon\022\005float\032\005%\027\267\3218\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\027\n\013is_training\022\004bool\032\002(\001\n\272\001\n\016FusedPadConv2D\022\n\n\005input\"\001T\022\014\n\010paddings\030\003\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\022\n\001T\022\004type:\007\n\0052\003\023\001\002\"&\n\004mode\022\006string:\026\n\024\022\007REFLECT\022\tSYMMETRIC\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\357\001\n\027FusedResizeAndPadConv2D\022\n\n\005input\"\001T\022\010\n\004size\030\003\022\014\n\010paddings\030\003\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\022\n\001T\022\004type:\007\n\0052\003\023\001\002\" \n\024resize_align_corners\022\004bool\032\002(\000\"&\n\004mode\022\006string:\026\n\024\022\007REFLECT\022\tSYMMETRIC\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\nW\n\006InTopK\022\017\n\013predictions\030\001\022\014\n\007targets\"\001T\032\r\n\tprecision\030\n\"\010\n\001k\022\003int\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\nW\n\010InTopKV2\022\017\n\013predictions\030\001\022\014\n\007targets\"\001T\022\006\n\001k\"\001T\032\r\n\tprecision\030\n\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\n2\n\006L2Loss\022\006\n\001t\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\222\001\n\003LRN\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\027\n\014depth_radius\022\003int\032\002\030\005\"\024\n\004bias\022\005float\032\005%\000\000\200?\"\025\n\005alpha\022\005float\032\005%\000\000\200?\"\024\n\004beta\022\005float\032\005%\000\000\000?\"\026\n\001T\022\004type\032\0020\001:\007\n\0052\003\023\016\001\n\301\001\n\007LRNGrad\022\020\n\013input_grads\"\001T\022\020\n\013input_image\"\001T\022\021\n\014output_image\"\001T\032\013\n\006output\"\001T\"\027\n\014depth_radius\022\003int\032\002\030\005\"\024\n\004bias\022\005float\032\005%\000\000\200?\"\025\n\005alpha\022\005float\032\005%\000\000\200?\"\024\n\004beta\022\005float\032\005%\000\000\000?\"\026\n\001T\022\004type\032\0020\001:\007\n\0052\003\023\016\001\n?\n\nLogSoftmax\022\013\n\006logits\"\001T\032\017\n\nlogsoftmax\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\324\001\n\007MaxPool\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\036\n\001T\022\004type\032\0020\001:\017\n\r2\013\023\016\001\002\003\t\004\005\006\021\013\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\":\n\013data_format\022\006string\032\006\022\004NHWC:\033\n\031\022\004NHWC\022\004NCHW\022\013NCHW_VECT_C\n\300\001\n\tMaxPool3D\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\005\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"\022\n\001T\022\004type:\007\n\0052\003\023\016\001\n\221\002\n\rMaxPool3DGrad\022\024\n\norig_input\"\006TInput\022\025\n\013orig_output\"\006TInput\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\005\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"\026\n\001T\022\004type\032\0020\001:\007\n\0052\003\023\016\001\"\033\n\006TInput\022\004type\032\0020\001:\007\n\0052\003\023\016\001\n\363\001\n\021MaxPool3DGradGrad\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\005\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\356\001\n\013MaxPoolGrad\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\037\n\001T\022\004type\032\0020\001:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\356\001\n\017MaxPoolGradGrad\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\326\001\n\021MaxPoolGradGradV2\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\t\n\004grad\"\001T\022\t\n\005ksize\030\003\022\013\n\007strides\030\003\032\013\n\006output\"\001T\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\336\001\n\031MaxPoolGradGradWithArgmax\022\n\n\005input\"\001T\022\t\n\004grad\"\001T\022\021\n\006argmax\"\007Targmax\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"\027\n\007Targmax\022\004type:\006\n\0042\002\003\t\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\326\001\n\rMaxPoolGradV2\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\t\n\004grad\"\001T\022\t\n\005ksize\030\003\022\013\n\007strides\030\003\032\013\n\006output\"\001T\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\037\n\001T\022\004type\032\0020\001:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\332\001\n\025MaxPoolGradWithArgmax\022\n\n\005input\"\001T\022\t\n\004grad\"\001T\022\021\n\006argmax\"\007Targmax\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"\027\n\007Targmax\022\004type:\006\n\0042\002\003\t\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\274\001\n\tMaxPoolV2\022\n\n\005input\"\001T\022\t\n\005ksize\030\003\022\013\n\007strides\030\003\032\013\n\006output\"\001T\"\036\n\001T\022\004type\032\0020\001:\017\n\r2\013\023\016\001\002\003\t\004\005\006\021\013\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\":\n\013data_format\022\006string\032\006\022\004NHWC:\033\n\031\022\004NHWC\022\004NCHW\022\013NCHW_VECT_C\n\317\001\n\021MaxPoolWithArgmax\022\n\n\005input\"\001T\032\013\n\006output\"\001T\032\021\n\006argmax\"\007Targmax\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\033\n\007Targmax\022\004type\032\0020\t:\006\n\0042\002\003\t\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n^\n\nNthElement\022\n\n\005input\"\001T\022\005\n\001n\030\003\032\013\n\006values\"\001T\"\023\n\007reverse\022\004bool\032\002(\000\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\315\001\n\020QuantizedAvgPool\022\n\n\005input\"\001T\022\r\n\tmin_input\030\001\022\r\n\tmax_input\030\001\032\013\n\006output\"\001T\032\016\n\nmin_output\030\001\032\016\n\nmax_output\030\001\"\024\n\001T\022\004type:\t\n\0072\005\013\014\r\017\020\"\022\n\005ksize\022\tlist(int)\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\231\003\n)QuantizedBatchNormWithGlobalNormalization\022\013\n\001t\"\006Tinput\022\t\n\005t_min\030\001\022\t\n\005t_max\030\001\022\013\n\001m\"\006Tinput\022\t\n\005m_min\030\001\022\t\n\005m_max\030\001\022\013\n\001v\"\006Tinput\022\t\n\005v_min\030\001\022\t\n\005v_max\030\001\022\016\n\004beta\"\006Tinput\022\014\n\010beta_min\030\001\022\014\n\010beta_max\030\001\022\017\n\005gamma\"\006Tinput\022\r\n\tgamma_min\030\001\022\r\n\tgamma_max\030\001\032\022\n\006result\"\010out_type\032\016\n\nresult_min\030\001\032\016\n\nresult_max\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\033\n\010out_type\022\004type:\t\n\0072\005\013\014\r\017\020\"\031\n\020variance_epsilon\022\005float\"!\n\031scale_after_normalization\022\004bool\n\336\001\n\020QuantizedBiasAdd\022\013\n\005input\"\002T1\022\n\n\004bias\"\002T2\022\r\n\tmin_input\030\001\022\r\n\tmax_input\030\001\022\014\n\010min_bias\030\001\022\014\n\010max_bias\030\001\032\022\n\006output\"\010out_type\032\013\n\007min_out\030\001\032\013\n\007max_out\030\001\"\025\n\002T1\022\004type:\t\n\0072\005\013\014\r\017\020\"\025\n\002T2\022\004type:\t\n\0072\005\013\014\r\017\020\"\033\n\010out_type\022\004type:\t\n\0072\005\013\014\r\017\020\n\333\002\n\017QuantizedConv2D\022\017\n\005input\"\006Tinput\022\021\n\006filter\"\007Tfilter\022\r\n\tmin_input\030\001\022\r\n\tmax_input\030\001\022\016\n\nmin_filter\030\001\022\016\n\nmax_filter\030\001\032\022\n\006output\"\010out_type\032\016\n\nmin_output\030\001\032\016\n\nmax_output\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\032\n\007Tfilter\022\004type:\t\n\0072\005\013\014\r\017\020\"\037\n\010out_type\022\004type\032\0020\r:\t\n\0072\005\013\014\r\017\020\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\315\001\n\020QuantizedMaxPool\022\n\n\005input\"\001T\022\r\n\tmin_input\030\001\022\r\n\tmax_input\030\001\032\013\n\006output\"\001T\032\016\n\nmin_output\030\001\032\016\n\nmax_output\030\001\"\024\n\001T\022\004type:\t\n\0072\005\013\014\r\017\020\"\022\n\005ksize\022\tlist(int)\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\306\001\n\rQuantizedRelu\022\022\n\010features\"\006Tinput\022\020\n\014min_features\030\001\022\020\n\014max_features\030\001\032\027\n\013activations\"\010out_type\032\023\n\017min_activations\030\001\032\023\n\017max_activations\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\037\n\010out_type\022\004type\032\0020\014:\t\n\0072\005\013\014\r\017\020\n\307\001\n\016QuantizedRelu6\022\022\n\010features\"\006Tinput\022\020\n\014min_features\030\001\022\020\n\014max_features\030\001\032\027\n\013activations\"\010out_type\032\023\n\017min_activations\030\001\032\023\n\017max_activations\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\037\n\010out_type\022\004type\032\0020\014:\t\n\0072\005\013\014\r\017\020\n\326\001\n\016QuantizedReluX\022\022\n\010features\"\006Tinput\022\r\n\tmax_value\030\001\022\020\n\014min_features\030\001\022\020\n\014max_features\030\001\032\027\n\013activations\"\010out_type\032\023\n\017min_activations\030\001\032\023\n\017max_activations\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\037\n\010out_type\022\004type\032\0020\014:\t\n\0072\005\013\014\r\017\020\nE\n\004Relu\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\034\n\001T\022\004type:\021\n\0172\r\001\002\003\004\005\006\t\016\021\023\026\027\013\nE\n\005Relu6\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\nW\n\tRelu6Grad\022\016\n\tgradients\"\001T\022\r\n\010features\"\001T\032\016\n\tbackprops\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\nV\n\010ReluGrad\022\016\n\tgradients\"\001T\022\r\n\010features\"\001T\032\016\n\tbackprops\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n<\n\004Selu\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\nM\n\010SeluGrad\022\016\n\tgradients\"\001T\022\014\n\007outputs\"\001T\032\016\n\tbackprops\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n9\n\007Softmax\022\013\n\006logits\"\001T\032\014\n\007softmax\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\nj\n\035SoftmaxCrossEntropyWithLogits\022\r\n\010features\"\001T\022\013\n\006labels\"\001T\032\t\n\004loss\"\001T\032\r\n\010backprop\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n@\n\010Softplus\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\nR\n\014SoftplusGrad\022\016\n\tgradients\"\001T\022\r\n\010features\"\001T\032\016\n\tbackprops\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n@\n\010Softsign\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\nR\n\014SoftsignGrad\022\016\n\tgradients\"\001T\022\r\n\010features\"\001T\032\016\n\tbackprops\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\223\001\n#SparseSoftmaxCrossEntropyWithLogits\022\r\n\010features\"\001T\022\021\n\006labels\"\007Tlabels\032\t\n\004loss\"\001T\032\r\n\010backprop\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\033\n\007Tlabels\022\004type\032\0020\t:\006\n\0042\002\003\t\n\201\001\n\004TopK\022\n\n\005input\"\001T\032\013\n\006values\"\001T\032\013\n\007indices\030\003\"\n\n\001k\022\003int(\001\"\022\n\006sorted\022\004bool\032\002(\001\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027B\026\010\007\022\022Use TopKV2 instead\nf\n\006TopKV2\022\n\n\005input\"\001T\022\005\n\001k\030\003\032\013\n\006values\"\001T\032\013\n\007indices\030\003\"\022\n\006sorted\022\004bool\032\002(\001\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027")
