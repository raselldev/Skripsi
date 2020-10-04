from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import collections

import numpy as np

from backend import context
from backend import execute
from backend.framework import constant_op
from backend.framework import tensor_util
from backend.framework import dtypes
from backend.framework import ops
from backend.core import op_def_pb2
from backend.framework import op_def_library
#from backend.framework import op_def_registry

def atrous_conv2d(value, filters, rate, padding, name=None):
  return convolution(
      input=value,
      filter=filters,
      padding=padding,
      dilation_rate=np.broadcast_to(rate, (2,)),
      name=name)

def convolution(
    input,  
    filter,  
    padding,
    strides=None,
    dilation_rate=None,
    name=None,
    data_format=None):
  with ops.name_scope(name, "convolution", [input, filter]) as name:
    input = ops.convert_to_tensor(input, name="input")  
    input_shape = input.get_shape()
    filter = ops.convert_to_tensor(filter, name="filter")  
    filter_shape = filter.get_shape()
    op = Convolution(
        input_shape,
        filter_shape,
        padding,
        strides=strides,
        dilation_rate=dilation_rate,
        name=name,
        data_format=data_format)
    return op(input, filter)

class Convolution(object):
  def __init__(self,
               input_shape,
               filter_shape,
               padding,
               strides=None,
               dilation_rate=None,
               name=None,
               data_format=None):
    num_total_dims = filter_shape.ndims
    if num_total_dims is None:
      num_total_dims = input_shape.ndims
    if num_total_dims is None:
      raise ValueError("rank of input or filter must be known")

    num_spatial_dims = num_total_dims - 2

    try:
      input_shape.with_rank(num_spatial_dims + 2)
    except ValueError:
      ValueError("input tensor must have rank %d" % (num_spatial_dims + 2))

    try:
      filter_shape.with_rank(num_spatial_dims + 2)
    except ValueError:
      ValueError("filter tensor must have rank %d" % (num_spatial_dims + 2))

    if data_format is None or not data_format.startswith("NC"):
      input_channels_dim = input_shape[num_spatial_dims + 1]
      spatial_dims = range(1, num_spatial_dims + 1)
    else:
      input_channels_dim = input_shape[1]
      spatial_dims = range(2, num_spatial_dims + 2)

    if not input_channels_dim.is_compatible_with(
        filter_shape[num_spatial_dims]):
      raise ValueError(
          "number of input channels does not match corresponding dimension of "
          "filter, {} != {}".format(input_channels_dim,
                                    filter_shape[num_spatial_dims]))

    strides, dilation_rate = _get_strides_and_dilation_rate(
        num_spatial_dims, strides, dilation_rate)

    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.data_format = data_format
    self.strides = strides
    self.name = name
    self.conv_op = _WithSpaceToBatch(
        input_shape,
        dilation_rate=dilation_rate,
        padding=padding,
        build_op=self._build_op,
        filter_shape=filter_shape,
        spatial_dims=spatial_dims,
        data_format=data_format)

  def _build_op(self, _, padding):
    return _NonAtrousConvolution(
        self.input_shape,
        filter_shape=self.filter_shape,
        padding=padding,
        data_format=self.data_format,
        strides=self.strides,
        name=self.name)

  def __call__(self, inp, filter):  
    return self.conv_op(inp, filter)

def _get_strides_and_dilation_rate(num_spatial_dims, strides, dilation_rate):
  if dilation_rate is None:
    dilation_rate = [1] * num_spatial_dims
  elif len(dilation_rate) != num_spatial_dims:
    raise ValueError("len(dilation_rate)=%d but should be %d" %
                     (len(dilation_rate), num_spatial_dims))
  dilation_rate = np.array(dilation_rate, dtype=np.int32)
  if np.any(dilation_rate < 1):
    raise ValueError("all values of dilation_rate must be positive")

  if strides is None:
    strides = [1] * num_spatial_dims
  elif len(strides) != num_spatial_dims:
    raise ValueError("len(strides)=%d but should be %d" % (len(strides),
                                                           num_spatial_dims))
  strides = np.array(strides, dtype=np.int32)
  if np.any(strides < 1):
    raise ValueError("all values of strides must be positive")

  if np.any(strides > 1) and np.any(dilation_rate > 1):
    raise ValueError(
        "strides > 1 not supported in conjunction with dilation_rate > 1")
  return strides, dilation_rate

class _WithSpaceToBatch(object):
  def __init__(self,
               input_shape,
               dilation_rate,
               padding,
               build_op,
               filter_shape=None,
               spatial_dims=None,
               data_format=None):
    dilation_rate = ops.convert_to_tensor(
        dilation_rate, dtypes.int32, name="dilation_rate")
    try:
      rate_shape = dilation_rate.get_shape().with_rank(1)
    except ValueError:
      raise ValueError("rate must be rank 1")

    if not dilation_rate.get_shape().is_fully_defined():
      raise ValueError("rate must have known shape")

    num_spatial_dims = rate_shape[0].value

    if data_format is not None and data_format.startswith("NC"):
      starting_spatial_dim = 2
    else:
      starting_spatial_dim = 1

    if spatial_dims is None:
      spatial_dims = range(starting_spatial_dim,
                           num_spatial_dims + starting_spatial_dim)
    orig_spatial_dims = list(spatial_dims)
    spatial_dims = sorted(set(int(x) for x in orig_spatial_dims))
    if spatial_dims != orig_spatial_dims or any(x < 1 for x in spatial_dims):
      raise ValueError(
          "spatial_dims must be a montonically increasing sequence of positive "
          "integers")  

    if data_format is not None and data_format.startswith("NC"):
      expected_input_rank = spatial_dims[-1]
    else:
      expected_input_rank = spatial_dims[-1] + 1

    try:
      input_shape.with_rank_at_least(expected_input_rank)
    except ValueError:
      ValueError("input tensor must have rank %d at least" %
                 (expected_input_rank))

    const_rate = tensor_util.constant_value(dilation_rate)
    rate_or_const_rate = dilation_rate
    if const_rate is not None:
      rate_or_const_rate = const_rate
      if np.any(const_rate < 1):
        raise ValueError("dilation_rate must be positive")
      if np.all(const_rate == 1):
        self.call = build_op(num_spatial_dims, padding)
        return

    if padding == "SAME":
      if filter_shape is None:
        raise ValueError("filter_shape must be specified for SAME padding")
      filter_shape = ops.convert_to_tensor(filter_shape, name="filter_shape")
      const_filter_shape = tensor_util.constant_value(filter_shape)
      if const_filter_shape is not None:
        filter_shape = const_filter_shape
        self.base_paddings = _with_space_to_batch_base_paddings(
            const_filter_shape, num_spatial_dims, rate_or_const_rate)
      else:
        self.num_spatial_dims = num_spatial_dims
        self.rate_or_const_rate = rate_or_const_rate
        self.base_paddings = None
    elif padding == "VALID":
      self.base_paddings = np.zeros([num_spatial_dims, 2], np.int32)
    else:
      raise ValueError("Invalid padding method %r" % padding)

    self.input_shape = input_shape
    self.spatial_dims = spatial_dims
    self.dilation_rate = dilation_rate
    self.data_format = data_format
    self.op = build_op(num_spatial_dims, "VALID")
    self.call = self._with_space_to_batch_call

  def _with_space_to_batch_call(self, inp, filter):  
    input_spatial_shape = None
    input_shape = self.input_shape
    spatial_dims = self.spatial_dims
    if input_shape.ndims is not None:
      input_shape_list = input_shape.as_list()
      input_spatial_shape = [input_shape_list[i] for i in spatial_dims]
    if input_spatial_shape is None or None in input_spatial_shape:
      input_shape_tensor = array_ops.shape(inp)
      input_spatial_shape = array_ops.stack(
          [input_shape_tensor[i] for i in spatial_dims])

    base_paddings = self.base_paddings
    if base_paddings is None:
      filter_shape = array_ops.shape(filter)
      base_paddings = _with_space_to_batch_base_paddings(
          filter_shape, self.num_spatial_dims, self.rate_or_const_rate)
    paddings, crops = array_ops.required_space_to_batch_paddings(
        input_shape=input_spatial_shape,
        base_paddings=base_paddings,
        block_shape=self.dilation_rate)

    dilation_rate = _with_space_to_batch_adjust(self.dilation_rate, 1,
                                                spatial_dims)
    paddings = _with_space_to_batch_adjust(paddings, 0, spatial_dims)
    crops = _with_space_to_batch_adjust(crops, 0, spatial_dims)
    input_converted = array_ops.space_to_batch_nd(
        input=inp, block_shape=dilation_rate, paddings=paddings)

    result = self.op(input_converted, filter)

    result_converted = array_ops.batch_to_space_nd(
        input=result, block_shape=dilation_rate, crops=crops)

    if self.data_format is not None and self.data_format.startswith("NC"):
      if not result_converted.shape[1].value and filter is not None:
        output_shape = result_converted.shape.as_list()
        output_shape[1] = filter.shape[-1]
        result_converted.set_shape(output_shape)

    return result_converted

  def __call__(self, inp, filter):  
    return self.call(inp, filter)

class _NonAtrousConvolution(object):
  def __init__(
      self,
      input_shape,
      filter_shape,  
      padding,
      data_format=None,
      strides=None,
      name=None):
    filter_shape = filter_shape.with_rank(input_shape.ndims)
    self.padding = padding
    self.name = name
    input_shape = input_shape.with_rank(filter_shape.ndims)
    if input_shape.ndims is None:
      raise ValueError("Rank of convolution must be known")
    if input_shape.ndims < 3 or input_shape.ndims > 5:
      raise ValueError(
          "`input` and `filter` must have rank at least 3 and at most 5")
    conv_dims = input_shape.ndims - 2
    if strides is None:
      strides = [1] * conv_dims
    elif len(strides) != conv_dims:
      raise ValueError("len(strides)=%d, but should be %d" % (len(strides),
                                                              conv_dims))
    if conv_dims == 1:
      if data_format is None:
        data_format = "NWC"
      elif data_format not in {"NCW", "NWC", "NCHW", "NHWC"}:
        raise ValueError("data_format must be \"NWC\" or \"NCW\".")
      self.strides = strides[0]
      self.data_format = data_format
      self.conv_op = self._conv1d
    elif conv_dims == 2:
      if data_format is None or data_format == "NHWC":
        data_format = "NHWC"
        strides = [1] + list(strides) + [1]
      elif data_format == "NCHW":
        strides = [1, 1] + list(strides)
      else:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
      self.strides = strides
      self.data_format = data_format
      self.conv_op = conv2d
    elif conv_dims == 3:
      if data_format is None or data_format == "NDHWC":
        strides = [1] + list(strides) + [1]
      elif data_format == "NCDHW":
        strides = [1, 1] + list(strides)
      else:
        raise ValueError("data_format must be \"NDHWC\" or \"NCDHW\". Have: %s"
                         % data_format)
      self.strides = strides
      self.data_format = data_format
      self.conv_op = gen_nn_ops.conv3d

  def _conv1d(self, input, filter, strides, padding, data_format, name):
    return conv1d(
        value=input,
        filters=filter,
        stride=strides,
        padding=padding,
        data_format=data_format,
        name=name)

  def __call__(self, inp, filter):  
    return self.conv_op(
        input=inp,
        filter=filter,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        name=self.name)


__fused_batch_norm_outputs = ["y", "batch_mean", "batch_variance",
                             "reserve_space_1", "reserve_space_2"]
_FusedBatchNormOutput = collections.namedtuple(
    "FusedBatchNorm", __fused_batch_norm_outputs)

_fused_batch_norm_grad_outputs = ["x_backprop", "scale_backprop",
                                 "offset_backprop", "reserve_space_3",
                                 "reserve_space_4"]
_FusedBatchNormGradOutput = collections.namedtuple(
    "FusedBatchNormGrad", _fused_batch_norm_grad_outputs)    


def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", dilations=[1, 1, 1, 1], name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(strides, (list, tuple)):
      raise TypeError(
          "Expected list for 'strides' argument to "
          "'conv2d' Op, not %r." % strides)
    #strides = [execute.make_int(_i, "strides") for _i in strides]
    #padding = execute.make_str(padding, "padding")
    if use_cudnn_on_gpu is None:
      use_cudnn_on_gpu = True
    use_cudnn_on_gpu = execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
    if data_format is None:
      data_format = "NHWC"
    if dilations is None:
      dilations = [1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
      raise TypeError(
          "Expected list for 'dilations' argument to "
          "'conv2d' Op, not %r." % dilations)
    dilations = [execute.make_int(_i, "dilations") for _i in dilations]
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
    execute.record_gradient(
      "Conv2D", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_backend.TFE_Py_FastPathExecute(
        _ctx.context_handle, _ctx._eagercontext.device_name, "Conv2D", name,
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

def relu(features, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Relu", features=features, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    execute.record_gradient(
      "Relu", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_backend.TFE_Py_FastPathExecute(
        _ctx.context_handle, _ctx._eagercontext.device_name, "Relu", name,
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

def _fused_batch_norm(x, scale, offset, mean, variance, epsilon=0.0001, data_format="NHWC", is_training=True, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if epsilon is None:
      epsilon = 0.0001
    #epsilon = execute.make_float(epsilon, "epsilon")
    if data_format is None:
      data_format = "NHWC"
#    data_format = execute.make_str(data_format, "data_format")
    if is_training is None:
      is_training = True
    is_training = execute.make_bool(is_training, "is_training")
    _, _, _op = _op_def_lib._apply_op_helper(
        "FusedBatchNorm", x=x, scale=scale, offset=offset, mean=mean,
        variance=variance, epsilon=epsilon, data_format=data_format,
        is_training=is_training, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "epsilon", _op.get_attr("epsilon"),
              "data_format", _op.get_attr("data_format"), "is_training",
              _op.get_attr("is_training"))
    execute.record_gradient(
      "FusedBatchNorm", _inputs_flat, _attrs, _result, name)
    _result = _FusedBatchNormOutput._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_backend.TFE_Py_FastPathExecute(
        _ctx.context_handle, _ctx._eagercontext.device_name,
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

def max_pool(input, ksize, strides, padding, data_format="NHWC", name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(ksize, (list, tuple)):
      raise TypeError(
          "Expected list for 'ksize' argument to "
          "'max_pool' Op, not %r." % ksize)
    ksize = [execute.make_int(_i, "ksize") for _i in ksize]
    if not isinstance(strides, (list, tuple)):
      raise TypeError(
          "Expected list for 'strides' argument to "
          "'max_pool' Op, not %r." % strides)
    #strides = [execute.make_int(_i, "strides") for _i in strides]
    #padding = execute.make_str(padding, "padding")
    if data_format is None:
      data_format = "NHWC"
    #data_format = execute.make_str(data_format, "data_format")
    _, _, _op = _op_def_lib._apply_op_helper(
        "MaxPool", input=input, ksize=ksize, strides=strides, padding=padding,
        data_format=data_format, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "ksize", _op.get_attr("ksize"),
              "strides", _op.get_attr("strides"), "padding",
              _op.get_attr("padding"), "data_format",
              _op.get_attr("data_format"))
    execute.record_gradient(
      "MaxPool", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_backend.TFE_Py_FastPathExecute(
        _ctx.context_handle, _ctx._eagercontext.device_name, "MaxPool",
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

def bias_add(value, bias, data_format="NHWC", name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if data_format is None:
      data_format = "NHWC"
    #data_format = execute.make_str(data_format, "data_format")
    _, _, _op = _op_def_lib._apply_op_helper(
        "BiasAdd", value=value, bias=bias, data_format=data_format, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "data_format",
              _op.get_attr("data_format"))
    execute.record_gradient(
      "BiasAdd", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_backend.TFE_Py_FastPathExecute(
        _ctx.context_handle, _ctx._eagercontext.device_name, "BiasAdd",
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

def conv2d_backprop_input(input_sizes, filter, out_backprop, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", dilations=[1, 1, 1, 1], name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(strides, (list, tuple)):
      raise TypeError(
          "Expected list for 'strides' argument to "
          "'conv2d_backprop_input' Op, not %r." % strides)
    #strides = [execute.make_int(_i, "strides") for _i in strides]
    #padding = execute.make_str(padding, "padding")
    if use_cudnn_on_gpu is None:
      use_cudnn_on_gpu = True
    use_cudnn_on_gpu = execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
    if data_format is None:
      data_format = "NHWC"
    #data_format = execute.make_str(data_format, "data_format")
    if dilations is None:
      dilations = [1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
      raise TypeError(
          "Expected list for 'dilations' argument to "
          "'conv2d_backprop_input' Op, not %r." % dilations)
    dilations = [execute.make_int(_i, "dilations") for _i in dilations]
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
    execute.record_gradient(
      "Conv2DBackpropInput", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_backend.TFE_Py_FastPathExecute(
        _ctx.context_handle, _ctx._eagercontext.device_name,
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

def conv2d_backprop_filter(input, filter_sizes, out_backprop, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", dilations=[1, 1, 1, 1], name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(strides, (list, tuple)):
      raise TypeError(
          "Expected list for 'strides' argument to "
          "'conv2d_backprop_filter' Op, not %r." % strides)
    #strides = [execute.make_int(_i, "strides") for _i in strides]
    #padding = execute.make_str(padding, "padding")
    if use_cudnn_on_gpu is None:
      use_cudnn_on_gpu = True
    use_cudnn_on_gpu = execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
    if data_format is None:
      data_format = "NHWC"
    #data_format = execute.make_str(data_format, "data_format")
    if dilations is None:
      dilations = [1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
      raise TypeError(
          "Expected list for 'dilations' argument to "
          "'conv2d_backprop_filter' Op, not %r." % dilations)
    dilations = [execute.make_int(_i, "dilations") for _i in dilations]
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
    execute.record_gradient(
      "Conv2DBackpropFilter", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_backend.TFE_Py_FastPathExecute(
        _ctx.context_handle, _ctx._eagercontext.device_name,
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

def bias_add_grad(out_backprop, data_format="NHWC", name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if data_format is None:
      data_format = "NHWC"
    #data_format = execute.make_str(data_format, "data_format")
    _, _, _op = _op_def_lib._apply_op_helper(
        "BiasAddGrad", out_backprop=out_backprop, data_format=data_format,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "data_format",
              _op.get_attr("data_format"))
    execute.record_gradient(
      "BiasAddGrad", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_backend.TFE_Py_FastPathExecute(
        _ctx.context_handle, _ctx._eagercontext.device_name, "BiasAddGrad",
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

def max_pool_grad(orig_input, orig_output, grad, ksize, strides, padding, data_format="NHWC", name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(ksize, (list, tuple)):
      raise TypeError(
          "Expected list for 'ksize' argument to "
          "'max_pool_grad' Op, not %r." % ksize)
    ksize = [execute.make_int(_i, "ksize") for _i in ksize]
    if not isinstance(strides, (list, tuple)):
      raise TypeError(
          "Expected list for 'strides' argument to "
          "'max_pool_grad' Op, not %r." % strides)
    #strides = [execute.make_int(_i, "strides") for _i in strides]
    #padding = execute.make_str(padding, "padding")
    if data_format is None:
      data_format = "NHWC"
    #data_format = execute.make_str(data_format, "data_format")
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
    execute.record_gradient(
      "MaxPoolGrad", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_backend.TFE_Py_FastPathExecute(
        _ctx.context_handle, _ctx._eagercontext.device_name, "MaxPoolGrad",
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

def relu_grad(gradients, features, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "ReluGrad", gradients=gradients, features=features, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    execute.record_gradient(
      "ReluGrad", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_backend.TFE_Py_FastPathExecute(
        _ctx.context_handle, _ctx._eagercontext.device_name, "ReluGrad",
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

def fused_batch_norm_grad(y_backprop, x, scale, reserve_space_1, reserve_space_2, epsilon=0.0001, data_format="NHWC", is_training=True, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if epsilon is None:
      epsilon = 0.0001
    #epsilon = execute.make_float(epsilon, "epsilon")
    if data_format is None:
      data_format = "NHWC"
    #data_format = execute.make_str(data_format, "data_format")
    if is_training is None:
      is_training = True
    is_training = execute.make_bool(is_training, "is_training")
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
    execute.record_gradient(
      "FusedBatchNormGrad", _inputs_flat, _attrs, _result, name)
    _result = _FusedBatchNormGradOutput._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_backend.TFE_Py_FastPathExecute(
        _ctx.context_handle, _ctx._eagercontext.device_name,
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


def fused_batch_norm(
    x,
    scale,
    offset,  # pylint: disable=invalid-name
    mean=None,
    variance=None,
    epsilon=0.001,
    data_format="NHWC",
    is_training=True,
    name=None):
  x = ops.convert_to_tensor(x, name="input")
  scale = ops.convert_to_tensor(scale, name="scale")
  offset = ops.convert_to_tensor(offset, name="offset")
  if is_training:
    if (mean is not None) or (variance is not None):
      raise ValueError("Both 'mean' and 'variance' must be None "
                       "if is_training is True.")
  if mean is None:
    mean = constant_op.constant([])
  if variance is None:
    variance = constant_op.constant([])
  min_epsilon = 1.001e-5
  epsilon = epsilon if epsilon > min_epsilon else min_epsilon
  if x.dtype == dtypes.float16 or x.dtype == dtypes.bfloat16:
    fused_batch_norm_func = nn.fused_batch_norm_v2
  else:
    fused_batch_norm_func = _fused_batch_norm
  y, batch_mean, batch_var, _, _ = fused_batch_norm_func(
      x,
      scale,
      offset,
      mean,
      variance,
      epsilon=epsilon,
      data_format=data_format,
      is_training=is_training,
      name=name)
  return y, batch_mean, batch_var


def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  #op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib

_op_def_lib = _InitOpDefLibrary(b"\n\274\001\n\007AvgPool\022\n\n\005value\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\301\001\n\tAvgPool3D\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\005\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\332\001\n\rAvgPool3DGrad\022\024\n\020orig_input_shape\030\003\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\005\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\325\001\n\013AvgPoolGrad\022\024\n\020orig_input_shape\030\003\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\343\001\n BatchNormWithGlobalNormalization\022\006\n\001t\"\001T\022\006\n\001m\"\001T\022\006\n\001v\"\001T\022\t\n\004beta\"\001T\022\n\n\005gamma\"\001T\032\013\n\006result\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\031\n\020variance_epsilon\022\005float\"!\n\031scale_after_normalization\022\004boolB#\010\t\022\037Use tf.nn.batch_normalization()\n\213\002\n$BatchNormWithGlobalNormalizationGrad\022\006\n\001t\"\001T\022\006\n\001m\"\001T\022\006\n\001v\"\001T\022\n\n\005gamma\"\001T\022\r\n\010backprop\"\001T\032\007\n\002dx\"\001T\032\007\n\002dm\"\001T\032\007\n\002dv\"\001T\032\007\n\002db\"\001T\032\007\n\002dg\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\031\n\020variance_epsilon\022\005float\"!\n\031scale_after_normalization\022\004boolB#\010\t\022\037Use tf.nn.batch_normalization()\n~\n\007BiasAdd\022\n\n\005value\"\001T\022\t\n\004bias\"\001T\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\n~\n\013BiasAddGrad\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\nQ\n\tBiasAddV1\022\n\n\005value\"\001T\022\t\n\004bias\"\001T\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\n\354\001\n\006Conv2D\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\034\n\020use_cudnn_on_gpu\022\004bool\032\002(\001\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\222\002\n\024Conv2DBackpropFilter\022\n\n\005input\"\001T\022\020\n\014filter_sizes\030\003\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\034\n\020use_cudnn_on_gpu\022\004bool\032\002(\001\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\221\002\n\023Conv2DBackpropInput\022\017\n\013input_sizes\030\003\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\034\n\020use_cudnn_on_gpu\022\004bool\032\002(\001\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\326\001\n\006Conv3D\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"!\n\tdilations\022\tlist(int)\032\t\n\007\032\005\001\001\001\001\001\n\344\001\n\024Conv3DBackpropFilter\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\022\n\001T\022\004type:\007\n\0052\003\023\001\002\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"!\n\tdilations\022\tlist(int)\032\t\n\007\032\005\001\001\001\001\001B\036\010\n\022\032Use Conv3DBackpropFilterV2\n\376\001\n\026Conv3DBackpropFilterV2\022\n\n\005input\"\001T\022\020\n\014filter_sizes\030\003\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"!\n\tdilations\022\tlist(int)\032\t\n\007\032\005\001\001\001\001\001\n\342\001\n\023Conv3DBackpropInput\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\022\n\001T\022\004type:\007\n\0052\003\023\001\002\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"!\n\tdilations\022\tlist(int)\032\t\n\007\032\005\001\001\001\001\001B\035\010\n\022\031Use Conv3DBackpropInputV2\n\237\002\n\025Conv3DBackpropInputV2\022\025\n\013input_sizes\"\006Tshape\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"!\n\tdilations\022\tlist(int)\032\t\n\007\032\005\001\001\001\001\001\"\032\n\006Tshape\022\004type\032\0020\003:\006\n\0042\002\003\t\nu\n\020DataFormatDimMap\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\"\034\n\nsrc_format\022\006string\032\006\022\004NHWC\"\034\n\ndst_format\022\006string\032\006\022\004NCHW\ny\n\024DataFormatVecPermute\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\"\034\n\nsrc_format\022\006string\032\006\022\004NHWC\"\034\n\ndst_format\022\006string\032\006\022\004NCHW\n\335\001\n\025DepthwiseConv2dNative\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\203\002\n#DepthwiseConv2dNativeBackpropFilter\022\n\n\005input\"\001T\022\020\n\014filter_sizes\030\003\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\202\002\n\"DepthwiseConv2dNativeBackpropInput\022\017\n\013input_sizes\030\003\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\245\001\n\nDilation2D\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\007strides\022\tlist(int)(\0010\004\"\026\n\005rates\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\317\001\n\030Dilation2DBackpropFilter\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\024\n\017filter_backprop\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\007strides\022\tlist(int)(\0010\004\"\026\n\005rates\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\312\001\n\027Dilation2DBackpropInput\022\n\n\005input\"\001T\022\013\n\006filter\"\001T\022\021\n\014out_backprop\"\001T\032\020\n\013in_backprop\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\007strides\022\tlist(int)(\0010\004\"\026\n\005rates\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n;\n\003Elu\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\nL\n\007EluGrad\022\016\n\tgradients\"\001T\022\014\n\007outputs\"\001T\032\016\n\tbackprops\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\211\002\n\021FractionalAvgPool\022\n\n\005value\"\001T\032\013\n\006output\"\001T\032\030\n\024row_pooling_sequence\030\t\032\030\n\024col_pooling_sequence\030\t\" \n\rpooling_ratio\022\013list(float)(\0010\004\"\031\n\rpseudo_random\022\004bool\032\002(\000\"\027\n\013overlapping\022\004bool\032\002(\000\"\031\n\rdeterministic\022\004bool\032\002(\000\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\023\n\001T\022\004type:\010\n\0062\004\001\002\003\t\n\266\001\n\025FractionalAvgPoolGrad\022\033\n\027orig_input_tensor_shape\030\t\022\021\n\014out_backprop\"\001T\022\030\n\024row_pooling_sequence\030\t\022\030\n\024col_pooling_sequence\030\t\032\013\n\006output\"\001T\"\027\n\013overlapping\022\004bool\032\002(\000\"\023\n\001T\022\004type:\010\n\0062\004\001\002\003\t\n\211\002\n\021FractionalMaxPool\022\n\n\005value\"\001T\032\013\n\006output\"\001T\032\030\n\024row_pooling_sequence\030\t\032\030\n\024col_pooling_sequence\030\t\" \n\rpooling_ratio\022\013list(float)(\0010\004\"\031\n\rpseudo_random\022\004bool\032\002(\000\"\027\n\013overlapping\022\004bool\032\002(\000\"\031\n\rdeterministic\022\004bool\032\002(\000\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\"\023\n\001T\022\004type:\010\n\0062\004\001\002\003\t\n\274\001\n\025FractionalMaxPoolGrad\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\021\n\014out_backprop\"\001T\022\030\n\024row_pooling_sequence\030\t\022\030\n\024col_pooling_sequence\030\t\032\013\n\006output\"\001T\"\027\n\013overlapping\022\004bool\032\002(\000\"\023\n\001T\022\004type:\010\n\0062\004\001\002\003\t\n\230\002\n\016FusedBatchNorm\022\006\n\001x\"\001T\022\n\n\005scale\"\001T\022\013\n\006offset\"\001T\022\t\n\004mean\"\001T\022\r\n\010variance\"\001T\032\006\n\001y\"\001T\032\017\n\nbatch_mean\"\001T\032\023\n\016batch_variance\"\001T\032\024\n\017reserve_space_1\"\001T\032\024\n\017reserve_space_2\"\001T\"\020\n\001T\022\004type:\005\n\0032\001\001\"\027\n\007epsilon\022\005float\032\005%\027\267\3218\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\027\n\013is_training\022\004bool\032\002(\001\n\300\002\n\022FusedBatchNormGrad\022\017\n\ny_backprop\"\001T\022\006\n\001x\"\001T\022\n\n\005scale\"\001T\022\024\n\017reserve_space_1\"\001T\022\024\n\017reserve_space_2\"\001T\032\017\n\nx_backprop\"\001T\032\023\n\016scale_backprop\"\001T\032\024\n\017offset_backprop\"\001T\032\024\n\017reserve_space_3\"\001T\032\024\n\017reserve_space_4\"\001T\"\020\n\001T\022\004type:\005\n\0032\001\001\"\027\n\007epsilon\022\005float\032\005%\027\267\3218\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\027\n\013is_training\022\004bool\032\002(\001\n\325\002\n\024FusedBatchNormGradV2\022\017\n\ny_backprop\"\001T\022\006\n\001x\"\001T\022\t\n\005scale\030\001\022\024\n\017reserve_space_1\"\001U\022\024\n\017reserve_space_2\"\001U\032\017\n\nx_backprop\"\001T\032\023\n\016scale_backprop\"\001U\032\024\n\017offset_backprop\"\001U\032\024\n\017reserve_space_3\"\001U\032\024\n\017reserve_space_4\"\001U\"\022\n\001T\022\004type:\007\n\0052\003\023\016\001\"\020\n\001U\022\004type:\005\n\0032\001\001\"\027\n\007epsilon\022\005float\032\005%\027\267\3218\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\027\n\013is_training\022\004bool\032\002(\001\n\256\002\n\020FusedBatchNormV2\022\006\n\001x\"\001T\022\n\n\005scale\"\001U\022\013\n\006offset\"\001U\022\t\n\004mean\"\001U\022\r\n\010variance\"\001U\032\006\n\001y\"\001T\032\017\n\nbatch_mean\"\001U\032\023\n\016batch_variance\"\001U\032\024\n\017reserve_space_1\"\001U\032\024\n\017reserve_space_2\"\001U\"\022\n\001T\022\004type:\007\n\0052\003\023\016\001\"\020\n\001U\022\004type:\005\n\0032\001\001\"\027\n\007epsilon\022\005float\032\005%\027\267\3218\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\027\n\013is_training\022\004bool\032\002(\001\n\272\001\n\016FusedPadConv2D\022\n\n\005input\"\001T\022\014\n\010paddings\030\003\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\022\n\001T\022\004type:\007\n\0052\003\023\001\002\"&\n\004mode\022\006string:\026\n\024\022\007REFLECT\022\tSYMMETRIC\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\357\001\n\027FusedResizeAndPadConv2D\022\n\n\005input\"\001T\022\010\n\004size\030\003\022\014\n\010paddings\030\003\022\013\n\006filter\"\001T\032\013\n\006output\"\001T\"\022\n\001T\022\004type:\007\n\0052\003\023\001\002\" \n\024resize_align_corners\022\004bool\032\002(\000\"&\n\004mode\022\006string:\026\n\024\022\007REFLECT\022\tSYMMETRIC\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\nW\n\006InTopK\022\017\n\013predictions\030\001\022\014\n\007targets\"\001T\032\r\n\tprecision\030\n\"\010\n\001k\022\003int\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\nW\n\010InTopKV2\022\017\n\013predictions\030\001\022\014\n\007targets\"\001T\022\006\n\001k\"\001T\032\r\n\tprecision\030\n\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\n2\n\006L2Loss\022\006\n\001t\"\001T\032\013\n\006output\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\222\001\n\003LRN\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\027\n\014depth_radius\022\003int\032\002\030\005\"\024\n\004bias\022\005float\032\005%\000\000\200?\"\025\n\005alpha\022\005float\032\005%\000\000\200?\"\024\n\004beta\022\005float\032\005%\000\000\000?\"\026\n\001T\022\004type\032\0020\001:\007\n\0052\003\023\016\001\n\301\001\n\007LRNGrad\022\020\n\013input_grads\"\001T\022\020\n\013input_image\"\001T\022\021\n\014output_image\"\001T\032\013\n\006output\"\001T\"\027\n\014depth_radius\022\003int\032\002\030\005\"\024\n\004bias\022\005float\032\005%\000\000\200?\"\025\n\005alpha\022\005float\032\005%\000\000\200?\"\024\n\004beta\022\005float\032\005%\000\000\000?\"\026\n\001T\022\004type\032\0020\001:\007\n\0052\003\023\016\001\n?\n\nLogSoftmax\022\013\n\006logits\"\001T\032\017\n\nlogsoftmax\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\324\001\n\007MaxPool\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\036\n\001T\022\004type\032\0020\001:\017\n\r2\013\023\016\001\002\003\t\004\005\006\021\013\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\":\n\013data_format\022\006string\032\006\022\004NHWC:\033\n\031\022\004NHWC\022\004NCHW\022\013NCHW_VECT_C\n\300\001\n\tMaxPool3D\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\005\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"\022\n\001T\022\004type:\007\n\0052\003\023\016\001\n\221\002\n\rMaxPool3DGrad\022\024\n\norig_input\"\006TInput\022\025\n\013orig_output\"\006TInput\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\005\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"\026\n\001T\022\004type\032\0020\001:\007\n\0052\003\023\016\001\"\033\n\006TInput\022\004type\032\0020\001:\007\n\0052\003\023\016\001\n\363\001\n\021MaxPool3DGradGrad\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\005\"\030\n\007strides\022\tlist(int)(\0010\005\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"0\n\013data_format\022\006string\032\007\022\005NDHWC:\020\n\016\022\005NDHWC\022\005NCDHW\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\356\001\n\013MaxPoolGrad\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\037\n\001T\022\004type\032\0020\001:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\356\001\n\017MaxPoolGradGrad\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\t\n\004grad\"\001T\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\326\001\n\021MaxPoolGradGradV2\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\t\n\004grad\"\001T\022\t\n\005ksize\030\003\022\013\n\007strides\030\003\032\013\n\006output\"\001T\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\336\001\n\031MaxPoolGradGradWithArgmax\022\n\n\005input\"\001T\022\t\n\004grad\"\001T\022\021\n\006argmax\"\007Targmax\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"\027\n\007Targmax\022\004type:\006\n\0042\002\003\t\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\326\001\n\rMaxPoolGradV2\022\017\n\norig_input\"\001T\022\020\n\013orig_output\"\001T\022\t\n\004grad\"\001T\022\t\n\005ksize\030\003\022\013\n\007strides\030\003\032\013\n\006output\"\001T\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"-\n\013data_format\022\006string\032\006\022\004NHWC:\016\n\014\022\004NHWC\022\004NCHW\"\037\n\001T\022\004type\032\0020\001:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\332\001\n\025MaxPoolGradWithArgmax\022\n\n\005input\"\001T\022\t\n\004grad\"\001T\022\021\n\006argmax\"\007Targmax\032\013\n\006output\"\001T\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"\027\n\007Targmax\022\004type:\006\n\0042\002\003\t\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\274\001\n\tMaxPoolV2\022\n\n\005input\"\001T\022\t\n\005ksize\030\003\022\013\n\007strides\030\003\032\013\n\006output\"\001T\"\036\n\001T\022\004type\032\0020\001:\017\n\r2\013\023\016\001\002\003\t\004\005\006\021\013\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\":\n\013data_format\022\006string\032\006\022\004NHWC:\033\n\031\022\004NHWC\022\004NCHW\022\013NCHW_VECT_C\n\317\001\n\021MaxPoolWithArgmax\022\n\n\005input\"\001T\032\013\n\006output\"\001T\032\021\n\006argmax\"\007Targmax\"\026\n\005ksize\022\tlist(int)(\0010\004\"\030\n\007strides\022\tlist(int)(\0010\004\"\033\n\007Targmax\022\004type\032\0020\t:\006\n\0042\002\003\t\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n^\n\nNthElement\022\n\n\005input\"\001T\022\005\n\001n\030\003\032\013\n\006values\"\001T\"\023\n\007reverse\022\004bool\032\002(\000\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\315\001\n\020QuantizedAvgPool\022\n\n\005input\"\001T\022\r\n\tmin_input\030\001\022\r\n\tmax_input\030\001\032\013\n\006output\"\001T\032\016\n\nmin_output\030\001\032\016\n\nmax_output\030\001\"\024\n\001T\022\004type:\t\n\0072\005\013\014\r\017\020\"\022\n\005ksize\022\tlist(int)\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\231\003\n)QuantizedBatchNormWithGlobalNormalization\022\013\n\001t\"\006Tinput\022\t\n\005t_min\030\001\022\t\n\005t_max\030\001\022\013\n\001m\"\006Tinput\022\t\n\005m_min\030\001\022\t\n\005m_max\030\001\022\013\n\001v\"\006Tinput\022\t\n\005v_min\030\001\022\t\n\005v_max\030\001\022\016\n\004beta\"\006Tinput\022\014\n\010beta_min\030\001\022\014\n\010beta_max\030\001\022\017\n\005gamma\"\006Tinput\022\r\n\tgamma_min\030\001\022\r\n\tgamma_max\030\001\032\022\n\006result\"\010out_type\032\016\n\nresult_min\030\001\032\016\n\nresult_max\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\033\n\010out_type\022\004type:\t\n\0072\005\013\014\r\017\020\"\031\n\020variance_epsilon\022\005float\"!\n\031scale_after_normalization\022\004bool\n\336\001\n\020QuantizedBiasAdd\022\013\n\005input\"\002T1\022\n\n\004bias\"\002T2\022\r\n\tmin_input\030\001\022\r\n\tmax_input\030\001\022\014\n\010min_bias\030\001\022\014\n\010max_bias\030\001\032\022\n\006output\"\010out_type\032\013\n\007min_out\030\001\032\013\n\007max_out\030\001\"\025\n\002T1\022\004type:\t\n\0072\005\013\014\r\017\020\"\025\n\002T2\022\004type:\t\n\0072\005\013\014\r\017\020\"\033\n\010out_type\022\004type:\t\n\0072\005\013\014\r\017\020\n\333\002\n\017QuantizedConv2D\022\017\n\005input\"\006Tinput\022\021\n\006filter\"\007Tfilter\022\r\n\tmin_input\030\001\022\r\n\tmax_input\030\001\022\016\n\nmin_filter\030\001\022\016\n\nmax_filter\030\001\032\022\n\006output\"\010out_type\032\016\n\nmin_output\030\001\032\016\n\nmax_output\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\032\n\007Tfilter\022\004type:\t\n\0072\005\013\014\r\017\020\"\037\n\010out_type\022\004type\032\0020\r:\t\n\0072\005\013\014\r\017\020\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\" \n\tdilations\022\tlist(int)\032\010\n\006\032\004\001\001\001\001\n\315\001\n\020QuantizedMaxPool\022\n\n\005input\"\001T\022\r\n\tmin_input\030\001\022\r\n\tmax_input\030\001\032\013\n\006output\"\001T\032\016\n\nmin_output\030\001\032\016\n\nmax_output\030\001\"\024\n\001T\022\004type:\t\n\0072\005\013\014\r\017\020\"\022\n\005ksize\022\tlist(int)\"\024\n\007strides\022\tlist(int)\"\"\n\007padding\022\006string:\017\n\r\022\004SAME\022\005VALID\n\306\001\n\rQuantizedRelu\022\022\n\010features\"\006Tinput\022\020\n\014min_features\030\001\022\020\n\014max_features\030\001\032\027\n\013activations\"\010out_type\032\023\n\017min_activations\030\001\032\023\n\017max_activations\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\037\n\010out_type\022\004type\032\0020\014:\t\n\0072\005\013\014\r\017\020\n\307\001\n\016QuantizedRelu6\022\022\n\010features\"\006Tinput\022\020\n\014min_features\030\001\022\020\n\014max_features\030\001\032\027\n\013activations\"\010out_type\032\023\n\017min_activations\030\001\032\023\n\017max_activations\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\037\n\010out_type\022\004type\032\0020\014:\t\n\0072\005\013\014\r\017\020\n\326\001\n\016QuantizedReluX\022\022\n\010features\"\006Tinput\022\r\n\tmax_value\030\001\022\020\n\014min_features\030\001\022\020\n\014max_features\030\001\032\027\n\013activations\"\010out_type\032\023\n\017min_activations\030\001\032\023\n\017max_activations\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\037\n\010out_type\022\004type\032\0020\014:\t\n\0072\005\013\014\r\017\020\nE\n\004Relu\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\034\n\001T\022\004type:\021\n\0172\r\001\002\003\004\005\006\t\016\021\023\026\027\013\nE\n\005Relu6\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\nW\n\tRelu6Grad\022\016\n\tgradients\"\001T\022\r\n\010features\"\001T\032\016\n\tbackprops\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\nV\n\010ReluGrad\022\016\n\tgradients\"\001T\022\r\n\010features\"\001T\032\016\n\tbackprops\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n<\n\004Selu\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\nM\n\010SeluGrad\022\016\n\tgradients\"\001T\022\014\n\007outputs\"\001T\032\016\n\tbackprops\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n9\n\007Softmax\022\013\n\006logits\"\001T\032\014\n\007softmax\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\nj\n\035SoftmaxCrossEntropyWithLogits\022\r\n\010features\"\001T\022\013\n\006labels\"\001T\032\t\n\004loss\"\001T\032\r\n\010backprop\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n@\n\010Softplus\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\nR\n\014SoftplusGrad\022\016\n\tgradients\"\001T\022\r\n\010features\"\001T\032\016\n\tbackprops\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n@\n\010Softsign\022\r\n\010features\"\001T\032\020\n\013activations\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\nR\n\014SoftsignGrad\022\016\n\tgradients\"\001T\022\r\n\010features\"\001T\032\016\n\tbackprops\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\n\223\001\n#SparseSoftmaxCrossEntropyWithLogits\022\r\n\010features\"\001T\022\021\n\006labels\"\007Tlabels\032\t\n\004loss\"\001T\032\r\n\010backprop\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\023\016\001\002\"\033\n\007Tlabels\022\004type\032\0020\t:\006\n\0042\002\003\t\n\201\001\n\004TopK\022\n\n\005input\"\001T\032\013\n\006values\"\001T\032\013\n\007indices\030\003\"\n\n\001k\022\003int(\001\"\022\n\006sorted\022\004bool\032\002(\001\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027B\026\010\007\022\022Use TopKV2 instead\nf\n\006TopKV2\022\n\n\005input\"\001T\022\005\n\001k\030\003\032\013\n\006values\"\001T\032\013\n\007indices\030\003\"\022\n\006sorted\022\004bool\032\002(\001\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027")
