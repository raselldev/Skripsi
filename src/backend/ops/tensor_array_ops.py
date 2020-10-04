from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import weakref

from backend import context
from backend.framework import tensor_shape
from backend.framework import ops
from backend.ops import math_ops
from backend.ops import array_ops
from backend.ops import gen_data_flow_ops

class TensorArray(object):
  def __init__(self,
               dtype,
               size=None,
               dynamic_size=None,
               clear_after_read=None,
               tensor_array_name=None,
               handle=None,
               flow=None,
               infer_shape=True,
               element_shape=None,
               colocate_with_first_write_call=True,
               name=None):
    if context.executing_eagerly():
      implementation = _EagerTensorArray
    else:
      implementation = _GraphTensorArray

    self._implementation = implementation(
        dtype,
        size=size,
        dynamic_size=dynamic_size,
        clear_after_read=clear_after_read,
        tensor_array_name=tensor_array_name,
        handle=handle,
        flow=flow,
        infer_shape=infer_shape,
        element_shape=element_shape,
        colocate_with_first_write_call=colocate_with_first_write_call,
        name=name)

    self._implementation.parent = weakref.ref(self)

  @property
  def flow(self):
    return self._implementation._flow

  @property
  def dtype(self):
    return self._implementation._dtype

  @property
  def handle(self):
    return self._implementation._handle

  @property
  def _infer_shape(self):
    return self._implementation._infer_shape

  @_infer_shape.setter
  def _infer_shape(self, infer_shape):
    self._implementation._infer_shape = infer_shape

  @property
  def _element_shape(self):
    return self._implementation._element_shape

  @_element_shape.setter
  def _element_shape(self, element_shape):
    self._implementation._element_shape = element_shape

  @property
  def _colocate_with_first_write_call(self):
    return self._implementation._colocate_with_first_write_call

  @property
  def _colocate_with(self):
    return self._implementation._colocate_with

  @_colocate_with.setter
  def _colocate_with(self, colocate_with):
    self._implementation._colocate_with = colocate_with

  def identity(self):
    return self._implementation.identity()

  def grad(self, source, flow=None, name=None):
    return self._implementation.grad(source, flow=flow, name=name)

  def read(self, index, name=None):
    return self._implementation.read(index, name=name)

  
  def write(self, index, value, name=None):
    return self._implementation.write(index, value, name=name)

  def stack(self, name=None):
    return self._implementation.stack(name=name)

  def gather(self, indices, name=None):
    return self._implementation.gather(indices, name=name)

  def concat(self, name=None):
    return self._implementation.concat(name=name)

  
  def unstack(self, value, name=None):
    return self._implementation.unstack(value, name=name)

  
  def scatter(self, indices, value, name=None):
    return self._implementation.scatter(indices, value, name=name)

  
  def split(self, value, lengths, name=None):
    return self._implementation.split(value, lengths, name=name)

  def size(self, name=None):
    return self._implementation.size(name=name)

  
  def close(self, name=None):
    return self._implementation.close(name=name)

class _GraphTensorArray(object):

  def __init__(self,
               dtype,
               size=None,
               dynamic_size=None,
               clear_after_read=None,
               tensor_array_name=None,
               handle=None,
               flow=None,
               infer_shape=True,
               element_shape=None,
               colocate_with_first_write_call=True,
               name=None):

    if clear_after_read is None:
      clear_after_read = True
    dynamic_size = dynamic_size or False

    self._dtype = dtype
    self._colocate_with_first_write_call = colocate_with_first_write_call
    if colocate_with_first_write_call:
      self._colocate_with = []
    else:
      self._colocate_with = None
    if element_shape is None:
      self._infer_shape = infer_shape
      self._element_shape = []
    else:
      self._infer_shape = True
      self._element_shape = [tensor_shape.TensorShape(element_shape)]
    with ops.name_scope(name, "TensorArray", [handle, size, flow]) as scope:
      if handle is not None:
        self._handle = handle
        if flow is None:
          raise ValueError("flow must not be None if handle is not None.")
        self._flow = flow
      else:
        def create():
          return gen_data_flow_ops.tensor_array_v3(
              dtype=dtype,
              size=size,
              element_shape=element_shape,
              identical_element_shapes=infer_shape,
              dynamic_size=dynamic_size,
              clear_after_read=clear_after_read,
              tensor_array_name=tensor_array_name,
              name=scope)
        if colocate_with_first_write_call:
          with ops.device(None), ops.colocate_with(None, ignore_existing=True):
            self._handle, self._flow = create()
        else:
          self._handle, self._flow = create()

  @property
  def flow(self):
    return self._flow

  @property
  def dtype(self):
    return self._dtype

  @property
  def handle(self):
    return self._handle

  def _merge_element_shape(self, shape):

    if self._element_shape:
      if not shape.is_compatible_with(self._element_shape[0]):
        raise ValueError(
            "Inconsistent shapes: saw %s but expected %s "
            "(and infer_shape=True)" % (shape, self._element_shape[0]))
      self._element_shape[0] = self._element_shape[0].merge_with(shape)
    else:
      self._element_shape.append(shape)

  @contextlib.contextmanager
  def _maybe_colocate_with(self, value):
    if not self._colocate_with_first_write_call:
      yield
    else:
      if not self._colocate_with:
        self._colocate_with.append(value)
      with ops.colocate_with(self._colocate_with[0]):
        yield

  def identity(self):
    
    flow = array_ops.identity(self._flow)
    ta = TensorArray(
        dtype=self._dtype, handle=self._handle, flow=flow,
        infer_shape=self._infer_shape,
        colocate_with_first_write_call=self._colocate_with_first_write_call)
    ta._element_shape = self._element_shape
    ta._colocate_with = self._colocate_with
    return ta

  def grad(self, source, flow=None, name=None):
    if flow is None:
      flow = self.flow
    with ops.name_scope(name, "TensorArrayGrad", [self._handle]):
      with ops.colocate_with(self._handle):
        g_handle, unused_flow = gen_data_flow_ops.tensor_array_grad_v3(
            handle=self._handle, source=source, flow_in=flow, name=name)
        with ops.control_dependencies([g_handle]):
          flow = array_ops.identity(flow, name="gradient_flow")
        g = TensorArray(
            dtype=self._dtype,
            handle=g_handle,
            flow=flow,
            infer_shape=self._infer_shape,
            colocate_with_first_write_call=False)
        g._element_shape = self._element_shape
        return g

  def read(self, index, name=None):
    
    value = gen_data_flow_ops.tensor_array_read_v3(
        handle=self._handle,
        index=index,
        flow_in=self._flow,
        dtype=self._dtype,
        name=name)
    if self._element_shape:
      value.set_shape(self._element_shape[0].dims)
    return value

  def write(self, index, value, name=None):
    
    with ops.name_scope(name, "TensorArrayWrite", [self._handle, index, value]):
      value = ops.convert_to_tensor(value, name="value")
      if self._infer_shape:
        self._merge_element_shape(value.shape)
      with self._maybe_colocate_with(value):
        flow_out = gen_data_flow_ops.tensor_array_write_v3(
            handle=self._handle,
            index=index,
            value=value,
            flow_in=self._flow,
            name=name)
      ta = TensorArray(
          dtype=self._dtype, handle=self._handle, flow=flow_out,
          colocate_with_first_write_call=self._colocate_with_first_write_call)
      ta._infer_shape = self._infer_shape
      ta._element_shape = self._element_shape
      ta._colocate_with = self._colocate_with
      return ta

  def stack(self, name=None):
    
    with ops.colocate_with(self._handle):
      with ops.name_scope(name, "TensorArrayStack", [self._handle]):
        return self.gather(math_ops.range(0, self.size()), name=name)

  def gather(self, indices, name=None):
    
    if self._element_shape:
      element_shape = self._element_shape[0]
    else:
      element_shape = tensor_shape.TensorShape(None)
    value = gen_data_flow_ops.tensor_array_gather_v3(
        handle=self._handle,
        indices=indices,
        flow_in=self._flow,
        dtype=self._dtype,
        name=name,
        element_shape=element_shape)
    if self._element_shape and self._element_shape[0].dims is not None:
      value.set_shape([None] + self._element_shape[0].dims)
    return value

  def concat(self, name=None):
    
    if self._element_shape and self._element_shape[0].dims is not None:
      element_shape_except0 = (
          tensor_shape.TensorShape(self._element_shape[0].dims[1:]))
    else:
      element_shape_except0 = tensor_shape.TensorShape(None)
    value, _ = gen_data_flow_ops.tensor_array_concat_v3(
        handle=self._handle,
        flow_in=self._flow,
        dtype=self._dtype,
        name=name,
        element_shape_except0=element_shape_except0)
    if self._element_shape and self._element_shape[0].dims is not None:
      value.set_shape([None] + self._element_shape[0].dims[1:])
    return value

  
  def unstack(self, value, name=None):
    with ops.name_scope(name, "TensorArrayUnstack", [self._handle, value]):
      num_elements = array_ops.shape(value)[0]
      return self.scatter(
          indices=math_ops.range(0, num_elements), value=value, name=name)

  
  def scatter(self, indices, value, name=None):
    with ops.name_scope(name, "TensorArrayScatter",
                        [self._handle, value, indices]):
      value = ops.convert_to_tensor(value, name="value")
      if self._infer_shape and not context.executing_eagerly():
        self._merge_element_shape(value.shape[1:])
      with self._maybe_colocate_with(value):
        flow_out = gen_data_flow_ops.tensor_array_scatter_v3(
            handle=self._handle,
            indices=indices,
            value=value,
            flow_in=self._flow,
            name=name)
      ta = TensorArray(
          dtype=self._dtype, handle=self._handle, flow=flow_out,
          colocate_with_first_write_call=self._colocate_with_first_write_call)
      ta._infer_shape = self._infer_shape
      ta._element_shape = self._element_shape
      ta._colocate_with = self._colocate_with
      return ta

  def split(self, value, lengths, name=None):
    with ops.name_scope(name, "TensorArraySplit",
                        [self._handle, value, lengths]):
      value = ops.convert_to_tensor(value, name="value")
      with self._maybe_colocate_with(value):
        lengths_64 = math_ops.to_int64(lengths)
        if self._infer_shape and not context.executing_eagerly():
          clengths = tensor_util.constant_value(lengths_64)
          if value.shape.dims is not None:
            if clengths is not None and clengths.max() == clengths.min():
              self._merge_element_shape(
                  tensor_shape.TensorShape([clengths[0]]).concatenate(
                      value.shape[1:]))
        flow_out = gen_data_flow_ops.tensor_array_split_v3(
            handle=self._handle,
            value=value,
            lengths=lengths_64,
            flow_in=self._flow,
            name=name)
      ta = TensorArray(
          dtype=self._dtype, handle=self._handle, flow=flow_out,
          colocate_with_first_write_call=self._colocate_with_first_write_call)
      ta._infer_shape = self._infer_shape
      ta._element_shape = self._element_shape
      ta._colocate_with = self._colocate_with
      return ta

  def size(self, name=None):
    return gen_data_flow_ops.tensor_array_size_v3(
        handle=self._handle, flow_in=self.flow, name=name)

  def close(self, name=None):
    return gen_data_flow_ops.tensor_array_close_v3(
        handle=self._handle, name=name)
