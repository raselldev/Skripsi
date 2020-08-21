from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import re
import sys
import threading
import contextlib

import numpy as np
import six
from six.moves import xrange


from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tensorflow as c_api
from tensorflow.python import context
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import errors_impl as errors
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import traceable_stack
from tensorflow.python.framework import registry
from tensorflow.python.ops import control_flow_util
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_stack
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import lock_util
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export


# Temporary global switches determining if we should enable the work-in-progress
# calls to the C API. These will be removed once all functionality is supported.
_USE_C_API = True
_USE_C_SHAPES = True



@contextlib.contextmanager
def stop_recording():
  try:
    pywrap_tensorflow.TFE_Py_TapeSetStopOnThread()
    yield
  finally:
    pywrap_tensorflow.TFE_Py_TapeSetRestartOnThread()



def tensor_id(tensor):
  return tensor._id  


class _UserDeviceSpec(object):
  def __init__(self, device_name_or_function):
    self._device_name_or_function = device_name_or_function

    self.display_name = str(self._device_name_or_function)
    if callable(self._device_name_or_function):
      dev_func = self._device_name_or_function
      func_name = function_utils.get_func_name(dev_func)
      func_code = function_utils.get_func_code(dev_func)
      if func_code:
        fname = func_code.co_filename
        lineno = func_code.co_firstlineno
      else:
        fname = "unknown"
        lineno = -1
      self.display_name = "%s<%s, %d>" % (func_name, fname, lineno)

    self.function = self._device_name_or_function
    if not (self._device_name_or_function is None or
            callable(self._device_name_or_function)):
      self.function = pydev.merge_device(self._device_name_or_function)



def _override_helper(clazz_object, operator, func):
  existing = getattr(clazz_object, operator, None)
  if existing is not None:
    # Check to see if this is a default method-wrapper or slot wrapper which
    # will be true for the comparison operators.
    if not isinstance(existing, type(object.__lt__)):
      raise ValueError("operator %s cannot be overwritten again on class %s." %
                       (operator, clazz_object))
  if operator not in Tensor.OVERLOADABLE_OPERATORS:
    raise ValueError("Overriding %s is disallowed" % operator)
  setattr(clazz_object, operator, func)


def _as_graph_element(obj):
  conv_fn = getattr(obj, "_as_graph_element", None)
  if conv_fn and callable(conv_fn):
    return conv_fn()
  return None


_TENSOR_LIKE_TYPES = tuple()


def is_dense_tensor_like(t):
  return isinstance(t, _TENSOR_LIKE_TYPES)


def register_dense_tensor_like_type(tensor_type):
  try:
    if not isinstance(tensor_type.name, property):
      raise TypeError("Type %s does not define a `name` property" %
                      tensor_type.__name__)
  except AttributeError:
    raise TypeError("Type %s does not define a `name` property" %
                    tensor_type.__name__)
  try:
    if not isinstance(tensor_type.dtype, property):
      raise TypeError("Type %s does not define a `dtype` property" %
                      tensor_type.__name__)
  except AttributeError:
    raise TypeError("Type %s does not define a `dtype` property" %
                    tensor_type.__name__)
  # We expect this list to be small, so choose quadratic complexity
  # for registration, so that we have a tuple that can be used for
  # more efficient `isinstance` checks later.
  global _TENSOR_LIKE_TYPES
  _TENSOR_LIKE_TYPES = tuple(list(_TENSOR_LIKE_TYPES) + [tensor_type])


def uid():
  return c_api.TFE_Py_UID()


def numpy_text(tensor, is_repr=False):
  if tensor.dtype.is_numpy_compatible:
    text = repr(tensor.numpy()) if is_repr else str(tensor.numpy())
  else:
    text = "<unprintable>"
  if "\n" in text:
    text = "\n" + text
  return text


class _TensorLike(object):
  pass


@tf_export("Tensor")
class Tensor(object):
  # List of Python operators that we allow to override.
  OVERLOADABLE_OPERATORS = {
      # Binary.
      "__add__",
      "__radd__",
      "__sub__",
      "__rsub__",
      "__mul__",
      "__rmul__",
      "__div__",
      "__rdiv__",
      "__truediv__",
      "__rtruediv__",
      "__floordiv__",
      "__rfloordiv__",
      "__mod__",
      "__rmod__",
      "__lt__",
      "__le__",
      "__gt__",
      "__ge__",
      "__and__",
      "__rand__",
      "__or__",
      "__ror__",
      "__xor__",
      "__rxor__",
      "__getitem__",
      "__pow__",
      "__rpow__",
      # Unary.
      "__invert__",
      "__neg__",
      "__abs__",
      "__matmul__",
      "__rmatmul__"
  }

  def __init__(self, op, value_index, dtype):
    if not isinstance(op, Operation):
      raise TypeError("op needs to be an Operation: %s" % op)
    self._op = op
    self._value_index = value_index
    self._dtype = dtypes.as_dtype(dtype)

    # This will be set by self.shape().
    self._shape_val = None

    # List of operations that use this Tensor as input.  We maintain this list
    # to easily navigate a computation graph.
    self._consumers = []

    if not _USE_C_SHAPES:
      # Attributes used for C++ shape inference. Not inspected, only forwarded.
      # If set, will be a HandleData object from cpp_shape_inference.proto.
      self._handle_data = None

    self._id = uid()

  @property
  def op(self):
    return self._op

  @property
  def dtype(self):
    return self._dtype

  @property
  def graph(self):
    return self._op.graph

  @property
  def name(self):
    if not self._op.name:
      raise ValueError("Operation was not named: %s" % self._op)
    return "%s:%d" % (self._op.name, self._value_index)

  @property
  def device(self):
    return self._op.device

  @property
  def shape(self):
    if self._shape_val is None:
      if _USE_C_SHAPES:
        self._shape_val = self._c_api_shape()
      else:
        # Call set_shape_and_handle_data_for_outputs in topological order on all
        # ops that are needed to compute self.op's shape. We do this instead of
        # having set_shape_and_handle_data_for_outputs recursively call
        # Operation.shape on self.op.inputs to overflowing the call stack.
        need_shapes = self._get_input_ops_without_shapes(self.op)
        need_shapes.sort(key=lambda op: op._id)
        for op in need_shapes:
          set_shape_and_handle_data_for_outputs(op)
    return self._shape_val

  def _get_input_ops_without_shapes(self, target_op):
    result = []
    stack = [self._op]
    visited = set()
    while stack:
      op = stack.pop()
      if op in visited: continue
      result.append(op)
      stack.extend(t.op for t in op.inputs if t._shape_val is None)
      visited.add(op)
    return result

  def _c_api_shape(self):
    c_graph = self._op._graph._c_graph  
    shape_vector, unknown_shape = c_api.TF_GraphGetTensorShapeHelper(
        c_graph, self._as_tf_output())
    if unknown_shape:
      return tensor_shape.unknown_shape()
    else:
      shape_vector = [None if d == -1 else d for d in shape_vector]
      return tensor_shape.TensorShape(shape_vector)

  @property
  def _shape(self):
    logging.warning("Tensor._shape is private, use Tensor.shape "
                    "instead. Tensor._shape will eventually be removed.")
    return self.shape

  @_shape.setter
  def _shape(self, value):
    raise ValueError(
        "Tensor._shape cannot be assigned, use Tensor.set_shape instead.")

  def __iter__(self):
    if not context.executing_eagerly():
      raise TypeError(
          "Tensor objects are only iterable when eager execution is "
          "enabled. To iterate over this tensor use tf.map_fn.")
    shape = self._shape_tuple()
    if shape is None:
      raise TypeError("Cannot iterate over a tensor with unknown shape.")
    if not shape:
      raise TypeError("Cannot iterate over a scalar tensor.")
    if shape[0] is None:
      raise TypeError(
          "Cannot iterate over a tensor with unknown first dimension.")
    for i in xrange(shape[0]):
      yield self[i]

  def _shape_as_list(self):
    if self.shape.ndims is not None:
      return [dim.value for dim in self.shape.dims]
    else:
      return None

  def _shape_tuple(self):
    shape = self._shape_as_list()
    if shape is None:
      return None
    return tuple(shape)

  def _rank(self):
    return self.shape.ndims

  def get_shape(self):
    return self.shape

  def set_shape(self, shape):
    if _USE_C_SHAPES:  
      # Reset cached shape.
      self._shape_val = None
    else:
      self._shape_val = self.shape.merge_with(shape)

    # Update C shape even if _USE_C_SHAPES = False, since we still want
    # set_shape to be reflected in the C API graph for when we run it.
    if not isinstance(shape, tensor_shape.TensorShape):
      shape = tensor_shape.TensorShape(shape)
    dim_list = []
    if shape.dims is None:
      unknown_shape = True
    else:
      unknown_shape = False
      for dim in shape.dims:
        if dim.value is None:
          dim_list.append(-1)
        else:
          dim_list.append(dim.value)
    try:
      c_api.TF_GraphSetTensorShape_wrapper(
          self._op._graph._c_graph,  
          self._as_tf_output(),
          dim_list,
          unknown_shape)
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(str(e))

  @property
  def value_index(self):
    return self._value_index

  def consumers(self):
    consumer_names = c_api.TF_OperationOutputConsumers_wrapper(
        self._as_tf_output())
    
    return [
        self.graph._get_operation_by_name_unsafe(name)
        for name in consumer_names
    ]
    # pylint: enable=protected-access

  def _as_node_def_input(self):
    if not self._op.name:
      raise ValueError("Operation was not named: %s" % self._op)
    if self._value_index == 0:
      return self._op.name
    else:
      return "%s:%d" % (self._op.name, self._value_index)

  def _as_tf_output(self):
    
    return c_api_util.tf_output(self.op._c_op, self.value_index)
    # pylint: enable=protected-access

  def __str__(self):
    return "Tensor(\"%s\"%s%s%s)" % (
        self.name, (", shape=%s" % self.get_shape())
        if self.get_shape().ndims is not None else "",
        (", dtype=%s" % self._dtype.name)
        if self._dtype else "", (", device=%s" % self.device)
        if self.device else "")

  def __repr__(self):
    return "<tf.Tensor '%s' shape=%s dtype=%s>" % (self.name, self.get_shape(),
                                                   self._dtype.name)

  def __hash__(self):
    return id(self)

  def __eq__(self, other):
    return id(self) == id(other)

  def __copy__(self):
    if self._shape_val is None:
      set_shape_and_handle_data_for_outputs(self.op)
    cls = self.__class__
    result = cls.__new__(cls)
    result.__dict__.update(self.__dict__)
    return result
  __array_priority__ = 100

  @staticmethod
  def _override_operator(operator, func):
    _override_helper(Tensor, operator, func)

  def __bool__(self):
    raise TypeError("Using a `tf.Tensor` as a Python `bool` is not allowed. "
                    "Use `if t is not None:` instead of `if t:` to test if a "
                    "tensor is defined, and use TensorFlow ops such as "
                    "tf.cond to execute subgraphs conditioned on the value of "
                    "a tensor.")

  def __nonzero__(self):
    raise TypeError("Using a `tf.Tensor` as a Python `bool` is not allowed. "
                    "Use `if t is not None:` instead of `if t:` to test if a "
                    "tensor is defined, and use TensorFlow ops such as "
                    "tf.cond to execute subgraphs conditioned on the value of "
                    "a tensor.")

  def eval(self, feed_dict=None, session=None):
    return _eval_using_default_session(self, feed_dict, self.graph, session)


class _EagerTensorBase(Tensor):
  @property
  def dtype(self):
    # Note: using the intern table directly here as this is
    # performance-sensitive in some models.
    return dtypes._INTERN_TABLE[self._datatype_enum()]  

  def numpy(self):
    if self.dtype == dtypes.resource:
      raise ValueError("Resource handles are not convertible to numpy.")
    return self._cpu_nograd()._numpy()  

  # __int__, __float__ and __index__ may copy the tensor to CPU and
  # only work for scalars; values are cast as per numpy.
  def __int__(self):
    return int(self.numpy())

  def __float__(self):
    return float(self.numpy())

  def __index__(self):
    return int(self.numpy())

  def __array__(self, dtype=None):
    return np.array(self.numpy(), dtype=dtype)

  def __format__(self, format_spec):
    return self.numpy().__format__(format_spec)

  def __reduce__(self):
    return (convert_to_tensor, (self.numpy(),))

  def _numpy(self):
    raise NotImplementedError()

  def __copy__(self):
    # Eager Tensors are immutable so it's safe to return themselves as a copy.
    return self

  def __deepcopy__(self, memo):
    # Eager Tensors are immutable so it's safe to return themselves as a copy.
    del memo
    return self

  def _datatype_enum(self):
    raise NotImplementedError()

  def _shape_tuple(self):
    raise NotImplementedError()

  def _rank(self):
    raise NotImplementedError()

  def _num_elements(self):
    raise NotImplementedError()

  def _copy_to_device(self, context, device):  # pylint: disable=redefined-outer-name
    raise NotImplementedError()

  def __str__(self):
    return "tf.Tensor(%s, shape=%s, dtype=%s)" % (numpy_text(self),
                                                  self.shape,
                                                  self.dtype.name)

  def __repr__(self):
    return "<tf.Tensor: id=%s, shape=%s, dtype=%s, numpy=%s>" % (
        self._id, self.shape, self.dtype.name, numpy_text(self, is_repr=True))

  @staticmethod
  def _override_operator(name, func):
    setattr(_EagerTensorBase, name, func)

  def _copy_nograd(self, ctx=None, device_name=None):
    if ctx is None:
      ctx = context.context()
    if device_name is None:
      device_name = ctx.device_name
    
    try:
      new_tensor = self._copy_to_device(context=ctx._handle, device=device_name)
    except core._NotOkStatusException as e:
      six.raise_from(core._status_to_exception(e.code, e.message), None)
    return new_tensor

  def _copy(self, ctx=None, device_name=None):
    new_tensor = self._copy_nograd(ctx, device_name)
    # Record the copy on tape and define backprop copy as well.
    if context.executing_eagerly():
      self_device = self.device
      def grad_fun(dresult):
        return [dresult._copy(device_name=self_device)]
      tape.record_operation("_copy", [new_tensor], [self], grad_fun)
    return new_tensor
    # pylint: enable=protected-access

  @property
  def shape(self):
    if self._tensor_shape is None:  # pylint: disable=access-member-before-definition
      # `_tensor_shape` is declared and defined in the definition of
      # `EagerTensor`, in C.
      self._tensor_shape = tensor_shape.TensorShape(self._shape_tuple())
    return self._tensor_shape

  def get_shape(self):
    return self.shape

  def _shape_as_list(self):
    return list(self._shape_tuple())

  @property
  def ndim(self):
    return self.shape.ndims

  def _cpu_nograd(self):
    return self._copy_nograd(context.context(), "CPU:0")

  def cpu(self):
    return self._copy(context.context(), "CPU:0")

  def gpu(self, gpu_index=0):
    return self._copy(context.context(), "GPU:" + str(gpu_index))

  def __bool__(self):
    if self._shape_tuple() != ():  # pylint: disable=g-explicit-bool-comparison
      raise ValueError(
          "Non-scalar tensor %s cannot be converted to boolean." % repr(self))
    if self.dtype != dtypes.bool:
      raise ValueError(
          "Non-boolean tensor %s cannot be converted to boolean." % repr(self))
    return bool(self.cpu().numpy())

  def __nonzero__(self):
    return self.__bool__()

  def set_shape(self, shape):
    if not self.shape.is_compatible_with(shape):
      raise ValueError(
          "Tensor's shape %s is not compatible with supplied shape %s" %
          (self.shape, shape))

  # Methods not supported / implemented for Eager Tensors.
  @property
  def op(self):
    raise AttributeError(
        "Tensor.op is meaningless when eager execution is enabled.")

  @property
  def graph(self):
    raise AttributeError(
        "Tensor.graph is meaningless when eager execution is enabled.")

  @property
  def name(self):
    raise AttributeError(
        "Tensor.name is meaningless when eager execution is enabled.")

  @property
  def value_index(self):
    raise AttributeError(
        "Tensor.value_index is meaningless when eager execution is enabled.")

  def consumers(self):
    raise NotImplementedError(
        "Tensor.consumers is meaningless when eager execution is enabled.")

  def _add_consumer(self, consumer):
    raise NotImplementedError(
        "_add_consumer not supported when eager execution is enabled.")

  def _as_node_def_input(self):
    raise NotImplementedError(
        "_as_node_def_input not supported when eager execution is enabled.")

  def _as_tf_output(self):
    raise NotImplementedError(
        "_as_tf_output not supported when eager execution is enabled.")

  def eval(self, feed_dict=None, session=None):
    raise NotImplementedError(
        "eval is not supported when eager execution is enabled, "
        "is .numpy() what you're looking for?"
    )


# This call creates an EagerTensor class, as a subclass of _EagerTensorBase, and
# registers it with the current module.
EagerTensor = c_api.TFE_Py_InitEagerTensor(_EagerTensorBase)


def _TensorTensorConversionFunction(t, dtype=None, name=None, as_ref=False):
  _ = name, as_ref
  if dtype and not dtype.is_compatible_with(t.dtype):
    raise ValueError(
        "Tensor conversion requested dtype %s for Tensor with dtype %s: %r" %
        (dtype.name, t.dtype.name, str(t)))
  return t


_tensor_conversion_func_registry = {
    0: [(Tensor, _TensorTensorConversionFunction)]
}
_tensor_conversion_func_cache = {}
_tensor_conversion_func_lock = threading.Lock()
register_dense_tensor_like_type(Tensor)


@tf_export("convert_to_tensor")
def convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None):
  return internal_convert_to_tensor(
      value=value,
      dtype=dtype,
      name=name,
      preferred_dtype=preferred_dtype,
      as_ref=False)


def _error_prefix(name):
  return "" if name is None else "%s: " % name


def internal_convert_to_tensor(value,
                               dtype=None,
                               name=None,
                               as_ref=False,
                               preferred_dtype=None,
                               ctx=None):
  if ctx is None: ctx = context.context()
  if isinstance(value, EagerTensor):
    if ctx.executing_eagerly():
      return value
    else:
      graph = get_default_graph()
      if not graph.building_function:
        raise RuntimeError("Attempting to capture an EagerTensor without "
                           "building a function.")
      return graph.capture(value, name=name)

  if dtype is not None:
    dtype = dtypes.as_dtype(dtype)
  unwrapped_type = type(value)
  conversion_func_list = _tensor_conversion_func_cache.get(unwrapped_type, None)
  if conversion_func_list is None:
    with _tensor_conversion_func_lock:
      conversion_func_list = []
      for _, funcs_at_priority in sorted(
          _tensor_conversion_func_registry.items()):
        for base_type, conversion_func in funcs_at_priority:
          if isinstance(value, base_type):
            conversion_func_list.append((base_type, conversion_func))
      _tensor_conversion_func_cache[unwrapped_type] = conversion_func_list

  for base_type, conversion_func in conversion_func_list:
    # If dtype is None but preferred_dtype is not None, we try to
    # cast to preferred_dtype first.
    ret = None
    if dtype is None and preferred_dtype is not None:
      try:
        ret = conversion_func(
            value, dtype=preferred_dtype, name=name, as_ref=as_ref)
      except (TypeError, ValueError, errors.UnimplementedError,
              errors.InvalidArgumentError):
        # Could not coerce the conversion to use the preferred dtype.
        ret = None

      if ret is not None and ret is not NotImplemented:
        if (ret.dtype.base_dtype !=
            dtypes.as_dtype(preferred_dtype).base_dtype):
          raise TypeError("convert_to_tensor did not convert to "
                          "the preferred dtype: %s vs %s " %
                          (ret.dtype.base_dtype,
                           dtypes.as_dtype(preferred_dtype).base_dtype))

    if ret is None:
      ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)

    if ret is NotImplemented:
      continue

    if not isinstance(ret, Tensor):
      raise RuntimeError(
          "%sConversion function %r for type %s returned non-Tensor: %r" %
          (_error_prefix(name), conversion_func, base_type, ret))
    if dtype and not dtype.is_compatible_with(ret.dtype):
      raise RuntimeError(
          "%sConversion function %r for type %s returned incompatible "
          "dtype: requested = %s, actual = %s" %
          (_error_prefix(name), conversion_func, base_type, dtype.name,
           ret.dtype.name))
    return ret
  raise TypeError("%sCannot convert %r with type %s to Tensor: "
                  "no conversion function registered." %
                  (_error_prefix(name), value, unwrapped_type))


def internal_convert_n_to_tensor(values,
                                 dtype=None,
                                 name=None,
                                 as_ref=False,
                                 preferred_dtype=None,
                                 ctx=None):
  if not isinstance(values, collections.Sequence):
    raise TypeError("values must be a list.")
  ret = []
  if ctx is None: ctx = context.context()
  for i, value in enumerate(values):
    n = None if name is None else "%s_%d" % (name, i)
    ret.append(
        internal_convert_to_tensor(
            value,
            dtype=dtype,
            name=n,
            as_ref=as_ref,
            preferred_dtype=preferred_dtype,
            ctx=ctx))
  return ret


def convert_n_to_tensor(values, dtype=None, name=None, preferred_dtype=None):
  return internal_convert_n_to_tensor(
      values=values,
      dtype=dtype,
      name=name,
      preferred_dtype=preferred_dtype,
      as_ref=False)


@tf_export("convert_to_tensor_or_indexed_slices")
def convert_to_tensor_or_indexed_slices(value, dtype=None, name=None):
  return internal_convert_to_tensor_or_indexed_slices(
      value=value, dtype=dtype, name=name, as_ref=False)


def internal_convert_to_tensor_or_indexed_slices(value,
                                                 dtype=None,
                                                 name=None,
                                                 as_ref=False):
  if isinstance(value, EagerTensor) and not context.executing_eagerly():
    return internal_convert_to_tensor(
        value, dtype=dtype, name=name, as_ref=as_ref)
  elif isinstance(value, _TensorLike):
    if dtype and not dtypes.as_dtype(dtype).is_compatible_with(value.dtype):
      raise ValueError(
          "Tensor conversion requested dtype %s for Tensor with dtype %s: %r" %
          (dtypes.as_dtype(dtype).name, value.dtype.name, str(value)))
    return value
  else:
    return internal_convert_to_tensor(
        value, dtype=dtype, name=name, as_ref=as_ref)


def internal_convert_n_to_tensor_or_indexed_slices(values,
                                                   dtype=None,
                                                   name=None,
                                                   as_ref=False):
  if not isinstance(values, collections.Sequence):
    raise TypeError("values must be a list.")
  ret = []
  for i, value in enumerate(values):
    if value is None:
      ret.append(value)
    else:
      n = None if name is None else "%s_%d" % (name, i)
      ret.append(
          internal_convert_to_tensor_or_indexed_slices(
              value, dtype=dtype, name=n, as_ref=as_ref))
  return ret


def convert_n_to_tensor_or_indexed_slices(values, dtype=None, name=None):
  return internal_convert_n_to_tensor_or_indexed_slices(
      values=values, dtype=dtype, name=name, as_ref=False)


# TODO(josh11b): Add ctx argument to conversion_func() signature.
@tf_export("register_tensor_conversion_function")
def register_tensor_conversion_function(base_type,
                                        conversion_func,
                                        priority=100):
  global _tensor_conversion_func_cache
  with _tensor_conversion_func_lock:
    if not (isinstance(base_type, type) or
            (isinstance(base_type, tuple) and
             all(isinstance(x, type) for x in base_type))):
      raise TypeError("base_type must be a type or a tuple of types.")
    if not callable(conversion_func):
      raise TypeError("conversion_func must be callable.")

    
    if context._context is not None and context.executing_eagerly(
    ) and isinstance(base_type, six.integer_types + (
        float,
        np.ndarray,
    )):
      raise TypeError(
          "Cannot register conversions for numpy arrays, python number types "
          "when executing eagerly.")

    try:
      funcs_at_priority = _tensor_conversion_func_registry[priority]
    except KeyError:
      funcs_at_priority = []
      _tensor_conversion_func_registry[priority] = funcs_at_priority
    funcs_at_priority.append((base_type, conversion_func))
    _tensor_conversion_func_cache = {}


@tf_export("IndexedSlices")
class IndexedSlices(object):
  def __init__(self, values, indices, dense_shape=None):
    """Creates an `IndexedSlices`."""
    _get_graph_from_inputs([values, indices, dense_shape])
    self._values = values
    self._indices = indices
    self._dense_shape = dense_shape

  @property
  def values(self):
    return self._values

  @property
  def indices(self):
    return self._indices

  @property
  def dense_shape(self):
    return self._dense_shape

  @property
  def name(self):
    return self.values.name

  @property
  def device(self):
    return self.values.device

  @property
  def op(self):
    return self.values.op

  @property
  def dtype(self):
    return self.values.dtype

  @property
  def graph(self):
    return self._values.graph

  def __str__(self):
    return "IndexedSlices(indices=%s, values=%s%s)" % (
        self._indices, self._values, (", dense_shape=%s" % self._dense_shape)
        if self._dense_shape is not None else "")

  def __neg__(self):
    return IndexedSlices(-self.values, self.indices, self.dense_shape)


IndexedSlicesValue = collections.namedtuple(
    "IndexedSlicesValue", ["values", "indices", "dense_shape"])


def _device_string(dev_spec):
  if isinstance(dev_spec, pydev.DeviceSpec):
    return dev_spec.to_string()
  else:
    return dev_spec


def _NodeDef(op_type, name, device=None, attrs=None):  # pylint: disable=redefined-outer-name
  node_def = node_def_pb2.NodeDef()
  node_def.op = compat.as_bytes(op_type)
  node_def.name = compat.as_bytes(name)
  if attrs is not None:
    for k, v in six.iteritems(attrs):
      node_def.attr[k].CopyFrom(v)
  if device is not None:
    if callable(device):
      node_def.device = device(node_def)
    else:
      node_def.device = _device_string(device)
  return node_def


# Copied from core/framework/node_def_util.cc
# TODO(mrry,josh11b): Consolidate this validation in C++ code.
_VALID_OP_NAME_REGEX = re.compile("^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$")
_VALID_SCOPE_NAME_REGEX = re.compile("^[A-Za-z0-9_.\\-/]*$")


def _create_c_op(graph, node_def, inputs, control_inputs):
  op_desc = c_api.TF_NewOperation(graph._c_graph,
                                  compat.as_str(node_def.op),
                                  compat.as_str(node_def.name))
  # Add inputs
  for op_input in inputs:
    if isinstance(op_input, (list, tuple)):
      c_api.TF_AddInputList(op_desc, [t._as_tf_output() for t in op_input])
    else:
      c_api.TF_AddInput(op_desc, op_input._as_tf_output())

  # Add control inputs
  for control_input in control_inputs:
    c_api.TF_AddControlInput(op_desc, control_input._c_op)
  # pylint: enable=protected-access

  # Add attrs
  for name, attr_value in node_def.attr.items():
    serialized = attr_value.SerializeToString()
    # TODO(skyewm): this creates and deletes a new TF_Status for every attr.
    # It might be worth creating a convenient way to re-use the same status.
    c_api.TF_SetAttrValueProto(op_desc, compat.as_str(name), serialized)

  try:
    c_op = c_api.TF_FinishOperation(op_desc)
  except errors.InvalidArgumentError as e:
    # Convert to ValueError for backwards compatibility.
    raise ValueError(str(e))

  return c_op


@tf_export("Operation")
class Operation(object):
  def __init__(self,
               node_def,
               g,
               inputs=None,
               output_types=None,
               control_inputs=None,
               input_types=None,
               original_op=None,
               op_def=None):

    if isinstance(node_def, node_def_pb2.NodeDef):
      if node_def.ByteSize() >= (1 << 31) or node_def.ByteSize() < 0:
        raise ValueError(
            "Cannot create a tensor proto whose content is larger than 2GB.")
      if not _VALID_OP_NAME_REGEX.match(node_def.name):
        raise ValueError("'%s' is not a valid node name" % node_def.name)
      c_op = None
    elif type(node_def).__name__ == "SwigPyObject":
      assert inputs is None
      assert output_types is None
      assert control_inputs is None
      assert input_types is None
      assert original_op is None
      assert op_def is None
      c_op = node_def
    else:
      raise TypeError("node_def needs to be a NodeDef: %s" % node_def)

    if not isinstance(g, Graph):
      raise TypeError("g needs to be a Graph: %s" % g)
    self._graph = g

    if inputs is None:
      inputs = []
    elif not isinstance(inputs, list):
      raise TypeError("inputs needs to be a list of Tensors: %s" % inputs)
    for a in inputs:
      if not isinstance(a, Tensor):
        raise TypeError("input needs to be a Tensor: %s" % a)
    if input_types is None:
      input_types = [i.dtype.base_dtype for i in inputs]
    else:
      if not all(
          x.is_compatible_with(i.dtype)
          for i, x in zip(inputs, input_types)):
        raise TypeError("In op '%s', input types (%s) are not compatible "
                        "with expected types (%s)" %
                        (node_def.name, [i.dtype for i in inputs],
                         input_types))

    # Build the list of control inputs.
    control_input_ops = []
    if control_inputs:
      for c in control_inputs:
        control_op = None
        if isinstance(c, Operation):
          control_op = c
        elif isinstance(c, (Tensor, IndexedSlices)):
          control_op = c.op
        else:
          raise TypeError("Control input must be an Operation, "
                          "a Tensor, or IndexedSlices: %s" % c)
        control_input_ops.append(control_op)

    # This will be set by self.inputs.
    self._inputs_val = None

    
    self._id_value = self._graph._next_id()
    self._original_op = original_op
    self._traceback = tf_stack.extract_stack()

    # List of _UserDevSpecs holding code location of device context manager
    # invocations and the users original argument to them.
    self._device_code_locations = None
    # Dict mapping op name to file and line information for op colocation
    # context managers.
    self._colocation_code_locations = None
    self._control_flow_context = self.graph._get_control_flow_context()
    # pylint: enable=protected-access

    # Initialize self._c_op.
    if c_op:
      self._c_op = c_op
    else:
      if op_def is None:
        op_def = self._graph._get_op_def(node_def.op)
      # TODO(skyewm): op_def_library.apply_op() flattens the incoming inputs.
      # Refactor so we don't have to do this here.
      grouped_inputs = self._reconstruct_sequence_inputs(
          op_def, inputs, node_def.attr)
      self._c_op = _create_c_op(self._graph, node_def, grouped_inputs,
                                control_input_ops)

    # Initialize self._outputs.
    num_outputs = c_api.TF_OperationNumOutputs(self._c_op)
    output_types = [
        c_api.TF_OperationOutputType(c_api_util.tf_output(self._c_op, i))
        for i in range(num_outputs)]
    self._outputs = [
        Tensor(self, i, output_type)
        for i, output_type in enumerate(output_types)
    ]

    self._graph._add_op(self)  

    if not c_op:
      self._control_flow_post_processing()

  def _control_flow_post_processing(self):
    for input_tensor in self.inputs:
      control_flow_util.CheckInputFromValidContext(self, input_tensor.op)
    if self._control_flow_context is not None:
      self._control_flow_context.AddOp(self)

  def _reconstruct_sequence_inputs(self, op_def, inputs, attrs):
    grouped_inputs = []
    i = 0
    for input_arg in op_def.input_arg:
      if input_arg.number_attr:
        input_len = attrs[input_arg.number_attr].i
        is_sequence = True
      elif input_arg.type_list_attr:
        input_len = len(attrs[input_arg.type_list_attr].list.type)
        is_sequence = True
      else:
        input_len = 1
        is_sequence = False

      if is_sequence:
        grouped_inputs.append(inputs[i:i + input_len])
      else:
        grouped_inputs.append(inputs[i])
      i += input_len

    assert i == len(inputs)
    return grouped_inputs

  def colocation_groups(self):
    default_colocation_group = [
        compat.as_bytes("loc:@%s" % self.name)
    ]
    try:
      class_attr = self.get_attr("_class")
    except ValueError:
      # This op has no explicit colocation group, so it is itself its
      # own root of a colocation group.
      return default_colocation_group

    attr_groups = [
        class_name for class_name in class_attr
        if class_name.startswith(b"loc:@")
    ]

    # If there are no colocation groups in the explicit _class field,
    # return the default colocation group.
    return attr_groups if attr_groups else default_colocation_group

  def values(self):
    return tuple(self.outputs)

  def _get_control_flow_context(self):
    return self._control_flow_context

  def _set_control_flow_context(self, ctx):
    self._control_flow_context = ctx

  @property
  def name(self):
    return c_api.TF_OperationName(self._c_op)

  @property
  def _id(self):
    return self._id_value

  @property
  def device(self):
    return c_api.TF_OperationDevice(self._c_op)

  @property
  def _device_assignments(self):
    return self._device_code_locations or []

  @property
  def _colocation_dict(self):
    locations_dict = self._colocation_code_locations or {}
    return locations_dict.copy()

  @property
  def _output_types(self):
    num_outputs = c_api.TF_OperationNumOutputs(self._c_op)
    output_types = [
        c_api.TF_OperationOutputType(self._tf_output(i))
        for i in xrange(num_outputs)
    ]
    if output_types:
      assert isinstance(output_types[0], int)
    return output_types

  def _tf_output(self, output_idx):
    tf_output = c_api.TF_Output()
    tf_output.oper = self._c_op
    tf_output.index = output_idx
    return tf_output

  def _tf_input(self, input_idx):
    tf_input = c_api.TF_Input()
    tf_input.oper = self._c_op
    tf_input.index = input_idx
    return tf_input

  def _set_device(self, device):  # pylint: disable=redefined-outer-name
    c_api.SetRequestedDevice(
        self._graph._c_graph,  
        self._c_op,  
        compat.as_str(_device_string(device)))

  def _update_input(self, index, tensor):
    if not isinstance(tensor, Tensor):
      raise TypeError("tensor must be a Tensor: %s" % tensor)
    _assert_same_graph(self, tensor)

    # Make sure output shapes are already computed for this op in case we create
    # a cycle (we cannot compute shapes for cycles). Usually shapes are computed
    # lazily upon request.
    if not _USE_C_SHAPES:
      set_shape_and_handle_data_for_outputs(self)

    # Reset cached inputs.
    self._inputs_val = None
    c_api.UpdateEdge(
        self._graph._c_graph,  
        tensor._as_tf_output(),  
        self._tf_input(index))

  def _add_control_inputs(self, ops):
    for op in ops:
      if not isinstance(op, Operation):
        raise TypeError("op must be an Operation: %s" % op)
      c_api.AddControlInput(self._graph._c_graph, self._c_op, op._c_op)  

  def _add_control_input(self, op):
    if not isinstance(op, Operation):
      raise TypeError("op must be an Operation: %s" % op)
    c_api.AddControlInput(self._graph._c_graph, self._c_op, op._c_op)  

  def _remove_all_control_inputs(self):
    c_api.RemoveAllControlInputs(self._graph._c_graph, self._c_op)  

  def __str__(self):
    return str(self.node_def)

  def __repr__(self):
    return "<tf.Operation '%s' type=%s>" % (self.name, self.type)

  @property
  def outputs(self):
    return self._outputs



  class _InputList(object):
    def __init__(self, inputs):
      self._inputs = inputs

    def __iter__(self):
      return iter(self._inputs)

    def __len__(self):
      return len(self._inputs)

    def __bool__(self):
      return bool(self._inputs)

    # Python 3 wants __bool__, Python 2.7 wants __nonzero__
    __nonzero__ = __bool__

    def __getitem__(self, i):
      return self._inputs[i]

# pylint: enable=protected-access

  @property
  def inputs(self):
    if self._inputs_val is None:
      tf_outputs = c_api.GetOperationInputs(self._c_op)
      
      retval = [
          self.graph._get_tensor_by_tf_output(tf_output)
          for tf_output in tf_outputs
      ]
      # pylint: enable=protected-access
      self._inputs_val = Operation._InputList(retval)
    return self._inputs_val

  @property
  def _inputs(self):
    logging.warning("Operation._inputs is private, use Operation.inputs "
                    "instead. Operation._inputs will eventually be removed.")
    return self.inputs

  @_inputs.setter
  def _inputs(self, value):
    raise ValueError("Cannot assign _inputs")

  @property
  def _input_types(self):
    num_inputs = c_api.TF_OperationNumInputs(self._c_op)
    input_types = [
        dtypes.as_dtype(c_api.TF_OperationInputType(self._tf_input(i)))
        for i in xrange(num_inputs)
    ]
    return input_types

  @_input_types.setter
  def _input_types(self, value):
    raise ValueError("Cannot assign _input_types")

  @property
  def control_inputs(self):
    control_c_ops = c_api.TF_OperationGetControlInputs_wrapper(self._c_op)
    
    return [
        self.graph._get_operation_by_name_unsafe(
            c_api.TF_OperationName(c_op)) for c_op in control_c_ops
    ]
    # pylint: enable=protected-access

  @property
  def _control_outputs(self):
    control_c_ops = c_api.TF_OperationGetControlOutputs_wrapper(self._c_op)
    
    return [
        self.graph._get_operation_by_name_unsafe(
            c_api.TF_OperationName(c_op)) for c_op in control_c_ops
    ]
    # pylint: enable=protected-access

  @property
  def _control_inputs(self):
    logging.warning("Operation._control_inputs is private, use "
                    "Operation.control_inputs instead. "
                    "Operation._control_inputs will eventually be removed.")
    return self.control_inputs

  @_control_inputs.setter
  def _control_inputs(self, value):
    logging.warning("Operation._control_inputs is private, use "
                    "Operation.control_inputs instead. "
                    "Operation._control_inputs will eventually be removed.")
    # Copy value because it may be self._control_inputs_val (in particular if
    # this is called from self._control_inputs += ...), and we don't want to
    # clear value below.
    value = copy.copy(value)
    self._remove_all_control_inputs()
    self._add_control_inputs(value)

  @property
  def type(self):
    return c_api.TF_OperationOpType(self._c_op)

  @property
  def graph(self):
    """The `Graph` that contains this operation."""
    return self._graph

  @property
  def node_def(self):
    
    
    with c_api_util.tf_buffer() as buf:
      c_api.TF_OperationToNodeDef(self._c_op, buf)
      data = c_api.TF_GetBuffer(buf)
    node_def = node_def_pb2.NodeDef()
    node_def.ParseFromString(compat.as_bytes(data))
    return node_def

  @property
  def _node_def(self):
    logging.warning("Operation._node_def is private, use Operation.node_def "
                    "instead. Operation._node_def will eventually be removed.")
    return self.node_def

  @property
  def op_def(self):
    return self._graph._get_op_def(self.type)

  @property
  def _op_def(self):
    logging.warning("Operation._op_def is private, use Operation.op_def "
                    "instead. Operation._op_def will eventually be removed.")
    return self.op_def

  @property
  def traceback(self):
    return tf_stack.convert_stack(self._traceback)

  @property
  def traceback_with_start_lines(self):
    return tf_stack.convert_stack(self._traceback,
                                  include_func_start_lineno=True)

  def _set_attr(self, attr_name, attr_value):
    buf = c_api.TF_NewBufferFromString(
        compat.as_bytes(attr_value.SerializeToString()))
    try:
      
      c_api.SetAttr(self._graph._c_graph, self._c_op, attr_name, buf)
      # pylint: enable=protected-access
    finally:
      c_api.TF_DeleteBuffer(buf)

  def get_attr(self, name):
    fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]
    try:
      with c_api_util.tf_buffer() as buf:
        c_api.TF_OperationGetAttrValueProto(self._c_op, name, buf)
        data = c_api.TF_GetBuffer(buf)
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(str(e))
    x = attr_value_pb2.AttrValue()
    x.ParseFromString(data)

    # Treat an empty oneof value as an empty list.
    if not x.WhichOneof("value"):
      return []
    if x.HasField("list"):
      for f in fields:
        if getattr(x.list, f):
          if f == "type":
            return [dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
          else:
            return list(getattr(x.list, f))
      return []
    else:
      for f in fields:
        if x.HasField(f):
          if f == "type":
            return dtypes.as_dtype(getattr(x, f))
          else:
            return getattr(x, f)
      assert False, "Unsupported field type in " + str(x)

  def run(self, feed_dict=None, session=None):
    _run_using_default_session(self, feed_dict, self.graph, session)

_gradient_registry = registry.Registry("gradient")


@tf_export("RegisterGradient")
class RegisterGradient(object):
  def __init__(self, op_type):
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string")
    self._op_type = op_type

  def __call__(self, f):
    _gradient_registry.register(f, self._op_type)
    return f


@tf_export("NoGradient", "NotDifferentiable")
def NotDifferentiable(op_type):
  if not isinstance(op_type, six.string_types):
    raise TypeError("op_type must be a string")
  _gradient_registry.register(None, op_type)


# Alias for the old name, will be eventually removed.
NoGradient = NotDifferentiable


def get_gradient_function(op):
  if not op.inputs:
    return None
  try:
    op_type = op.get_attr("_gradient_op_type")
  except ValueError:
    op_type = op.type
  return _gradient_registry.lookup(op_type)


_shape_registry = registry.Registry("shape functions")
_default_shape_function_registry = registry.Registry("default shape functions")

# These are set to common_shapes.call_cpp_shape_fn by op generated code
# (generated by python_op_gen.cc).
# It is set outside ops.py to avoid a circular dependency.
_call_cpp_shape_fn = None
_call_cpp_shape_fn_and_require_op = None


def _set_call_cpp_shape_fn(call_cpp_shape_fn):
  global _call_cpp_shape_fn, _call_cpp_shape_fn_and_require_op
  if _call_cpp_shape_fn:
    return  # already registered

  def call_without_requiring(op):
    return call_cpp_shape_fn(op, require_shape_fn=False)

  _call_cpp_shape_fn = call_without_requiring

  def call_with_requiring(op):
    return call_cpp_shape_fn(op, require_shape_fn=True)

  _call_cpp_shape_fn_and_require_op = call_with_requiring


class RegisterShape(object):
  def __init__(self, op_type):
    """Saves the `op_type` as the `Operation` type."""
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string")
    self._op_type = op_type

  def __call__(self, f):
    """Registers "f" as the shape function for "op_type"."""
    if f is None:
      assert _call_cpp_shape_fn
      try:
        _default_shape_function_registry.register(_call_cpp_shape_fn,
                                                  self._op_type)
      except KeyError:
        pass
    else:
      _shape_registry.register(f, self._op_type)
    return f

def _set_shape_and_handle_data_for_outputs_c_api(op):
  assert not _USE_C_SHAPES
  for output in op.outputs:
    output._shape_val = output._c_api_shape()
    # Set the resource handle data for compatibility with the Python shape
    # inference code.
    serialized = c_api.GetHandleShapeAndType(op._graph._c_graph,  
                                             output._as_tf_output())
    if serialized:
      output._handle_data = (
          cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData
          .FromString(compat.as_bytes(serialized)))
    else:
      output._handle_data = None


def set_shape_and_handle_data_for_outputs(op):
  if _USE_C_SHAPES: return

  if op.graph._is_function(op.type):
    for output in op.outputs:
      output._shape_val = tensor_shape.unknown_shape()
    return

  try:
    shape_func = _shape_registry.lookup(op.type)
  except LookupError:
    try:
      shape_func = _default_shape_function_registry.lookup(op.type)
    except LookupError:
      shape_func = _call_cpp_shape_fn_and_require_op

  shapes = shape_func(op)
  if shapes is None:
    raise RuntimeError(
        "Shape function for op %s did not return any shapes" % op)
  elif isinstance(shapes, dict):
    # Returned by call_cpp_shape_fn
    shapes_dict = shapes
    shapes = shapes_dict["shapes"]
    handle_datas = shapes_dict["handle_data"]
    for output, handle_data in zip(op.outputs, handle_datas):
      # Don't override any existing handle data that may have been manually set.
      
      if output._handle_data is None:
        output._handle_data = handle_data
      # pylint: enable=protected-access

  if len(op.outputs) != len(shapes):
    raise RuntimeError(
        "Shape function for op %s returned %d shapes but expected %d %s %s" %
        (op, len(shapes), len(op.outputs), shape_func.__name__, str(shapes)))
  for output, s in zip(op.outputs, shapes):
    output._shape_val = tensor_shape.unknown_shape()
    output._shape_val = output._shape_val.merge_with(s)


class OpStats(object):
  def __init__(self, statistic_type, value=None):
    self.statistic_type = statistic_type
    self.value = value

  @property
  def statistic_type(self):
    return self._statistic_type

  @statistic_type.setter
  def statistic_type(self, statistic_type):
    self._statistic_type = statistic_type

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self._value = value

  def __iadd__(self, other):
    if other.statistic_type != self.statistic_type:
      raise ValueError("Can't add an OpStat of type %s to one of %s." %
                       (self.statistic_type, other.statistic_type))
    if self.value is None:
      self.value = other.value
    elif other.value is not None:
      self._value += other.value
    return self


_stats_registry = registry.Registry("statistical functions")


class RegisterStatistics(object):
  def __init__(self, op_type, statistic_type):
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string.")
    if "," in op_type:
      raise TypeError("op_type must not contain a comma.")
    self._op_type = op_type
    if not isinstance(statistic_type, six.string_types):
      raise TypeError("statistic_type must be a string.")
    if "," in statistic_type:
      raise TypeError("statistic_type must not contain a comma.")
    self._statistic_type = statistic_type

  def __call__(self, f):
    _stats_registry.register(f, self._op_type + "," + self._statistic_type)
    return f


def get_stats_for_node_def(graph, node, statistic_type):
  try:
    stats_func = _stats_registry.lookup(node.op + "," + statistic_type)
    result = stats_func(graph, node)
  except LookupError:
    result = OpStats(statistic_type)
  return result


def _name_from_scope_name(name):
  return name[:-1] if (name and name[-1] == "/") else name


_MUTATION_LOCK_GROUP = 0
_SESSION_RUN_LOCK_GROUP = 1

@tf_export("Graph")
class Graph(object):
  def __init__(self):
    self._lock = threading.RLock()
    self._group_lock = lock_util.GroupLock(num_groups=2)
    self._nodes_by_id = dict()  # GUARDED_BY(self._lock)
    self._next_id_counter = 0  # GUARDED_BY(self._lock)
    self._nodes_by_name = dict()  # GUARDED_BY(self._lock)
    self._version = 0  # GUARDED_BY(self._lock)
    # Maps a name used in the graph to the next id to use for that name.
    self._names_in_use = {}
    self._stack_state_is_thread_local = False
    self._thread_local = threading.local()
    # Functions that will be applied to choose a device if none is specified.
    # After switch_to_thread_local(), self._thread_local._device_function_stack
    # is used instead.
    self._graph_device_function_stack = traceable_stack.TraceableStack()
    # Default original_op applied to new ops.
    self._default_original_op = None
    # Current control flow context. It could be either CondContext or
    # WhileContext defined in ops/control_flow_ops.py
    self._control_flow_context = None
    # A new node will depend of the union of all of the nodes in the stack.
    # After switch_to_thread_local(),
    # self._thread_local._control_dependencies_stack is used instead.
    self._graph_control_dependencies_stack = []
    # Arbitrary collections of objects.
    self._collections = {}
    # The graph-level random seed
    self._seed = None
    # A dictionary of attributes that should be applied to all ops.
    self._attr_scope_map = {}
    # A map from op type to the kernel label that should be used.
    self._op_to_kernel_label_map = {}
    # A map from op type to an alternative op type that should be used when
    # computing gradients.
    self._gradient_override_map = {}
    # True if the graph is considered "finalized".  In that case no
    # new operations can be added.
    self._finalized = False
    # Functions defined in the graph
    self._functions = collections.OrderedDict()
    # Default GraphDef versions
    
    self._building_function = False
    # Stack of colocate_with ops. After switch_to_thread_local(),
    # self._thread_local._colocation_stack is used instead.
    self._graph_colocation_stack = traceable_stack.TraceableStack()
    # Set of tensors that are dangerous to feed!
    self._unfeedable_tensors = set()
    # Set of operations that are dangerous to fetch!
    self._unfetchable_ops = set()
    # A map of tensor handle placeholder to tensor dtype.
    self._handle_feeders = {}
    # A map from tensor handle to its read op.
    self._handle_readers = {}
    # A map from tensor handle to its move op.
    self._handle_movers = {}
    # A map from tensor handle to its delete op.
    self._handle_deleters = {}
    # Allow optimizers and other objects to pseudo-uniquely key graphs (this key
    # will be shared when defining function graphs, for example, so optimizers
    # being called inside function definitions behave as if they were seeing the
    # actual outside graph).
    self._graph_key = "grap-key-%d/" % (uid(),)
    # A string with the last reduction method passed to
    # losses.compute_weighted_loss(), or None.
    self._last_loss_reduction = None
    self._container = ""
    self._registered_ops = op_def_registry.get_registered_ops()

    # TODO(skyewm): fold as much of the above as possible into the C
    # implementation
    self._scoped_c_graph = c_api_util.ScopedTFGraph()
    # The C API requires all ops to have shape functions. Disable this
    # requirement (many custom ops do not have shape functions, and we don't
    # want to break these existing cases).
    c_api.SetRequireShapeInferenceFns(self._c_graph, False)

  # Note: this method is private because the API of tf.Graph() is public and
  # frozen, and this functionality is still not ready for public visibility.
  @tf_contextlib.contextmanager
  def _variable_creator_scope(self, creator):
    # This step makes a copy of the existing stack, and it also initializes
    # self._thread_local._variable_creator_stack if it doesn't exist yet.
    old = list(self._variable_creator_stack)
    self._thread_local._variable_creator_stack.append(creator)  
    try:
      yield
    finally:
      self._thread_local._variable_creator_stack = old  

  # Note: this method is private because the API of tf.Graph() is public and
  # frozen, and this functionality is still not ready for public visibility.
  @property
  def _variable_creator_stack(self):
    if not hasattr(self._thread_local, "_variable_creator_stack"):
      self._thread_local._variable_creator_stack = []  
    return list(self._thread_local._variable_creator_stack)  

  @_variable_creator_stack.setter
  def _variable_creator_stack(self, variable_creator_stack):
    self._thread_local._variable_creator_stack = variable_creator_stack  

  def _check_not_finalized(self):
    if self._finalized:
      raise RuntimeError("Graph is finalized and cannot be modified.")

  def _add_op(self, op):
    self._check_not_finalized()
    if not isinstance(op, (Tensor, Operation)):
      raise TypeError("op must be a Tensor or Operation: %s" % op)
    with self._lock:
      
      if op._id in self._nodes_by_id:
        raise ValueError("cannot add an op with id %d as it already "
                         "exists in the graph" % op._id)
      if op.name in self._nodes_by_name:
        raise ValueError("cannot add op with name %s as that name "
                         "is already used" % op.name)
      self._nodes_by_id[op._id] = op
      self._nodes_by_name[op.name] = op
      self._version = max(self._version, op._id)
      # pylint: enable=protected-access

  @property
  def _c_graph(self):
    if self._scoped_c_graph:
      return self._scoped_c_graph.graph
    return None

  @property
  def version(self):
    if self._finalized:
      return self._version

    with self._lock:
      return self._version

  @property
  def graph_def_versions(self):
    with c_api_util.tf_buffer() as buf:
      c_api.TF_GraphVersions(self._c_graph, buf)
      data = c_api.TF_GetBuffer(buf)
    version_def = versions_pb2.VersionDef()
    version_def.ParseFromString(compat.as_bytes(data))
    return version_def

  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, seed):
    self._seed = seed

  @property
  def finalized(self):
    return self._finalized

  def finalize(self):
    self._finalized = True

  def _unsafe_unfinalize(self):
    self._finalized = False

  def _get_control_flow_context(self):
    return self._control_flow_context

  def _set_control_flow_context(self, ctx):
    self._control_flow_context = ctx

  def _copy_functions_to_graph_def(self, graph_def, starting_bytesize):
    bytesize = starting_bytesize
    for f in self._functions.values():
      bytesize += f.definition.ByteSize()
      if bytesize >= (1 << 31) or bytesize < 0:
        raise ValueError("GraphDef cannot be larger than 2GB.")
      graph_def.library.function.extend([f.definition])
      if f.grad_func_name:
        grad_def = function_pb2.GradientDef()
        grad_def.function_name = f.name
        grad_def.gradient_func = f.grad_func_name
        graph_def.library.gradient.extend([grad_def])

  def _as_graph_def(self, from_version=None, add_shapes=False):
    with self._lock:
      with c_api_util.tf_buffer() as buf:
        c_api.TF_GraphToGraphDef(self._c_graph, buf)
        data = c_api.TF_GetBuffer(buf)
      graph = graph_pb2.GraphDef()
      graph.ParseFromString(compat.as_bytes(data))
      # Strip the experimental library field iff it's empty.
      if not graph.library.function:
        graph.ClearField("library")

      if add_shapes:
        for node in graph.node:
          op = self._nodes_by_name[node.name]
          if op.outputs:
            node.attr["_output_shapes"].list.shape.extend(
                [output.get_shape().as_proto() for output in op.outputs])
    return graph, self._version

  def as_graph_def(self, from_version=None, add_shapes=False):
    result, _ = self._as_graph_def(from_version, add_shapes)
    return result

  def _is_function(self, name):
    return compat.as_str(name) in self._functions

  def _get_function(self, name):
    return self._functions.get(compat.as_str(name), None)

  def _add_function(self, function):
    name = function.name
    # Sanity checks on gradient definition.
    if (function.grad_func_name is not None) and (function.python_grad_func is
                                                  not None):
      raise ValueError("Gradient defined twice for function %s" % name)

    if not function._c_func:
      serialized = function.definition.SerializeToString()
      c_func = c_api.TF_FunctionImportFunctionDef(serialized)
      function._c_func = c_api_util.ScopedTFFunction(c_func)
    gradient = (function._grad_func._c_func.func if function._grad_func
                else None)
    c_api.TF_GraphCopyFunction(self._c_graph, function._c_func.func, gradient)
    # pylint: enable=protected-access

    self._functions[compat.as_str(name)] = function

    # Need a new-enough consumer to support the functions we add to the graph.
    if self._graph_def_versions.min_consumer < 12:
      self._graph_def_versions.min_consumer = 12

  @property
  def building_function(self):
    return self._building_function

  # Helper functions to create operations.
  @deprecated_args(None,
                   "Shapes are always computed; don't use the compute_shapes "
                   "as it has no effect.", "compute_shapes")
  def create_op(
      self,
      op_type,
      inputs,
      dtypes,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_shapes=True,
      compute_device=True):
    del compute_shapes

    self._check_not_finalized()
    for idx, a in enumerate(inputs):
      if not isinstance(a, Tensor):
        raise TypeError("Input #%d is not a tensor: %s" % (idx, a))
    if name is None:
      name = op_type
    # If a names ends with a '/' it is a "name scope" and we use it as-is,
    # after removing the trailing '/'.
    if name and name[-1] == "/":
      name = _name_from_scope_name(name)
    else:
      name = self.unique_name(name)

    node_def = _NodeDef(op_type, name, device=None, attrs=attrs)

    input_ops = set([t.op for t in inputs])
    control_inputs = self._control_dependencies_for_inputs(input_ops)
    # _create_op_helper mutates the new Operation. `_mutation_lock` ensures a
    # Session.run call cannot occur between creating and mutating the op.
    with self._mutation_lock():
      ret = Operation(
          node_def,
          self,
          inputs=inputs,
          output_types=dtypes,
          control_inputs=control_inputs,
          input_types=input_types,
          original_op=self._default_original_op,
          op_def=op_def)
      self._create_op_helper(ret, compute_device=compute_device)
    return ret

  def _create_op_from_tf_operation(self, c_op, compute_device=True):
    self._check_not_finalized()
    ret = Operation(c_op, self)

    name_key = ret.name.lower()
    if name_key not in self._names_in_use:
      self._names_in_use[name_key] = 1
    self._create_op_helper(ret, compute_device=compute_device)
    return ret

  def _make_colocation_conflict_message(self, op, colocation_op):  
    op_info = error_interpolation.compute_field_dict(op)
    coloc_op_info = error_interpolation.compute_field_dict(colocation_op)
    msg = ("Tried to colocate op '{op_name}'{op_loc} having device '{op_dev}' "
           "with op '{coloc_op_name}'{coloc_op_loc} which had an incompatible "
           "device '{coloc_op_dev}'.\n\n{op_summary}\n\n{coloc_op_summary}"
           .format(op_name=op.name,
                   op_loc=op_info["defined_at"],
                   op_dev=op.device,
                   op_summary=op_info["devs_and_colocs"],
                   coloc_op_name=colocation_op.name,
                   coloc_op_loc=coloc_op_info["defined_at"],
                   coloc_op_dev=colocation_op.device,
                   coloc_op_summary=coloc_op_info["devs_and_colocs"]))
    return msg

  def _create_op_helper(self, op, compute_device=True):
    """Common logic for creating an op in this graph."""
    # Apply any additional attributes requested. Do not overwrite any existing
    # attributes.
    for key, value in self._attr_scope_map.items():
      try:
        op.get_attr(key)
      except ValueError:
        if callable(value):
          value = value(op.node_def)
          if not isinstance(value, (type(None), attr_value_pb2.AttrValue)):
            raise TypeError(
                "Callable for scope map key '%s' must return either None or "
                "an AttrValue protocol buffer; but it returned: %s" % (key,
                                                                       value))
        if value:
          op._set_attr(key, value)  

    # Apply a kernel label if one has been specified for this op type.
    try:
      kernel_label = self._op_to_kernel_label_map[op.type]
      op._set_attr("_kernel",  
                   attr_value_pb2.AttrValue(s=compat.as_bytes(kernel_label)))
    except KeyError:
      pass

    # Apply the overriding op type for gradients if one has been specified for
    # this op type.
    try:
      mapped_op_type = self._gradient_override_map[op.type]
      op._set_attr("_gradient_op_type",  
                   attr_value_pb2.AttrValue(s=compat.as_bytes(mapped_op_type)))
    except KeyError:
      pass

    self._record_op_seen_by_control_dependencies(op)

    if compute_device:
      self._apply_device_functions(op)

    # Snapshot the colocation stack metadata before we might generate error
    # messages using it.  Note that this snapshot depends on the actual stack
    # and is independent of the op's _class attribute.
    
    op._colocation_code_locations = self._snapshot_colocation_stack_metadata()
    # pylint: enable=protected-access

    if self._colocation_stack:
      all_colocation_groups = []
      for colocation_op in self._colocation_stack.peek_objs():
        all_colocation_groups.extend(colocation_op.colocation_groups())
        if colocation_op.device:
          if (op.device and pydev.canonical_name(op.device) !=
              pydev.canonical_name(colocation_op.device)):
            msg = self._make_colocation_conflict_message(op, colocation_op)
            logging.warning(msg)
          else:
            op._set_device(colocation_op.device)  

      all_colocation_groups = sorted(set(all_colocation_groups))
      
      op._set_attr("_class", attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(s=all_colocation_groups)))
      # pylint: enable=protected-access

    # Sets "container" attribute if
    # (1) self._container is not None
    # (2) "is_stateful" is set in OpDef
    # (3) "container" attribute is in OpDef
    # (4) "container" attribute is None
    if self._container and op.op_def.is_stateful:
      try:
        container_attr = op.get_attr("container")
      except ValueError:
        # "container" attribute is not in OpDef
        pass
      else:
        if not container_attr:
          op._set_attr("container", attr_value_pb2.AttrValue(  
              s=compat.as_bytes(self._container)))

  def _add_new_tf_operations(self, compute_devices=True):
    """Creates `Operations` in this graph for any new TF_Operations.

    This is useful for when TF_Operations are indirectly created by the C API
    outside of the Operation constructor (e.g. by TF_ImportGraphDef,
    TF_FinishWhile). This ensures there are corresponding Operations for all
    TF_Operations in the underlying TF_Graph.

    Args:
      compute_devices: (Optional.) If True, device functions will be executed
        to compute the device properties of each new Operation.

    Returns:
      A list of the new `Operation` objects.
    """
    # Create all Operation objects before accessing their inputs since an op may
    # be created before its inputs.
    new_ops = [
        self._create_op_from_tf_operation(c_op, compute_device=compute_devices)
        for c_op in c_api_util.new_tf_operations(self)
    ]

    
    for op in new_ops:
      # Operations created by the C API always retrieve shapes from the C API so
      # we preserve the shapes of ops created in import_graph_def (from the
      # "_output_shapes" attr of the imported NodeDef).
      if not _USE_C_SHAPES:
        _set_shape_and_handle_data_for_outputs_c_api(op)
      new_control_inputs = self._control_dependencies_for_inputs(op.inputs)
      op._add_control_inputs(new_control_inputs)
      op._control_flow_post_processing()
    # pylint: enable=protected-access

    return new_ops

  def as_graph_element(self, obj, allow_tensor=True, allow_operation=True):
    """Returns the object referred to by `obj`, as an `Operation` or `Tensor`.

    This function validates that `obj` represents an element of this
    graph, and gives an informative error message if it is not.

    This function is the canonical way to get/validate an object of
    one of the allowed types from an external argument reference in the
    Session API.

    This method may be called concurrently from multiple threads.

    Args:
      obj: A `Tensor`, an `Operation`, or the name of a tensor or operation.
        Can also be any object with an `_as_graph_element()` method that returns
        a value of one of these types.
      allow_tensor: If true, `obj` may refer to a `Tensor`.
      allow_operation: If true, `obj` may refer to an `Operation`.

    Returns:
      The `Tensor` or `Operation` in the Graph corresponding to `obj`.

    Raises:
      TypeError: If `obj` is not a type we support attempting to convert
        to types.
      ValueError: If `obj` is of an appropriate type but invalid. For
        example, an invalid string.
      KeyError: If `obj` is not an object in the graph.
    """
    if self._finalized:
      return self._as_graph_element_locked(obj, allow_tensor, allow_operation)

    with self._lock:
      return self._as_graph_element_locked(obj, allow_tensor, allow_operation)

  def _as_graph_element_locked(self, obj, allow_tensor, allow_operation):
    """See `Graph.as_graph_element()` for details."""
    # The vast majority of this function is figuring
    # out what an API user might be doing wrong, so
    # that we can give helpful error messages.
    #
    # Ideally, it would be nice to split it up, but we
    # need context to generate nice error messages.

    if allow_tensor and allow_operation:
      types_str = "Tensor or Operation"
    elif allow_tensor:
      types_str = "Tensor"
    elif allow_operation:
      types_str = "Operation"
    else:
      raise ValueError("allow_tensor and allow_operation can't both be False.")

    temp_obj = _as_graph_element(obj)
    if temp_obj is not None:
      obj = temp_obj

    # If obj appears to be a name...
    if isinstance(obj, compat.bytes_or_text_types):
      name = compat.as_str(obj)

      if ":" in name and allow_tensor:
        # Looks like a Tensor name and can be a Tensor.
        try:
          op_name, out_n = name.split(":")
          out_n = int(out_n)
        except:
          raise ValueError("The name %s looks a like a Tensor name, but is "
                           "not a valid one. Tensor names must be of the "
                           "form \"<op_name>:<output_index>\"." % repr(name))
        if op_name in self._nodes_by_name:
          op = self._nodes_by_name[op_name]
        else:
          raise KeyError("The name %s refers to a Tensor which does not "
                         "exist. The operation, %s, does not exist in the "
                         "graph." % (repr(name), repr(op_name)))
        try:
          return op.outputs[out_n]
        except:
          raise KeyError("The name %s refers to a Tensor which does not "
                         "exist. The operation, %s, exists but only has "
                         "%s outputs." % (repr(name), repr(op_name),
                                          len(op.outputs)))

      elif ":" in name and not allow_tensor:
        # Looks like a Tensor name but can't be a Tensor.
        raise ValueError("Name %s appears to refer to a Tensor, not a %s." %
                         (repr(name), types_str))

      elif ":" not in name and allow_operation:
        # Looks like an Operation name and can be an Operation.
        if name not in self._nodes_by_name:
          raise KeyError("The name %s refers to an Operation not in the "
                         "graph." % repr(name))
        return self._nodes_by_name[name]

      elif ":" not in name and not allow_operation:
        # Looks like an Operation name but can't be an Operation.
        if name in self._nodes_by_name:
          # Yep, it's an Operation name
          err_msg = ("The name %s refers to an Operation, not a %s." %
                     (repr(name), types_str))
        else:
          err_msg = ("The name %s looks like an (invalid) Operation name, "
                     "not a %s." % (repr(name), types_str))
        err_msg += (" Tensor names must be of the form "
                    "\"<op_name>:<output_index>\".")
        raise ValueError(err_msg)

    elif isinstance(obj, Tensor) and allow_tensor:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Tensor %s is not an element of this graph." % obj)
      return obj
    elif isinstance(obj, Operation) and allow_operation:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Operation %s is not an element of this graph." % obj)
      return obj
    else:
      # We give up!
      raise TypeError("Can not convert a %s into a %s." % (type(obj).__name__,
                                                           types_str))

  def get_operations(self):
    """Return the list of operations in the graph.

    You can modify the operations in place, but modifications
    to the list such as inserts/delete have no effect on the
    list of operations known to the graph.

    This method may be called concurrently from multiple threads.

    Returns:
      A list of Operations.
    """
    if self._finalized:
      return list(self._nodes_by_id.values())

    with self._lock:
      return list(self._nodes_by_id.values())

  def get_operation_by_name(self, name):

    if not isinstance(name, six.string_types):
      raise TypeError("Operation names are strings (or similar), not %s." %
                      type(name).__name__)
    return self.as_graph_element(name, allow_tensor=False, allow_operation=True)

  def _get_operation_by_name_unsafe(self, name):
    if self._finalized:
      return self._nodes_by_name[name]

    with self._lock:
      return self._nodes_by_name[name]

  def _get_operation_by_tf_operation(self, tf_oper):
    op_name = c_api.TF_OperationName(tf_oper)
    return self._get_operation_by_name_unsafe(op_name)

  def get_tensor_by_name(self, name):
    # Names should be strings.
    if not isinstance(name, six.string_types):
      raise TypeError("Tensor names are strings (or similar), not %s." %
                      type(name).__name__)
    return self.as_graph_element(name, allow_tensor=True, allow_operation=False)

  def _get_tensor_by_tf_output(self, tf_output):
    op = self._get_operation_by_tf_operation(tf_output.oper)
    return op.outputs[tf_output.index]

  def _next_id(self):
    self._check_not_finalized()
    with self._lock:
      self._next_id_counter += 1
      return self._next_id_counter

  @property
  def _last_id(self):
    return self._next_id_counter

  def _get_op_def(self, type):  # pylint: disable=redefined-builtin
    with c_api_util.tf_buffer() as buf:
      
      c_api.TF_GraphGetOpDef(self._c_graph, compat.as_bytes(type), buf)
      # pylint: enable=protected-access
      data = c_api.TF_GetBuffer(buf)
    op_def = op_def_pb2.OpDef()
    op_def.ParseFromString(compat.as_bytes(data))
    return op_def

  def as_default(self):
    return _default_graph_stack.get_controller(self)

  @property
  def collections(self):
    """Returns the names of the collections known to this graph."""
    return list(self._collections)

  def add_to_collection(self, name, value):
    """Stores `value` in the collection with the given `name`.

    Note that collections are not sets, so it is possible to add a value to
    a collection several times.

    Args:
      name: The key for the collection. The `GraphKeys` class
        contains many standard names for collections.
      value: The value to add to the collection.
    """  # pylint: disable=g-doc-exception
    self._check_not_finalized()
    with self._lock:
      if name not in self._collections:
        self._collections[name] = [value]
      else:
        self._collections[name].append(value)

  def add_to_collections(self, names, value):
    """Stores `value` in the collections given by `names`.

    Note that collections are not sets, so it is possible to add a value to
    a collection several times. This function makes sure that duplicates in
    `names` are ignored, but it will not check for pre-existing membership of
    `value` in any of the collections in `names`.

    `names` can be any iterable, but if `names` is a string, it is treated as a
    single collection name.

    Args:
      names: The keys for the collections to add to. The `GraphKeys` class
        contains many standard names for collections.
      value: The value to add to the collections.
    """
    # Make sure names are unique, but treat strings as a single collection name
    names = (names,) if isinstance(names, six.string_types) else set(names)
    for name in names:
      self.add_to_collection(name, value)

  def get_collection_ref(self, name):
    """Returns a list of values in the collection with the given `name`.

    If the collection exists, this returns the list itself, which can
    be modified in place to change the collection.  If the collection does
    not exist, it is created as an empty list and the list is returned.

    This is different from `get_collection()` which always returns a copy of
    the collection list if it exists and never creates an empty collection.

    Args:
      name: The key for the collection. For example, the `GraphKeys` class
        contains many standard names for collections.

    Returns:
      The list of values in the collection with the given `name`, or an empty
      list if no value has been added to that collection.
    """  # pylint: disable=g-doc-exception
    with self._lock:
      coll_list = self._collections.get(name, None)
      if coll_list is None:
        coll_list = []
        self._collections[name] = coll_list
      return coll_list

  def get_collection(self, name, scope=None):
    """Returns a list of values in the collection with the given `name`.

    This is different from `get_collection_ref()` which always returns the
    actual collection list if it exists in that it returns a new list each time
    it is called.

    Args:
      name: The key for the collection. For example, the `GraphKeys` class
        contains many standard names for collections.
      scope: (Optional.) A string. If supplied, the resulting list is filtered
        to include only items whose `name` attribute matches `scope` using
        `re.match`. Items without a `name` attribute are never returned if a
        scope is supplied. The choice of `re.match` means that a `scope` without
        special tokens filters by prefix.

    Returns:
      The list of values in the collection with the given `name`, or
      an empty list if no value has been added to that collection. The
      list contains the values in the order under which they were
      collected.
    """  # pylint: disable=g-doc-exception
    with self._lock:
      collection = self._collections.get(name, None)
      if collection is None:
        return []
      if scope is None:
        return list(collection)
      else:
        c = []
        regex = re.compile(scope)
        for item in collection:
          if hasattr(item, "name") and regex.match(item.name):
            c.append(item)
        return c

  def get_all_collection_keys(self):
    """Returns a list of collections used in this graph."""
    with self._lock:
      return [x for x in self._collections if isinstance(x, six.string_types)]

  def clear_collection(self, name):
    """Clears all values in a collection.

    Args:
      name: The key for the collection. The `GraphKeys` class contains many
        standard names for collections.
    """
    self._check_not_finalized()
    with self._lock:
      if name in self._collections:
        del self._collections[name]

  @tf_contextlib.contextmanager
  def _original_op(self, op):
    """Python 'with' handler to help annotate ops with their originator.

    An op may have an 'original_op' property that indicates the op on which
    it was based. For example a replica op is based on the op that was
    replicated and a gradient op is based on the op that was differentiated.

    All ops created in the scope of this 'with' handler will have
    the given 'op' as their original op.

    Args:
      op: The Operation that all ops created in this scope will have as their
        original op.

    Yields:
      Nothing.
    """
    old_original_op = self._default_original_op
    self._default_original_op = op
    try:
      yield
    finally:
      self._default_original_op = old_original_op

  @property
  def _name_stack(self):
    # This may be called from a thread where name_stack doesn't yet exist.
    if not hasattr(self._thread_local, "_name_stack"):
      self._thread_local._name_stack = ""
    return self._thread_local._name_stack

  @_name_stack.setter
  def _name_stack(self, name_stack):
    self._thread_local._name_stack = name_stack

  # pylint: disable=g-doc-return-or-yield,line-too-long
  @tf_contextlib.contextmanager
  def name_scope(self, name):
    r"""Returns a context manager that creates hierarchical names for operations.

    A graph maintains a stack of name scopes. A `with name_scope(...):`
    statement pushes a new name onto the stack for the lifetime of the context.

    The `name` argument will be interpreted as follows:

    * A string (not ending with '/') will create a new name scope, in which
      `name` is appended to the prefix of all operations created in the
      context. If `name` has been used before, it will be made unique by
      calling `self.unique_name(name)`.
    * A scope previously captured from a `with g.name_scope(...) as
      scope:` statement will be treated as an "absolute" name scope, which
      makes it possible to re-enter existing scopes.
    * A value of `None` or the empty string will reset the current name scope
      to the top-level (empty) name scope.

    For example:

    ```python
    with tf.Graph().as_default() as g:
      c = tf.constant(5.0, name="c")
      assert c.op.name == "c"
      c_1 = tf.constant(6.0, name="c")
      assert c_1.op.name == "c_1"

      # Creates a scope called "nested"
      with g.name_scope("nested") as scope:
        nested_c = tf.constant(10.0, name="c")
        assert nested_c.op.name == "nested/c"

        # Creates a nested scope called "inner".
        with g.name_scope("inner"):
          nested_inner_c = tf.constant(20.0, name="c")
          assert nested_inner_c.op.name == "nested/inner/c"

        # Create a nested scope called "inner_1".
        with g.name_scope("inner"):
          nested_inner_1_c = tf.constant(30.0, name="c")
          assert nested_inner_1_c.op.name == "nested/inner_1/c"

          # Treats `scope` as an absolute name scope, and
          # switches to the "nested/" scope.
          with g.name_scope(scope):
            nested_d = tf.constant(40.0, name="d")
            assert nested_d.op.name == "nested/d"

            with g.name_scope(""):
              e = tf.constant(50.0, name="e")
              assert e.op.name == "e"
    ```

    The name of the scope itself can be captured by `with
    g.name_scope(...) as scope:`, which stores the name of the scope
    in the variable `scope`. This value can be used to name an
    operation that represents the overall result of executing the ops
    in a scope. For example:

    ```python
    inputs = tf.constant(...)
    with g.name_scope('my_layer') as scope:
      weights = tf.Variable(..., name="weights")
      biases = tf.Variable(..., name="biases")
      affine = tf.matmul(inputs, weights) + biases
      output = tf.nn.relu(affine, name=scope)
    ```

    NOTE: This constructor validates the given `name`. Valid scope
    names match one of the following regular expressions:

        [A-Za-z0-9.][A-Za-z0-9_.\\-/]* (for scopes at the root)
        [A-Za-z0-9_.\\-/]* (for other scopes)

    Args:
      name: A name for the scope.

    Returns:
      A context manager that installs `name` as a new name scope.

    Raises:
      ValueError: If `name` is not a valid scope name, according to the rules
        above.
    """
    if name:
      if isinstance(name, compat.bytes_or_text_types):
        name = compat.as_str(name)

      if self._name_stack:
        # Scopes created in a nested scope may have initial characters
        # that are illegal as the initial character of an op name
        # (viz. '-', '\', '/', and '_').
        if not _VALID_SCOPE_NAME_REGEX.match(name):
          raise ValueError("'%s' is not a valid scope name" % name)
      else:
        # Scopes created in the root must match the more restrictive
        # op name regex, which constrains the initial character.
        if not _VALID_OP_NAME_REGEX.match(name):
          raise ValueError("'%s' is not a valid scope name" % name)
    old_stack = self._name_stack
    if not name:  # Both for name=None and name="" we re-set to empty scope.
      new_stack = None
    elif name[-1] == "/":
      new_stack = _name_from_scope_name(name)
    else:
      new_stack = self.unique_name(name)
    self._name_stack = new_stack
    try:
      yield "" if new_stack is None else new_stack + "/"
    finally:
      self._name_stack = old_stack

  # pylint: enable=g-doc-return-or-yield,line-too-long

  def unique_name(self, name, mark_as_used=True):
    """Return a unique operation name for `name`.

    Note: You rarely need to call `unique_name()` directly.  Most of
    the time you just need to create `with g.name_scope()` blocks to
    generate structured names.

    `unique_name` is used to generate structured names, separated by
    `"/"`, to help identify operations when debugging a graph.
    Operation names are displayed in error messages reported by the
    TensorFlow runtime, and in various visualization tools such as
    TensorBoard.

    If `mark_as_used` is set to `True`, which is the default, a new
    unique name is created and marked as in use. If it's set to `False`,
    the unique name is returned without actually being marked as used.
    This is useful when the caller simply wants to know what the name
    to be created will be.

    Args:
      name: The name for an operation.
      mark_as_used: Whether to mark this name as being used.

    Returns:
      A string to be passed to `create_op()` that will be used
      to name the operation being created.
    """
    if self._name_stack:
      name = self._name_stack + "/" + name

    # For the sake of checking for names in use, we treat names as case
    # insensitive (e.g. foo = Foo).
    name_key = name.lower()
    i = self._names_in_use.get(name_key, 0)
    # Increment the number for "name_key".
    if mark_as_used:
      self._names_in_use[name_key] = i + 1
    if i > 0:
      base_name_key = name_key
      # Make sure the composed name key is not already used.
      while name_key in self._names_in_use:
        name_key = "%s_%d" % (base_name_key, i)
        i += 1
      # Mark the composed name_key as used in case someone wants
      # to call unique_name("name_1").
      if mark_as_used:
        self._names_in_use[name_key] = 1

      # Return the new name with the original capitalization of the given name.
      name = "%s_%d" % (name, i-1)
    return name

  def get_name_scope(self):
    """Returns the current name scope.

    For example:

    ```python
    with tf.name_scope('scope1'):
      with tf.name_scope('scope2'):
        print(tf.get_default_graph().get_name_scope())
    ```
    would print the string `scope1/scope2`.

    Returns:
      A string representing the current name scope.
    """
    return self._name_stack

  @tf_contextlib.contextmanager
  def _colocate_with_for_gradient(self, op, gradient_uid,
                                  ignore_existing=False):
    with self.colocate_with(op, ignore_existing):
      if gradient_uid is not None and self._control_flow_context is not None:
        self._control_flow_context.EnterGradientColocation(op, gradient_uid)
        try:
          yield
        finally:
          self._control_flow_context.ExitGradientColocation(op, gradient_uid)
      else:
        yield

  @tf_contextlib.contextmanager
  def colocate_with(self, op, ignore_existing=False):
    """Returns a context manager that specifies an op to colocate with.

    Note: this function is not for public use, only for internal libraries.

    For example:

    ```python
    a = tf.Variable([1.0])
    with g.colocate_with(a):
      b = tf.constant(1.0)
      c = tf.add(a, b)
    ```

    `b` and `c` will always be colocated with `a`, no matter where `a`
    is eventually placed.

    **NOTE** Using a colocation scope resets any existing device constraints.

    If `op` is `None` then `ignore_existing` must be `True` and the new
    scope resets all colocation and device constraints.

    Args:
      op: The op to colocate all created ops with, or `None`.
      ignore_existing: If true, only applies colocation of this op within
        the context, rather than applying all colocation properties
        on the stack.  If `op` is `None`, this value must be `True`.

    Raises:
      ValueError: if op is None but ignore_existing is False.

    Yields:
      A context manager that specifies the op with which to colocate
      newly created ops.
    """
    if op is None and not ignore_existing:
      raise ValueError("Trying to reset colocation (op is None) but "
                       "ignore_existing is not True")

    if op is not None and not isinstance(op, Operation):
      # We always want to colocate with the reference op.
      op = internal_convert_to_tensor_or_indexed_slices(op, as_ref=True).op

    # By default, colocate_with resets the device function stack,
    # since colocate_with is typically used in specific internal
    # library functions where colocation is intended to be "stronger"
    # than device functions.
    #
    # In the future, a caller may specify that device_functions win
    # over colocation, in which case we can add support.
    device_fn_tmp = self._device_function_stack
    self._device_function_stack = traceable_stack.TraceableStack()

    if ignore_existing:
      current_stack = self._colocation_stack
      self._colocation_stack = traceable_stack.TraceableStack()

    if op is not None:
      # offset refers to the stack frame used for storing code location.
      # We use 4, the sum of 1 to use our caller's stack frame and 3
      # to jump over layers of context managers above us.
      self._colocation_stack.push_obj(op, offset=4)

    try:
      yield
    finally:
      # Restore device function stack
      self._device_function_stack = device_fn_tmp
      if op is not None:
        self._colocation_stack.pop_obj()

      # Reset the colocation stack if requested.
      if ignore_existing:
        self._colocation_stack = current_stack

  def _add_device_to_stack(self, device_name_or_function, offset=0):
    """Add device to stack manually, separate from a context manager."""
    total_offset = 1 + offset
    spec = _UserDeviceSpec(device_name_or_function)
    self._device_function_stack.push_obj(spec, offset=total_offset)
    return spec

  @tf_contextlib.contextmanager
  def device(self, device_name_or_function):
    
    """Returns a context manager that specifies the default device to use.

    The `device_name_or_function` argument may either be a device name
    string, a device function, or None:

    * If it is a device name string, all operations constructed in
      this context will be assigned to the device with that name, unless
      overridden by a nested `device()` context.
    * If it is a function, it will be treated as a function from
      Operation objects to device name strings, and invoked each time
      a new Operation is created. The Operation will be assigned to
      the device with the returned name.
    * If it is None, all `device()` invocations from the enclosing context
      will be ignored.

    For information about the valid syntax of device name strings, see
    the documentation in
    [`DeviceNameUtils`](https://www.tensorflow.org/code/tensorflow/core/util/device_name_utils.h).

    For example:

    ```python
    with g.device('/device:GPU:0'):
      # All operations constructed in this context will be placed
      # on GPU 0.
      with g.device(None):
        # All operations constructed in this context will have no
        # assigned device.

    # Defines a function from `Operation` to device string.
    def matmul_on_gpu(n):
      if n.type == "MatMul":
        return "/device:GPU:0"
      else:
        return "/cpu:0"

    with g.device(matmul_on_gpu):
      # All operations of type "MatMul" constructed in this context
      # will be placed on GPU 0; all other operations will be placed
      # on CPU 0.
    ```

    **N.B.** The device scope may be overridden by op wrappers or
    other library code. For example, a variable assignment op
    `v.assign()` must be colocated with the `tf.Variable` `v`, and
    incompatible device scopes will be ignored.

    Args:
      device_name_or_function: The device name or function to use in
        the context.

    Yields:
      A context manager that specifies the default device to use for newly
      created ops.
    """
    self._add_device_to_stack(device_name_or_function, offset=2)
    try:
      yield
    finally:
      self._device_function_stack.pop_obj()

  def _apply_device_functions(self, op):
    """Applies the current device function stack to the given operation."""
    # Apply any device functions in LIFO order, so that the most recently
    # pushed function has the first chance to apply a device to the op.
    # We apply here because the result can depend on the Operation's
    # signature, which is computed in the Operation constructor.
    
    for device_spec in self._device_function_stack.peek_objs():
      if device_spec.function is None:
        break
      op._set_device(device_spec.function(op))
    op._device_code_locations = self._snapshot_device_function_stack_metadata()
    # pylint: enable=protected-access

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def container(self, container_name):
    """Returns a context manager that specifies the resource container to use.

    Stateful operations, such as variables and queues, can maintain their
    states on devices so that they can be shared by multiple processes.
    A resource container is a string name under which these stateful
    operations are tracked. These resources can be released or cleared
    with `tf.Session.reset()`.

    For example:

    ```python
    with g.container('experiment0'):
      # All stateful Operations constructed in this context will be placed
      # in resource container "experiment0".
      v1 = tf.Variable([1.0])
      v2 = tf.Variable([2.0])
      with g.container("experiment1"):
        # All stateful Operations constructed in this context will be
        # placed in resource container "experiment1".
        v3 = tf.Variable([3.0])
        q1 = tf.FIFOQueue(10, tf.float32)
      # All stateful Operations constructed in this context will be
      # be created in the "experiment0".
      v4 = tf.Variable([4.0])
      q1 = tf.FIFOQueue(20, tf.float32)
      with g.container(""):
        # All stateful Operations constructed in this context will be
        # be placed in the default resource container.
        v5 = tf.Variable([5.0])
        q3 = tf.FIFOQueue(30, tf.float32)

    # Resets container "experiment0", after which the state of v1, v2, v4, q1
    # will become undefined (such as uninitialized).
    tf.Session.reset(target, ["experiment0"])
    ```

    Args:
      container_name: container name string.

    Returns:
      A context manager for defining resource containers for stateful ops,
        yields the container name.
    """
    original_container = self._container
    self._container = container_name
    try:
      yield self._container
    finally:
      self._container = original_container

  # pylint: enable=g-doc-return-or-yield

  class _ControlDependenciesController(object):
    """Context manager for `control_dependencies()`."""

    def __init__(self, graph, control_inputs):
      """Create a new `_ControlDependenciesController`.

      A `_ControlDependenciesController` is the context manager for
      `with tf.control_dependencies()` blocks.  These normally nest,
      as described in the documentation for `control_dependencies()`.

      The `control_inputs` argument list control dependencies that must be
      added to the current set of control dependencies.  Because of
      uniquification the set can be empty even if the caller passed a list of
      ops.  The special value `None` indicates that we want to start a new
      empty set of control dependencies instead of extending the current set.

      In that case we also clear the current control flow context, which is an
      additional mechanism to add control dependencies.

      Args:
        graph: The graph that this controller is managing.
        control_inputs: List of ops to use as control inputs in addition
          to the current control dependencies.  None to indicate that
          the dependencies should be cleared.
      """
      self._graph = graph
      if control_inputs is None:
        self._control_inputs_val = []
        self._new_stack = True
      else:
        self._control_inputs_val = control_inputs
        self._new_stack = False
      self._seen_nodes = set()
      self._old_stack = None
      self._old_control_flow_context = None



    def __enter__(self):
      if self._new_stack:
        # Clear the control_dependencies graph.
        self._old_stack = self._graph._control_dependencies_stack
        self._graph._control_dependencies_stack = []
        # Clear the control_flow_context too.
        self._old_control_flow_context = self._graph._get_control_flow_context()
        self._graph._set_control_flow_context(None)
      self._graph._push_control_dependencies_controller(self)

    def __exit__(self, unused_type, unused_value, unused_traceback):
      self._graph._pop_control_dependencies_controller(self)
      if self._new_stack:
        self._graph._control_dependencies_stack = self._old_stack
        self._graph._set_control_flow_context(self._old_control_flow_context)

# pylint: enable=protected-access

    @property
    def control_inputs(self):
      return self._control_inputs_val

    def add_op(self, op):
      self._seen_nodes.add(op)

    def op_in_group(self, op):
      return op in self._seen_nodes

  def _push_control_dependencies_controller(self, controller):
    self._control_dependencies_stack.append(controller)

  def _pop_control_dependencies_controller(self, controller):
    assert self._control_dependencies_stack[-1] is controller
    self._control_dependencies_stack.pop()

  def _current_control_dependencies(self):
    ret = set()
    for controller in self._control_dependencies_stack:
      for op in controller.control_inputs:
        ret.add(op)
    return ret

  def _control_dependencies_for_inputs(self, input_ops):
    """For an op that takes `input_ops` as inputs, compute control inputs.

    The returned control dependencies should yield an execution that
    is equivalent to adding all control inputs in
    self._control_dependencies_stack to a newly created op. However,
    this function attempts to prune the returned control dependencies
    by observing that nodes created within the same `with
    control_dependencies(...):` block may have data dependencies that make
    the explicit approach redundant.

    Args:
      input_ops: The data input ops for an op to be created.

    Returns:
      A list of control inputs for the op to be created.
    """
    ret = []
    for controller in self._control_dependencies_stack:
      # If any of the input_ops already depends on the inputs from controller,
      # we say that the new op is dominated (by that input), and we therefore
      # do not need to add control dependencies for this controller's inputs.
      dominated = False
      for op in input_ops:
        if controller.op_in_group(op):
          dominated = True
          break
      if not dominated:
        # Don't add a control input if we already have a data dependency on i.
        # NOTE(mrry): We do not currently track transitive data dependencies,
        #   so we may add redundant control inputs.
        ret.extend([c for c in controller.control_inputs if c not in input_ops])
    return ret

  def _record_op_seen_by_control_dependencies(self, op):
    """Record that the given op depends on all registered control dependencies.

    Args:
      op: An Operation.
    """
    for controller in self._control_dependencies_stack:
      controller.add_op(op)

  def control_dependencies(self, control_inputs):
    if control_inputs is None:
      return self._ControlDependenciesController(self, None)
    # First convert the inputs to ops, and deduplicate them.
    # NOTE(mrry): Other than deduplication, we do not currently track direct
    #   or indirect dependencies between control_inputs, which may result in
    #   redundant control inputs.
    control_ops = []
    current = self._current_control_dependencies()
    for c in control_inputs:
      if isinstance(c, IndexedSlices):
        c = c.op
      c = self.as_graph_element(c)
      if isinstance(c, Tensor):
        c = c.op
      elif not isinstance(c, Operation):
        raise TypeError("Control input must be Operation or Tensor: %s" % c)
      if c not in current:
        control_ops.append(c)
        current.add(c)
    return self._ControlDependenciesController(self, control_ops)

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def _attr_scope(self, attr_map):
    """EXPERIMENTAL: A context manager for setting attributes on operators.

    This context manager can be used to add additional
    attributes to operators within the scope of the context.

    For example:

       with ops.Graph().as_default() as g:
         f_1 = Foo()  # No extra attributes
         with g._attr_scope({"_a": tf.attr_value_pb2.AttrValue(b=False)}):
           f_2 = Foo()  # Additional attribute _a=False
           with g._attr_scope({"_a": tf.attr_value_pb2.AttrValue(b=True)}):
             f_3 = Foo()  # Additional attribute _a=False
             with g._attr_scope({"_a": None}):
               f_4 = Foo()  # No additional attributes.

    Args:
      attr_map: A dictionary mapping attr name strings to
        AttrValue protocol buffers or None.

    Returns:
      A context manager that sets the kernel label to be used for one or more
      ops created in that context.

    Raises:
      TypeError: If attr_map is not a dictionary mapping
        strings to AttrValue protobufs.
    """
    if not isinstance(attr_map, dict):
      raise TypeError("attr_map must be a dictionary mapping "
                      "strings to AttrValue protocol buffers")
    # The saved_attrs dictionary stores any currently-set labels that
    # will be overridden by this context manager.
    saved_attrs = {}
    # Install the given attribute
    for name, attr in attr_map.items():
      if not (isinstance(name, six.string_types) and
              (isinstance(attr, (type(None), attr_value_pb2.AttrValue)) or
               callable(attr))):
        raise TypeError("attr_map must be a dictionary mapping "
                        "strings to AttrValue protocol buffers or "
                        "callables that emit AttrValue protocol buffers")
      try:
        saved_attrs[name] = self._attr_scope_map[name]
      except KeyError:
        pass
      if attr is None:
        del self._attr_scope_map[name]
      else:
        self._attr_scope_map[name] = attr
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the attributes set for this context, and restore any saved
      # attributes.
      for name, attr in attr_map.items():
        try:
          self._attr_scope_map[name] = saved_attrs[name]
        except KeyError:
          del self._attr_scope_map[name]

  # pylint: enable=g-doc-return-or-yield

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def _kernel_label_map(self, op_to_kernel_label_map):
    """EXPERIMENTAL: A context manager for setting kernel labels.

    This context manager can be used to select particular
    implementations of kernels within the scope of the context.

    For example:

        with ops.Graph().as_default() as g:
          f_1 = Foo()  # Uses the default registered kernel for the Foo op.
          with g.kernel_label_map({"Foo": "v_2"}):
            f_2 = Foo()  # Uses the registered kernel with label "v_2"
                         # for the Foo op.
            with g.kernel_label_map({"Foo": "v_3"}):
              f_3 = Foo()  # Uses the registered kernel with label "v_3"
                           # for the Foo op.
              with g.kernel_label_map({"Foo": ""}):
                f_4 = Foo()  # Uses the default registered kernel
                             # for the Foo op.

    Args:
      op_to_kernel_label_map: A dictionary mapping op type strings to
        kernel label strings.

    Returns:
      A context manager that sets the kernel label to be used for one or more
      ops created in that context.

    Raises:
      TypeError: If op_to_kernel_label_map is not a dictionary mapping
        strings to strings.
    """
    if not isinstance(op_to_kernel_label_map, dict):
      raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_labels dictionary stores any currently-set labels that
    # will be overridden by this context manager.
    saved_labels = {}
    # Install the given label
    for op_type, label in op_to_kernel_label_map.items():
      if not (isinstance(op_type, six.string_types) and
              isinstance(label, six.string_types)):
        raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_labels[op_type] = self._op_to_kernel_label_map[op_type]
      except KeyError:
        pass
      self._op_to_kernel_label_map[op_type] = label
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, label in op_to_kernel_label_map.items():
        try:
          self._op_to_kernel_label_map[op_type] = saved_labels[op_type]
        except KeyError:
          del self._op_to_kernel_label_map[op_type]

  # pylint: enable=g-doc-return-or-yield

  # pylint: disable=g-doc-return-or-yield
  @tf_contextlib.contextmanager
  def gradient_override_map(self, op_type_map):
    """EXPERIMENTAL: A context manager for overriding gradient functions.

    This context manager can be used to override the gradient function
    that will be used for ops within the scope of the context.

    For example:

    ```python
    @tf.RegisterGradient("CustomSquare")
    def _custom_square_grad(op, grad):
      # ...

    with tf.Graph().as_default() as g:
      c = tf.constant(5.0)
      s_1 = tf.square(c)  # Uses the default gradient for tf.square.
      with g.gradient_override_map({"Square": "CustomSquare"}):
        s_2 = tf.square(s_2)  # Uses _custom_square_grad to compute the
                              # gradient of s_2.
    ```

    Args:
      op_type_map: A dictionary mapping op type strings to alternative op
        type strings.

    Returns:
      A context manager that sets the alternative op type to be used for one
      or more ops created in that context.

    Raises:
      TypeError: If `op_type_map` is not a dictionary mapping strings to
        strings.
    """
    if not isinstance(op_type_map, dict):
      raise TypeError("op_type_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_mappings dictionary stores any currently-set mappings that
    # will be overridden by this context manager.
    saved_mappings = {}
    # Install the given label
    for op_type, mapped_op_type in op_type_map.items():
      if not (isinstance(op_type, six.string_types) and
              isinstance(mapped_op_type, six.string_types)):
        raise TypeError("op_type_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_mappings[op_type] = self._gradient_override_map[op_type]
      except KeyError:
        pass
      self._gradient_override_map[op_type] = mapped_op_type
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, mapped_op_type in op_type_map.items():
        try:
          self._gradient_override_map[op_type] = saved_mappings[op_type]
        except KeyError:
          del self._gradient_override_map[op_type]

  # pylint: enable=g-doc-return-or-yield

  def prevent_feeding(self, tensor):
    """Marks the given `tensor` as unfeedable in this graph."""
    self._unfeedable_tensors.add(tensor)

  def is_feedable(self, tensor):
    """Returns `True` if and only if `tensor` is feedable."""
    return tensor not in self._unfeedable_tensors

  def prevent_fetching(self, op):
    """Marks the given `op` as unfetchable in this graph."""
    self._unfetchable_ops.add(op)

  def is_fetchable(self, tensor_or_op):
    """Returns `True` if and only if `tensor_or_op` is fetchable."""
    if isinstance(tensor_or_op, Tensor):
      return tensor_or_op.op not in self._unfetchable_ops
    else:
      return tensor_or_op not in self._unfetchable_ops

  def switch_to_thread_local(self):
    """Make device, colocation and dependencies stacks thread-local.

    Device, colocation and dependencies stacks are not thread-local be default.
    If multiple threads access them, then the state is shared.  This means that
    one thread may affect the behavior of another thread.

    After this method is called, the stacks become thread-local.  If multiple
    threads access them, then the state is not shared.  Each thread uses its own
    value; a thread doesn't affect other threads by mutating such a stack.

    The initial value for every thread's stack is set to the current value
    of the stack when `switch_to_thread_local()` was first called.
    """
    if not self._stack_state_is_thread_local:
      self._stack_state_is_thread_local = True

  @property
  def _device_function_stack(self):
    if self._stack_state_is_thread_local:
      # This may be called from a thread where device_function_stack doesn't yet
      # exist.
      
      if not hasattr(self._thread_local, "_device_function_stack"):
        stack_copy_for_this_thread = self._graph_device_function_stack.copy()
        self._thread_local._device_function_stack = stack_copy_for_this_thread
      return self._thread_local._device_function_stack
      # pylint: enable=protected-access
    else:
      return self._graph_device_function_stack

  @property
  def _device_functions_outer_to_inner(self):
    user_device_specs = self._device_function_stack.peek_objs()
    device_functions = [spec.function for spec in user_device_specs]
    device_functions_outer_to_inner = list(reversed(device_functions))
    return device_functions_outer_to_inner

  def _snapshot_device_function_stack_metadata(self):
    """Return device function stack as a list of TraceableObjects.

    Returns:
      [traceable_stack.TraceableObject, ...] where each TraceableObject's .obj
      member is a displayable name for the user's argument to Graph.device, and
      the filename and lineno members point to the code location where
      Graph.device was called directly or indirectly by the user.
    """
    traceable_objects = self._device_function_stack.peek_traceable_objs()
    snapshot = []
    for obj in traceable_objects:
      obj_copy = obj.copy_metadata()
      obj_copy.obj = obj.obj.display_name
      snapshot.append(obj_copy)
    return snapshot

  @_device_function_stack.setter
  def _device_function_stack(self, device_function_stack):
    if self._stack_state_is_thread_local:
      
      self._thread_local._device_function_stack = device_function_stack
      # pylint: enable=protected-access
    else:
      self._graph_device_function_stack = device_function_stack

  @property
  def _colocation_stack(self):
    """Return thread-local copy of colocation stack."""
    if self._stack_state_is_thread_local:
      # This may be called from a thread where colocation_stack doesn't yet
      # exist.
      
      if not hasattr(self._thread_local, "_colocation_stack"):
        stack_copy_for_this_thread = self._graph_colocation_stack.copy()
        self._thread_local._colocation_stack = stack_copy_for_this_thread
      return self._thread_local._colocation_stack
      # pylint: enable=protected-access
    else:
      return self._graph_colocation_stack

  def _snapshot_colocation_stack_metadata(self):
    """Return colocation stack metadata as a dictionary."""
    traceable_objects = self._colocation_stack.peek_traceable_objs()
    return {obj.obj.name: obj.copy_metadata() for obj in traceable_objects}

  @_colocation_stack.setter
  def _colocation_stack(self, colocation_stack):
    if self._stack_state_is_thread_local:
      
      self._thread_local._colocation_stack = colocation_stack
      # pylint: enable=protected-access
    else:
      self._graph_colocation_stack = colocation_stack

  @property
  def _control_dependencies_stack(self):
    if self._stack_state_is_thread_local:
      # This may be called from a thread where control_dependencies_stack
      # doesn't yet exist.
      if not hasattr(self._thread_local, "_control_dependencies_stack"):
        self._thread_local._control_dependencies_stack = (
            self._graph_control_dependencies_stack[:])
      return self._thread_local._control_dependencies_stack
    else:
      return self._graph_control_dependencies_stack

  @_control_dependencies_stack.setter
  def _control_dependencies_stack(self, control_dependencies):
    if self._stack_state_is_thread_local:
      self._thread_local._control_dependencies_stack = control_dependencies
    else:
      self._graph_control_dependencies_stack = control_dependencies

  @property
  def _distribution_strategy_stack(self):
    """A stack to maintain distribution strategy context for each thread."""
    if not hasattr(self._thread_local, "_distribution_strategy_stack"):
      self._thread_local._distribution_strategy_stack = []  
    return self._thread_local._distribution_strategy_stack  

  @_distribution_strategy_stack.setter
  def _distribution_strategy_stack(self, _distribution_strategy_stack):
    self._thread_local._distribution_strategy_stack = (  
        _distribution_strategy_stack)

  def _mutation_lock(self):
    """Returns a lock to guard code that creates & mutates ops.

    See the comment for self._group_lock for more info.
    """
    return self._group_lock.group(_MUTATION_LOCK_GROUP)

  def _session_run_lock(self):
    """Returns a lock to guard code for Session.run.

    See the comment for self._group_lock for more info.
    """
    return self._group_lock.group(_SESSION_RUN_LOCK_GROUP)


# TODO(agarwal): currently device directives in an outer eager scope will not
# apply to inner graph mode code. Fix that.


@tf_export("device")
def device(device_name_or_function):
  """Wrapper for `Graph.device()` using the default graph.

  See
  `tf.Graph.device`
  for more details.

  Args:
    device_name_or_function: The device name or function to use in
      the context.

  Returns:
    A context manager that specifies the default device to use for newly
    created ops.

  Raises:
    RuntimeError: If eager execution is enabled and a function is passed in.
  """
  if context.executing_eagerly():
    # TODO(agarwal): support device functions in EAGER mode.
    if callable(device_name_or_function):
      raise RuntimeError(
          "tf.device does not support functions when eager execution "
          "is enabled.")
    return context.device(device_name_or_function)
  else:
    return get_default_graph().device(device_name_or_function)


@tf_export("container")
def container(container_name):
  """Wrapper for `Graph.container()` using the default graph.

  Args:
    container_name: The container string to use in the context.

  Returns:
    A context manager that specifies the default container to use for newly
    created stateful ops.
  """
  return get_default_graph().container(container_name)


def _colocate_with_for_gradient(op, gradient_uid, ignore_existing=False):
  if context.executing_eagerly():
    if op is not None:
      return device(op.device)
    else:
      return _NullContextmanager()
  else:
    default_graph = get_default_graph()
    if isinstance(op, EagerTensor):
      if default_graph.building_function:
        return default_graph.device(op.device)
      else:
        raise ValueError("Encountered an Eager-defined Tensor during graph "
                         "construction, but a function was not being built.")
    return default_graph._colocate_with_for_gradient(
        op, gradient_uid=gradient_uid, ignore_existing=ignore_existing)


@tf_export("colocate_with")
def colocate_with(op, ignore_existing=False):
  return _colocate_with_for_gradient(op, None, ignore_existing=ignore_existing)


@tf_export("control_dependencies")
def control_dependencies(control_inputs):
  if context.executing_eagerly():
    if control_inputs:
      # Excute any pending callables.
      for control in control_inputs:
        if callable(control):
          control()
    return _NullContextmanager()
  else:
    return get_default_graph().control_dependencies(control_inputs)


class _DefaultStack(threading.local):
  """A thread-local stack of objects for providing implicit defaults."""

  def __init__(self):
    super(_DefaultStack, self).__init__()
    self._enforce_nesting = True
    self.stack = []

  def get_default(self):
    return self.stack[-1] if len(self.stack) >= 1 else None

  def reset(self):
    self.stack = []

  def is_cleared(self):
    return not self.stack

  @property
  def enforce_nesting(self):
    return self._enforce_nesting

  @enforce_nesting.setter
  def enforce_nesting(self, value):
    self._enforce_nesting = value

  @tf_contextlib.contextmanager
  def get_controller(self, default):
    """A context manager for manipulating a default stack."""
    self.stack.append(default)
    try:
      yield default
    finally:
      # stack may be empty if reset() was called
      if self.stack:
        if self._enforce_nesting:
          if self.stack[-1] is not default:
            raise AssertionError(
                "Nesting violated for default stack of %s objects" %
                type(default))
          self.stack.pop()
        else:
          self.stack.remove(default)


_default_session_stack = _DefaultStack()  


def default_session(session):
  """Python "with" handler for defining a default session.

  This function provides a means of registering a session for handling
  Tensor.eval() and Operation.run() calls. It is primarily intended for use
  by session.Session, but can be used with any object that implements
  the Session.run() interface.

  Use with the "with" keyword to specify that Tensor.eval() and Operation.run()
  invocations within the scope of a block should be executed by a particular
  session.

  The default session applies to the current thread only, so it is always
  possible to inspect the call stack and determine the scope of a default
  session. If you create a new thread, and wish to use the default session
  in that thread, you must explicitly add a "with ops.default_session(sess):"
  block in that thread's function.

  Example:
    The following code examples are equivalent:

    # 1. Using the Session object directly:
    sess = ...
    c = tf.constant(5.0)
    sess.run(c)

    # 2. Using default_session():
    sess = ...
    with ops.default_session(sess):
      c = tf.constant(5.0)
      result = c.eval()

    # 3. Overriding default_session():
    sess = ...
    with ops.default_session(sess):
      c = tf.constant(5.0)
      with ops.default_session(...):
        c.eval(session=sess)

  Args:
    session: The session to be installed as the default session.

  Returns:
    A context manager for the default session.
  """
  return _default_session_stack.get_controller(session)


@tf_export("get_default_session")
def get_default_session():
  """Returns the default session for the current thread.

  The returned `Session` will be the innermost session on which a
  `Session` or `Session.as_default()` context has been entered.

  NOTE: The default session is a property of the current thread. If you
  create a new thread, and wish to use the default session in that
  thread, you must explicitly add a `with sess.as_default():` in that
  thread's function.

  Returns:
    The default `Session` being used in the current thread.
  """
  return _default_session_stack.get_default()


def _eval_using_default_session(tensors, feed_dict, graph, session=None):
  if session is None:
    session = get_default_session()
    if session is None:
      raise ValueError("Cannot evaluate tensor using `eval()`: No default "
                       "session is registered. Use `with "
                       "sess.as_default()` or pass an explicit session to "
                       "`eval(session=sess)`")
    if session.graph is not graph:
      raise ValueError("Cannot use the default session to evaluate tensor: "
                       "the tensor's graph is different from the session's "
                       "graph. Pass an explicit session to "
                       "`eval(session=sess)`.")
  else:
    if session.graph is not graph:
      raise ValueError("Cannot use the given session to evaluate tensor: "
                       "the tensor's graph is different from the session's "
                       "graph.")
  return session.run(tensors, feed_dict)


def _run_using_default_session(operation, feed_dict, graph, session=None):
  """Uses the default session to run "operation".

  Args:
    operation: The Operation to be run.
    feed_dict: A dictionary that maps Tensor objects (or tensor names) to lists,
      numpy ndarrays, TensorProtos, or strings.
    graph: The graph in which "operation" is defined.
    session: (Optional) A different session to use to run "operation".

  Raises:
    ValueError: If no default session is available; the default session
      does not have "graph" as its graph; or if "session" is specified,
      and it does not have "graph" as its graph.
  """
  if session is None:
    session = get_default_session()
    if session is None:
      raise ValueError("Cannot execute operation using `run()`: No default "
                       "session is registered. Use `with "
                       "sess.as_default():` or pass an explicit session to "
                       "`run(session=sess)`")
    if session.graph is not graph:
      raise ValueError("Cannot use the default session to execute operation: "
                       "the operation's graph is different from the "
                       "session's graph. Pass an explicit session to "
                       "run(session=sess).")
  else:
    if session.graph is not graph:
      raise ValueError("Cannot use the given session to execute operation: "
                       "the operation's graph is different from the session's "
                       "graph.")
  session.run(operation, feed_dict)


class _DefaultGraphStack(_DefaultStack):  
  """A thread-local stack of objects for providing an implicit default graph."""

  def __init__(self):
    super(_DefaultGraphStack, self).__init__()
    self._global_default_graph = None

  def get_default(self):
    """Override that returns a global default if the stack is empty."""
    ret = super(_DefaultGraphStack, self).get_default()
    if ret is None:
      ret = self._GetGlobalDefaultGraph()
    return ret

  def _GetGlobalDefaultGraph(self):
    if self._global_default_graph is None:
      # TODO(mrry): Perhaps log that the default graph is being used, or set
      #   provide some other feedback to prevent confusion when a mixture of
      #   the global default graph and an explicit graph are combined in the
      #   same process.
      self._global_default_graph = Graph()
    return self._global_default_graph

  def reset(self):
    super(_DefaultGraphStack, self).reset()
    self._global_default_graph = None

  @tf_contextlib.contextmanager
  def get_controller(self, default):
    context.context().context_switches.push(
        default.building_function, default.as_default)
    try:
      with super(_DefaultGraphStack, self).get_controller(
          default) as g, context.graph_mode():
        yield g
    finally:
      # If an exception is raised here it may be hiding a related exception in
      # the try-block (just above).
      context.context().context_switches.pop()


_default_graph_stack = _DefaultGraphStack()


# pylint: disable=g-doc-return-or-yield,line-too-long
@tf_export("init_scope")
@tf_contextlib.contextmanager
def init_scope():
  # pylint: enable=g-doc-return-or-yield,line-too-long

  if context.executing_eagerly():
    # Fastpath.
    with stop_recording():
      yield
  else:
    # Retrieve the active name scope: entering an `init_scope` preserves
    # the name scope of the current context.
    default_graph = get_default_graph()
    scope = default_graph.get_name_scope()
    if scope and scope[-1] != "/":
      # Names that end with trailing slashes are treated by `name_scope` as
      # absolute.
      scope = scope + "/"
    inner_device_stack = default_graph._device_function_stack  

    outer_context = None
    if not _default_graph_stack.stack:
      # If the default graph stack is empty, then we cannot be building a
      # function. Install the global graph (which, in this case, is also the
      # default graph) as the outer context.
      if default_graph.building_function:
        raise RuntimeError("The global graph is building a function.")
      outer_context = default_graph.as_default
    else:
      # Find a context that is not building a function.
      for stack_entry in reversed(context.context().context_switches.stack):
        if not stack_entry.is_building_function:
          outer_context = stack_entry.enter_context_fn
          break

      if outer_context is None:
        # As a last resort, obtain the global default graph; this graph doesn't
        # necessarily live on the graph stack (and hence it doesn't necessarily
        # live on the context stack), but it is stored in the graph stack's
        # encapsulating object.
        outer_context = _default_graph_stack._GetGlobalDefaultGraph().as_default  

    if outer_context is None:
      # Sanity check; this shouldn't be triggered.
      raise RuntimeError("All graphs are building functions, and no "
                         "eager context was previously active.")

    outer_graph = None
    outer_device_stack = None
    try:
      with outer_context(), name_scope(scope), control_dependencies(
          None), stop_recording():
        if not context.executing_eagerly():
          # The device stack is preserved when lifting into a graph. Eager
          # execution doesn't implement device stacks and in particular it
          # doesn't support device functions, so in general it's not possible
          # to do the same when lifting into the eager context.
          outer_graph = get_default_graph()
          outer_device_stack = outer_graph._device_function_stack  
          outer_graph._device_function_stack = inner_device_stack  
        yield
    finally:
      # If an exception is raised here it may be hiding a related exception in
      # try-block (just above).
      if outer_graph is not None:
        outer_graph._device_function_stack = outer_device_stack  


@tf_export("enable_eager_execution")
def enable_eager_execution(config=None,
                           device_policy=None,
                           execution_mode=None):
  if context.default_execution_mode != context.EAGER_MODE:
    return enable_eager_execution_internal(
        config=config,
        device_policy=device_policy,
        execution_mode=execution_mode,
        server_def=None)


def enable_eager_execution_internal(config=None,
                                    device_policy=None,
                                    execution_mode=None,
                                    server_def=None):
  if config is not None and not isinstance(config, config_pb2.ConfigProto):
    raise TypeError(
        "config must be a tf.ConfigProto, but got %s" % type(config))
  if device_policy not in (None, context.DEVICE_PLACEMENT_EXPLICIT,
                           context.DEVICE_PLACEMENT_WARN,
                           context.DEVICE_PLACEMENT_SILENT,
                           context.DEVICE_PLACEMENT_SILENT_FOR_INT32):
    raise ValueError(
        "device_policy must be one of None, tf.contrib.eager.DEVICE_PLACEMENT_*"
    )
  if execution_mode not in (None, context.SYNC, context.ASYNC):
    raise ValueError(
        "execution_mode must be one of None, tf.contrib.eager.SYNC, "
        "tf.contrib.eager.ASYNC")
  if context.default_execution_mode == context.GRAPH_MODE:
    graph_mode_has_been_used = (
        _default_session_stack.stack
        or len(get_default_graph().get_operations()) > 0)  # pylint: disable=g-explicit-length-test
    if graph_mode_has_been_used:
      raise ValueError(
          "tf.enable_eager_execution must be called at program startup.")
  context.default_execution_mode = context.EAGER_MODE
  
  if context._context is None:
    context._context = context.Context(
        config=config,
        device_policy=device_policy,
        execution_mode=execution_mode,
        server_def=server_def)
  elif ((config is not None and config is not context._context._config) or
        (device_policy is not None and
         device_policy is not context._context._device_policy) or
        (execution_mode is not None and
         execution_mode is not context._context._execution_mode)):
    raise ValueError("Trying to change the options of an active eager"
                     " execution. Context config: %s, specified config:"
                     " %s. Context device policy: %s, specified device"
                     " policy: %s. Context execution mode: %s, "
                     " specified execution mode %s." %
                     (context._context._config, config,
                      context._context._device_policy, device_policy,
                      context._context._execution_mode, execution_mode))
  else:
    raise ValueError(
        "tf.enable_eager_execution must be called at program startup.")

  # Monkey patch to get rid of an unnecessary conditional since the context is
  # now initialized.
  context.context = context.context_safe


def eager_run(main=None, argv=None):
  """Runs the program with an optional main function and argv list.

  The program will run with eager execution enabled.

  Example:
  ```python
  import tensorflow as tf
  # Import subject to future changes:
  from tensorflow.contrib.eager.python import tfe

  def main(_):
    u = tf.constant(6.0)
    v = tf.constant(7.0)
    print(u * v)

  if __name__ == "__main__":
    tfe.run()
  ```

  Args:
    main: the main function to run.
    argv: the arguments to pass to it.
  """
  enable_eager_execution()
  app.run(main, argv)


@tf_export("reset_default_graph")
def reset_default_graph():
  """Clears the default graph stack and resets the global default graph.

  NOTE: The default graph is a property of the current thread. This
  function applies only to the current thread.  Calling this function while
  a `tf.Session` or `tf.InteractiveSession` is active will result in undefined
  behavior. Using any previously created `tf.Operation` or `tf.Tensor` objects
  after calling this function will result in undefined behavior.
  Raises:
    AssertionError: If this function is called within a nested graph.
  """
  if not _default_graph_stack.is_cleared():
    raise AssertionError("Do not use tf.reset_default_graph() to clear "
                         "nested graphs. If you need a cleared graph, "
                         "exit the nesting and create a new graph.")
  _default_graph_stack.reset()


@tf_export("get_default_graph")
def get_default_graph():
  return _default_graph_stack.get_default()

def has_default_graph():
  """Returns True if there is a default graph."""
  return len(_default_graph_stack.stack) >= 1


def get_name_scope():
  """Returns the current name scope in the default_graph.

  For example:

  ```python
  with tf.name_scope('scope1'):
    with tf.name_scope('scope2'):
      print(tf.get_name_scope())
  ```
  would print the string `scope1/scope2`.

  Returns:
    A string representing the current name scope.
  """
  if context.executing_eagerly():
    return context.context().scope_name.rstrip("/")
  return get_default_graph().get_name_scope()


def _assert_same_graph(original_item, item):
  """Fail if the 2 items are from different graphs.

  Args:
    original_item: Original item to check against.
    item: Item to check.

  Raises:
    ValueError: if graphs do not match.
  """
  if original_item.graph is not item.graph:
    raise ValueError("%s must be from the same graph as %s." % (item,
                                                                original_item))


def _get_graph_from_inputs(op_input_list, graph=None):
  if get_default_graph().building_function:
    return get_default_graph()

  op_input_list = tuple(op_input_list)  # Handle generators correctly
  if graph and not isinstance(graph, Graph):
    raise TypeError("Input graph needs to be a Graph: %s" % graph)

  # 1. We validate that all of the inputs are from the same graph. This is
  #    either the supplied graph parameter, or the first one selected from one
  #    the graph-element-valued inputs. In the latter case, we hold onto
  #    that input in original_graph_element so we can provide a more
  #    informative error if a mismatch is found.
  original_graph_element = None
  for op_input in op_input_list:
    # Determine if this is a valid graph_element.
    # TODO(josh11b): Note that we exclude subclasses of Tensor. Need to clean this
    # up.
    graph_element = None
    if (isinstance(op_input, (Operation, _TensorLike)) and
        ((not isinstance(op_input, Tensor)) or type(op_input) == Tensor)):  # pylint: disable=unidiomatic-typecheck
      graph_element = op_input
    else:
      graph_element = _as_graph_element(op_input)

    if graph_element is not None:
      if not graph:
        original_graph_element = graph_element
        graph = graph_element.graph
      elif original_graph_element is not None:
        _assert_same_graph(original_graph_element, graph_element)
      elif graph_element.graph is not graph:
        raise ValueError("%s is not from the passed-in graph." % graph_element)

  # 2. If all else fails, we use the default graph, which is always there.
  return graph or get_default_graph()


@tf_export("GraphKeys")
class GraphKeys(object):

  # Key to collect Variable objects that are global (shared across machines).
  # Default collection for all variables, except local ones.
  GLOBAL_VARIABLES = "variables"
  # Key to collect local variables that are local to the machine and are not
  # saved/restored.
  LOCAL_VARIABLES = "local_variables"
  # Key to collect local variables which are used to accumulate interal state
  # to be used in tf.metrics.*.
  METRIC_VARIABLES = "metric_variables"
  # Key to collect model variables defined by layers.
  MODEL_VARIABLES = "model_variables"
  # Key to collect Variable objects that will be trained by the
  # optimizers.
  TRAINABLE_VARIABLES = "trainable_variables"
  # Key to collect summaries.
  SUMMARIES = "summaries"
  # Key to collect QueueRunners.
  QUEUE_RUNNERS = "queue_runners"
  # Key to collect table initializers.
  TABLE_INITIALIZERS = "table_initializer"
  # Key to collect asset filepaths. An asset represents an external resource
  # like a vocabulary file.
  ASSET_FILEPATHS = "asset_filepaths"
  # Key to collect Variable objects that keep moving averages.
  MOVING_AVERAGE_VARIABLES = "moving_average_variables"
  # Key to collect regularization losses at graph construction.
  REGULARIZATION_LOSSES = "regularization_losses"
  # Key to collect concatenated sharded variables.
  CONCATENATED_VARIABLES = "concatenated_variables"
  # Key to collect savers.
  SAVERS = "savers"
  # Key to collect weights
  WEIGHTS = "weights"
  # Key to collect biases
  BIASES = "biases"
  # Key to collect activations
  ACTIVATIONS = "activations"
  # Key to collect update_ops
  UPDATE_OPS = "update_ops"
  # Key to collect losses
  LOSSES = "losses"
  # Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
  SAVEABLE_OBJECTS = "saveable_objects"
  # Key to collect all shared resources used by the graph which need to be
  # initialized once per cluster.
  RESOURCES = "resources"
  # Key to collect all shared resources used in this graph which need to be
  # initialized once per session.
  LOCAL_RESOURCES = "local_resources"
  # Trainable resource-style variables.
  TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables"

  # Key to indicate various ops.
  INIT_OP = "init_op"
  LOCAL_INIT_OP = "local_init_op"
  READY_OP = "ready_op"
  READY_FOR_LOCAL_INIT_OP = "ready_for_local_init_op"
  SUMMARY_OP = "summary_op"
  GLOBAL_STEP = "global_step"

  # Used to count the number of evaluations performed during a single evaluation
  # run.
  EVAL_STEP = "eval_step"
  TRAIN_OP = "train_op"

  # Key for control flow context.
  COND_CONTEXT = "cond_context"
  WHILE_CONTEXT = "while_context"

  # Used to store v2 summary names.
  _SUMMARY_COLLECTION = "_SUMMARY_V2"

  # List of all collections that keep track of variables.
  _VARIABLE_COLLECTIONS = [
      GLOBAL_VARIABLES,
      LOCAL_VARIABLES,
      METRIC_VARIABLES,
      MODEL_VARIABLES,
      TRAINABLE_VARIABLES,
      MOVING_AVERAGE_VARIABLES,
      CONCATENATED_VARIABLES,
      TRAINABLE_RESOURCE_VARIABLES,
  ]

  # Key for streaming model ports.
  # NOTE(yuanbyu): internal and experimental.
  _STREAMING_MODEL_PORTS = "streaming_model_ports"

  @decorator_utils.classproperty
  @deprecation.deprecated(None, "Use `tf.GraphKeys.GLOBAL_VARIABLES` instead.")
  def VARIABLES(cls):  # pylint: disable=no-self-argument
    return cls.GLOBAL_VARIABLES


def dismantle_graph(graph):
  """Cleans up reference cycles from a `Graph`.

  Helpful for making sure the garbage collector doesn't need to run after a
  temporary `Graph` is no longer needed.

  Args:
    graph: A `Graph` object to destroy. Neither it nor any of its ops are usable
      after this function runs.
  """
  memory.dismantle_ordered_dict(graph._functions)  

  # Now clean up Operation<->Graph reference cycles by clearing all of the
  # attributes for the Graph and its ops.
  graph_operations = graph.get_operations()
  for op in graph_operations:
    op.__dict__ = {}
  graph.__dict__ = {}


@tf_export("add_to_collection")
def add_to_collection(name, value):
  """Wrapper for `Graph.add_to_collection()` using the default graph.

  See `tf.Graph.add_to_collection`
  for more details.

  Args:
    name: The key for the collection. For example, the `GraphKeys` class
      contains many standard names for collections.
    value: The value to add to the collection.

  @compatibility(eager)
  Collections are only supported in eager when variables are created inside an
  EagerVariableStore (e.g. as part of a layer or template).
  @end_compatibility
  """
  get_default_graph().add_to_collection(name, value)

@tf_export("add_to_collections")
def add_to_collections(names, value):
  """Wrapper for `Graph.add_to_collections()` using the default graph.

  See `tf.Graph.add_to_collections`
  for more details.

  Args:
    names: The key for the collections. The `GraphKeys` class
      contains many standard names for collections.
    value: The value to add to the collections.

  @compatibility(eager)
  Collections are only supported in eager when variables are created inside an
  EagerVariableStore (e.g. as part of a layer or template).
  @end_compatibility
  """
  get_default_graph().add_to_collections(names, value)


@tf_export("get_collection_ref")
def get_collection_ref(key):
  return get_default_graph().get_collection_ref(key)


@tf_export("get_collection")
def get_collection(key, scope=None):
  return get_default_graph().get_collection(key, scope)


def get_all_collection_keys():
  """Returns a list of collections used in the default graph."""
  return get_default_graph().get_all_collection_keys()


name_scope_cache = {}


# Named like a function for backwards compatibility with the
# @tf_contextlib.contextmanager version, which was switched to a class to avoid
# some object creation overhead.
@tf_export("name_scope", "keras.backend.name_scope")
class name_scope(object):  # pylint: disable=invalid-name

  @property
  def name(self):
    return self._name

  def __init__(self, name, default_name=None, values=None):
    self._name = default_name if name is None else name
    self._default_name = default_name
    self._values = values
    self._ctx = context.context()
    self._in_eager_mode = self._ctx.executing_eagerly()

  def __enter__(self):
    if self._in_eager_mode:
      self._old_name = self._ctx.scope_name
      if not self._name:
        scope_name = ""
      else:
        cache_key = self._name, self._old_name, self._default_name
        if cache_key in name_scope_cache:
          self._ctx.scope_name = name_scope_cache[cache_key]
          return self._ctx.scope_name
        elif self._name[-1] == "/":
          scope_name = self._name
        else:
          name_with_trailing_slash = self._name + "/"
          scope_name = (
              self._old_name + name_with_trailing_slash
              if self._old_name else name_with_trailing_slash)
        name_scope_cache[cache_key] = scope_name
      self._ctx.scope_name = scope_name
      return scope_name
    else:
      if self._name is None and self._values is not None:
        raise ValueError(
            "At least one of name (%s) and default_name (%s) must be provided."
            % (self._name, self._default_name))
      if self._values is None:
        self._values = []
      g = _get_graph_from_inputs(self._values)
      self._g_manager = g.as_default()
      self._g_manager.__enter__()
      try:
        self._name_scope = g.name_scope(self._name)
        return self._name_scope.__enter__()
      except:
        self._g_manager.__exit__(*sys.exc_info())
        raise

  def __exit__(self, type_arg, value_arg, traceback_arg):
    if self._in_eager_mode:
      self._ctx.scope_name = self._old_name
    else:
      self._name_scope.__exit__(type_arg, value_arg, traceback_arg)
      self._g_manager.__exit__(type_arg, value_arg, traceback_arg)
    return False

_proto_function_registry = registry.Registry("proto functions")

def register_proto_function(collection_name,
                            proto_type=None,
                            to_proto=None,
                            from_proto=None):
  if to_proto and not callable(to_proto):
    raise TypeError("to_proto must be callable.")
  if from_proto and not callable(from_proto):
    raise TypeError("from_proto must be callable.")

  _proto_function_registry.register((proto_type, to_proto, from_proto),
                                    collection_name)

