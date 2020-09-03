from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import ops
from tensorflow.core import variable_pb2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables


class EagerResourceDeleter(object):
  def __init__(self, handle, handle_device):
    if not isinstance(handle, ops.Tensor):
      raise ValueError(
          ("Passed handle=%s to EagerResourceDeleter. Was expecting a handle "
           "Tensor." % (handle,)))
    self._handle = handle
    self._handle_device = handle_device

  def __del__(self):
    try:
      with context.eager_mode():
        with ops.device(self._handle_device):
          gen_resource_variable_ops.destroy_resource_op(
              self._handle, ignore_lookup_error=True)
    except TypeError:
      pass
    except AttributeError:
      pass

def shape_safe_assign_variable_handle(handle, shape, value, name=None):
  with _handle_graph(handle):
    value_tensor = ops.convert_to_tensor(value)
  shape.assert_is_compatible_with(value_tensor.shape)
  return gen_resource_variable_ops.assign_variable_op(handle,
                                                      value_tensor,
                                                      name=name)

class ResourceVariable(variables.RefVariable):
  def __init__(self,
               initial_value=None,
               trainable=True,
               collections=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               dtype=None,
               variable_def=None,
               import_scope=None,
               constraint=None):
    if variable_def:
      if initial_value is not None:
        raise ValueError("variable_def and initial_value are mutually "
                         "exclusive.")
      if context.executing_eagerly():
        raise ValueError("Creating ResourceVariable from variable_def is "
                         "not supported when eager execution is enabled.")
      self._init_from_proto(variable_def, import_scope=import_scope)
    else:
      self._init_from_args(
          initial_value=initial_value,
          trainable=trainable,
          collections=collections,
          validate_shape=validate_shape,
          caching_device=caching_device,
          name=name,
          dtype=dtype,
          constraint=constraint)

  def _init_from_args(self,
                      initial_value=None,
                      trainable=True,
                      collections=None,
                      validate_shape=True,
                      caching_device=None,
                      name=None,
                      dtype=None,
                      constraint=None):
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)

    if isinstance(initial_value, ops.Tensor) and hasattr(
        initial_value, "graph") and initial_value.graph.building_function:
      raise ValueError("Tensor-typed variable initializers must either be "
                       "wrapped in an init_scope or callable "
                       "(e.g., `tf.Variable(lambda : "
                       "tf.truncated_normal([10, 40]))`) when building "
                       "functions. Please file a feature request if this "
                       "restriction inconveniences you.")

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
          "collections argument to Variable constructor must be a list, tuple, "
          "or set. Got %s of type %s" % (collections, type(collections)))
    if constraint is not None and not callable(constraint):
      raise ValueError("The `constraint` argument must be a callable.")

    if isinstance(initial_value, checkpointable.CheckpointInitialValue):
      self._maybe_initialize_checkpointable()
      self._update_uid = initial_value.checkpoint_position.restore_uid
      initial_value = initial_value.wrapped_value

    self._trainable = trainable
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    self._save_slice_info = None
    # Store the graph key so optimizers know how to only retrieve variables from
    # this graph.
    self._graph_key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
      with ops.name_scope(name, "Variable", []
                          if init_from_fn else [initial_value]) as name:
        # pylint: disable=protected-access
        handle_name = ops._name_from_scope_name(name)
        if self._in_graph_mode:
          shared_name = handle_name
        else:
          # When in eager mode use a uid for the shared_name, to prevent
          # accidental sharing.
          shared_name = "%s_%d" % (handle_name, ops.uid())
        # Use attr_scope and device(None) to simulate the behavior of
        # colocate_with when the variable we want to colocate with doesn't
        # yet exist.
        attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                s=[compat.as_bytes("loc:@%s" % handle_name)]))
        with ops.get_default_graph()._attr_scope({"_class": attr}):
          with ops.name_scope("Initializer"), ops.device(None):
            initial_value = ops.convert_to_tensor(
                initial_value() if init_from_fn else initial_value,
                name="initial_value", dtype=dtype)
          self._handle = eager_safe_variable_handle(
              shape=initial_value.get_shape(),
              dtype=initial_value.dtype.base_dtype,
              shared_name=shared_name,
              name=name,
              graph_mode=self._in_graph_mode)
        self._shape = initial_value.shape
        # pylint: disable=protected-access
        if (self._in_graph_mode and initial_value is not None and
            initial_value.op._get_control_flow_context() is not None):
          raise ValueError(
              "Initializer for variable %s is from inside a control-flow "
              "construct, such as a loop or conditional. When creating a "
              "variable inside a loop or conditional, use a lambda as the "
              "initializer." % name)
        # pylint: enable=protected-access
        self._unique_id = shared_name
        self._initial_value = initial_value if self._in_graph_mode else None
        self._handle_name = handle_name + ":0"
        self._dtype = initial_value.dtype.base_dtype
        self._constraint = constraint

        if self._in_graph_mode:
          with ops.name_scope("IsInitialized"):
            self._is_initialized_op = (
                gen_resource_variable_ops.var_is_initialized_op(self._handle))
          if initial_value is not None:
            with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
              self._initializer_op = (
                  gen_resource_variable_ops.assign_variable_op(
                      self._handle,
                      self._try_guard_against_uninitialized_dependencies(
                          initial_value),
                      name=n))
          with ops.name_scope("Read"), ops.colocate_with(self._handle):
            # Manually assign reads to the handle's device to avoid log
            # messages.
            with ops.device(self._handle.device):
              value = self._read_variable_op()
            self._graph_element = value
            if caching_device is not None:
              # Variables may be created in a tf.device() or ops.colocate_with()
              # context. At the same time, users would expect caching device to
              # be independent of this context, and/or would not expect the
              # current device context to be merged with the caching device
              # spec.  Therefore we reset the colocation stack before creating
              # the cached value. Note that resetting the colocation stack will
              # also reset the device stack.
              with ops.colocate_with(None, ignore_existing=True):
                with ops.device(caching_device):
                  self._cached_value = array_ops.identity(value)
            else:
              self._cached_value = None
        else:
          gen_resource_variable_ops.assign_variable_op(self._handle,
                                                       initial_value)
          self._is_initialized_op = None
          self._initializer_op = None
          self._graph_element = None
          if caching_device:
            with ops.device(caching_device):
              self._cached_value = self._read_variable_op()
          else:
            self._cached_value = None
        if not context.executing_eagerly():
          # Eager variables are only added to collections if they are part of an
          # eager variable store (otherwise in an interactive session they would
          # hog memory and cause OOM). This is done in ops/variable_scope.py.
          ops.add_to_collections(collections, self)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)

    if not self._in_graph_mode:
      # After the handle has been created, set up a way to clean it up when
      # executing eagerly. We'll hold the only reference to the deleter, so that
      # when this object is garbage collected the deleter will be too. This
      # means ResourceVariables can be part of reference cycles without those
      # cycles being uncollectable, and means that no __del__ will be defined at
      # all in graph mode.
      self._handle_deleter = EagerResourceDeleter(
          handle=self._handle, handle_device=self._handle.device)
    self._cached_shape_as_list = None

  def _init_from_proto(self, variable_def, import_scope=None):
    assert not context.executing_eagerly()
    self._in_graph_mode = True
    assert isinstance(variable_def, variable_pb2.VariableDef)
    if not variable_def.is_resource:
      raise ValueError("Trying to restore Variable as ResourceVariable.")

    # Create from variable_def.
    g = ops.get_default_graph()
    self._handle = g.as_graph_element(
        ops.prepend_name_scope(
            variable_def.variable_name, import_scope=import_scope))
    self._shape = tensor_shape.TensorShape(
        self._handle.op.get_attr("shape"))
    self._handle_name = self._handle.name
    self._unique_id = self._handle_name
    self._initializer_op = g.as_graph_element(
        ops.prepend_name_scope(
            variable_def.initializer_name, import_scope=import_scope))
    # Check whether initial_value_name exists for backwards compatibility.
    if (hasattr(variable_def, "initial_value_name") and
        variable_def.initial_value_name):
      self._initial_value = g.as_graph_element(
          ops.prepend_name_scope(variable_def.initial_value_name,
                                 import_scope=import_scope))
    else:
      self._initial_value = None
    self._trainable = getattr(variable_def, "trainable", True)
    if variable_def.snapshot_name:
      snapshot = g.as_graph_element(
          ops.prepend_name_scope(
              variable_def.snapshot_name, import_scope=import_scope))
      self._cached_value = snapshot
      while snapshot.op.type != "ReadVariableOp":
        snapshot = snapshot.op.inputs[0]
      self._graph_element = snapshot
    else:
      self._cached_value = None
      # Legacy case for protos without the snapshot name; assume it's the
      # following.
      self._graph_element = g.get_tensor_by_name(
          self._handle.op.name + "/Read/ReadVariableOp:0")
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = variables.Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def,
          import_scope=import_scope)
    else:
      self._save_slice_info = None
    self._caching_device = None
    self._dtype = dtypes.as_dtype(self._handle.op.get_attr("dtype"))
    self._constraint = None
    self._cached_shape_as_list = None

  def _assign_dependencies(self):
    """Makes assignments depend on the cached value, if any.

    This prevents undefined behavior with reads not ordered wrt writes.

    Yields:
      None.
    """
    if self._cached_value is not None:
      with ops.control_dependencies([self._cached_value]):
        yield
    else:
      yield

  def __nonzero__(self):
    return self.__bool__()

  def __bool__(self):
    return bool(self.read_value())

  def __copy__(self):
    return self

  def __deepcopy__(self, memo):
    if not context.executing_eagerly():
      raise NotImplementedError(
          "__deepcopy__() is only available when eager execution is enabled.")
    copied_variable = ResourceVariable(
        initial_value=self.read_value(),
        trainable=self._trainable,
        constraint=self._constraint,
        dtype=self._dtype,
        name=self._shared_name + "_copy")
    memo[self._unique_id] = copied_variable
    return copied_variable

  @property
  def dtype(self):
    """The dtype of this variable."""
    return self._dtype

  @property
  def device(self):
    """The device this variable is on."""
    return self._handle.device

  @property
  def graph(self):
    """The `Graph` of this variable."""
    return self._handle.graph

  @property
  def name(self):
    """The name of the handle for this variable."""
    return self._handle_name

  @property
  def shape(self):
    """The shape of this variable."""
    return self._shape

  def _shape_as_list(self):
    if self._cached_shape_as_list:
      return self._cached_shape_as_list
    if self.shape.ndims is None:
      return None
    self._cached_shape_as_list = [dim.value for dim in self.shape.dims]
    return self._cached_shape_as_list

  def _shape_tuple(self):
    shape = self._shape_as_list()
    if shape is None:
      return None
    return tuple(shape)

  @property
  def create(self):
    """The op responsible for initializing this variable."""
    if not self._in_graph_mode:
      raise RuntimeError("Calling create is not supported when eager execution"
                         " is enabled.")
    return self._initializer_op

  @property
  def handle(self):
    """The handle by which this variable can be accessed."""
    return self._handle

  def value(self):
    """A cached operation which reads the value of this variable."""
    if self._cached_value is not None:
      return self._cached_value
    with ops.colocate_with(None, ignore_existing=True):
      with ops.device(self._handle.device):
        return self._read_variable_op()

  def _as_graph_element(self):
    """Conversion function for Graph.as_graph_element()."""
    return self._graph_element

  @property
  def initializer(self):
    """The op responsible for initializing this variable."""
    return self._initializer_op

  @property
  def initial_value(self):
    """Returns the Tensor used as the initial value for the variable."""
    if context.executing_eagerly():
      raise RuntimeError("initial_value not supported in EAGER mode.")
    return self._initial_value

  @property
  def constraint(self):
    """Returns the constraint function associated with this variable.

    Returns:
      The constraint function that was passed to the variable constructor.
      Can be `None` if no constraint was passed.
    """
    return self._constraint

  @property
  def op(self):
    """The op for this variable."""
    return self._handle.op

  def eval(self, session=None):
    """Evaluates and returns the value of this variable."""
    if context.executing_eagerly():
      raise RuntimeError("Trying to eval in EAGER mode")
    return self._graph_element.eval(session=session)

  def numpy(self):
    if context.executing_eagerly():
      return self.read_value().numpy()
    raise NotImplementedError(
        "numpy() is only available when eager execution is enabled.")

  def count_up_to(self, limit):
    """Increments this variable until it reaches `limit`.

    When that Op is run it tries to increment the variable by `1`. If
    incrementing the variable would bring it above `limit` then the Op raises
    the exception `OutOfRangeError`.

    If no error is raised, the Op outputs the value of the variable before
    the increment.

    This is essentially a shortcut for `count_up_to(self, limit)`.

    Args:
      limit: value at which incrementing the variable raises an error.

    Returns:
      A `Tensor` that will hold the variable value before the increment. If no
      other Op modifies this variable, the values produced will all be
      distinct.
    """
    return gen_state_ops.resource_count_up_to(self.handle, limit=limit,
                                              T=self.dtype)

  def _set_save_slice_info(self, save_slice_info):
    """Sets the slice info for this `ResourceVariable`.

    Args:
      save_slice_info: A `Variable.SaveSliceInfo` object.
    """
    self._save_slice_info = save_slice_info

  def _get_save_slice_info(self):
    return self._save_slice_info

  def _read_variable_op(self):
    if self.trainable:
      tape.variable_accessed(self)
    result = gen_resource_variable_ops.read_variable_op(self._handle,
                                                        self._dtype)
    if not context.executing_eagerly():
      # Note that if a control flow context is active the input of the read op
      # might not actually be the handle. This line bypasses it.
      tape.record_operation(
          "ReadVariableOp", [result], [self._handle], lambda x: [x])
    return result

  def read_value(self):
    """Constructs an op which reads the value of this variable.

    Should be used when there are multiple reads, or when it is desirable to
    read the value only after some condition is true.

    Returns:
     the read operation.
    """
    with ops.name_scope("Read"):
      # Ensure we read the variable in the same device as the handle.
      with ops.device(self._handle.device):
        value = self._read_variable_op()
    # Return an identity so it can get placed on whatever device the context
    # specifies instead of the device where the variable is.
    return array_ops.identity(value)

  def sparse_read(self, indices, name=None):
    """Reads the value of this variable sparsely, using `gather`."""
    with ops.name_scope("Gather" if name is None else name) as name:
      if self.trainable:
        tape.variable_accessed(self)
      value = gen_resource_variable_ops.resource_gather(
          self._handle, indices, dtype=self._dtype, name=name)
    return array_ops.identity(value)

  def to_proto(self, export_scope=None):
    """Converts a `ResourceVariable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Raises:
      RuntimeError: If run in EAGER mode.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    """
    if context.executing_eagerly():
      raise RuntimeError("to_proto not supported in EAGER mode.")
    if export_scope is None or self.handle.name.startswith(export_scope):
      var_def = variable_pb2.VariableDef()
      var_def.variable_name = ops.strip_name_scope(self.handle.name,
                                                   export_scope)
      if self._initial_value is not None:
        # This is inside an if-statement for backwards compatibility, since
        # self._initial_value might be None for variables constructed from old
        # protos.
        var_def.initial_value_name = ops.strip_name_scope(
            self._initial_value.name, export_scope)
      var_def.initializer_name = ops.strip_name_scope(self.initializer.name,
                                                      export_scope)
      if self._cached_value is not None:
        var_def.snapshot_name = ops.strip_name_scope(self._cached_value.name,
                                                     export_scope)
      else:
        # Store the graph_element here
        var_def.snapshot_name = ops.strip_name_scope(self._graph_element.name,
                                                     export_scope)
      var_def.is_resource = True
      var_def.trainable = self.trainable
      if self._save_slice_info:
        var_def.save_slice_info_def.MergeFrom(
            self._save_slice_info.to_proto(export_scope=export_scope))
      return var_def
    else:
      return None

  def from_proto(variable_def, import_scope=None):
    if context.executing_eagerly():
      raise RuntimeError("from_proto not supported in EAGER mode.")
    return ResourceVariable(
        variable_def=variable_def, import_scope=import_scope)

  
  def _OverloadAllOperators():  # pylint: disable=invalid-name
    """Register overloads for all operators."""
    for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
      ResourceVariable._OverloadOperator(operator)
    # For slicing, bind getitem differently than a tensor (use SliceHelperVar
    # instead)
    # pylint: disable=protected-access
    setattr(ResourceVariable, "__getitem__", array_ops._SliceHelperVar)

  def _AsTensor(self):
    return self.value()

  def _ref(self):
    """Unsupported."""
    raise NotImplementedError("ResourceVariable does not implement _ref()")

  def set_shape(self, shape):
    """Unsupported."""
    raise NotImplementedError("ResourceVariable does not implement set_shape()")

  
  def _OverloadOperator(operator):  # pylint: disable=invalid-name
    """Defer an operator overload to `ops.Tensor`.

    We pull the operator out of ops.Tensor dynamically to avoid ordering issues.

    Args:
      operator: string. The operator name.
    """

    tensor_oper = getattr(ops.Tensor, operator)
    def _run_op(a, *args):
      # pylint: disable=protected-access
      value = a._AsTensor()
      return tensor_oper(value, *args)

    # Propagate __doc__ to wrapper
    try:
      _run_op.__doc__ = tensor_oper.__doc__
    except AttributeError:
      pass

    setattr(ResourceVariable, operator, _run_op)

  __array_priority__ = 100

  def is_initialized(self, name=None):
    """Checks whether a resource variable has been initialized.

    Outputs boolean scalar indicating whether the tensor has been initialized.

    Args:
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `bool`.
    """
    return gen_resource_variable_ops.var_is_initialized_op(self.handle, name)

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    """Subtracts a value from this variable.

    Args:
      delta: A `Tensor`. The value to subtract from this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name to use for the operation.
      read_value: A `bool`. Whether to read and return the new value of the
          variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    """
    # TODO(apassos): this here and below is not atomic. Consider making it
    # atomic if there's a way to do so without a performance cost for those who
    # don't need it.
    with _handle_graph(self.handle), self._assign_dependencies():
      assign_sub_op = gen_resource_variable_ops.assign_sub_variable_op(
          self.handle, ops.convert_to_tensor(delta, dtype=self.dtype),
          name=name)
    if read_value:
      return self._lazy_read(assign_sub_op)
    return assign_sub_op

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    """Adds a value to this variable.

    Args:
      delta: A `Tensor`. The value to add to this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name to use for the operation.
      read_value: A `bool`. Whether to read and return the new value of the
          variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    """
    with _handle_graph(self.handle), self._assign_dependencies():
      assign_add_op = gen_resource_variable_ops.assign_add_variable_op(
          self.handle, ops.convert_to_tensor(delta, dtype=self.dtype),
          name=name)
    if read_value:
      return self._lazy_read(assign_add_op)
    return assign_add_op

  def _lazy_read(self, op):
    if self.trainable:
      tape.variable_accessed(self)
    return _UnreadVariable(
        handle=self._handle, dtype=self.dtype, shape=self._shape,
        in_graph_mode=self._in_graph_mode,
        deleter=self._handle_deleter if not self._in_graph_mode else None,
        parent_op=op, unique_id=self._unique_id)

  def assign(self, value, use_locking=None, name=None, read_value=True):
    """Assigns a new value to this variable.

    Args:
      value: A `Tensor`. The new value for this variable.
      use_locking: If `True`, use locking during the assignment.
      name: The name to use for the assignment.
      read_value: A `bool`. Whether to read and return the new value of the
          variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    """
    # Note: not depending on the cached value here since this can used to
    # initialize the variable.
    with _handle_graph(self.handle):
      value_tensor = ops.convert_to_tensor(value, dtype=self.dtype)
      self._shape.assert_is_compatible_with(value_tensor.shape)
      assign_op = gen_resource_variable_ops.assign_variable_op(
          self.handle, value_tensor, name=name)
      if read_value:
        return self._lazy_read(assign_op)
    return assign_op

  def __reduce__(self):
    return (ResourceVariable, (self.numpy(),))

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    """Subtracts `IndexedSlices` from this variable.

    Args:
      sparse_delta: `IndexedSlices` to be subtracted from this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.

    Raises:
      ValueError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return self._lazy_read(gen_resource_variable_ops.resource_scatter_sub(
        self.handle, sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    """Adds `IndexedSlices` from this variable.

    Args:
      sparse_delta: `IndexedSlices` to be added to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.

    Raises:
      ValueError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return self._lazy_read(gen_resource_variable_ops.resource_scatter_add(
        self.handle, sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    """Assigns `IndexedSlices` to this variable.

    Args:
      sparse_delta: `IndexedSlices` to be assigned to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.

    Raises:
      ValueError: if `sparse_delta` is not an `IndexedSlices`.
    """
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return self._lazy_read(gen_resource_variable_ops.resource_scatter_update(
        self.handle, sparse_delta.indices,
        ops.convert_to_tensor(sparse_delta.values, self.dtype), name=name))

  def scatter_nd_sub(self, indices, updates, name=None):
    """Applies sparse subtraction to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = ref.scatter_nd_sub(indices, updates)
        with tf.Session() as sess:
          print sess.run(op)
    ```

    The resulting update to ref would look like this:

        [1, -9, 3, -6, -6, 6, 7, -4]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.

    Raises:
      ValueError: if `sparse_delta` is not an `IndexedSlices`.
    """
    return self._lazy_read(gen_state_ops.resource_scatter_nd_sub(
        self.handle, indices, ops.convert_to_tensor(updates, self.dtype),
        name=name))

  def scatter_nd_add(self, indices, updates, name=None):
    """Applies sparse addition to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        add = ref.scatter_nd_add(indices, updates)
        with tf.Session() as sess:
          print sess.run(add)
    ```

    The resulting update to ref would look like this:

        [1, 13, 3, 14, 14, 6, 7, 20]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.

    Raises:
      ValueError: if `sparse_delta` is not an `IndexedSlices`.
    """
    return self._lazy_read(gen_state_ops.resource_scatter_nd_add(
        self.handle, indices, ops.convert_to_tensor(updates, self.dtype),
        name=name))

  def scatter_nd_update(self, indices, updates, name=None):
    """Applies sparse assignment to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = ref.scatter_nd_update(indices, updates)
        with tf.Session() as sess:
          print sess.run(op)
    ```

    The resulting update to ref would look like this:

        [1, 11, 3, 10, 9, 6, 7, 12]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.

    Raises:
      ValueError: if `sparse_delta` is not an `IndexedSlices`.
    """
    return self._lazy_read(gen_state_ops.resource_scatter_nd_update(
        self.handle, indices, ops.convert_to_tensor(updates, self.dtype),
        name=name))

  def _strided_slice_assign(self, begin, end, strides, value, name, begin_mask,
                            end_mask, ellipsis_mask, new_axis_mask,
                            shrink_axis_mask):
    with _handle_graph(self.handle), self._assign_dependencies():
      return self._lazy_read(
          gen_array_ops.resource_strided_slice_assign(
              ref=self.handle,
              begin=begin,
              end=end,
              strides=strides,
              value=ops.convert_to_tensor(value, dtype=self.dtype),
              name=name,
              begin_mask=begin_mask,
              end_mask=end_mask,
              ellipsis_mask=ellipsis_mask,
              new_axis_mask=new_axis_mask,
              shrink_axis_mask=shrink_axis_mask))

  def __int__(self):
    if self.dtype != dtypes.int32 and self.dtype != dtypes.int64:
      raise TypeError("Non-integer variable can't be converted to integer.")
    return int(self.value().numpy())

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    del name
    if dtype is not None and dtype != self.dtype:
      return NotImplemented
    if as_ref:
      return self.read_value().op.inputs[0]
    else:
      return self.value()

  def __iadd__(self, unused_other):
    raise RuntimeError("Variable += value not supported. Use "
                       "variable.assign_add(value) to modify the variable "
                       "value and variable = variable + value to get a new "
                       "Tensor object.")

  def __isub__(self, unused_other):
    raise RuntimeError("Variable -= value not supported. Use "
                       "variable.assign_sub(value) to modify the variable "
                       "value and variable = variable - value to get a new "
                       "Tensor object.")

  def __imul__(self, unused_other):
    raise RuntimeError("Variable *= value not supported. Use "
                       "`var.assign(var * value)` to modify the variable or "
                       "`var = var * value` to get a new Tensor object.")

  def __idiv__(self, unused_other):
    raise RuntimeError("Variable /= value not supported. Use "
                       "`var.assign(var / value)` to modify the variable or "
                       "`var = var / value` to get a new Tensor object.")

  def __itruediv__(self, unused_other):
    raise RuntimeError("Variable /= value not supported. Use "
                       "`var.assign(var / value)` to modify the variable or "
                       "`var = var / value` to get a new Tensor object.")

  def __irealdiv__(self, unused_other):
    raise RuntimeError("Variable /= value not supported. Use "
                       "`var.assign(var / value)` to modify the variable or "
                       "`var = var / value` to get a new Tensor object.")

  def __ipow__(self, unused_other):
    raise RuntimeError("Variable **= value not supported. Use "
                       "`var.assign(var ** value)` to modify the variable or "
                       "`var = var ** value` to get a new Tensor object.")


pywrap_tensorflow.TFE_Py_RegisterResourceVariableType(ResourceVariable)
math_ops._resource_variable_type = ResourceVariable  # pylint: disable=protected-access


def _dense_var_to_tensor(var, dtype=None, name=None, as_ref=False):
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


class _UnreadVariable(ResourceVariable):
  """Represents a future for a read of a variable.

  Pretends to be the tensor if anyone looks.
  """

  def __init__(self, handle, dtype,  # pylint: disable=super-init-not-called
               shape, in_graph_mode, deleter, parent_op, unique_id):
    # We do not call super init on purpose.
    self._trainable = False
    self._save_slice_info = None
    self._graph_key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    self._in_graph_mode = in_graph_mode
    self._handle = handle
    self._shape = shape
    self._initial_value = None
    if isinstance(self._handle, ops.EagerTensor):
      self._handle_name = ""
    else:
      self._handle_name = self._handle.name
    self._unique_id = unique_id
    self._dtype = dtype
    self._constraint = None
    self._cached_value = None
    self._is_initialized_op = None
    self._initializer_op = None
    self._parent_op = parent_op
    if context.executing_eagerly():
      self._graph_element = None
    else:
      self._graph_element = self.read_value()
    self._handle_deleter = deleter

  @property
  def name(self):
    if self._in_graph_mode:
      return self._parent_op.name
    else:
      return "UnreadVariable"

  def value(self):
    return self._read_variable_op()

  def read_value(self):
    return self._read_variable_op()

  def _read_variable_op(self):
    with ops.control_dependencies([self._parent_op]):
      return gen_resource_variable_ops.read_variable_op(self._handle,
                                                        self._dtype)

  def set_shape(self, shape):
    self._shape = shape
    self._cached_shape_as_list = None

  @property
  def op(self):
    """The op for this variable."""
    return self._parent_op

ops.register_tensor_conversion_function(_UnreadVariable, _dense_var_to_tensor)
ops.register_dense_tensor_like_type(_UnreadVariable)


class _MixedPrecisionVariable(ResourceVariable):
  """Represents a variable that can return in desired dtype when read.

  In mixed precision training, it is usually desirable to use different dtypes
  for variables and computation. This class will be used to wrap created
  ResourceVariable when mixed precision training is enabled. It allows layers to
  perform computation in a different dtype than their variable dtypes, in order
  to achieve higher performance without causing quality loss.
  """

  def __init__(self, var, read_dtype):
    """Creates a MixedPrecisionVariable.

    Args:
      var: A ResourceVariable instance.
      read_dtype: A tf.DType, the returned dtype when read, default to None.
        Casting is performed if read_dtype is not None and differs from
        var.dtype.
    Returns:
      An MixedPrecisionVariable instance.
    Raises:
      ValueError: if var is not a ResourceVariable instance, or read_dtype is
        not a tf.DType instance.
    """
    # pylint: disable=super-init-not-called
    # We do not call super init on purpose.
    if not isinstance(var, ResourceVariable):
      raise ValueError("InvalidArgument: var must be a ResourceVariable type.")
    if not isinstance(read_dtype, dtypes.DType):
      raise ValueError("InvalidArgument: read_dtype must be a tf.DType type.")

    self._var = var
    self._trainable = var.trainable
    self._save_slice_info = None
    self._graph_key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    self._in_graph_mode = var._in_graph_mode  # pylint: disable=protected-access
    self._handle = var.handle
    self._shape = var.shape
    self._initial_value = None
    if isinstance(self.handle, ops.EagerTensor):
      self._handle_name = ""
    else:
      self._handle_name = self.handle.name
    self._unique_id = var._unique_id  # pylint: disable=protected-access
    self._dtype = var.dtype
    self._constraint = None
    self._cached_value = None
    self._is_initialized_op = var._is_initialized_op  # pylint: disable=protected-access
    self._initializer_op = var._initializer_op  # pylint: disable=protected-access
    # This needs to be set before read_value() is called.
    self._read_dtype = read_dtype
    if context.executing_eagerly():
      self._graph_element = None
    else:
      self._graph_element = self.read_value()
    self._handle_deleter = (
        var._handle_deleter if not self._in_graph_mode  # pylint: disable=protected-access
        else None)
    # pylint: enable=super-init-not-called

  @property
  def name(self):
    return self._var.name

  def value(self):
    return self._read_variable_op()

  def read_value(self):
    return self._read_variable_op()

  def _read_variable_op(self):
    with ops.colocate_with(self._handle):
      res = gen_resource_variable_ops.read_variable_op(self._handle,
                                                       self._dtype)
      if self._read_dtype != self._dtype:
        return math_ops.cast(res, self._read_dtype)
      else:
        return res

  def set_shape(self, shape):
    self._shape = shape
    self._cached_shape_as_list = None

  @property
  def op(self):
    """The op for this variable."""
    return self._var.op

  @property
  def read_dtype(self):
    """The dtype of the returned tensor when reading the var."""
    return self._read_dtype

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    del name
    dtype = dtype or self.read_dtype
    if dtype != self.read_dtype or as_ref:
      return NotImplemented
    else:
      res = self.value()
    return res

  def _should_act_as_resource_variable(self):
    """To pass resource_variable_ops.is_resource_variable check."""
    pass

ops.register_tensor_conversion_function(ResourceVariable, _dense_var_to_tensor)
ops.register_tensor_conversion_function(
    variables.Variable, variables.Variable._TensorConversionFunction)  # pylint: disable=protected-access

# pylint: disable=protected-access
ResourceVariable._OverloadAllOperators()
ops.register_dense_tensor_like_type(ResourceVariable)


@ops.RegisterGradient("ReadVariableOp")
def _ReadGrad(_, grad):
  """Gradient for read op."""
  return grad


@ops.RegisterGradient("ResourceGather")
def _GatherGrad(op, grad):
  """Gradient for gather op."""
  # Build appropriately shaped IndexedSlices
  handle = op.inputs[0]
  indices = op.inputs[1]
  params_shape = gen_resource_variable_ops.variable_shape(handle)
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[1:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return (ops.IndexedSlices(values, indices, params_shape), None)


def _to_proto_fn(v, export_scope=None):
  """Converts Variable and ResourceVariable to VariableDef for collections."""
  return v.to_proto(export_scope=export_scope)


def _from_proto_fn(v, import_scope=None):
  """Creates Variable or ResourceVariable from VariableDef as needed."""
  if v.is_resource:
    return ResourceVariable.from_proto(v, import_scope=import_scope)
  return variables.Variable.from_proto(v, import_scope=import_scope)


ops.register_proto_function(
    ops.GraphKeys.GLOBAL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.TRAINABLE_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.LOCAL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.MODEL_VARIABLES,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)
ops.register_proto_function(
    ops.GraphKeys.GLOBAL_STEP,
    proto_type=variable_pb2.VariableDef,
    to_proto=_to_proto_fn,
    from_proto=_from_proto_fn)


def is_resource_variable(var):
  """"Returns True if `var` is to be considered a ResourceVariable."""
  return isinstance(var, ResourceVariable) or hasattr(
      var, "_should_act_as_resource_variable")
