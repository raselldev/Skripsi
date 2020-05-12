from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum

import six

from tensorflow.python.util import compat
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.ops import state_ops
from tensorflow.python import context
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.training import base as checkpointable


@tf_export("VariableSynchronization")
class VariableSynchronization(enum.Enum):
  """Indicates when a distributed variable will be synced.

  * `AUTO`: Indicates that the synchronization will be determined by the current
    `DistributionStrategy` (eg. With `MirroredStrategy` this would be
    `ON_WRITE`).
  * `NONE`: Indicates that there will only be one copy of the variable, so
    there is no need to sync.
  * `ON_WRITE`: Indicates that the variable will be updated across devices
    every time it is written.
  * `ON_READ`: Indicates that the variable will be aggregated across devices
    when it is read (eg. when checkpointing or when evaluating an op that uses
    the variable).
  """
  AUTO = 0
  NONE = 1
  ON_WRITE = 2
  ON_READ = 3


@tf_export("VariableAggregation")
class VariableAggregation(enum.Enum):
  """Indicates how a distributed variable will be aggregated.

  `tf.contrib.distribute.DistributionStrategy` distributes a model by making
  multiple copies (called "towers") acting data-parallel on different elements
  of the input batch. When performing some variable-update operation, say
  `var.assign_add(x)`, in a model, we need to resolve how to combine the
  different values for `x` computed in the different towers.

  * `NONE`: This is the default, giving an error if you use a
    variable-update operation with multiple towers.
  * `SUM`: Add the updates across towers.
  * `MEAN`: Take the arithmetic mean ("average") of the updates across towers.
  * `ONLY_FIRST_TOWER`: This is for when every tower is performing the same
    update, but we only want to perform the update once. Used, e.g., for the
    global step counter.
  """
  NONE = 0
  SUM = 1
  MEAN = 2
  ONLY_FIRST_TOWER = 3


class VariableMetaclass(type):
  """Metaclass to allow construction of tf.Variable to be overridden."""

  def _variable_v1_call(cls,
                        initial_value=None,
                        trainable=None,
                        collections=None,
                        validate_shape=True,
                        caching_device=None,
                        name=None,
                        variable_def=None,
                        dtype=None,
                        expected_shape=None,
                        import_scope=None,
                        constraint=None,
                        use_resource=None,
                        synchronization=VariableSynchronization.AUTO,
                        aggregation=VariableAggregation.NONE):
    """Call on Variable class. Useful to force the signature."""
    previous_getter = lambda **kwargs: default_variable_creator(None, **kwargs)
    for getter in ops.get_default_graph()._variable_creator_stack:  # pylint: disable=protected-access
      previous_getter = _make_getter(getter, previous_getter)

    # Reset `aggregation` that is explicitly set as `None` to the enum NONE.
    if aggregation is None:
      aggregation = VariableAggregation.NONE
    return previous_getter(
        initial_value=initial_value,
        trainable=trainable,
        collections=collections,
        validate_shape=validate_shape,
        caching_device=caching_device,
        name=name,
        variable_def=variable_def,
        dtype=dtype,
        expected_shape=expected_shape,
        import_scope=import_scope,
        constraint=constraint,
        use_resource=use_resource,
        synchronization=synchronization,
        aggregation=aggregation)

  def _variable_v2_call(cls,
                        initial_value=None,
                        trainable=None,
                        validate_shape=True,
                        caching_device=None,
                        name=None,
                        variable_def=None,
                        dtype=None,
                        import_scope=None,
                        constraint=None,
                        synchronization=VariableSynchronization.AUTO,
                        aggregation=VariableAggregation.NONE):
    """Call on Variable class. Useful to force the signature."""
    previous_getter = lambda **kws: default_variable_creator_v2(None, **kws)
    for getter in ops.get_default_graph()._variable_creator_stack:  # pylint: disable=protected-access
      previous_getter = _make_getter(getter, previous_getter)

    # Reset `aggregation` that is explicitly set as `None` to the enum NONE.
    if aggregation is None:
      aggregation = VariableAggregation.NONE
    return previous_getter(
        initial_value=initial_value,
        trainable=trainable,
        validate_shape=validate_shape,
        caching_device=caching_device,
        name=name,
        variable_def=variable_def,
        dtype=dtype,
        import_scope=import_scope,
        constraint=constraint,
        synchronization=synchronization,
        aggregation=aggregation)

  def __call__(cls, *args, **kwargs):
    if cls is VariableV1:
      return cls._variable_v1_call(*args, **kwargs)
    elif cls is Variable:
      return cls._variable_v2_call(*args, **kwargs)
    else:
      return super(VariableMetaclass, cls).__call__(*args, **kwargs)

@tf_export(v2=["Variable"])
class Variable(six.with_metaclass(VariableMetaclass,
                                  checkpointable.CheckpointableBase)):
  """See the [Variables Guide](https://tensorflow.org/guide/variables).

  A variable maintains state in the graph across calls to `run()`. You add a
  variable to the graph by constructing an instance of the class `Variable`.

  The `Variable()` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  If you want to change the shape of a variable later you have to use an
  `assign` Op with `validate_shape=False`.

  Just like any `Tensor`, variables created with `Variable()` can be used as
  inputs for other Ops in the graph. Additionally, all the operators
  overloaded for the `Tensor` class are carried over to variables, so you can
  also add nodes to the graph by just doing arithmetic on variables.

  ```python
  import tensorflow as tf

  # Create a variable.
  w = tf.Variable(<initial-value>, name=<optional-name>)

  # Use the variable in the graph like any Tensor.
  y = tf.matmul(w, ...another variable or tensor...)

  # The overloaded operators are available too.
  z = tf.sigmoid(w + y)

  # Assign a new value to the variable with `assign()` or a related method.
  w.assign(w + 1.0)
  w.assign_add(1.0)
  ```

  When you launch the graph, variables have to be explicitly initialized before
  you can run Ops that use their value. You can initialize a variable by
  running its *initializer op*, restoring the variable from a save file, or
  simply running an `assign` Op that assigns a value to the variable. In fact,
  the variable *initializer op* is just an `assign` Op that assigns the
  variable's initial value to the variable itself.

  ```python
  # Launch the graph in a session.
  with tf.Session() as sess:
      # Run the variable initializer.
      sess.run(w.initializer)
      # ...you now can run ops that use the value of 'w'...
  ```

  The most common initialization pattern is to use the convenience function
  `global_variables_initializer()` to add an Op to the graph that initializes
  all the variables. You then run that Op after launching the graph.

  ```python
  # Add an Op to initialize global variables.
  init_op = tf.global_variables_initializer()

  # Launch the graph in a session.
  with tf.Session() as sess:
      # Run the Op that initializes global variables.
      sess.run(init_op)
      # ...you can now run any Op that uses variable values...
  ```

  If you need to create a variable with an initial value dependent on another
  variable, use the other variable's `initialized_value()`. This ensures that
  variables are initialized in the right order.

  All variables are automatically collected in the graph where they are
  created. By default, the constructor adds the new variable to the graph
  collection `GraphKeys.GLOBAL_VARIABLES`. The convenience function
  `global_variables()` returns the contents of that collection.

  When building a machine learning model it is often convenient to distinguish
  between variables holding the trainable model parameters and other variables
  such as a `global step` variable used to count training steps. To make this
  easier, the variable constructor supports a `trainable=<bool>` parameter. If
  `True`, the new variable is also added to the graph collection
  `GraphKeys.TRAINABLE_VARIABLES`. The convenience function
  `trainable_variables()` returns the contents of this collection. The
  various `Optimizer` classes use this collection as the default list of
  variables to optimize.

  WARNING: tf.Variable objects by default have a non-intuitive memory model. A
  Variable is represented internally as a mutable Tensor which can
  non-deterministically alias other Tensors in a graph. The set of operations
  which consume a Variable and can lead to aliasing is undetermined and can
  change across TensorFlow versions. Avoid writing code which relies on the
  value of a Variable either changing or not changing as other operations
  happen. For example, using Variable objects or simple functions thereof as
  predicates in a `tf.cond` is dangerous and error-prone:

  ```
  v = tf.Variable(True)
  tf.cond(v, lambda: v.assign(False), my_false_fn)  # Note: this is broken.
  ```

  Here replacing adding `use_resource=True` when constructing the variable will
  fix any nondeterminism issues:
  ```
  v = tf.Variable(True, use_resource=True)
  tf.cond(v, lambda: v.assign(False), my_false_fn)
  ```

  To use the replacement for variables which does
  not have these issues:

  * Add `use_resource=True` when constructing `tf.Variable`;
  * Call `tf.get_variable_scope().set_use_resource(True)` inside a
    `tf.variable_scope` before the `tf.get_variable()` call.
  """

  def __init__(self,
               initial_value=None,
               trainable=True,
               validate_shape=True,
               caching_device=None,
               name=None,
               variable_def=None,
               dtype=None,
               import_scope=None,
               constraint=None,
               synchronization=VariableSynchronization.AUTO,
               aggregation=VariableAggregation.NONE):
    """Creates a new variable with value `initial_value`.

    The new variable is added to the graph collections listed in `collections`,
    which defaults to `[GraphKeys.GLOBAL_VARIABLES]`.

    If `trainable` is `True` the variable is also added to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES`.

    This constructor creates both a `variable` Op and an `assign` Op to set the
    variable to its initial value.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
      trainable: If `True`, the default, GradientTapes automatically watch uses
        of this variable.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Optional device string describing where the Variable
        should be cached for reading.  Defaults to the Variable's device.
        If not `None`, caches on another device.  Typical use is to cache
        on the device where the Ops using the Variable reside, to deduplicate
        copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      variable_def: `VariableDef` protocol buffer. If not `None`, recreates
        the Variable object with its contents, referencing the variable's nodes
        in the graph, which must already exist. The graph is not changed.
        `variable_def` and the other arguments are mutually exclusive.
      dtype: If set, initial_value will be converted to the given type.
        If `None`, either the datatype will be kept (if `initial_value` is
        a Tensor), or `convert_to_tensor` will decide.
      import_scope: Optional `string`. Name scope to add to the
        `Variable.` Only used when initializing from protocol buffer.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value
        (which must have the same shape). Constraints are not safe to
        use when doing asynchronous distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize. If `synchronization` is set to `ON_READ`,
        `trainable` must not be set to `True`.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

    Raises:
      ValueError: If both `variable_def` and initial_value are specified.
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
      RuntimeError: If eager execution is enabled.
    """
    raise NotImplementedError

  def __repr__(self):
    raise NotImplementedError

  def value(self):
    """Returns the last snapshot of this variable.

    You usually do not need to call this method as all ops that need the value
    of the variable call it automatically through a `convert_to_tensor()` call.

    Returns a `Tensor` which holds the value of the variable.  You can not
    assign a new value to this tensor as it is not a reference to the variable.

    To avoid copies, if the consumer of the returned value is on the same device
    as the variable, this actually returns the live value of the variable, not
    a copy.  Updates to the variable are seen by the consumer.  If the consumer
    is on a different device it will get a copy of the variable.

    Returns:
      A `Tensor` containing the value of the variable.
    """
    raise NotImplementedError

  def read_value(self):
    """Returns the value of this variable, read in the current context.

    Can be different from value() if it's on another device, with control
    dependencies, etc.

    Returns:
      A `Tensor` containing the value of the variable.
    """
    raise NotImplementedError

  def set_shape(self, shape):
    """Overrides the shape for this variable.

    Args:
      shape: the `TensorShape` representing the overridden shape.
    """
    raise NotImplementedError

  @property
  def trainable(self):
    raise NotImplementedError

  def eval(self, session=None):
    """In a session, computes and returns the value of this variable.

    This is not a graph construction method, it does not add ops to the graph.

    This convenience method requires a session where the graph
    containing this variable has been launched. If no session is
    passed, the default session is used.  See `tf.Session` for more
    information on launching a graph and on sessions.

    ```python
    v = tf.Variable([1, 2])
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # Usage passing the session explicitly.
        print(v.eval(sess))
        # Usage with the default session.  The 'with' block
        # above makes 'sess' the default session.
        print(v.eval())
    ```

    Args:
      session: The session to use to evaluate this variable. If
        none, the default session is used.

    Returns:
      A numpy `ndarray` with a copy of the value of this variable.
    """
    raise NotImplementedError

  def initialized_value(self):
    """Returns the value of the initialized variable.

    You should use this instead of the variable itself to initialize another
    variable with a value that depends on the value of this variable.

    ```python
    # Initialize 'v' with a random tensor.
    v = tf.Variable(tf.truncated_normal([10, 40]))
    # Use `initialized_value` to guarantee that `v` has been
    # initialized before its value is used to initialize `w`.
    # The random values are picked only once.
    w = tf.Variable(v.initialized_value() * 2.0)
    ```

    Returns:
      A `Tensor` holding the value of this variable after its initializer
      has run.
    """
    raise NotImplementedError

  @property
  def initial_value(self):
    """Returns the Tensor used as the initial value for the variable.

    Note that this is different from `initialized_value()` which runs
    the op that initializes the variable before returning its value.
    This method returns the tensor that is used by the op that initializes
    the variable.

    Returns:
      A `Tensor`.
    """
    raise NotImplementedError

  @property
  def constraint(self):
    """Returns the constraint function associated with this variable.

    Returns:
      The constraint function that was passed to the variable constructor.
      Can be `None` if no constraint was passed.
    """
    raise NotImplementedError

  def assign(self, value, use_locking=False, name=None, read_value=True):
    """Assigns a new value to the variable.

    This is essentially a shortcut for `assign(self, value)`.

    Args:
      value: A `Tensor`. The new value for this variable.
      use_locking: If `True`, use locking during the assignment.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the
        new value of the variable; if False will return the assign op.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the assignment has completed.
    """
    raise NotImplementedError

  def assign_add(self, delta, use_locking=False, name=None, read_value=True):
    """Adds a value to this variable.

     This is essentially a shortcut for `assign_add(self, delta)`.

    Args:
      delta: A `Tensor`. The value to add to this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the
        new value of the variable; if False will return the assign op.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the addition has completed.
    """
    raise NotImplementedError

  def assign_sub(self, delta, use_locking=False, name=None, read_value=True):
    """Subtracts a value from this variable.

    This is essentially a shortcut for `assign_sub(self, delta)`.

    Args:
      delta: A `Tensor`. The value to subtract from this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the
        new value of the variable; if False will return the assign op.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the subtraction has completed.
    """
    raise NotImplementedError

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
    raise NotImplementedError

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    """Adds `IndexedSlices` to this variable.

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
    raise NotImplementedError

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
    raise NotImplementedError

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
    raise NotImplementedError

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
    raise NotImplementedError

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
        op = ref.scatter_nd_assign(indices, updates)
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
    raise NotImplementedError

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
    raise NotImplementedError

  def load(self, value, session=None):
  
    raise NotImplementedError

  # Conversion to tensor.
  @staticmethod
  def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):  # pylint: disable=invalid-name
    """Utility function for converting a Variable to a Tensor."""
    _ = name
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          "Incompatible type conversion requested to type '%s' for variable "
          "of type '%s'" % (dtype.name, v.dtype.name))
    if as_ref:
      return v._ref()  # pylint: disable=protected-access
    else:
      return v.value()

  @staticmethod
  def _OverloadAllOperators():  # pylint: disable=invalid-name
    """Register overloads for all operators."""
    for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
      Variable._OverloadOperator(operator)
    # For slicing, bind getitem differently than a tensor (use SliceHelperVar
    # instead)
    # pylint: disable=protected-access
    setattr(Variable, "__getitem__", array_ops._SliceHelperVar)

  @staticmethod
  def _OverloadOperator(operator):  # pylint: disable=invalid-name
    

    def _run_op(a, *args):
      # pylint: disable=protected-access
      return getattr(ops.Tensor, operator)(a._AsTensor(), *args)
    # Propagate __doc__ to wrapper
    try:
      _run_op.__doc__ = getattr(ops.Tensor, operator).__doc__
    except AttributeError:
      pass

    setattr(Variable, operator, _run_op)

  __array_priority__ = 100

  @property
  def name(self):
    """The name of this variable."""
    raise NotImplementedError

  @property
  def initializer(self):
    """The initializer operation for this variable."""
    raise NotImplementedError

  @property
  def device(self):
    """The device of this variable."""
    raise NotImplementedError

  @property
  def dtype(self):
    """The `DType` of this variable."""
    raise NotImplementedError

  @property
  def op(self):
    """The `Operation` of this variable."""
    raise NotImplementedError

  @property
  def graph(self):
    """The `Graph` of this variable."""
    raise NotImplementedError

  @property
  def shape(self):
    raise NotImplementedError

  def get_shape(self):
    raise NotImplementedError

  def to_proto(self, export_scope=None):
    raise NotImplementedError

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    """Returns a `Variable` object created from `variable_def`."""
    return RefVariable(variable_def=variable_def,
                       import_scope=import_scope)

  class SaveSliceInfo(object):

    def __init__(self,
                 full_name=None,
                 full_shape=None,
                 var_offset=None,
                 var_shape=None,
                 save_slice_info_def=None,
                 import_scope=None):
      if save_slice_info_def:
        assert isinstance(save_slice_info_def, variable_pb2.SaveSliceInfoDef)
        self.full_name = ops.prepend_name_scope(
            save_slice_info_def.full_name, import_scope=import_scope)
        self.full_shape = [i for i in save_slice_info_def.full_shape]
        self.var_offset = [i for i in save_slice_info_def.var_offset]
        self.var_shape = [i for i in save_slice_info_def.var_shape]
      else:
        self.full_name = full_name
        self.full_shape = full_shape
        self.var_offset = var_offset
        self.var_shape = var_shape

    @property
    def spec(self):
      """Computes the spec string used for saving."""
      full_shape_str = " ".join(["%d" % d for d in self.full_shape]) + " "
      sl_spec = ":".join([
          "%d,%d" % (o, s) for o, s in zip(self.var_offset, self.var_shape)
      ])
      return full_shape_str + sl_spec

    def to_proto(self, export_scope=None):
      if (export_scope is None or
          self.full_name.startswith(export_scope)):
        save_slice_info_def = variable_pb2.SaveSliceInfoDef()
        save_slice_info_def.full_name = ops.strip_name_scope(
            self.full_name, export_scope)
        for i in self.full_shape:
          save_slice_info_def.full_shape.append(i)
        for i in self.var_offset:
          save_slice_info_def.var_offset.append(i)
        for i in self.var_shape:
          save_slice_info_def.var_shape.append(i)
        return save_slice_info_def
      else:
        return None

  def __iadd__(self, other):
    raise NotImplementedError

  def __isub__(self, other):
    raise NotImplementedError

  def __imul__(self, other):
    raise NotImplementedError

  def __idiv__(self, other):
    raise NotImplementedError

  def __itruediv__(self, other):
    raise NotImplementedError

  def __irealdiv__(self, other):
    raise NotImplementedError

  def __ipow__(self, other):
    raise NotImplementedError

@tf_export(v1=["Variable"])
class VariableV1(Variable):
  def __init__(self,  # pylint: disable=super-init-not-called
               initial_value=None,
               trainable=True,
               collections=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               variable_def=None,
               dtype=None,
               expected_shape=None,
               import_scope=None,
               constraint=None,
               use_resource=None,
               synchronization=VariableSynchronization.AUTO,
               aggregation=VariableAggregation.NONE):
    """
    """

  SaveSliceInfo = Variable.SaveSliceInfo


class RefVariable(VariableV1):
  def __init__(self,
               initial_value=None,
               trainable=True,
               collections=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               variable_def=None,
               dtype=None,
               expected_shape=None,
               import_scope=None,
               constraint=None):
    self._in_graph_mode = True
    if variable_def:
      if initial_value:
        raise ValueError("variable_def and initial_value are mutually "
                         "exclusive.")
      self._init_from_proto(variable_def, import_scope=import_scope)
    else:
      # Create from initial_value.
      self._init_from_args(
          initial_value=initial_value,
          trainable=trainable,
          collections=collections,
          validate_shape=validate_shape,
          caching_device=caching_device,
          name=name,
          dtype=dtype,
          expected_shape=expected_shape,
          constraint=constraint)

  def __repr__(self):
    if context.executing_eagerly() and not self._in_graph_mode:
      return "<tf.Variable '%s' shape=%s dtype=%s, numpy=%s>" % (
          self.name, self.get_shape(), self.dtype.name,
          ops.numpy_text(self.read_value(), is_repr=True))
    else:
      return "<tf.Variable '%s' shape=%s dtype=%s>" % (
          self.name, self.get_shape(), self.dtype.name)

  def _init_from_args(self,
                      initial_value=None,
                      trainable=True,
                      collections=None,
                      validate_shape=True,
                      caching_device=None,
                      name=None,
                      dtype=None,
                      expected_shape=None,
                      constraint=None):
    
    _ = expected_shape
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
          "collections argument to Variable constructor must be a list, tuple, "
          "or set. Got %s of type %s" % (collections, type(collections)))
    if constraint is not None and not callable(constraint):
      raise ValueError("The `constraint` argument must be a callable.")

    # Store the graph key so optimizers know how to only retrieve variables from
    # this graph.
    self._graph_key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    if isinstance(initial_value, checkpointable.CheckpointInitialValue):
      self._maybe_initialize_checkpointable()
      self._update_uid = initial_value.checkpoint_position.restore_uid
      initial_value = initial_value.wrapped_value

    self._trainable = trainable
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    with ops.init_scope():
      # Ensure that we weren't lifted into the eager context.
      if context.executing_eagerly():
        raise RuntimeError(
            "RefVariable not supported when eager execution is enabled. ")
      with ops.name_scope(name, "Variable", [] if init_from_fn else
                          [initial_value]) as name:

        if init_from_fn:
          # Use attr_scope and device(None) to simulate the behavior of
          # colocate_with when the variable we want to colocate with doesn't
          # yet exist.
          true_name = ops._name_from_scope_name(name)  # pylint: disable=protected-access
          attr = attr_value_pb2.AttrValue(
              list=attr_value_pb2.AttrValue.ListValue(
                  s=[compat.as_bytes("loc:@%s" % true_name)]))
          # pylint: disable=protected-access
          with ops.get_default_graph()._attr_scope({"_class": attr}):
            with ops.name_scope("Initializer"):
              self._initial_value = ops.convert_to_tensor(
                  initial_value(), name="initial_value", dtype=dtype)
              shape = (self._initial_value.get_shape()
                       if validate_shape else tensor_shape.unknown_shape())
            self._variable = state_ops.variable_op_v2(
                shape,
                self._initial_value.dtype.base_dtype,
                name=name)
          # pylint: enable=protected-access

        # Or get the initial value from a Tensor or Python object.
        else:
          self._initial_value = ops.convert_to_tensor(
              initial_value, name="initial_value", dtype=dtype)
          # pylint: disable=protected-access
          if self._initial_value.op._get_control_flow_context() is not None:
            raise ValueError(
                "Initializer for variable %s is from inside a control-flow "
                "construct, such as a loop or conditional. When creating a "
                "variable inside a loop or conditional, use a lambda as the "
                "initializer." % name)
          # pylint: enable=protected-access
          shape = (self._initial_value.get_shape()
                   if validate_shape else tensor_shape.unknown_shape())
          # In this case, the variable op can't be created until after the
          # initial_value has been converted to a Tensor with a known type.
          self._variable = state_ops.variable_op_v2(
              shape,
              self._initial_value.dtype.base_dtype,
              name=name)

        # Manually overrides the variable's shape with the initial value's.
        if validate_shape:
          initial_value_shape = self._initial_value.get_shape()
          if not initial_value_shape.is_fully_defined():
            raise ValueError("initial_value must have a shape specified: %s" %
                             self._initial_value)

        # If 'initial_value' makes use of other variables, make sure we don't
        # have an issue if these other variables aren't initialized first by
        # using their initialized_value() method.
        self._initializer_op = state_ops.assign(
            self._variable,
            self._try_guard_against_uninitialized_dependencies(
                self._initial_value),
            validate_shape=validate_shape).op

        # TODO(vrv): Change this class to not take caching_device, but
        # to take the op to colocate the snapshot with, so we can use
        # colocation rather than devices.
        if caching_device is not None:
          with ops.device(caching_device):
            self._snapshot = array_ops.identity(self._variable, name="read")
        else:
          with ops.colocate_with(self._variable.op):
            self._snapshot = array_ops.identity(self._variable, name="read")
      ops.add_to_collections(collections, self)

    self._caching_device = caching_device
    self._save_slice_info = None
    self._constraint = constraint

  def _init_from_proto(self, variable_def, import_scope=None):
    assert isinstance(variable_def, variable_pb2.VariableDef)
    # Create from variable_def.
    g = ops.get_default_graph()
    self._variable = g.as_graph_element(
        ops.prepend_name_scope(variable_def.variable_name,
                               import_scope=import_scope))
    self._initializer_op = g.as_graph_element(
        ops.prepend_name_scope(variable_def.initializer_name,
                               import_scope=import_scope))
    # Tests whether initial_value_name exists first for backwards compatibility.
    if (hasattr(variable_def, "initial_value_name") and
        variable_def.initial_value_name):
      self._initial_value = g.as_graph_element(
          ops.prepend_name_scope(variable_def.initial_value_name,
                                 import_scope=import_scope))
    else:
      self._initial_value = None
    self._trainable = getattr(variable_def, "trainable", True)
    self._snapshot = g.as_graph_element(
        ops.prepend_name_scope(variable_def.snapshot_name,
                               import_scope=import_scope))
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def,
          import_scope=import_scope)
    else:
      self._save_slice_info = None
    self._caching_device = None
    self._constraint = None

  def _as_graph_element(self):
    return self._variable

  def _AsTensor(self):
    return self._snapshot

  def __iter__(self):
    raise TypeError("'Variable' object is not iterable.")

  def value(self):
    return self._snapshot

  def read_value(self):
    return array_ops.identity(self._variable, name="read")

  def _ref(self):
    return self._variable

  def set_shape(self, shape):
    self._ref().set_shape(shape)
    self.value().set_shape(shape)

  @property
  def trainable(self):
    return self._trainable

  def eval(self, session=None):
    return self._variable.eval(session=session)

  def initialized_value(self):
    with ops.init_scope():
      return control_flow_ops.cond(is_variable_initialized(self),
                                   self.read_value,
                                   lambda: self.initial_value)

  @property
  def initial_value(self):
    return self._initial_value

  @property
  def constraint(self):
    return self._constraint

  def assign(self, value, use_locking=False, name=None, read_value=True):
    assign = state_ops.assign(self._variable, value, use_locking=use_locking,
                              name=name)
    if read_value:
      return assign
    return assign.op

  def assign_add(self, delta, use_locking=False, name=None, read_value=True):
    assign = state_ops.assign_add(
        self._variable, delta, use_locking=use_locking, name=name)
    if read_value:
      return assign
    return assign.op

  def assign_sub(self, delta, use_locking=False, name=None, read_value=True):
    assign = state_ops.assign_sub(
        self._variable, delta, use_locking=use_locking, name=name)
    if read_value:
      return assign
    return assign.op

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return gen_state_ops.scatter_sub(
        self._variable,
        sparse_delta.indices,
        sparse_delta.values,
        use_locking=use_locking,
        name=name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return gen_state_ops.scatter_add(
        self._variable,
        sparse_delta.indices,
        sparse_delta.values,
        use_locking=use_locking,
        name=name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    if not isinstance(sparse_delta, ops.IndexedSlices):
      raise ValueError("sparse_delta is not IndexedSlices: %s" % sparse_delta)
    return gen_state_ops.scatter_update(
        self._variable,
        sparse_delta.indices,
        sparse_delta.values,
        use_locking=use_locking,
        name=name)

  def scatter_nd_sub(self, indices, updates, name=None):
    return gen_state_ops.scatter_nd_sub(
        self._variable, indices, updates, use_locking=True, name=name)

  def scatter_nd_add(self, indices, updates, name=None):
    return gen_state_ops.scatter_nd_add(
        self._variable, indices, updates, use_locking=True, name=name)

  def scatter_nd_update(self, indices, updates, name=None):
    return gen_state_ops.scatter_nd_update(
        self._variable, indices, updates, use_locking=True, name=name)

  def _strided_slice_assign(self,
                            begin,
                            end,
                            strides,
                            value,
                            name,
                            begin_mask,
                            end_mask,
                            ellipsis_mask,
                            new_axis_mask,
                            shrink_axis_mask):
    return gen_array_ops.strided_slice_assign(ref=self._ref(),
                                              begin=begin,
                                              end=end,
                                              strides=strides,
                                              value=value,
                                              name=name,
                                              begin_mask=begin_mask,
                                              end_mask=end_mask,
                                              ellipsis_mask=ellipsis_mask,
                                              new_axis_mask=new_axis_mask,
                                              shrink_axis_mask=shrink_axis_mask)

  def count_up_to(self, limit):
    return state_ops.count_up_to(self._variable, limit=limit)

  def load(self, value, session=None):
    if context.executing_eagerly():
      self.assign(value)
    else:
      session = session or ops.get_default_session()
      if session is None:
        raise ValueError(
            "Either session argument should be provided or default session "
            "should be established")
      session.run(self._initializer_op, {self._initializer_op.inputs[1]: value})

  @staticmethod
  def _TensorConversionFunction(v, dtype=None, name=None, as_ref=False):  # pylint: disable=invalid-name

    _ = name
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          "Incompatible type conversion requested to type '%s' for variable "
          "of type '%s'" % (dtype.name, v.dtype.name))
    if as_ref:
      return v._ref()  # pylint: disable=protected-access
    else:
      return v.value()

  @staticmethod
  def _OverloadAllOperators():  # pylint: disable=invalid-name
    for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
      Variable._OverloadOperator(operator)  # pylint: disable=protected-access
    # For slicing, bind getitem differently than a tensor (use SliceHelperVar
    # instead)
    # pylint: disable=protected-access
    setattr(Variable, "__getitem__", array_ops._SliceHelperVar)

  @staticmethod
  def _OverloadOperator(operator):  # pylint: disable=invalid-name
    def _run_op(a, *args):
      # pylint: disable=protected-access
      return getattr(ops.Tensor, operator)(a._AsTensor(), *args)
    # Propagate __doc__ to wrapper
    try:
      _run_op.__doc__ = getattr(ops.Tensor, operator).__doc__
    except AttributeError:
      pass

    setattr(Variable, operator, _run_op)

  def _gather_saveables_for_checkpoint(self):
    return {checkpointable.VARIABLE_VALUE_KEY: self}

  def _try_guard_against_uninitialized_dependencies(self, initial_value):
    if not isinstance(initial_value, ops.Tensor):
      raise TypeError("initial_value needs to be a Tensor: %s" % initial_value)

    # Don't modify initial_value if it contains any cyclic dependencies.
    def has_cycle(op, path):
      """Detect cycles in the dependencies of `initial_value`."""
      if op.name in path:
        return True
      path.add(op.name)
      for op_input in op.inputs:
        if has_cycle(op_input.op, path):
          return True
      for op_control_input in op.control_inputs:
        if has_cycle(op_control_input, path):
          return True
      path.remove(op.name)
      return False
    if has_cycle(initial_value.op, path=set()):
      return initial_value

    return self._safe_initial_value_from_tensor(initial_value, op_cache={})

  def _safe_initial_value_from_tensor(self, tensor, op_cache):
    op = tensor.op
    new_op = op_cache.get(op.name)
    if new_op is None:
      new_op = self._safe_initial_value_from_op(op, op_cache)
      op_cache[op.name] = new_op
    return new_op.outputs[tensor.value_index]

  def _safe_initial_value_from_op(self, op, op_cache):
    op_type = op.node_def.op
    if op_type in ("IsVariableInitialized", "VarIsInitializedOp",
                   "ReadVariableOp"):
      return op

    # Attempt to find the initialized_value of any variable reference / handles.
    # TODO(b/70206927): Fix handling of ResourceVariables.
    if op_type in ("Variable", "VariableV2", "VarHandleOp"):
      initialized_value = self._find_initialized_value_for_variable(op)
      return op if initialized_value is None else initialized_value.op

    # Recursively build initializer expressions for inputs.
    modified = False
    new_op_inputs = []
    for op_input in op.inputs:
      new_op_input = self._safe_initial_value_from_tensor(op_input, op_cache)
      new_op_inputs.append(new_op_input)
      modified = modified or (new_op_input != op_input)

    # If at least one input was modified, replace the op.
    if modified:
      new_op_type = op_type
      if new_op_type == "RefSwitch":
        new_op_type = "Switch"
      new_op_name = op.node_def.name + "_" + self.name
      new_op_name = new_op_name.replace(":", "_")
      return self.graph.create_op(
          new_op_type, new_op_inputs,
          op._output_types,  # pylint: disable=protected-access
          name=new_op_name, attrs=op.node_def.attr)

    return op


  __array_priority__ = 100

  @property
  def name(self):
    return self._variable.name

  @property
  def _shared_name(self):
    return self.name[:-2]

  @property
  def initializer(self):
    return self._initializer_op

  @property
  def device(self):
    return self._variable.device

  @property
  def dtype(self):
    return self._variable.dtype

  @property
  def op(self):
    return self._variable.op

  @property
  def graph(self):
    return self._variable.graph

  @property
  def shape(self):
    return self._variable.get_shape()

  def get_shape(self):
    return self.shape

  def to_proto(self, export_scope=None):
    if (export_scope is None or
        self._variable.name.startswith(export_scope)):
      var_def = variable_pb2.VariableDef()
      var_def.variable_name = ops.strip_name_scope(
          self._variable.name, export_scope)
      if self._initial_value is not None:
        # For backwards compatibility.
        var_def.initial_value_name = ops.strip_name_scope(
            self._initial_value.name, export_scope)
      var_def.trainable = self.trainable
      var_def.initializer_name = ops.strip_name_scope(
          self.initializer.name, export_scope)
      var_def.snapshot_name = ops.strip_name_scope(
          self._snapshot.name, export_scope)
      if self._save_slice_info:
        var_def.save_slice_info_def.MergeFrom(self._save_slice_info.to_proto(
            export_scope=export_scope))
      return var_def
    else:
      return None

  def __iadd__(self, other):
    logging.log_first_n(
        logging.WARN,
        "Variable += will be deprecated. Use variable.assign_add"
        " if you want assignment to the variable value or 'x = x + y'"
        " if you want a new python Tensor object.", 1)
    return self + other

  def __isub__(self, other):
    logging.log_first_n(
        logging.WARN,
        "Variable -= will be deprecated. Use variable.assign_sub"
        " if you want assignment to the variable value or 'x = x - y'"
        " if you want a new python Tensor object.", 1)
    return self - other

  def __imul__(self, other):
    logging.log_first_n(
        logging.WARN,
        "Variable *= will be deprecated. Use `var.assign(var * other)`"
        " if you want assignment to the variable value or `x = x * y`"
        " if you want a new python Tensor object.", 1)
    return self * other

  def __idiv__(self, other):
    logging.log_first_n(
        logging.WARN,
        "Variable /= will be deprecated. Use `var.assign(var / other)`"
        " if you want assignment to the variable value or `x = x / y`"
        " if you want a new python Tensor object.", 1)
    return self / other

  def __itruediv__(self, other):
    logging.log_first_n(
        logging.WARN,
        "Variable /= will be deprecated. Use `var.assign(var / other)`"
        " if you want assignment to the variable value or `x = x / y`"
        " if you want a new python Tensor object.", 1)
    return self / other

  def __irealdiv__(self, other):
    logging.log_first_n(
        logging.WARN,
        "Variable /= will be deprecated. Use `var.assign(var / other)`"
        " if you want assignment to the variable value or `x = x / y`"
        " if you want a new python Tensor object.", 1)
    return self / other

  def __ipow__(self, other):
    logging.log_first_n(
        logging.WARN,
        "Variable **= will be deprecated. Use `var.assign(var ** other)`"
        " if you want assignment to the variable value or `x = x ** y`"
        " if you want a new python Tensor object.", 1)
    return self ** other

  def _set_save_slice_info(self, save_slice_info):
    self._save_slice_info = save_slice_info

  def _get_save_slice_info(self):
    return self._save_slice_info


def trainable_variables(scope=None):
  return ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope)

def _all_saveable_objects(scope=None):
  return (ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope) +
          ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS, scope))

