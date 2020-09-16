from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as collections_lib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import sys
import threading
import traceback

import six
from six import iteritems
from six.moves import xrange  # pylint: disable=redefined-builtin

from backend.python.framework import dtypes
from backend.python import context
#from backend.python import tf_logging as logging
from backend.python.framework import ops
from backend.python.framework import tensor_shape
from backend.python.ops import variables
from backend.python.ops import init_ops
from backend.python.util import function_utils

__all__ = [
    "AUTO_REUSE", "VariableScope", "get_variable_scope", "get_variable",
    "get_local_variable", "variable_scope", "variable_op_scope",
    "no_regularizer", "VariableSynchronization", "VariableAggregation"
]

VariableSynchronization = variables.VariableSynchronization

VariableAggregation = variables.VariableAggregation 

def default_variable_creator(next_creator=None, **kwargs):
  assert next_creator is None
  initial_value = kwargs.get("initial_value", None)
  trainable = kwargs.get("trainable", None)
  collections = kwargs.get("collections", None)
  validate_shape = kwargs.get("validate_shape", True)
  caching_device = kwargs.get("caching_device", None)
  name = kwargs.get("name", None)
  variable_def = kwargs.get("variable_def", None)
  dtype = kwargs.get("dtype", None)
  expected_shape = kwargs.get("expected_shape", None)
  import_scope = kwargs.get("import_scope", None)
  constraint = kwargs.get("constraint", None)
  use_resource = kwargs.get("use_resource", None)

  # Set trainable value based on synchronization value.
  synchronization = kwargs.get("synchronization", VariableSynchronization.AUTO)
  trainable = _get_trainable_value(
      synchronization=synchronization, trainable=trainable)

  if use_resource is None:
    use_resource = get_variable_scope().use_resource
  if use_resource is None:
    use_resource = _DEFAULT_USE_RESOURCE
  use_resource = use_resource or context.executing_eagerly()
  if use_resource:
    return resource_variable_ops.ResourceVariable(
        initial_value=initial_value, trainable=trainable,
        collections=collections, validate_shape=validate_shape,
        caching_device=caching_device, name=name, dtype=dtype,
        constraint=constraint, variable_def=variable_def,
        import_scope=import_scope)
  else:
    return variables.RefVariable(
        initial_value=initial_value, trainable=trainable,
        collections=collections, validate_shape=validate_shape,
        caching_device=caching_device, name=name, dtype=dtype,
        constraint=constraint, variable_def=variable_def,
        expected_shape=expected_shape, import_scope=import_scope)

variables.default_variable_creator = default_variable_creator

def _get_trainable_value(synchronization, trainable):
  if synchronization == VariableSynchronization.ON_READ:
    if trainable:
      raise ValueError(
          "Synchronization value can be set to "
          "VariableSynchronization.ON_READ only for non-trainable variables. "
          "You have specified trainable=True and "
          "synchronization=VariableSynchronization.ON_READ.")
    else:
      # Set trainable to be false when variable is to be synced on read.
      trainable = False
  elif trainable is None:
    trainable = True
  return trainable

def get_variable_scope():
  return get_variable_scope_store().current_scope

def get_variable_scope_store():
  scope_store = ops.get_collection(_VARSCOPESTORE_KEY)

  if not scope_store:
    scope_store = _VariableScopeStore()
    ops.add_to_collection(_VARSCOPESTORE_KEY, scope_store)
  else:
    scope_store = scope_store[0]

  return scope_store

_VARSCOPESTORE_KEY = ("__varscope",)

class _VariableScopeStore(threading.local):
  def __init__(self):
    super(_VariableScopeStore, self).__init__()
    self.current_scope = VariableScope(False)
    self.variable_scopes_count = {}

  def open_variable_scope(self, scope_name):
    if scope_name in self.variable_scopes_count:
      self.variable_scopes_count[scope_name] += 1
    else:
      self.variable_scopes_count[scope_name] = 1

  def close_variable_subscopes(self, scope_name):
    for k in list(self.variable_scopes_count.keys()):
      if scope_name is None or k.startswith(scope_name + "/"):
        self.variable_scopes_count[k] = 0

  def variable_scope_count(self, scope_name):
    return self.variable_scopes_count.get(scope_name, 0)

class VariableScope(object):
  def __init__(self,
               reuse,
               name="",
               initializer=None,
               regularizer=None,
               caching_device=None,
               partitioner=None,
               custom_getter=None,
               name_scope="",
               dtype=dtypes.float32,
               use_resource=None,
               constraint=None):
    self._name = name
    self._initializer = initializer
    self._regularizer = regularizer
    self._reuse = reuse
    self._caching_device = caching_device
    self._partitioner = partitioner
    self._custom_getter = custom_getter
    self._name_scope = name_scope
    self._dtype = dtype
    self._use_resource = use_resource
    self._constraint = constraint
    if context.executing_eagerly():
      if self._caching_device is not None:
        raise NotImplementedError("Caching devices is not yet supported "
                                  "when eager execution is enabled.")
      if self._partitioner is not None:
        raise NotImplementedError("Partitioned variables are not yet supported "
                                  "when eager execution is enabled.")
      self._reuse = AUTO_REUSE
      self._use_resource = True

  @property
  def name(self):
    return self._name

  @property
  def original_name_scope(self):
    return self._name_scope

  @property
  def reuse(self):
    return self._reuse

  @property
  def initializer(self):
    return self._initializer

  @property
  def dtype(self):
    return self._dtype

  @property
  def use_resource(self):
    return self._use_resource

  @property
  def regularizer(self):
    return self._regularizer

  @property
  def caching_device(self):
    return self._caching_device

  @property
  def partitioner(self):
    return self._partitioner

  @property
  def custom_getter(self):
    return self._custom_getter

  @property
  def constraint(self):
    return self._constraint

  def reuse_variables(self):
    self._reuse = True

  def set_initializer(self, initializer):
    self._initializer = initializer

  def set_dtype(self, dtype):
    self._dtype = dtype

  def set_use_resource(self, use_resource):
    if context.executing_eagerly() and not use_resource:
      raise ValueError("When eager execution is enabled, "
                       "use_resource cannot be set to false.")
    self._use_resource = use_resource

  def set_regularizer(self, regularizer):
    self._regularizer = regularizer

  def set_caching_device(self, caching_device):
    if context.executing_eagerly():
      raise NotImplementedError("Caching devices are not yet supported "
                                "when eager execution is enabled.")
    self._caching_device = caching_device

  def set_partitioner(self, partitioner):
    if partitioner and context.executing_eagerly():
      raise NotImplementedError("Partitioned variables are not yet supported "
                                "when eager execution is enabled.")
    self._partitioner = partitioner

  def set_custom_getter(self, custom_getter):
    self._custom_getter = custom_getter

  def get_collection(self, name):
    scope = self._name + "/" if self._name else ""
    return ops.get_collection(name, scope)

  def trainable_variables(self):
    return self.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)

  def global_variables(self):
    return self.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

  def local_variables(self):
    return self.get_collection(ops.GraphKeys.LOCAL_VARIABLES)

  def get_variable(self,
                   var_store,
                   name,
                   shape=None,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   reuse=None,
                   trainable=None,
                   collections=None,
                   caching_device=None,
                   partitioner=None,
                   validate_shape=True,
                   use_resource=None,
                   custom_getter=None,
                   constraint=None,
                   synchronization=VariableSynchronization.AUTO,
                   aggregation=VariableAggregation.NONE):
    if regularizer is None:
      regularizer = self._regularizer
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if custom_getter is None:
      custom_getter = self._custom_getter
    if context.executing_eagerly():
      reuse = False
      use_resource = True
    else:
      if reuse is None:
        reuse = self._reuse
      if use_resource is None:
        use_resource = self._use_resource

    full_name = self.name + "/" + name if self.name else name
    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None):
      # Check that `initializer` dtype and `dtype` are consistent before
      # replacing them with defaults.
      if (dtype is not None and initializer is not None and
          not callable(initializer)):
        init_dtype = ops.convert_to_tensor(initializer).dtype.base_dtype
        if init_dtype != dtype:
          raise ValueError("Initializer type '%s' and explicit dtype '%s' "
                           "don't match." % (init_dtype, dtype))
      if initializer is None:
        initializer = self._initializer
      if constraint is None:
        constraint = self._constraint
      if dtype is None:
        dtype = self._dtype
      return var_store.get_variable(
          full_name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          partitioner=partitioner,
          validate_shape=validate_shape,
          use_resource=use_resource,
          custom_getter=custom_getter,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)

  def _get_partitioned_variable(self,
                                var_store,
                                name,
                                shape=None,
                                dtype=None,
                                initializer=None,
                                regularizer=None,
                                trainable=None,
                                collections=None,
                                caching_device=None,
                                partitioner=None,
                                validate_shape=True,
                                use_resource=None,
                                constraint=None):
    if context.executing_eagerly():
      raise NotImplementedError("Partitioned variables are not yet supported "
                                "when eager execution is enabled.")
    if initializer is None:
      initializer = self._initializer
    if regularizer is None:
      regularizer = self._regularizer
    if constraint is None:
      constraint = self._constraint
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if dtype is None:
      dtype = self._dtype
    if use_resource is None:
      use_resource = self._use_resource

    if self._custom_getter is not None:
      raise ValueError(
          "Private access to _get_partitioned_variable is not allowed when "
          "a custom getter is set.  Current custom getter: %s.  "
          "It is likely that you're using create_partitioned_variables.  "
          "If so, consider instead using get_variable with a non-empty "
          "partitioner parameter instead." % self._custom_getter)

    if partitioner is None:
      raise ValueError("No partitioner was specified")

    # This allows the variable scope name to be used as the variable name if
    # this function is invoked with an empty name arg, for backward
    # compatibility with create_partitioned_variables().
    full_name_list = []
    if self.name:
      full_name_list.append(self.name)
    if name:
      full_name_list.append(name)
    full_name = "/".join(full_name_list)

    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None):
      # pylint: disable=protected-access
      return var_store._get_partitioned_variable(
          full_name, shape=shape, dtype=dtype, initializer=initializer,
          regularizer=regularizer, reuse=self.reuse, trainable=trainable,
          collections=collections, caching_device=caching_device,
          partitioner=partitioner, validate_shape=validate_shape,
          use_resource=use_resource, constraint=constraint)
      # pylint: enable=protected-access

_DEFAULT_USE_RESOURCE = False

class variable_scope(object):
  def __init__(self,
               name_or_scope,
               default_name=None,
               values=None,
               initializer=None,
               regularizer=None,
               caching_device=None,
               partitioner=None,
               custom_getter=None,
               reuse=None,
               dtype=None,
               use_resource=None,
               constraint=None,
               auxiliary_name_scope=True):
    self._name_or_scope = name_or_scope
    self._default_name = default_name
    self._values = values
    self._initializer = initializer
    self._regularizer = regularizer
    self._caching_device = caching_device
    self._partitioner = partitioner
    self._custom_getter = custom_getter
    self._reuse = reuse
    self._dtype = dtype
    self._use_resource = use_resource
    self._constraint = constraint
    if self._default_name is None and self._name_or_scope is None:
      raise TypeError("If default_name is None then name_or_scope is required")
    if self._reuse is False:
      # We don't allow non-inheriting scopes, False = None here.
      self._reuse = None
    if not (self._reuse is True
            or self._reuse is None
            or self._reuse is AUTO_REUSE):
      raise ValueError("The reuse parameter must be True or False or None.")
    if self._values is None:
      self._values = []
    self._in_graph_mode = not context.executing_eagerly()
    if self._in_graph_mode:
      self._graph = ops._get_graph_from_inputs(self._values)  # pylint: disable=protected-access
    self._cached_pure_variable_scope = None
    self._current_name_scope = None
    if not isinstance(auxiliary_name_scope, bool):
      raise TypeError("The auxiliary_name_scope must be `True` or `False`, "
                      "while get {}".format(auxiliary_name_scope))
    self._auxiliary_name_scope = auxiliary_name_scope

  def __enter__(self):
    # If the default graph is building a function, then we should not replace it
    # with the cached graph.
    if ops.get_default_graph().building_function:
      self._building_function = True
    else:
      self._building_function = False
    if self._in_graph_mode and not self._building_function:
      self._graph_context_manager = self._graph.as_default()
      self._graph_context_manager.__enter__()
    if self._cached_pure_variable_scope is not None:
      # Fast path for re-entering variable_scopes. We've held on to the pure
      # variable scope from a previous successful __enter__, so we avoid some
      # overhead by re-using that object.
      if self._current_name_scope is not None:
        self._current_name_scope.__enter__()
      return self._cached_pure_variable_scope.__enter__()

    try:
      return self._enter_scope_uncached()
    except:
      if self._graph_context_manager is not None:
        self._graph_context_manager.__exit__(*sys.exc_info())
      raise

  def _enter_scope_uncached(self):
    if self._auxiliary_name_scope:
      # Create a new name scope later
      current_name_scope = None
    else:
      # Reenter the current name scope
      name_scope = ops.get_name_scope()
      if name_scope:
        # Hack to reenter
        name_scope += "/"
        current_name_scope = ops.name_scope(name_scope)
      else:
        # Root scope
        current_name_scope = ops.name_scope(name_scope)

    # IMPORTANT: Only assign to self._cached_pure_variable_scope and
    # self._current_name_scope after successful __enter__() calls.
    if self._name_or_scope is not None:
      if not isinstance(self._name_or_scope,
                        (VariableScope,) + six.string_types):
        raise TypeError("VariableScope: name_or_scope must be a string or "
                        "VariableScope.")
      if isinstance(self._name_or_scope, six.string_types):
        name_scope = self._name_or_scope
      else:
        name_scope = self._name_or_scope.name.split("/")[-1]
      if name_scope or current_name_scope:
        current_name_scope = current_name_scope or ops.name_scope(name_scope)
        try:
          current_name_scope_name = current_name_scope.__enter__()
        except:
          current_name_scope.__exit__(*sys.exc_info())
          raise
        self._current_name_scope = current_name_scope
        if isinstance(self._name_or_scope, six.string_types):
          old_name_scope = current_name_scope_name
        else:
          old_name_scope = self._name_or_scope.original_name_scope
        pure_variable_scope = _pure_variable_scope(
            self._name_or_scope,
            reuse=self._reuse,
            initializer=self._initializer,
            regularizer=self._regularizer,
            caching_device=self._caching_device,
            partitioner=self._partitioner,
            custom_getter=self._custom_getter,
            old_name_scope=old_name_scope,
            dtype=self._dtype,
            use_resource=self._use_resource,
            constraint=self._constraint)
        try:
          entered_pure_variable_scope = pure_variable_scope.__enter__()
        except:
          pure_variable_scope.__exit__(*sys.exc_info())
          raise
        self._cached_pure_variable_scope = pure_variable_scope
        return entered_pure_variable_scope
      else:
        self._current_name_scope = None
        # This can only happen if someone is entering the root variable scope.
        pure_variable_scope = _pure_variable_scope(
            self._name_or_scope,
            reuse=self._reuse,
            initializer=self._initializer,
            regularizer=self._regularizer,
            caching_device=self._caching_device,
            partitioner=self._partitioner,
            custom_getter=self._custom_getter,
            dtype=self._dtype,
            use_resource=self._use_resource,
            constraint=self._constraint)
        try:
          entered_pure_variable_scope = pure_variable_scope.__enter__()
        except:
          pure_variable_scope.__exit__(*sys.exc_info())
          raise
        self._cached_pure_variable_scope = pure_variable_scope
        return entered_pure_variable_scope

    else:  # Here name_or_scope is None. Using default name, but made unique.
      if self._reuse:
        raise ValueError("reuse=True cannot be used without a name_or_scope")
      current_name_scope = current_name_scope or ops.name_scope(
          self._default_name)
      try:
        current_name_scope_name = current_name_scope.__enter__()
      except:
        current_name_scope.__exit__(*sys.exc_info())
        raise
      self._current_name_scope = current_name_scope
      unique_default_name = _get_unique_variable_scope(self._default_name)
      pure_variable_scope = _pure_variable_scope(
          unique_default_name,
          initializer=self._initializer,
          regularizer=self._regularizer,
          caching_device=self._caching_device,
          partitioner=self._partitioner,
          custom_getter=self._custom_getter,
          old_name_scope=current_name_scope_name,
          dtype=self._dtype,
          use_resource=self._use_resource,
          constraint=self._constraint)
      try:
        entered_pure_variable_scope = pure_variable_scope.__enter__()
      except:
        pure_variable_scope.__exit__(*sys.exc_info())
        raise
      self._cached_pure_variable_scope = pure_variable_scope
      return entered_pure_variable_scope

  def __exit__(self, type_arg, value_arg, traceback_arg):
    self._cached_pure_variable_scope.__exit__(
        type_arg, value_arg, traceback_arg)
    if self._current_name_scope:
      self._current_name_scope.__exit__(type_arg, value_arg, traceback_arg)
    if self._in_graph_mode and not self._building_function:
      self._graph_context_manager.__exit__(type_arg, value_arg, traceback_arg)

def _get_unique_variable_scope(prefix):
  var_scope_store = get_variable_scope_store()
  current_scope = get_variable_scope()
  name = current_scope.name + "/" + prefix if current_scope.name else prefix
  if var_scope_store.variable_scope_count(name) == 0:
    return prefix
  idx = 1
  while var_scope_store.variable_scope_count(name + ("_%d" % idx)) > 0:
    idx += 1
  return prefix + ("_%d" % idx)

class _pure_variable_scope(object):  # pylint: disable=invalid-name
  def __init__(self,
               name_or_scope,
               reuse=None,
               initializer=None,
               regularizer=None,
               caching_device=None,
               partitioner=None,
               custom_getter=None,
               old_name_scope=None,
               dtype=dtypes.float32,
               use_resource=None,
               constraint=None):
    self._name_or_scope = name_or_scope
    self._reuse = reuse
    self._initializer = initializer
    self._regularizer = regularizer
    self._caching_device = caching_device
    self._partitioner = partitioner
    self._custom_getter = custom_getter
    self._old_name_scope = old_name_scope
    self._dtype = dtype
    self._use_resource = use_resource
    self._constraint = constraint
    self._var_store = _get_default_variable_store()
    self._var_scope_store = get_variable_scope_store()
    if isinstance(self._name_or_scope, VariableScope):
      self._new_name = self._name_or_scope.name
      name_scope = self._name_or_scope._name_scope  # pylint: disable=protected-access
      # Handler for the case when we jump to a shared scope.  We create a new
      #   VariableScope (self._var_scope_object) that contains a copy of the
      #   provided shared scope, possibly with changed reuse and initializer, if
      #   the user requested this.
      variable_scope_object = VariableScope(
          self._name_or_scope.reuse if not self._reuse else self._reuse,
          name=self._new_name,
          initializer=self._name_or_scope.initializer,
          regularizer=self._name_or_scope.regularizer,
          caching_device=self._name_or_scope.caching_device,
          partitioner=self._name_or_scope.partitioner,
          dtype=self._name_or_scope.dtype,
          custom_getter=self._name_or_scope.custom_getter,
          name_scope=name_scope,
          use_resource=self._name_or_scope.use_resource,
          constraint=self._constraint)
      if self._initializer is not None:
        variable_scope_object.set_initializer(self._initializer)
      if self._regularizer is not None:
        variable_scope_object.set_regularizer(self._regularizer)
      if self._caching_device is not None:
        variable_scope_object.set_caching_device(self._caching_device)
      if self._partitioner is not None:
        variable_scope_object.set_partitioner(self._partitioner)
      if self._custom_getter is not None:
        variable_scope_object.set_custom_getter(
            _maybe_wrap_custom_getter(
                self._custom_getter, self._name_or_scope.custom_getter))
      if self._dtype is not None:
        variable_scope_object.set_dtype(self._dtype)
      if self._use_resource is not None:
        variable_scope_object.set_use_resource(self._use_resource)
      self._cached_variable_scope_object = variable_scope_object

  def __enter__(self):
    self._old = self._var_scope_store.current_scope
    if isinstance(self._name_or_scope, VariableScope):
      self._var_scope_store.open_variable_scope(self._new_name)
      self._old_subscopes = copy.copy(
          self._var_scope_store.variable_scopes_count)
      variable_scope_object = self._cached_variable_scope_object
    else:
      # Handler for the case when we just prolong current variable scope.
      #   VariableScope with name extended by the provided one, and inherited
      #   reuse and initializer (except if the user provided values to set).
      self._new_name = (
          self._old.name + "/" + self._name_or_scope if self._old.name
          else self._name_or_scope)
      self._reuse = (self._reuse
                     or self._old.reuse)  # Re-using is inherited by sub-scopes.
      if self._old_name_scope is None:
        name_scope = self._name_or_scope
      else:
        name_scope = self._old_name_scope
      variable_scope_object = VariableScope(
          self._reuse,
          name=self._new_name,
          initializer=self._old.initializer,
          regularizer=self._old.regularizer,
          caching_device=self._old.caching_device,
          partitioner=self._old.partitioner,
          dtype=self._old.dtype,
          use_resource=self._old.use_resource,
          custom_getter=self._old.custom_getter,
          name_scope=name_scope,
          constraint=self._constraint)
      if self._initializer is not None:
        variable_scope_object.set_initializer(self._initializer)
      if self._regularizer is not None:
        variable_scope_object.set_regularizer(self._regularizer)
      if self._caching_device is not None:
        variable_scope_object.set_caching_device(self._caching_device)
      if self._partitioner is not None:
        variable_scope_object.set_partitioner(self._partitioner)
      if self._custom_getter is not None:
        variable_scope_object.set_custom_getter(
            _maybe_wrap_custom_getter(self._custom_getter,
                                      self._old.custom_getter))
      if self._dtype is not None:
        variable_scope_object.set_dtype(self._dtype)
      if self._use_resource is not None:
        variable_scope_object.set_use_resource(self._use_resource)
      self._var_scope_store.open_variable_scope(self._new_name)
    self._var_scope_store.current_scope = variable_scope_object
    return variable_scope_object

  def __exit__(self, type_arg, value_arg, traceback_arg):
    # If jumping out from a non-prolonged scope, restore counts.
    if isinstance(self._name_or_scope, VariableScope):
      self._var_scope_store.variable_scopes_count = self._old_subscopes
    else:
      self._var_scope_store.close_variable_subscopes(self._new_name)
    self._var_scope_store.current_scope = self._old

def _get_default_variable_store():
  store = ops.get_collection(_VARSTORE_KEY)
  if store:
    return store[0]
  store = _VariableStore()
  ops.add_to_collection(_VARSTORE_KEY, store)
  return store

_VARSTORE_KEY = ("__variable_store",)

class _VariableStore(object):
  def __init__(self):
    self._vars = {}  # A dictionary of the stored TensorFlow variables.
    self._partitioned_vars = {}  # A dict of the stored PartitionedVariables.
    self._store_eager_variables = False

  def get_variable(self,
                   name,
                   shape=None,
                   dtype=dtypes.float32,
                   initializer=None,
                   regularizer=None,
                   reuse=None,
                   trainable=None,
                   collections=None,
                   caching_device=None,
                   partitioner=None,
                   validate_shape=True,
                   use_resource=None,
                   custom_getter=None,
                   constraint=None,
                   synchronization=VariableSynchronization.AUTO,
                   aggregation=VariableAggregation.NONE):
    if custom_getter is not None and not callable(custom_getter):
      raise ValueError(
          "Passed a custom_getter which is not callable: %s" % custom_getter)

    with ops.init_scope():
      if context.executing_eagerly():
        # Variable creation and initialization takes place in `init_scope`s;
        # as such, if an `init_scope` lifts us into the eager context, then we
        # need to use `ResourceVariable`s.
        use_resource = True

    # Note that it's fine to reuse eager variables whose initialization was
    # lifted from a function-building graph into the eager context (that's why
    # the following clause is not wrapped in an `init_scope`); lifted variables
    # are tracked by the graph's `VariableStore`.
    if context.executing_eagerly():
      if not self._store_eager_variables and reuse:
        raise RuntimeError(
            "When eager execution is enabled variable reuse is only supported"
            " when an EagerVariableStore is active. See the documentation on"
            " EagerVariableStore for example usage.")
      if self._store_eager_variables:
        reuse = AUTO_REUSE

    # If a *_ref type is passed in an error would be triggered further down the
    # stack. We prevent this using base_dtype to get a non-ref version of the
    # type, before doing anything else. When _ref types are removed in favor of
    # resources, this line can be removed.
    try:
      dtype = dtype.base_dtype
    except AttributeError:
      # .base_dtype not existing means that we will try and use the raw dtype
      # which was passed in - this might be a NumPy type which is valid.
      pass

    # This is the main logic of get_variable.  However, custom_getter
    # may override this logic.  So we save it as a callable and pass
    # it to custom_getter.
    # Note: the parameters of _true_getter, and their documentation, match
    # *exactly* item-for-item with the docstring of this method.
    def _true_getter(  # pylint: disable=missing-docstring
        name,
        shape=None,
        dtype=dtypes.float32,
        initializer=None,
        regularizer=None,
        reuse=None,
        trainable=None,
        collections=None,
        caching_device=None,
        partitioner=None,
        validate_shape=True,
        use_resource=None,
        constraint=None,
        synchronization=VariableSynchronization.AUTO,
        aggregation=VariableAggregation.NONE):
      is_scalar = (shape is not None
                   and isinstance(shape, collections_lib.Sequence)
                   and not shape)
      # Partitioned variable case
      if partitioner is not None and not is_scalar:
        if not callable(partitioner):
          raise ValueError(
              "Partitioner must be callable, but received: %s" % partitioner)
        with ops.name_scope(None):
          return self._get_partitioned_variable(name=name,
                                                shape=shape,
                                                dtype=dtype,
                                                initializer=initializer,
                                                regularizer=regularizer,
                                                reuse=reuse,
                                                trainable=trainable,
                                                collections=collections,
                                                caching_device=caching_device,
                                                partitioner=partitioner,
                                                validate_shape=validate_shape,
                                                use_resource=use_resource,
                                                constraint=constraint)

      # Special case for partitioned variable to allow reuse without having to
      # specify partitioner.
      if (reuse is True and partitioner is None
          and name in self._partitioned_vars):
        return self._get_partitioned_variable(name=name,
                                              shape=shape,
                                              dtype=dtype,
                                              initializer=initializer,
                                              regularizer=regularizer,
                                              reuse=reuse,
                                              trainable=trainable,
                                              collections=collections,
                                              caching_device=caching_device,
                                              partitioner=None,
                                              validate_shape=validate_shape,
                                              use_resource=use_resource,
                                              constraint=constraint)

      # Single variable case
      if "%s/part_0" % name in self._vars:
        raise ValueError(
            "No partitioner was provided, but a partitioned version of the "
            "variable was found: %s/part_0. Perhaps a variable of the same "
            "name was already created with partitioning?" % name)

      return self._get_single_variable(
          name=name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          validate_shape=validate_shape,
          use_resource=use_resource,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)

    # Set trainable value based on synchronization value.
    trainable = _get_trainable_value(
        synchronization=synchronization, trainable=trainable)

    if custom_getter is not None:
      # Handle backwards compatibility with getter arguments that were added
      # to the API after users started writing custom getters.
      custom_getter_kwargs = {
          "getter": _true_getter,
          "name": name,
          "shape": shape,
          "dtype": dtype,
          "initializer": initializer,
          "regularizer": regularizer,
          "reuse": reuse,
          "trainable": trainable,
          "collections": collections,
          "caching_device": caching_device,
          "partitioner": partitioner,
          "validate_shape": validate_shape,
          "use_resource": use_resource,
          "synchronization": synchronization,
          "aggregation": aggregation,
      }
      # `fn_args` and `has_kwargs` can handle functions, `functools.partial`,
      # `lambda`.
      if ("constraint" in function_utils.fn_args(custom_getter) or
          function_utils.has_kwargs(custom_getter)):
        custom_getter_kwargs["constraint"] = constraint
      return custom_getter(**custom_getter_kwargs)
    else:
      return _true_getter(
          name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          partitioner=partitioner,
          validate_shape=validate_shape,
          use_resource=use_resource,
          constraint=constraint,
          synchronization=synchronization,
          aggregation=aggregation)

  def _get_partitioned_variable(self,
                                name,
                                partitioner,
                                shape=None,
                                dtype=dtypes.float32,
                                initializer=None,
                                regularizer=None,
                                reuse=None,
                                trainable=None,
                                collections=None,
                                caching_device=None,
                                validate_shape=True,
                                use_resource=None,
                                constraint=None):
    if context.executing_eagerly():
      raise NotImplementedError("Partitioned variables are not yet supported "
                                "when eager execution is enabled.")

    initializing_from_value = initializer is not None and isinstance(
        initializer, ops.Tensor)
    reuse_without_partition = reuse and not partitioner

    if name in self._vars:
      raise ValueError(
          "A partitioner was provided, but an unpartitioned version of the "
          "variable was found: %s.  Perhaps a variable of the same name was "
          "already created without partitioning?" % name)

    shape = tensor_shape.as_shape(shape)
    if initializing_from_value:
      shape = shape.merge_with(initializer.get_shape())

    if not reuse_without_partition:
      if not shape.is_fully_defined():
        raise ValueError("Shape of a new partitioned variable (%s) must be "
                         "fully defined, but instead was %s." % (name, shape))

      if shape.ndims < 1:
        raise ValueError("A partitioned Variable must have rank at least 1, "
                         "shape: %s" % shape)

      partitions = partitioner(shape=shape, dtype=dtype)

      if not isinstance(partitions, collections_lib.Sequence):
        raise ValueError("Partitioner must return a sequence, but saw: %s"
                         % partitions)

      if len(partitions) != shape.ndims:
        raise ValueError(
            "Partitioner returned a partition list that does not match the "
            "Variable's rank: %s vs. %s" % (partitions, shape))

      if any([p < 1 for p in partitions]):
        raise ValueError(
            "Partitioner returned zero partitions for some axes: %s" %
            partitions)

    if name in self._partitioned_vars:
      if reuse is False:
        raise ValueError(
            "Partitioned variable with name %s already exists. Did you mean to "
            "set reuse=True or reuse=tf.AUTO_REUSE in VarScope?"
            % name)

      existing_var = self._partitioned_vars[name]
      if not shape.is_compatible_with(existing_var.get_shape()):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified shape %s "
            "and found shape %s."
            % (name, shape, existing_var.get_shape()))
      if not dtype.is_compatible_with(existing_var.dtype):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified dtype %s "
            "and found dtype %s."
            % (name, dtype.name, existing_var.dtype.name))

      # pylint: disable=protected-access
      if (not reuse_without_partition and
          existing_var._get_partitions() != partitions):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified partitions "
            "%s and found partitions %s." %
            (name, partitions, existing_var._get_partitions()))
      # pylint: enable=protected-access

      return existing_var

    if reuse is True:
      raise ValueError("PartitionedVariable %s does not exist, or was not "
                       "created with tf.get_variable(). Did you mean to set "
                       "reuse=False or reuse=tf.AUTO_REUSE in VarScope?" % name)

    slice_dim, slice_shape = _compute_slice_dim_and_shape(
        shape.as_list(), partitions)

    vs = []
    num_slices = partitions[slice_dim]
    num_slices_with_excess = shape[slice_dim].value % num_slices

    slice_offset = [0] * shape.ndims

    if "%s/part_0" % name in self._vars:
      if "%s/part_%d" % (name, num_slices - 1) not in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but %s/part_%d was not."
            % (num_slices, name, name, num_slices - 1))
      if "%s/part_%d" % (name, num_slices) in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but so was the extra shard %s/part_%d."
            % (num_slices, name, name, num_slices))

    for i in xrange(num_slices):
      var_shape = slice_shape[:]
      var_offset = slice_offset[:]
      partition_info = _PartitionInfo(
          full_shape=shape.as_list(), var_offset=var_offset)
      if i < num_slices_with_excess:
        var_shape[slice_dim] += 1
      slice_offset[slice_dim] += var_shape[slice_dim]

      var_full_name = "%s/part_%d" % (name, i)
      with ops.name_scope(var_full_name + "/PartitionedInitializer"):
        # Create the tensor to initialize the variable with default value.
        if initializer is None:
          init, initializing_from_value = self._get_default_initializer(
              name=name, shape=shape, dtype=dtype)
          if initializing_from_value:
            init_shape = None
          else:
            init_shape = var_shape
        elif callable(initializer):
          init = initializer
          init_shape = var_shape
        elif isinstance(initializer, ops.Tensor):
          init = array_ops.slice(initializer, var_offset, var_shape)
          # Use the dtype of the given tensor.
          dtype = init.dtype.base_dtype
          init_shape = None
        else:
          init = ops.convert_to_tensor(initializer, dtype=dtype)
          init = array_ops.slice(init, var_offset, var_shape)
          init_shape = None

      with ops.name_scope(None):
        var = self._get_single_variable(
            name=var_full_name,
            shape=init_shape,
            dtype=dtype,
            initializer=init,
            partition_info=partition_info,
            regularizer=regularizer,
            reuse=reuse,
            trainable=trainable,
            collections=collections,
            caching_device=caching_device,
            validate_shape=validate_shape,
            use_resource=use_resource,
            constraint=constraint)

      # pylint: disable=protected-access
      var._set_save_slice_info(variables.Variable.SaveSliceInfo(
          name, shape.as_list(), var_offset, var_shape))
      vs.append(var)
      # pylint: enable=protected-access

      # pylint: disable=protected-access
    partitioned_var = variables.PartitionedVariable(name=name,
                                                    shape=shape,
                                                    dtype=dtype,
                                                    variable_list=vs,
                                                    partitions=partitions)
    # pylint: enable=protected-access

    self._partitioned_vars[name] = partitioned_var
    return partitioned_var

  def _get_single_variable(self,
                           name,
                           shape=None,
                           dtype=dtypes.float32,
                           initializer=None,
                           regularizer=None,
                           partition_info=None,
                           reuse=None,
                           trainable=None,
                           collections=None,
                           caching_device=None,
                           validate_shape=True,
                           use_resource=None,
                           constraint=None,
                           synchronization=VariableSynchronization.AUTO,
                           aggregation=VariableAggregation.NONE):
    # Set to true if initializer is a constant.
    initializing_from_value = False
    if initializer is not None and not callable(initializer):
      initializing_from_value = True
    if shape is not None and initializing_from_value:
      raise ValueError("If initializer is a constant, do not specify shape.")

    dtype = dtypes.as_dtype(dtype)
    shape = tensor_shape.as_shape(shape)

    if name in self._vars:
      # Here we handle the case when returning an existing variable.
      if reuse is False:
        tb = self._vars[name].op.traceback[::-1]
        # Throw away internal tf entries and only take a few lines.
        tb = [x for x in tb if "tensorflow/python" not in x[0]][:3]
        raise ValueError("Variable %s already exists, disallowed."
                         " Did you mean to set reuse=True or "
                         "reuse=tf.AUTO_REUSE in VarScope? "
                         "Originally defined at:\n\n%s" % (
                             name, "".join(traceback.format_list(tb))))
      found_var = self._vars[name]
      if not shape.is_compatible_with(found_var.get_shape()):
        raise ValueError("Trying to share variable %s, but specified shape %s"
                         " and found shape %s." % (name, shape,
                                                   found_var.get_shape()))
      if not dtype.is_compatible_with(found_var.dtype):
        dtype_str = dtype.name
        found_type_str = found_var.dtype.name
        raise ValueError("Trying to share variable %s, but specified dtype %s"
                         " and found dtype %s." % (name, dtype_str,
                                                   found_type_str))
      return found_var

    # The code below handles only the case of creating a new variable.
    if reuse is True:
      raise ValueError("Variable %s does not exist, or was not created with "
                       "tf.get_variable(). Did you mean to set "
                       "reuse=tf.AUTO_REUSE in VarScope?" % name)

    # Create the tensor to initialize the variable with default value.
    if initializer is None:
      initializer, initializing_from_value = self._get_default_initializer(
          name=name, shape=shape, dtype=dtype)
    # Enter an init scope when creating the initializer.
    with ops.init_scope():
      if initializing_from_value:
        init_val = initializer
        variable_dtype = None
      else:
        # Instantiate initializer if provided initializer is a type object.
        if isinstance(initializer, type(init_ops.Initializer)):
          initializer = initializer(dtype=dtype)
        if shape and shape.is_fully_defined():
          init_val = lambda: initializer(  # pylint: disable=g-long-lambda
              shape.as_list(), dtype=dtype, partition_info=partition_info)
        elif not tf_inspect.getargspec(initializer).args:
          init_val = initializer
        else:
          raise ValueError("You can only pass an initializer function that "
                           "expects no arguments to its callable when the "
                           "shape is not fully defined. The given initializer "
                           "function expects the following args %s" %
                           tf_inspect.getargspec(initializer).args)
        variable_dtype = dtype.base_dtype

    # Create the variable.
    if use_resource is None:
      # Set the default value if unspecified.
      use_resource = _DEFAULT_USE_RESOURCE
    v = variables.VariableV1(
        initial_value=init_val,
        name=name,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        dtype=variable_dtype,
        validate_shape=validate_shape,
        constraint=constraint,
        use_resource=use_resource,
        synchronization=synchronization,
        aggregation=aggregation)
    if context.executing_eagerly() and self._store_eager_variables:
      if collections:
        ops.add_to_collections(collections, v)
      else:
        ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, v)
      if trainable:
        ops.add_to_collection(ops.GraphKeys.TRAINABLE_VARIABLES, v)

    if not context.executing_eagerly() or self._store_eager_variables:
      # In eager mode we do not want to keep default references to Variable
      # objects as this will prevent their memory from being released.
      self._vars[name] = v
    #logging.vlog(1, "Created variable %s with shape %s and init %s", v.name,                 format(shape), initializer)

    # Run the regularizer if requested and save the resulting loss.
    if regularizer:
      with ops.colocate_with(v):
        with ops.name_scope(name + "/Regularizer/"):
          with ops.init_scope():
            loss = regularizer(v)
        if loss is not None:
          if context.executing_eagerly():
            v_name = "v_%s" % type(v)
            loss_name = "loss_%s" % type(loss)
          else:
            v_name = v.name
            loss_name = loss.name
          #logging.vlog(1, "Applied regularizer to %s and added the result %s "   "to REGULARIZATION_LOSSES.", v_name, loss_name)
          ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, loss)
    return v

  # Initialize variable when no initializer provided
  def _get_default_initializer(self, name, shape=None, dtype=dtypes.float32):
    del shape
    # If dtype is DT_FLOAT, provide a uniform unit scaling initializer
    if dtype.is_floating:
      initializer = init_ops.glorot_uniform_initializer()
      initializing_from_value = False
    # If dtype is DT_INT/DT_UINT, provide a default value `zero`
    # If dtype is DT_BOOL, provide a default value `FALSE`
    elif (dtype.is_integer or dtype.is_unsigned or dtype.is_bool
          or dtype == dtypes.string):
      initializer = init_ops.zeros_initializer()
      initializing_from_value = False
    # NOTES:Do we need to support for handling DT_STRING and DT_COMPLEX here?
    else:
      raise ValueError("An initializer for variable %s of %s is required"
                       % (name, dtype.base_dtype))

    return initializer, initializing_from_value

def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 use_resource=None,
                 custom_getter=None,
                 constraint=None,
                 synchronization=VariableSynchronization.AUTO,
                 aggregation=VariableAggregation.NONE):
  return get_variable_scope().get_variable(
      _get_default_variable_store(),
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      trainable=trainable,
      collections=collections,
      caching_device=caching_device,
      partitioner=partitioner,
      validate_shape=validate_shape,
      use_resource=use_resource,
      custom_getter=custom_getter,
      constraint=constraint,
      synchronization=synchronization,
      aggregation=aggregation)

def _maybe_wrap_custom_getter(custom_getter, old_getter):
  if old_getter is None:
    return custom_getter

  # The new custom_getter should call the old one
  def wrapped_custom_getter(getter, *args, **kwargs):
    # Call:
    #  custom_getter(
    #    lambda: old_getter(true_getter, ...), *args, **kwargs)
    # which means custom_getter will call old_getter, which
    # will call the true_getter, perform any intermediate
    # processing, and return the results to the current
    # getter, which will also perform additional processing.
    return custom_getter(
        functools.partial(old_getter, getter),
        *args, **kwargs)
  return wrapped_custom_getter

