#Contains the base Layer class, from which all layers inherit
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from backend.python import base_layer
from backend.python.ops import variable_scope as vs
from backend.python.framework import ops
from backend.python.framework import dtypes

InputSpec = base_layer.InputSpec  # pylint: disable=invalid-name

class Layer(base_layer.Layer):
  def __init__(self, trainable=True, name=None, dtype=None,
               **kwargs):
    self._use_resource_variables = False
    scope = kwargs.pop('_scope', None)
    self._reuse = kwargs.pop('_reuse', None)

    # Avoid an incorrect lint error
    self._trainable_weights = []
    self.built = False

    super(Layer, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                **kwargs)

    self._graph = None
    if scope:
      with vs.variable_scope(scope) as captured_scope:
        self._scope = captured_scope
    else:
      self._scope = None
    self._current_scope = None

  def _init_set_name(self, name):
    # Determine layer name (non-unique).
    if isinstance(name, vs.VariableScope):
      base_name = name.name
    else:
      base_name = name
      self._name = name
    if not name:
      self._name, base_name = self._make_unique_name()
    self._base_name = base_name

  def _make_unique_name(self, name_uid_map=None, avoid_names=None,
                        namespace='', zero_based=False):
    base_name = base_layer.to_snake_case(self.__class__.__name__)
    name = base_layer.unique_layer_name(base_name,
                                        name_uid_map=name_uid_map,
                                        avoid_names=avoid_names,
                                        namespace=namespace,
                                        zero_based=zero_based)
    return (name, base_name)

  def _name_scope(self):
    """Determines op naming for the Layer."""
    return self._current_scope.original_name_scope

  def _set_scope(self, scope=None):
    if self._scope is None:
      # If constructed with _scope=None, lazy setting of scope.
      if self._reuse:
        with vs.variable_scope(
            scope if scope is not None else self._base_name) as captured_scope:
          self._scope = captured_scope
      else:
        with vs.variable_scope(
            scope, default_name=self._base_name) as captured_scope:
          self._scope = captured_scope

  def add_weight(self,
                 name,
                 shape,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 constraint=None,
                 use_resource=None,
                 synchronization=vs.VariableSynchronization.AUTO,
                 aggregation=vs.VariableAggregation.NONE,
                 partitioner=None):
    
    if dtype is None:
      dtype = self.dtype or dtypes.float32

    self._set_scope(None)
    reuse = self.built or self._reuse
    with vs.variable_scope(
        self._scope, reuse=reuse, auxiliary_name_scope=False) as scope:
      self._current_scope = scope
      with ops.name_scope(self._name_scope()):
        use_resource = (use_resource or
                        self._use_resource_variables or
                        scope.use_resource)
        variable = super(Layer, self).add_weight(
            name,
            shape,
            dtype=dtypes.as_dtype(dtype),
            initializer=initializer,
            trainable=trainable,
            constraint=constraint,
            partitioner=partitioner,
            use_resource=use_resource,
            synchronization=synchronization,
            aggregation=aggregation,
            getter=vs.get_variable)
    return variable

  def __call__(self, inputs, *args, **kwargs):
    self._set_scope(kwargs.pop('scope', None))

    if self.built:
      self._always_reuse_variable_scope = vs.variable_scope(
            self._scope, reuse=True, auxiliary_name_scope=False)
      scope_context_manager = self._always_reuse_variable_scope

    else:
      scope_context_manager = vs.variable_scope(
          self._scope, reuse=self._reuse, auxiliary_name_scope=False)

    with scope_context_manager as scope:
      self._current_scope = scope
      outputs = super(Layer, self).__call__(inputs, *args, **kwargs)
    return outputs
