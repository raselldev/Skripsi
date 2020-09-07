from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading


from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import distribution_strategy_context



_update_device = threading.local()


def get_update_device():
  try:
    return _update_device.current
  except AttributeError:
    return None


class UpdateContext(object):

  def __init__(self, device):
    self._device = device
    self._old_device = None

  def __enter__(self):
    self._old_device = get_update_device()
    _update_device.current = self._device

  def __exit__(self, exception_type, exception_value, traceback):
    del exception_type, exception_value, traceback
    _update_device.current = self._old_device


# ------------------------------------------------------------------------------
# Public utility functions.


def get_loss_reduction():
  loss_reduction = ops.get_default_graph()._last_loss_reduction  # pylint: disable=protected-access
  return variable_scope.VariableAggregation.MEAN


# ------------------------------------------------------------------------------
# Internal API for validating the current thread mode


def _require_cross_tower_context(distribution_strategy):
  context = _get_per_thread_mode()
  if context.cross_tower_context is distribution_strategy: return
  # We have an error to report, figure out the right message.
  if context.distribution_strategy is not distribution_strategy:
    if (context.distribution_strategy is
        distribution_strategy_context._get_default_distribution_strategy()):  # pylint: disable=protected-access
      raise RuntimeError(
          'Need to be inside "with distribution_strategy.scope()" for %s' %
          (distribution_strategy,))
    else:
      raise RuntimeError(
          "Mixing different DistributionStrategy objects: %s is not %s" %
          (context.distribution_strategy, distribution_strategy))
  assert context.cross_tower_context is None
  raise RuntimeError("Method requires being in cross-tower context, use "
                     "get_tower_context().merge_call()")


def require_tower_context(tower_ctx):
  context = _get_per_thread_mode()
  if context.tower_context is tower_ctx: return
  # We have an error to report, figure out the right message.
  if context.tower_context is None:
    raise RuntimeError("Need to be inside `call_for_each_tower()`")
  if context.distribution_strategy is tower_ctx.distribution_strategy:
    # Two different TowerContexts with the same DistributionStrategy.
    raise RuntimeError("Mismatching tower context.")
  raise RuntimeError(
      "Mismatching DistributionStrategy objects: %s is not %s." %
      (context.distribution_strategy, tower_ctx.distribution_strategy))


def _require_distribution_strategy_scope(distribution_strategy):
  context = _get_per_thread_mode()
  if context.distribution_strategy is distribution_strategy: return
  # We have an error to report, figure out the right message.
  if (context.distribution_strategy is
      distribution_strategy_context._get_default_distribution_strategy()):  # pylint: disable=protected-access
    raise RuntimeError(
        'Need to be inside "with distribution_strategy.scope()" for %s' %
        (distribution_strategy,))
  else:
    raise RuntimeError(
        "Mixing different DistributionStrategy objects: %s is not %s" %
        (context.distribution_strategy, distribution_strategy))




class _CurrentDistributionContext(object):

  def __init__(self,
               distribution_strategy,
               var_creator_scope,
               var_scope=None,
               default_device=None):
    self._context = distribution_strategy_context._CrossTowerThreadMode(  # pylint: disable=protected-access
        distribution_strategy)
    self._var_creator_scope = var_creator_scope
    self._var_scope = var_scope
    if default_device:
      self._device_scope = ops.device(default_device)
    else:
      self._device_scope = None

  def __enter__(self):
    _push_per_thread_mode(self._context)
    if self._var_scope:
      self._var_scope.__enter__()
    self._var_creator_scope.__enter__()
    if self._device_scope:
      self._device_scope.__enter__()
    return self._context.distribution_strategy

  def __exit__(self, exception_type, exception_value, traceback):
    if self._device_scope:
      self._device_scope.__exit__(exception_type, exception_value, traceback)
    self._var_creator_scope.__exit__(exception_type, exception_value, traceback)
    if self._var_scope:
      self._var_scope.__exit__(exception_type, exception_value, traceback)
    _pop_per_thread_mode()


class _SameScopeAgainContext(object):

  def __init__(self, distribution_strategy):
    self._distribution_strategy = distribution_strategy

  def __enter__(self):
    return self._distribution_strategy

  def __exit__(self, exception_type, exception_value, traceback):
    del exception_type, exception_value, traceback


# ------------------------------------------------------------------------------
# Base classes for all distribution strategies.


class DistributionStrategy(object):
  def __init__(self):
    self._default_device = None

  def scope(self):
    if distribution_strategy_context.has_distribution_strategy():
      _require_cross_tower_context(self)
      return _SameScopeAgainContext(self)

    def creator_with_resource_vars(*args, **kwargs):
      _require_distribution_strategy_scope(self)
      kwargs["use_resource"] = True
      return self._create_variable(*args, **kwargs)

    def disable_partitioned_variables(getter, *args, **kwargs):
      if kwargs.pop("partitioner", None) is not None:
        tf_logging.log_first_n(
            tf_logging.WARN, "Partitioned variables are disabled when using "
            "DistributionStrategy.", 1)
      return getter(*args, **kwargs)

    return _CurrentDistributionContext(
        self, variable_scope.variable_creator_scope(creator_with_resource_vars),
        variable_scope.variable_scope(
            variable_scope.get_variable_scope(),
            custom_getter=disable_partitioned_variables),
        self._default_device)

  def _create_variable(self, next_creator, *args, **kwargs):
    # Note: should support "colocate_with" argument.
    raise NotImplementedError("must be implemented in descendants")

  def read_var(self, v):
    raise NotImplementedError("must be implemented in descendants")

  def colocate_vars_with(self, colocate_with_variable):
    def create_colocated_variable(next_creator, *args, **kwargs):
      _require_distribution_strategy_scope(self)
      kwargs["use_resource"] = True
      kwargs["colocate_with"] = colocate_with_variable
      return next_creator(*args, **kwargs)

    _require_distribution_strategy_scope(self)
    return variable_scope.variable_creator_scope(create_colocated_variable)

  def _call_dataset_fn(self, dataset_fn):
    result = dataset_fn()
    if not isinstance(result, dataset_ops.Dataset):
      raise ValueError(
          "dataset_fn() must return a tf.data.Dataset when using a "
          "DistributionStrategy.")
    return result

  def distribute_dataset(self, dataset_fn):
    raise NotImplementedError("must be implemented in descendants")

  def broadcast(self, tensor, destinations=None):
    _require_cross_tower_context(self)
    return self._broadcast(tensor, destinations)

  def _broadcast(self, tensor, destinations):
    raise NotImplementedError("must be implemented in descendants")

  def initialize(self):
    if eager_context.executing_eagerly():
      return
    else:
      return []

  def finalize(self):
    if eager_context.executing_eagerly():
      return
    else:
      return []

  def run_steps_on_dataset(self, fn, iterator, iterations=1,
                           initial_loop_values=None):
    _require_cross_tower_context(self)
    return self._run_steps_on_dataset(fn, iterator, iterations,
                                      initial_loop_values)

  def _run_steps_on_dataset(self, fn, iterator, iterations,
                            initial_loop_values):
    raise NotImplementedError("must be implemented in descendants")

  def call_for_each_tower(self, fn, *args, **kwargs):
    _require_cross_tower_context(self)
    return self._call_for_each_tower(fn, *args, **kwargs)

  def _call_for_each_tower(self, fn, *args, **kwargs):
    raise NotImplementedError("must be implemented in descendants")

  def reduce(self, aggregation, value, destinations):
    _require_cross_tower_context(self)
    assert aggregation in [
        variable_scope.VariableAggregation.SUM,
        variable_scope.VariableAggregation.MEAN,
        variable_scope.VariableAggregation.ONLY_FIRST_TOWER
    ]
    return self._reduce(aggregation, value, destinations)

  def _reduce(self, aggregation, value, destinations):
    raise NotImplementedError("must be implemented in descendants")

  def batch_reduce(self, aggregation, value_destination_pairs):
    
    _require_cross_tower_context(self)
    assert aggregation in [
        variable_scope.VariableAggregation.SUM,
        variable_scope.VariableAggregation.MEAN,
        variable_scope.VariableAggregation.ONLY_FIRST_TOWER
    ]
    return self._batch_reduce(aggregation, value_destination_pairs)

  def _batch_reduce(self, aggregation, value_destination_pairs):
    return [
        self.reduce(aggregation, t, destinations=v)
        for t, v in value_destination_pairs
    ]

  def update(self, var, fn, *args, **kwargs):
    _require_cross_tower_context(self)
    options = {"grouped": kwargs.pop("grouped", True)}
    return self._update(var, options, fn, *args, **kwargs)

  def _update(self, var, options, fn, *args, **kwargs):
    raise NotImplementedError("must be implemented in descendants")

  def update_non_slot(self, colocate_with, fn, *args, **kwargs):
    _require_cross_tower_context(self)
    options = {"grouped": kwargs.pop("grouped", True)}
    return self._update_non_slot(colocate_with, options, fn, *args, **kwargs)

  def _update_non_slot(self, colocate_with, options, fn, *args, **kwargs):
    raise NotImplementedError("must be implemented in descendants")

  def unwrap(self, value):
    return self._unwrap(value)

  def value_container(self, value):
    raise NotImplementedError("must be implemented in descendants")

  def _unwrap(self, distributed_value):
    raise NotImplementedError("must be implemented in descendants")

  def group(self, value, name=None):
    value = nest.flatten(self.unwrap(value))

    if len(value) != 1 or name is not None:
      return control_flow_ops.group(value, name=name)
    # Special handling for the common case of one op.
    v, = value
    if hasattr(v, "op"):
      v = v.op
    return v

  @property
  def is_single_tower(self):
    raise NotImplementedError("must be implemented in descendants")

  @property
  def num_towers(self):
    raise NotImplementedError("must be implemented in descendants")

  @property
  def worker_devices(self):
    
    raise NotImplementedError("must be implemented in descendants")

  @property
  def parameter_devices(self):
    
    raise NotImplementedError("must be implemented in descendants")

  def non_slot_devices(self, var_list):
    raise NotImplementedError("must be implemented in descendants")

  @property
  def worker_device_index(self):
    _require_cross_tower_context(self)
    return self._worker_device_index()

  def _worker_device_index(self):
    raise NotImplementedError("must be implemented in descendants")

  @property
  def between_graph(self):
    raise NotImplementedError("must be implemented in descendants")

  def configure(self,
                session_config=None,
                cluster_spec=None,
                task_type=None,
                task_id=None):
    del session_config, cluster_spec, task_type, task_id

  @property
  def should_init(self):
    raise NotImplementedError("must be implemented in descendants")

  @property
  def should_checkpoint(self):
    raise NotImplementedError("must be implemented in descendants")

  @property
  def should_save_summary(self):
    raise NotImplementedError("must be implemented in descendants")

class TowerContext(object):

  def __init__(self, distribution_strategy, tower_id):
    self._distribution_strategy = distribution_strategy
    self._thread_context = distribution_strategy_context._InTowerThreadMode(  # pylint: disable=protected-access
        self)
    self._tower_id = tower_id

  def __enter__(self):
    _push_per_thread_mode(self._thread_context)

  def __exit__(self, exception_type, exception_value, traceback):
    _pop_per_thread_mode()

  def merge_call(self, merge_fn, *args, **kwargs):
    require_tower_context(self)
    return self._merge_call(merge_fn, *args, **kwargs)

  def _merge_call(self, merge_fn, *args, **kwargs):
    _push_per_thread_mode(  # thread-local, so not needed with multiple threads
        distribution_strategy_context._CrossTowerThreadMode(  # pylint: disable=protected-access
            self._distribution_strategy))
    try:
      return merge_fn(self._distribution_strategy, *args, **kwargs)
    finally:
      _pop_per_thread_mode()

  @property
  def is_single_tower(self):
    require_tower_context(self)
    return self._distribution_strategy.is_single_tower

  @property
  def num_towers(self):
    return self._distribution_strategy.num_towers

  @property
  def tower_id(self):
    require_tower_context(self)
    return self._tower_id

  @property
  def distribution_strategy(self):
    return self._distribution_strategy

  @property
  def device(self):
    require_tower_context(self)
    return device_util.current()

  # TODO(josh11b): Implement `start_all_reduce(method, t)` for efficient
  # all-reduce. It would return a function returning the result of reducing `t`
  # across all towers. The caller would wait to call this function until they
  # needed the reduce result, allowing an efficient implementation:
  # * With eager execution, the reduction could be performed asynchronously
  #   in the background, not blocking until the result was needed.
  # * When constructing a graph, it could batch up all reduction requests up
  #   to that point that the first result is needed. Most likely this can be
  #   implemented in terms of `merge_call()` and `batch_reduce()`.

# ------------------------------------------------------------------------------


class _DefaultDistributionStrategy(DistributionStrategy):

  def scope(self):
    if distribution_strategy_context.has_distribution_strategy():
      raise RuntimeError("Must not nest DistributionStrategy scopes.")

    def creator(next_creator, *args, **kwargs):
      _require_distribution_strategy_scope(self)
      return next_creator(*args, **kwargs)

    return _CurrentDistributionContext(
        self, variable_scope.variable_creator_scope(creator))

  def colocate_vars_with(self, colocate_with_variable):
    _require_distribution_strategy_scope(self)
    return ops.colocate_with(colocate_with_variable)

  def distribute_dataset(self, dataset_fn):
    return self._call_dataset_fn(dataset_fn)

  def _broadcast(self, tensor, destinations):
    if destinations is None:
      return tensor
    else:
      raise NotImplementedError("TODO")

  def _call_for_each_tower(self, fn, *args, **kwargs):
    # We don't run `fn` in multiple threads in _DefaultDistributionStrategy.
    kwargs.pop("run_concurrently", None)
    with TowerContext(self, tower_id=0):
      return fn(*args, **kwargs)

  def _reduce(self, aggregation, value, destinations):
    # TODO(josh11b): Use destinations?
    del aggregation, destinations
    return value

  def _update(self, var, options, fn, *args, **kwargs):
    # The implementations of _update() and _update_non_slot() are identical
    # except _update() passes `var` as the first argument to `fn()`.
    return self._update_non_slot(var, options, fn, var, *args, **kwargs)

  def _update_non_slot(self, colocate_with, options, fn, *args, **kwargs):
    should_group = options.pop("grouped")
    assert not options  # Validate that we are processing all of the options.
    # TODO(josh11b): Figure out what we should be passing to UpdateContext()
    # once that value is used for something.
    with ops.colocate_with(colocate_with), UpdateContext(colocate_with):
      result = fn(*args, **kwargs)
      if should_group:
        return result
      else:
        return nest.map_structure(self._unwrap, result)

  def read_var(self, tower_local_var):
    return array_ops.identity(tower_local_var)

  def _unwrap(self, distributed_value):
    return [distributed_value]

  def value_container(self, value):
    return value

  @property
  def is_single_tower(self):
    return True

  @property
  def num_towers(self):
    return 1

  @property
  def worker_devices(self):
    raise RuntimeError(
        "worker_devices() method unsupported by _DefaultDistributionStrategy.")

  @property
  def parameter_devices(self):
    raise RuntimeError("parameter_devices() method unsupported by "
                       "_DefaultDistributionStrategy.")

  def non_slot_devices(self, var_list):
    return min(var_list, key=lambda x: x.name)

  def _worker_device_index(self):
    raise RuntimeError("worker_device_index() method unsupported by "
                       "_DefaultDistributionStrategy.")


# ------------------------------------------------------------------------------
# Deprecated, use v.assign_add(amount) instead.  Internal API, so expect
# it to be deleted soon.



def increment_var(v, amount=1):
  def update(vu):
    return vu.assign_add(amount, read_value=False)

  def merge_fn(dist, vm):
    return dist.update(vm, update)

  tower_context = distribution_strategy_context.get_tower_context()
  return tower_context.merge_call(merge_fn, v)


# ------------------------------------------------------------------------------
# We haven't yet implemented deserialization for DistributedVariables.
# So here we catch any attempts to deserialize variables
# when using distribution strategies.
# pylint: disable=protected-access
_original_from_proto = resource_variable_ops._from_proto_fn


def _from_proto_fn(v, import_scope=None):
  if distribution_strategy_context.has_distribution_strategy():
    raise NotImplementedError(
        "Deserialization of variables is not yet supported when using"
        "distributed strategies.")
  else:
    return _original_from_proto(v, import_scope=import_scope)

resource_variable_ops._from_proto_fn = _from_proto_fn
# pylint: enable=protected-access


#-------------------------------------------------------------------------------
# Shorthand for some methods from distribution_strategy_context.
_push_per_thread_mode = distribution_strategy_context._push_per_thread_mode  # pylint: disable=protected-access
_get_per_thread_mode = distribution_strategy_context._get_per_thread_mode  # pylint: disable=protected-access
_pop_per_thread_mode = distribution_strategy_context._pop_per_thread_mode  # pylint: disable=protected-access
