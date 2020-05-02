from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import losses_impl
from tensorflow.python.framework import ops
from tensorflow.python.training import distribution_strategy_context


def get_loss_reduction():
  loss_reduction = ops.get_default_graph()._last_loss_reduction  # pylint: disable=protected-access
  if loss_reduction == losses_impl.Reduction.SUM:
    return variable_scope.VariableAggregation.SUM
  return variable_scope.VariableAggregation.MEAN

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

  # TODO(josh11b): `PerDeviceDataset` currently only implements a few methods of
  # Dataset API such as make_one_shot_iterator and make_initializable_iterator.
  # Extend to implement more functionality of datasets.
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
    # TODO(josh11b): More docstring
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