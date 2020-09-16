from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading


from backend.python.framework import ops
from backend.python.ops import variable_scope
from backend.python.ops import resource_variable_ops
from backend.python.training import distribution_strategy_context



_update_device = threading.local()



class _CurrentDistributionContext(object):

  def __init__(self,
               distribution_strategy,
               var_creator_scope,
               var_scope=None,
               default_device=None):
    self._context = distribution_strategy_context._CrossTowerThreadMode(  
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

class DistributionStrategy(object):
  def __init__(self):
    self._default_device = None

  def scope(self):
    if distribution_strategy_context.has_distribution_strategy():
      context = _get_per_thread_mode()
      return _SameScopeAgainContext(self)

    def creator_with_resource_vars(*args, **kwargs):
      context = _get_per_thread_mode()
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


  def colocate_vars_with(self, colocate_with_variable):
    def create_colocated_variable(next_creator, *args, **kwargs):
      kwargs["use_resource"] = True
      kwargs["colocate_with"] = colocate_with_variable
      return next_creator(*args, **kwargs)
    return variable_scope.variable_creator_scope(create_colocated_variable)

  def _call_dataset_fn(self, dataset_fn):
    result = dataset_fn()
    if not isinstance(result, dataset_ops.Dataset):
      raise ValueError(
          "dataset_fn() must return a tf.data.Dataset when using a "
          "DistributionStrategy.")
    return result


  def broadcast(self, tensor, destinations=None):
    context = _get_per_thread_mode()
    return self._broadcast(tensor, destinations)


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
    context = _get_per_thread_mode()
    return self._run_steps_on_dataset(fn, iterator, iterations,
                                      initial_loop_values)


  def call_for_each_tower(self, fn, *args, **kwargs):
    context = _get_per_thread_mode()
    return self._call_for_each_tower(fn, *args, **kwargs)


  def reduce(self, aggregation, value, destinations):
    context = _get_per_thread_mode()
    assert aggregation in [
        variable_scope.VariableAggregation.SUM,
        variable_scope.VariableAggregation.MEAN,
        variable_scope.VariableAggregation.ONLY_FIRST_TOWER
    ]
    return self._reduce(aggregation, value, destinations)


  def batch_reduce(self, aggregation, value_destination_pairs):
    
    context = _get_per_thread_mode()
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
    context = _get_per_thread_mode()
    options = {"grouped": kwargs.pop("grouped", True)}
    return self._update(var, options, fn, *args, **kwargs)


  def update_non_slot(self, colocate_with, fn, *args, **kwargs):
    context = _get_per_thread_mode()
    options = {"grouped": kwargs.pop("grouped", True)}
    return self._update_non_slot(colocate_with, options, fn, *args, **kwargs)


  def unwrap(self, value):
    return self._unwrap(value)


  def group(self, value, name=None):
    value = nest.flatten(self.unwrap(value))

    if len(value) != 1 or name is not None:
      return control_flow_ops.group(value, name=name)
    v, = value
    if hasattr(v, "op"):
      v = v.op
    return v


  @property
  def worker_device_index(self):
    context = _get_per_thread_mode()
    return self._worker_device_index()

  def configure(self,
                session_config=None,
                cluster_spec=None,
                task_type=None,
                task_id=None):
    del session_config, cluster_spec, task_type, task_id

class TowerContext(object):

  def __init__(self, distribution_strategy, tower_id):
    self._distribution_strategy = distribution_strategy
    self._thread_context = distribution_strategy_context._InTowerThreadMode(  
        self)
    self._tower_id = tower_id

  def __enter__(self):
    _push_per_thread_mode(self._thread_context)

  def __exit__(self, exception_type, exception_value, traceback):
    _pop_per_thread_mode()

  def merge_call(self, merge_fn, *args, **kwargs):
    context = _get_per_thread_mode()
    return self._merge_call(merge_fn, *args, **kwargs)

  def _merge_call(self, merge_fn, *args, **kwargs):
    _push_per_thread_mode(
        distribution_strategy_context._CrossTowerThreadMode(  
            self._distribution_strategy))
    try:
      return merge_fn(self._distribution_strategy, *args, **kwargs)
    finally:
      _pop_per_thread_mode()

  @property
  def is_single_tower(self):
    context = _get_per_thread_mode()
    return self._distribution_strategy.is_single_tower

  @property
  def num_towers(self):
    return self._distribution_strategy.num_towers

  @property
  def tower_id(self):
    context = _get_per_thread_mode()
    return self._tower_id

  @property
  def distribution_strategy(self):
    return self._distribution_strategy

  @property
  def device(self):
    context = _get_per_thread_mode()
    return device_util.current()

class _DefaultDistributionStrategy(DistributionStrategy):

  def scope(self):
    def creator(next_creator, *args, **kwargs):
      context = _get_per_thread_mode()
      return next_creator(*args, **kwargs)

    return _CurrentDistributionContext(
        self, variable_scope.variable_creator_scope(creator))

  def colocate_vars_with(self, colocate_with_variable):
    context = _get_per_thread_mode()
    return ops.colocate_with(colocate_with_variable)

  def distribute_dataset(self, dataset_fn):
    return self._call_dataset_fn(dataset_fn)

  def _call_for_each_tower(self, fn, *args, **kwargs):
    kwargs.pop("run_concurrently", None)
    with TowerContext(self, tower_id=0):
      return fn(*args, **kwargs)

  def _reduce(self, aggregation, value, destinations):
    del aggregation, destinations
    return value

  def _update(self, var, options, fn, *args, **kwargs):
    return self._update_non_slot(var, options, fn, var, *args, **kwargs)

  
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

  def non_slot_devices(self, var_list):
    return min(var_list, key=lambda x: x.name)


def increment_var(v, amount=1):
  def update(vu):
    return vu.assign_add(amount, read_value=False)

  def merge_fn(dist, vm):
    return dist.update(vm, update)

  tower_context = distribution_strategy_context.get_tower_context()
  return tower_context.merge_call(merge_fn, v)

_original_from_proto = resource_variable_ops._from_proto_fn

def _from_proto_fn(v, import_scope=None):
  return _original_from_proto(v, import_scope=import_scope)

resource_variable_ops._from_proto_fn = _from_proto_fn
_push_per_thread_mode = distribution_strategy_context._push_per_thread_mode  
_get_per_thread_mode = distribution_strategy_context._get_per_thread_mode  
_pop_per_thread_mode = distribution_strategy_context._pop_per_thread_mode  
