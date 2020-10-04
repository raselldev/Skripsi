from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


from backend import context
from backend.framework import ops
from backend.framework import dtypes
from backend.ops import control_flow_ops
from backend.ops import resource_variable_ops
from backend.ops import gradients_impl as gradients
from backend.ops import variables
from backend.ops import variable_scope
#from backend.python.training import slot_creator
from backend.training import distribute
from backend.training import base as checkpointable

def get_loss_reduction():
  loss_reduction = ops.get_default_graph()._last_loss_reduction  
  return variable_scope.VariableAggregation.MEAN

def _get_processor(v):
  if context.executing_eagerly():
    if isinstance(v, ops.Tensor):
      return _TensorProcessor(v)
    else:
      return _DenseResourceVariableProcessor(v)
  if v.op.type == "VarHandleOp":
    return _DenseResourceVariableProcessor(v)
  if isinstance(v, variables.Variable):
    return _RefVariableProcessor(v)
  if isinstance(v, ops.Tensor):
    return _TensorProcessor(v)
  raise NotImplementedError("Trying to optimize unsupported type ", v)

def _var_key(var):
  if hasattr(var, "op"):
    return (var.op.graph, var.op.name)
  return var._unique_id  

class _RefVariableProcessor(object):
  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v._ref()  

  def update_op(self, optimizer, g):
    if isinstance(g, ops.Tensor):
      update_op = optimizer._apply_dense(g, self._v)  
      if self._v.constraint is not None:
        with ops.control_dependencies([update_op]):
          return self._v.assign(self._v.constraint(self._v))
      else:
        return update_op

class Optimizer(
    checkpointable.CheckpointableBase):
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2

  def __init__(self, use_locking, name):
    if not name:
      raise ValueError("Must specify the optimizer name")
    self._use_locking = use_locking
    self._name = name
    self._slots = {}
    self._non_slot_dict = {}
    self._deferred_slot_restorations = {}

  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
    grads_and_vars = self.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    return self.apply_gradients(grads_and_vars, global_step=global_step,
                                name=name)

  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    if callable(loss):
      with backprop.GradientTape() as tape:
        if var_list is not None:
          tape.watch(var_list)
        loss_value = loss()

        if (distribute_lib.get_loss_reduction() ==
            variable_scope.VariableAggregation.MEAN):
          num_towers = distribute.get_distribution_strategy(
          ).num_towers
          if num_towers > 1:
            loss_value *= (1. / num_towers)

      if var_list is None:
        var_list = tape.watched_variables()
      with ops.control_dependencies([loss_value]):
        grads = tape.gradient(loss_value, var_list, grad_loss)
      return list(zip(grads, var_list))


    if (get_loss_reduction() ==
        variable_scope.VariableAggregation.MEAN):
      num_towers = distribute.get_distribution_strategy(
      ).num_towers
      if num_towers > 1:
        loss *= (1. / num_towers)

    self._assert_valid_dtypes([loss])
    if grad_loss is not None:
      self._assert_valid_dtypes([grad_loss])
    if var_list is None:
      var_list = (
          variables.trainable_variables() +
          ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    else:
      var_list = nest.flatten(var_list)
    
    var_list += ops.get_collection(ops.GraphKeys._STREAMING_MODEL_PORTS)
   
    processors = [_get_processor(v) for v in var_list]

    var_refs = [p.target() for p in processors]
    grads = gradients.gradients(
        loss, var_refs, grad_ys=grad_loss,
        gate_gradients=(gate_gradients == Optimizer.GATE_OP),
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops)
    if gate_gradients == Optimizer.GATE_GRAPH:
      grads = control_flow_ops.tuple(grads)
    grads_and_vars = list(zip(grads, var_list))
    self._assert_valid_dtypes(
        [v for g, v in grads_and_vars
         if g is not None and v.dtype != dtypes.resource])
    return grads_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    if distribute.has_distribution_strategy():
      grads_and_vars = get_filtered_grad_fn(lambda: grads_and_vars)()


    grads_and_vars = tuple(grads_and_vars)
    converted_grads_and_vars = []
    for g, v in grads_and_vars:
      p = _get_processor(v)
      converted_grads_and_vars.append((g, v, p))

    converted_grads_and_vars = tuple(converted_grads_and_vars)
    var_list = [v for g, v, _ in converted_grads_and_vars if g is not None]

    with ops.init_scope():
      self._create_slots(var_list)
    update_ops = []
    with ops.name_scope(name, self._name) as name:
      self._prepare()
      for grad, var, processor in converted_grads_and_vars:
        if grad is None:
          continue
        if context.executing_eagerly() or isinstance(
            var,
            resource_variable_ops.ResourceVariable) and not var._in_graph_mode:  
          scope_name = ""
        else:
          scope_name = var.op.name
        with ops.name_scope("update_" + scope_name), ops.colocate_with(var):
          update_ops.append(processor.update_op(self, grad))
      if global_step is None:
        apply_updates = self._finish(update_ops, name)
      
      return apply_updates

  def _distributed_apply(self,
                         distribution,
                         grads_and_vars,
                         global_step=None,
                         name=None):
    reduced_grads = distribution.batch_reduce(
        variable_scope.VariableAggregation.SUM, grads_and_vars)
    var_list = [v for _, v in grads_and_vars]
    grads_and_vars = zip(reduced_grads, var_list)
    self._create_slots(var_list)

    def update(v, g):
      assert v is not None
      p = _get_processor(v)

      scope_name = "" if context.executing_eagerly() else v.op.name
      with ops.name_scope("update_" + scope_name):
        return p.update_op(self, g)

    with ops.name_scope(name, self._name) as name:
      self._prepare()

      update_ops = [
          op
          for grad, var in grads_and_vars
          for op in distribution.update(var, update, grad, grouped=False)
      ]

      non_slot_devices = distribution.non_slot_devices(var_list)
      finish_updates = distribution.update_non_slot(
          non_slot_devices, finish, self, update_ops, grouped=False)
      if global_step is None:
        apply_updates = distribution.group(finish_updates, name=name)
      

      if not context.executing_eagerly():
        if isinstance(apply_updates, ops.Tensor):
          apply_updates = apply_updates.op
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        if apply_updates not in train_op:
          train_op.append(apply_updates)

      return apply_updates

  def get_slot(self, var, name):
    
    named_slots = self._slots.get(name, None)
    if not named_slots:
      return None

    if hasattr(var, "_distributed_container"):
      distributed_container = var._distributed_container()
      assert distributed_container is not None
      if context.executing_eagerly():
        key = distributed_container._unique_id
      else:
        key = (distributed_container.graph, distributed_container._shared_name)
     
      mirrored_slot = named_slots.get(key, None)
      if mirrored_slot is None: return None
      return mirrored_slot.get(device=var.device)

    return named_slots.get(_var_key(var), None)

  def get_slot_names(self):
    return sorted(self._slots.keys())

  def variables(self):
    current_graph = ops.get_default_graph()

    def _from_current_graph(variable):
      if variable._in_graph_mode:  
        return variable.op.graph is current_graph
      else:
        return variable._graph_key == current_graph._graph_key  

    optimizer_variables = [v for v in self._non_slot_variables()
                           if _from_current_graph(v)]
    for _, variable_dict in self._slots.items():
      for _, slot_for_variable in variable_dict.items():
        if _from_current_graph(slot_for_variable):
          optimizer_variables.append(slot_for_variable)
    return sorted(optimizer_variables, key=lambda v: v.name)

  def _create_non_slot_variable(self, initial_value, name, colocate_with):
    eager = context.executing_eagerly()
    graph = None if eager else colocate_with.graph

    key = (name, graph)
    v = self._non_slot_dict.get(key, None)
    if v is None:
      self._maybe_initialize_checkpointable()
      distribution_strategy = (
          distribute.get_distribution_strategy())
      with distribution_strategy.colocate_vars_with(colocate_with):
        if eager:
          restored_initial_value = self._preload_simple_restoration(
              name=name, shape=None)
          if restored_initial_value is not None:
            initial_value = restored_initial_value
        v = variable_scope.variable(initial_value, name=name, trainable=False)
      self._handle_deferred_dependencies(name=name, checkpointable=v)
      self._non_slot_dict[key] = v

    return v

  def _assert_valid_dtypes(self, tensors):
    valid_dtypes = self._valid_dtypes()
    for t in tensors:
      dtype = t.dtype.base_dtype

  def _valid_dtypes(self):
    return set(
        [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64])

  def _create_slots(self, var_list):
    pass

  def _resource_apply_dense(self, grad, handle):
    raise NotImplementedError()

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    summed_grad, unique_indices = _deduplicate_indexed_slices(
        values=grad, indices=indices)
    return self._resource_apply_sparse(summed_grad, handle, unique_indices)

  def _resource_apply_sparse(self, grad, handle, indices):
    raise NotImplementedError()

  def _apply_sparse_duplicate_indices(self, grad, var):
    summed_values, unique_indices = _deduplicate_indexed_slices(
        values=grad.values, indices=grad.indices)
    gradient_no_duplicate_indices = ops.IndexedSlices(
        indices=unique_indices,
        values=summed_values,
        dense_shape=grad.dense_shape)
    return self._apply_sparse(gradient_no_duplicate_indices, var)

  def _apply_sparse(self, grad, var):
    raise NotImplementedError()

  def _finish(self, update_ops, name_scope):
    return control_flow_ops.group(*update_ops, name=name_scope)

  def _slot_dict(self, slot_name):
    named_slots = self._slots.get(slot_name, None)
    if named_slots is None:
      named_slots = {}
      self._slots[slot_name] = named_slots
    return named_slots

  def _get_or_make_slot(self, var, val, slot_name, op_name):
    named_slots = self._slot_dict(slot_name)
    if _var_key(var) not in named_slots:
      new_slot_variable = slot_creator.create_slot(var, val, op_name)
      self._restore_slot_variable(
          slot_name=slot_name, variable=var,
          slot_variable=new_slot_variable)
      named_slots[_var_key(var)] = new_slot_variable
    return named_slots[_var_key(var)]

  def _get_or_make_slot_with_initializer(self, var, initializer, shape, dtype,
                                         slot_name, op_name):
    named_slots = self._slot_dict(slot_name)
    if _var_key(var) not in named_slots:
      new_slot_variable = slot_creator.create_slot_with_initializer(
          var, initializer, shape, dtype, op_name)
      self._restore_slot_variable(
          slot_name=slot_name, variable=var,
          slot_variable=new_slot_variable)
      named_slots[_var_key(var)] = new_slot_variable
    return named_slots[_var_key(var)]

  def _zeros_slot(self, var, slot_name, op_name):
    named_slots = self._slot_dict(slot_name)
    if _var_key(var) not in named_slots:
      new_slot_variable = slot_creator.create_zeros_slot(var, op_name)
      self._restore_slot_variable(
          slot_name=slot_name, variable=var,
          slot_variable=new_slot_variable)
      named_slots[_var_key(var)] = new_slot_variable
    return named_slots[_var_key(var)]

  def _restore_slot_variable(self, slot_name, variable, slot_variable):
    variable_key = _var_key(variable)
    deferred_restorations = self._deferred_slot_restorations.get(
        slot_name, {}).pop(variable_key, [])
    deferred_restorations.sort(key=lambda position: position.restore_uid,
                               reverse=True)
    for checkpoint_position in deferred_restorations:
      checkpoint_position.restore(slot_variable)

  def _call_if_callable(self, param):
    return param() if callable(param) else param

class RMSPropOptimizer(Optimizer):
  """Optimizer that implements the RMSProp algorithm.
  See the
  [paper](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
  """

  def __init__(self,
               learning_rate,
               decay=0.9,
               momentum=0.0,
               epsilon=1e-10,
               use_locking=False,
               centered=False,
               name="RMSProp"):
    super(RMSPropOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._decay = decay
    self._momentum = momentum
    self._epsilon = epsilon
    self._centered = centered

    # Tensors for learning rate and momentum.  Created in _prepare.
    self._learning_rate_tensor = None
    self._decay_tensor = None
    self._momentum_tensor = None
    self._epsilon_tensor = None

  def _create_slots(self, var_list):
    for v in var_list:
      if v.get_shape().is_fully_defined():
        init_rms = init_ops.ones_initializer(dtype=v.dtype.base_dtype)
      else:
        init_rms = array_ops.ones_like(v)
      self._get_or_make_slot_with_initializer(v, init_rms, v.get_shape(),
                                              v.dtype.base_dtype, "rms",
                                              self._name)
      if self._centered:
        self._zeros_slot(v, "mg", self._name)
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._learning_rate)
    decay = self._call_if_callable(self._decay)
    momentum = self._call_if_callable(self._momentum)
    epsilon = self._call_if_callable(self._epsilon)

    self._learning_rate_tensor = ops.convert_to_tensor(lr, name="learning_rate")
    self._decay_tensor = ops.convert_to_tensor(decay, name="decay")
    self._momentum_tensor = ops.convert_to_tensor(momentum, name="momentum")
    self._epsilon_tensor = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return training_ops.apply_centered_rms_prop(
          var,
          mg,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad,
          use_locking=self._use_locking).op
    else:
      return training_ops.apply_rms_prop(
          var,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad,
          use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return training_ops.resource_apply_centered_rms_prop(
          var.handle,
          mg.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype.base_dtype),
          grad,
          use_locking=self._use_locking)
    else:
      return training_ops.resource_apply_rms_prop(
          var.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype.base_dtype),
          grad,
          use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return training_ops.sparse_apply_centered_rms_prop(
          var,
          mg,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad.values,
          grad.indices,
          use_locking=self._use_locking)
    else:
      return training_ops.sparse_apply_rms_prop(
          var,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad.values,
          grad.indices,
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return training_ops.resource_sparse_apply_centered_rms_prop(
          var.handle,
          mg.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._decay_tensor, grad.dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype),
          grad,
          indices,
          use_locking=self._use_locking)
    else:
      return training_ops.resource_sparse_apply_rms_prop(
          var.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._decay_tensor, grad.dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype),
          grad,
          indices,
          use_locking=self._use_locking)
