from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl as gradients
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.framework import dtypes
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python import context
from tensorflow.python.training import base as checkpointable


class Optimizer(
    checkpointable.CheckpointableBase):
  

  # Values for gate_gradients.
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2

  def __init__(self, use_locking, name):
    if not name:
      raise ValueError("Must specify the optimizer name")
    self._use_locking = use_locking
    self._name = name
    # Dictionary of slots.
    #  {slot_name :
    #      {_var_key(variable_to_train): slot_for_the_variable, ... },
    #   ... }
    self._slots = {}
    self._non_slot_dict = {}
   
    self._deferred_slot_restorations = {}

  def _create_slots(self, var_list):
    for v in var_list:
      if v.get_shape().is_fully_defined():
        init_rms = init_ops.ones_initializer(dtype=v.dtype.base_dtype)

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

        # Scale loss if using a "mean" loss reduction and multiple towers.
        # Have to be careful to call distribute_lib.get_loss_reduction()
        # *after* loss() is evaluated, so we know what loss reduction it uses.
        # TODO(josh11b): Test that we handle weight decay in a reasonable way.
        if (distribute_lib.get_loss_reduction() ==
            variable_scope.VariableAggregation.MEAN):
          num_towers = distribution_strategy_context.get_distribution_strategy(
          ).num_towers
          if num_towers > 1:
            loss_value *= (1. / num_towers)

      if var_list is None:
        var_list = tape.watched_variables()
      # TODO(jhseu): Figure out why GradientTape's gradients don't require loss
      # to be executed.
      with ops.control_dependencies([loss_value]):
        grads = tape.gradient(loss_value, var_list, grad_loss)
      return list(zip(grads, var_list))

    # Non-callable/Tensor loss case
    if context.executing_eagerly():
      raise RuntimeError(
          "`loss` passed to Optimizer.compute_gradients should "
          "be a function when eager execution is enabled.")

    # Scale loss if using a "mean" loss reduction and multiple towers.
    if (distribute_lib.get_loss_reduction() ==
        variable_scope.VariableAggregation.MEAN):
      num_towers = distribution_strategy_context.get_distribution_strategy(
      ).num_towers
      if num_towers > 1:
        loss *= (1. / num_towers)

    if gate_gradients not in [Optimizer.GATE_NONE, Optimizer.GATE_OP,
                              Optimizer.GATE_GRAPH]:
      raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, "
                       "Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not %s" %
                       gate_gradients)
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
    # pylint: enable=protected-access
    processors = [_get_processor(v) for v in var_list]
    if not var_list:
      raise ValueError("No variables to optimize.")
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

  def _assert_valid_dtypes(self, tensors):
    valid_dtypes = self._valid_dtypes()
    for t in tensors:
      dtype = t.dtype.base_dtype
      if dtype not in valid_dtypes:
        raise ValueError(
            "Invalid type %r for %s, expected: %s." % (
                dtype, t.name, [v for v in valid_dtypes]))

  def _valid_dtypes(self):
    return set(
        [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64])

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    if distribution_strategy_context.get_cross_tower_context():
      raise RuntimeError("Use `_distributed_apply()` instead of "
                         "`apply_gradients()` in a cross-tower context.")
    # TODO(isaprykin): Get rid of `has_distribution_strategy()` check by
    # always calling _distributed_apply(), using the default distribution
    # as needed.
    if distribution_strategy_context.has_distribution_strategy():
      grads_and_vars = get_filtered_grad_fn(lambda: grads_and_vars)()
      return distribution_strategy_context.get_tower_context().merge_call(
          self._distributed_apply, grads_and_vars, global_step, name)

    # No DistributionStrategy case.
    grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
    if not grads_and_vars:
      raise ValueError("No variables provided.")
    converted_grads_and_vars = []
    for g, v in grads_and_vars:
      if g is not None:
        try:
          # Convert the grad to Tensor or IndexedSlices if necessary.
          g = ops.convert_to_tensor_or_indexed_slices(g)
        except TypeError:
          raise TypeError(
              "Gradient must be convertible to a Tensor"
              " or IndexedSlices, or None: %s" % g)
        if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
          raise TypeError(
              "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
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
      else:
        with ops.control_dependencies([self._finish(update_ops, "update")]):
          with ops.colocate_with(global_step):
            if isinstance(global_step, resource_variable_ops.ResourceVariable):
              # TODO(apassos): the implicit read in assign_add is slow; consider
              # making it less so.
              apply_updates = resource_variable_ops.assign_add_variable_op(
                  global_step.handle,
                  ops.convert_to_tensor(1, dtype=global_step.dtype),
                  name=name)
            else:
              apply_updates = state_ops.assign_add(global_step, 1, name=name)

      if not context.executing_eagerly():
        if isinstance(apply_updates, ops.Tensor):
          apply_updates = apply_updates.op
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        if apply_updates not in train_op:
          train_op.append(apply_updates)

      return apply_updates

  def _call_if_callable(self, param):
    """Call the function if param is callable."""
    return param() if callable(param) else param

  def _finish(self, update_ops, name_scope):
    return control_flow_ops.group(*update_ops, name=name_scope)
  
def _get_processor(v):
  if context.executing_eagerly():
    if isinstance(v, ops.Tensor):
      return _TensorProcessor(v)
    else:
      return _DenseResourceVariableProcessor(v)
  if isinstance(
      v, resource_variable_ops.ResourceVariable) and not v._in_graph_mode:  
    # True if and only if `v` was initialized eagerly.
    return _DenseResourceVariableProcessor(v)
  if v.op.type == "VarHandleOp":
    return _DenseResourceVariableProcessor(v)
  if isinstance(v, variables.Variable):
    return _RefVariableProcessor(v)
  if isinstance(v, ops.Tensor):
    return _TensorProcessor(v)
  raise NotImplementedError("Trying to optimize unsupported type ", v)

class _RefVariableProcessor(object):

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v._ref()  
