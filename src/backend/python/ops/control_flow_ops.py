from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import functools
import os

import six

from backend.python import context
from backend.python import execute
from backend.core import control_flow_pb2
from backend.python.framework import constant_op
from backend.python.framework import sparse_tensor
from backend.python.framework import ops
from backend.python.framework import op_def_library as _op_def_library
from backend.python.framework import op_def_registry as _op_def_registry
from backend.python.ops import array_ops
from backend.python.ops import math_ops
from backend.python.ops import tensor_array_ops
from backend.python.ops import control_flow_util as util
from backend.util import nest

from backend.core import op_def_pb2 as _op_def_pb2



ENABLE_COND_V2 = os.getenv("TF_ENABLE_COND_V2", "0") != "0"
ENABLE_WHILE_V2 = os.getenv("TF_ENABLE_WHILE_V2", "0") != "0"

_basetuple = tuple

def exit(data, name=None):  
  data = ops.internal_convert_to_tensor_or_indexed_slices(data, as_ref=True)
  if isinstance(data, ops.Tensor):
      return _exit(data, name)

def _exit(data, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Exit", data=data, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    execute.record_gradient(
      "Exit", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result


def cond(pred,
         true_fn=None,
         false_fn=None,
         strict=False,
         name=None,
         fn1=None,
         fn2=None):
  if ENABLE_COND_V2 and not context.executing_eagerly():
    return cond_v2(pred, true_fn, false_fn, name)

  with ops.name_scope(name, "cond", [pred]):
    p_2, p_1 = switch(pred, pred)
    pivot_1 = array_ops.identity(p_1, name="switch_t")
    pivot_2 = array_ops.identity(p_2, name="switch_f")
    pred = array_ops.identity(pred, name="pred_id")
    for tensor in [p_1, p_2, pivot_1, pivot_2, pred]:
      tensor.op.graph.prevent_fetching(tensor.op)

    # Build the graph for the true branch in a new context.
    context_t = CondContext(pred, pivot_1, branch=1)
    try:
      context_t.Enter()
      orig_res_t, res_t = context_t.BuildCondBranch(true_fn)
      if orig_res_t is None:
        raise ValueError("true_fn must have a return value.")
      context_t.ExitResult(res_t)
    finally:
      context_t.Exit()

    # Build the graph for the false branch in a new context.
    context_f = CondContext(pred, pivot_2, branch=0)
    try:
      context_f.Enter()
      orig_res_f, res_f = context_f.BuildCondBranch(false_fn)
      context_f.ExitResult(res_f)
    finally:
      context_f.Exit()

    if not strict:
      orig_res_t = _UnpackIfSingleton(orig_res_t)
      orig_res_f = _UnpackIfSingleton(orig_res_f)

    res_t_flat = nest.flatten(res_t)
    res_f_flat = nest.flatten(res_f)

    for x, y in zip(res_t_flat, res_f_flat):
      assert ((isinstance(x, ops.IndexedSlices) and
               isinstance(y, ops.IndexedSlices)) or
              (isinstance(x, sparse_tensor.SparseTensor) and
               isinstance(y, sparse_tensor.SparseTensor)) or
              (isinstance(x, ops.Tensor) and isinstance(y, ops.Tensor)))
      val_x = x if isinstance(x, ops.Tensor) else x.values
      val_y = y if isinstance(y, ops.Tensor) else y.values
      if val_x.dtype.base_dtype != val_y.dtype.base_dtype:
        raise ValueError(
            "Outputs of true_fn and false_fn must have the same type: %s, %s" %
            (val_x.dtype.name, val_y.dtype.name))

    merges = [merge(pair)[0] for pair in zip(res_f_flat, res_t_flat)]
    merges = _convert_flows_to_tensorarrays(nest.flatten(orig_res_t), merges)

    # Only add non-nested conds to the collection. Any nested control flow will
    # be encapsulated in the root context.
    assert context_t.outer_context == context_f.outer_context
    if context_t.outer_context is None:
      ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_t)
      ops.add_to_collection(ops.GraphKeys.COND_CONTEXT, context_f)

    merges = nest.pack_sequence_as(structure=orig_res_t, flat_sequence=merges)

    # Singleton lists and tuples are automatically unpacked if strict == False.
    if not strict:
      merges = _UnpackIfSingleton(merges)
    return merges

class ControlFlowContext(object):
  def __init__(self, values_def=None, import_scope=None):
    self._nested_contexts = []
    self._outer_context = ops.get_default_graph()._get_control_flow_context()
    if self._outer_context:
      self._outer_context._nested_contexts.append(self)  
    self._context_stack = []
    if values_def:
      self._init_values_from_proto(values_def, import_scope=import_scope)
    else:
      # The names of tensors that have been already seen in this context.
      self._values = set()
      # The keys are the names of tensors referenced by but external to this
      # context. Each value is the Tensor that should be used by this context to
      # access the key value (e.g. a switch output guarding a cond input value).
      self._external_values = {}

  def _init_values_from_proto(self, values_def, import_scope=None):
    assert isinstance(values_def, control_flow_pb2.ValuesDef)
    self._values = set(
        ops.prepend_name_scope(value, import_scope)
        for value in values_def.values)
    g = ops.get_default_graph()
    self._external_values = {}
    for k, v in values_def.external_values.items():
      k = ops.prepend_name_scope(k, import_scope)
      self._external_values[k] = g.as_graph_element(
          ops.prepend_name_scope(v, import_scope))
    op_names = set([
        op.split(":")[0]
        for op in self._values - set(self._external_values.keys())
    ])
    for op in op_names:
      
      g.as_graph_element(op)._set_control_flow_context(self)
      

  @property
  def name(self):
    return self._name

  @property
  def outer_context(self):
    return self._outer_context

  @property
  def grad_state(self):
    raise NotImplementedError("Abstract method")

  @property
  def back_prop(self):
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def to_control_flow_context_def(self, context_def, export_scope=None):
    raise NotImplementedError("Abstract method")

  def _to_values_def(self, export_scope=None):
    values_def = control_flow_pb2.ValuesDef()
    values_def.values.extend(
        [ops.strip_name_scope(v, export_scope) for v in sorted(self._values)])
    for k, v in self._external_values.items():
      k = ops.strip_name_scope(k, export_scope)
      values_def.external_values[k] = ops.strip_name_scope(v.name, export_scope)
    return values_def

  def AddName(self, name):
    self._values.add(name)

  
  def Enter(self):
    graph = ops.get_default_graph()
    self._context_stack.append(graph._get_control_flow_context())
    graph._set_control_flow_context(self)

  def Exit(self):
    graph = ops.get_default_graph()
    last_context = self._context_stack.pop()
    graph._set_control_flow_context(last_context)

  def EnterGradientColocation(self, op, gradient_uid):
    if self._outer_context:
      self._outer_context.EnterGradientColocation(op, gradient_uid)

  def ExitGradientColocation(self, op, gradient_uid):
    if self._outer_context:
      self._outer_context.ExitGradientColocation(op, gradient_uid)

  def ExitResult(self, result):
    if self._outer_context:
      nest.map_structure(lambda x: self._outer_context.AddName(x.name), result)

  def GetWhileContext(self):
    if self._outer_context:
      return self._outer_context.GetWhileContext()
    return None

  def _IsInOuterContext(self, op):
    op_ctxt = util.GetOutputContext(op)
    outer_ctxt = self.outer_context
    while outer_ctxt != op_ctxt:
      if outer_ctxt is None:
        return False
      outer_ctxt = outer_ctxt.outer_context
    return True

  def _RemoveExternalControlEdges(self, op):
    while_ctxt = self.GetWhileContext()
    # A control input of `op` is internal if it is in the same while
    # loop context as the enclosing while loop context of self.
    if while_ctxt is None:
      internal_control_inputs = op.control_inputs
    else:
      internal_control_inputs = []
      for x in op.control_inputs:
        ctxt = util.GetOutputContext(x)
        if ctxt is not None and ctxt.GetWhileContext() == while_ctxt:
          internal_control_inputs.append(x)
    external_control_inputs = []
    if len(internal_control_inputs) != len(op.control_inputs):
      external_control_inputs = list(set(op.control_inputs)
                                     - set(internal_control_inputs))
      op._remove_all_control_inputs()
      op._add_control_inputs(internal_control_inputs)
    return internal_control_inputs, external_control_inputs


  def AddInnerOp(self, op):
    if self._outer_context:
      self._outer_context.AddInnerOp(op)

  def GetControlPivot(self):
    return None

  def IsWhileContext(self):
    return False

  def IsCondContext(self):
    return False

  def IsXLAContext(self):
    return False

  def __str__(self):
    return self.name

class CondContext(ControlFlowContext):
  def __init__(self,
               pred=None,
               pivot=None,
               branch=None,
               name="cond_text",
               context_def=None,
               import_scope=None):
    self._name = ops.get_default_graph().unique_name(name)

    if context_def:
      self._init_from_proto(context_def, import_scope=import_scope)
    else:
      # Initializes the default fields.
      ControlFlowContext.__init__(self)
      self._pred = pred  # The boolean tensor for the cond predicate
      self._pivot = pivot  # The predicate tensor in this branch
      self._branch = branch  # 0 or 1 representing this branch

      # Values considered to have been already seen in this context. pred is not
      # included in this context.
      self._values.add(pred.name)
      self._external_values[pred.name] = pred
      self._values.add(pivot.name)
      pivot.op._set_control_flow_context(self)  

  def _init_from_proto(self, context_def, import_scope=None):
    assert isinstance(context_def, control_flow_pb2.CondContextDef)
    # Create from context_def.
    g = ops.get_default_graph()
    self._name = ops.prepend_name_scope(context_def.context_name, import_scope)
    self._pred = g.as_graph_element(
        ops.prepend_name_scope(context_def.pred_name, import_scope))
    self._pivot = g.as_graph_element(
        ops.prepend_name_scope(context_def.pivot_name, import_scope))
    self._branch = context_def.branch
    super(CondContext, self).__init__(values_def=context_def.values_def,
                                      import_scope=import_scope)

  @property
  def pred(self):
    return self._pred

  @property
  def pivot(self):
    return self._pivot

  @property
  def branch(self):
    return self._branch

  @property
  def grad_state(self):
    if self.GetWhileContext():
      return self.GetWhileContext().grad_state
    return None

  @property
  def back_prop(self):
    if self.GetWhileContext():
      self.GetWhileContext().back_prop
    return False

  def GetControlPivot(self):
    return self._pivot

  def to_proto(self, export_scope=None):
    if (export_scope is None or self.name.startswith(export_scope)):
      context_def = control_flow_pb2.CondContextDef()
      context_def.context_name = ops.strip_name_scope(self.name, export_scope)
      context_def.pred_name = ops.strip_name_scope(self._pred.name,
                                                   export_scope)
      context_def.pivot_name = ops.strip_name_scope(self._pivot.name,
                                                    export_scope)
      context_def.branch = self._branch
      context_def.values_def.MergeFrom(super(CondContext, self)._to_values_def(
          export_scope))
      for nested in self._nested_contexts:
        nested_def = context_def.nested_contexts.add()
        nested.to_control_flow_context_def(nested_def)

      return context_def
    else:
      return None

  @staticmethod
  def from_proto(context_def, import_scope=None):
    ret = CondContext(context_def=context_def,
                      import_scope=import_scope)

    ret.Enter()
    for nested_def in context_def.nested_contexts:
      from_control_flow_context_def(nested_def, import_scope=import_scope)
    ret.Exit()
    return ret

  def to_control_flow_context_def(self, context_def, export_scope=None):
    context_def.cond_ctxt.CopyFrom(self.to_proto(export_scope=export_scope))

  def AddValue(self, val):
    if val.name in self._values:
      # Use the real value if it comes from outer context. This is needed in
      # particular for nested conds.
      result = self._external_values.get(val.name)
      result = val if result is None else result
    else:
      result = val
      self._values.add(val.name)
      if self._outer_context:
        result = self._outer_context.AddValue(val)
        self._values.add(result.name)
        self._external_values[result.name] = result
      with ops.control_dependencies(None):
        result = _SwitchRefOrTensor(result, self._pred)[self._branch]
        if self._outer_context:
          self._outer_context.AddInnerOp(result.op)

      result.op.graph.prevent_fetching(result.op)
      
      result.op._set_control_flow_context(self)
      

      self._values.add(result.name)
      self._external_values[val.name] = result
    return result

  def AddOp(self, op):
    self._AddOpInternal(op)

  def _AddOpInternal(self, op):
    if not op.inputs:
      # If we're in a while loop, remove any control inputs from outside the
      # loop.
      self._RemoveExternalControlEdges(op)

      if not any(util.OpInContext(input_op, self)
                 for input_op in op.control_inputs):
        
        op._add_control_input(self._pivot.op)
        
    else:
      # Make each input to 'op' available in this CondContext. If an input is
      # already part of this context there's nothing to do, but if it's
      # external, AddValue() will handle adding the appropriate Switch node and
      # other bookkeeping.
      for index in range(len(op.inputs)):
        x = op.inputs[index]
        if op.type == "Merge" and x.op.type == "NextIteration":
          # Edge case: if we're importing a while loop inside this CondContext,
          # AddValue() will not correctly handle the NextIteration inputs to
          # Merge node. The problem is that the NextIteration should also be
          # part of this context, but if we're importing it won't have been
          # processed and added to the context yet, so AddValue() will try to
          # add a Switch which results in an invalid graph. Instead, we use the
          # NextIteration input as-is here, and it will eventually be added to
          # the context via AddOp().
          real_x = x
        else:
          real_x = self.AddValue(x)
        if real_x != x:
          
          op._update_input(index, real_x)
          
      # Remove any external control dependency on this op.
      self._RemoveExternalControlEdges(op)
      
      if op.graph._is_function(op.type) or op.type == "SymbolicGradient":
        op._add_control_input(self._pivot.op)
      

    # Mark op's outputs as seen by this context and any outer contexts.
    output_names = [x.name for x in op.outputs]
    ctxt = self
    while ctxt is not None:
      
      ctxt._values.update(output_names)
      ctxt = ctxt._outer_context
      

    if self._outer_context or not util.IsLoopExit(op):
      op.graph.prevent_fetching(op)

    if self._outer_context:
      self._outer_context.AddInnerOp(op)

  def _ProcessOutputTensor(self, val):
    real_val = val
    if val.name not in self._values:
      # Handle the special case of lambda: x
      self._values.add(val.name)
      if self._outer_context:
        real_val = self._outer_context.AddValue(val)
        self._values.add(real_val.name)
        self._external_values[real_val.name] = real_val
      real_val = _SwitchRefOrTensor(real_val, self._pred)[self._branch]
      self._external_values[val.name] = real_val
    else:
      external_val = self._external_values.get(val.name)
      if external_val is not None:
        real_val = external_val
    return real_val

  def _BuildCondTensor(self, v):
    if isinstance(v, ops.Operation):
      # Use pivot as the proxy for this op.
      return with_dependencies([v], self._pivot)
    elif isinstance(v, (ops.IndexedSlices, sparse_tensor.SparseTensor)):
      values = self._ProcessOutputTensor(v.values)
      indices = self._ProcessOutputTensor(v.indices)
      if isinstance(v, ops.IndexedSlices):
        dense_shape = v.dense_shape
        if dense_shape is not None:
          dense_shape = self._ProcessOutputTensor(dense_shape)
        return ops.IndexedSlices(values, indices, dense_shape)
      else:
        dense_shape = self._ProcessOutputTensor(v.dense_shape)
        return sparse_tensor.SparseTensor(indices, values, dense_shape)
    else:
      v = nest.map_structure(_convert_tensorarray_to_flow, v)
      return self._ProcessOutputTensor(ops.convert_to_tensor(v))

  def BuildCondBranch(self, fn):
    pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  
    original_result = fn()
    post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  
    if len(post_summaries) > len(pre_summaries):
      new_summaries = post_summaries[len(pre_summaries):]
      summary_ref = ops.get_collection_ref(ops.GraphKeys._SUMMARY_COLLECTION)  
      summary_ref[:] = pre_summaries
      with ops.control_dependencies(new_summaries):
        if original_result is None:
          return no_op(), None
        else:
          original_result = nest.map_structure(array_ops.identity,
                                               original_result)
    if original_result is None:
      return None, None

    result = nest.map_structure(self._BuildCondTensor, original_result)
    if not isinstance(result, (list, _basetuple)):
      result = [result]
    return original_result, result

  def IsCondContext(self):
    return True

def _SwitchRefOrTensor(data, pred, name="Switch"):
  data = ops.convert_to_tensor_or_indexed_slices(data, name="data")
  # NOTE(vrv): ops.colocate_with(data, ignore_existing=True) below
  # addresses the following scenario.
  #
  # Assume you execute Optimizer.apply_gradients() in a branch of a cond().
  #
  # 1. The update op is created inside a `with ops.colocate(var):` block
  #
  # 2. Some tensor `data` is captured and a switch is created in a
  #    `with ops.colocate_with(data):` block.
  #
  # with ops.colocate_with(var):
  #  with ops.colocate_with(data):
  #    op = ...
  #
  # var and data may be pinned to different devices, so we want to ops
  # created within ops.colocate_with(data) to ignore the existing stack.
  with ops.colocate_with(data, ignore_existing=True):
    if isinstance(data, ops.Tensor):
      if data.dtype._is_ref_dtype:  
        return ref_switch(data, pred, name=name)
    return switch(data, pred, name=name)

def _convert_tensorarray_to_flow(tensor_or_tensor_array):
  if isinstance(tensor_or_tensor_array, tensor_array_ops.TensorArray):
    return tensor_or_tensor_array.flow
  else:
    return tensor_or_tensor_array

def _UnpackIfSingleton(res):
  if isinstance(res, (list, _basetuple)) and len(res) == 1:
    return res[0]
  else:
    return res

def _convert_flows_to_tensorarrays(tensors_or_tensorarrays, tensors_or_flows):
  if len(tensors_or_tensorarrays) != len(tensors_or_flows):
    raise ValueError(
        "Lengths of original Tensor list and new list do not match: %d vs. %d" %
        (len(tensors_or_tensorarrays), len(tensors_or_flows)))
  return [
      _make_tensor_array(ta, t_or_flow)
      if isinstance(ta, tensor_array_ops.TensorArray) else t_or_flow
      for (ta, t_or_flow) in zip(tensors_or_tensorarrays, tensors_or_flows)
  ]

def while_loop(cond,
               body,
               loop_vars,
               shape_invariants=None,
               parallel_iterations=10,
               back_prop=True,
               swap_memory=False,
               name=None,
               maximum_iterations=None,
               return_same_structure=False):
  if ENABLE_WHILE_V2 and not context.executing_eagerly():
    if not _while_v2:
      raise ValueError("The while_v2 module is not set. Did you forget to "
                       "import tensorflow.python.ops."
                       "while_v2?")
    return _while_v2.while_loop(cond, body, loop_vars, name)

  with ops.name_scope(name, "while", loop_vars):
    if not loop_vars:
      raise ValueError("No loop variables provided")
    if not callable(cond):
      raise TypeError("cond must be callable.")
    if not callable(body):
      raise TypeError("body must be callable.")
    if parallel_iterations < 1:
      raise TypeError("parallel_iterations must be a positive integer.")

    if maximum_iterations is not None:
      maximum_iterations = ops.convert_to_tensor(
          maximum_iterations, name="maximum_iterations")
      if maximum_iterations.shape.ndims != 0:
        raise ValueError("maximum_iterations must be a scalar, saw shape: %s" %
                         maximum_iterations.shape)

      counter = constant_op.constant(
          0, dtype=maximum_iterations.dtype, name="iteration_counter")
      orig_cond = cond
      orig_body = body
      if len(loop_vars) == 1:
        loop_vars = (counter, loop_vars[0])
        cond = lambda i, lv: (  # pylint: disable=g-long-lambda
            math_ops.logical_and(i < maximum_iterations, orig_cond(lv)))
        body = lambda i, lv: (i + 1, orig_body(lv))
      else:
        loop_vars = (counter, loop_vars)
        cond = lambda i, lv: (  # pylint: disable=g-long-lambda
            math_ops.logical_and(i < maximum_iterations, orig_cond(*lv)))
        body = lambda i, lv: (i + 1, orig_body(*lv))

    if context.executing_eagerly():
      try_to_pack = len(loop_vars) == 1
      packed = False  # whether the body result was packed into a 1-item tuple

      while cond(*loop_vars):
        loop_vars = body(*loop_vars)
        if try_to_pack and not isinstance(loop_vars, (list, _basetuple)):
          packed = True
          loop_vars = (loop_vars,)
      if maximum_iterations is not None:
        return loop_vars[1]
      else:
        return loop_vars[0] if packed else loop_vars

    if shape_invariants is not None:
      if maximum_iterations is not None:
        shape_invariants = (tensor_shape.TensorShape([]), shape_invariants)
      nest.assert_same_structure(loop_vars, shape_invariants)

    loop_context = WhileContext(
        maximum_iterations=maximum_iterations,
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory)
    # Only add non-nested loops to the collection. Any nested control flow will
    # be encapsulated in the root context.
    if loop_context.outer_context is None:
      ops.add_to_collection(ops.GraphKeys.WHILE_CONTEXT, loop_context)
    result = loop_context.BuildLoop(cond, body, loop_vars, shape_invariants,
                                    return_same_structure)
    if maximum_iterations is not None:
      return result[1]
    else:
      return result

class WhileContext(ControlFlowContext):
  def __init__(self,
               maximum_iterations=None,
               parallel_iterations=10,
               back_prop=True,
               swap_memory=False,
               name="while_context",
               grad_state=None,
               context_def=None,
               import_scope=None):
    if context_def:
      self._init_from_proto(context_def, import_scope=import_scope)
    else:
      ControlFlowContext.__init__(self)
      self._init_from_args(maximum_iterations, parallel_iterations, back_prop,
                           swap_memory, name)
    # The gradient loop state.
    self._grad_state = grad_state

  def _init_from_args(self, maximum_iterations, parallel_iterations, back_prop,
                      swap_memory, name):
    if not isinstance(parallel_iterations, int) or (parallel_iterations <= 0):
      raise ValueError("`parallel_iterations` must be a positive integer: "
                       "%s" % parallel_iterations)
    self._name = ops.get_default_graph().unique_name(name)
    self._maximum_iterations = maximum_iterations
    self._parallel_iterations = parallel_iterations
    self._back_prop = back_prop
    self._swap_memory = swap_memory
    # We use this node to control constants created by the pred lambda.
    self._pivot_for_pred = None
    # We use this node to control constants created by the body lambda.
    self._pivot_for_body = None
    # The boolean tensor for loop termination condition. Used in code
    # generation for gradient computation
    self._pivot = None
    # The list of exit tensors for loop variables.
    self._loop_exits = []
    # The list of enter tensors for loop variables.
    self._loop_enters = []
    self._graph = ops.get_default_graph()

  def _init_from_proto(self, context_def, import_scope=None):
    assert isinstance(context_def, control_flow_pb2.WhileContextDef)
    # Create from context_def.
    g = ops.get_default_graph()
    self._name = ops.prepend_name_scope(context_def.context_name, import_scope)
    if context_def.maximum_iterations_name:
      self._maximum_iterations = g.as_graph_element(
          ops.prepend_name_scope(context_def.maximum_iterations_name,
                                 import_scope))
    else:
      self._maximum_iterations = None
    self._parallel_iterations = context_def.parallel_iterations
    self._back_prop = context_def.back_prop
    self._swap_memory = context_def.swap_memory
    self._pivot_for_pred = g.as_graph_element(
        ops.prepend_name_scope(context_def.pivot_for_pred_name, import_scope))
    # We use this node to control constants created by the body lambda.
    self._pivot_for_body = g.as_graph_element(
        ops.prepend_name_scope(context_def.pivot_for_body_name, import_scope))
    # The boolean tensor for loop termination condition. Used in code
    # generation for gradient computation.
    self._pivot = g.as_graph_element(
        ops.prepend_name_scope(context_def.pivot_name, import_scope))
    # The list of exit tensors for loop variables.
    self._loop_exits = [
        g.as_graph_element(ops.prepend_name_scope(exit_name, import_scope))
        for exit_name in context_def.loop_exit_names
    ]
    # The list of enter tensors for loop variables.
    self._loop_enters = [
        g.as_graph_element(ops.prepend_name_scope(enter_name, import_scope))
        for enter_name in context_def.loop_enter_names
    ]
    super(WhileContext, self).__init__(
        values_def=context_def.values_def, import_scope=import_scope)

    # import_scope causes self.name to be different from the original serialized
    # context's name. Rewrite "frame_name" attrs with the new name.
    if import_scope:
      for tensor_name in self._values:
        op = g.as_graph_element(tensor_name).op
        if util.IsLoopEnter(op):
          
          op._set_attr("frame_name",
                       attr_value_pb2.AttrValue(s=compat.as_bytes(self.name)))
          
    self._graph = ops.get_default_graph()

  @property
  def maximum_iterations(self):
    return self._maximum_iterations

  @property
  def parallel_iterations(self):
    return self._parallel_iterations

  @property
  def back_prop(self):
    return self._back_prop

  @property
  def swap_memory(self):
    return self._swap_memory

  @property
  def pivot(self):
    return self._pivot

  @property
  def loop_enters(self):
    return self._loop_enters

  @property
  def loop_exits(self):
    return self._loop_exits

  @property
  def grad_state(self):
    return self._grad_state

  def GetWhileContext(self):
    return self

  def GetControlPivot(self):
    if self._pivot_for_body is not None:
      return self._pivot_for_body
    return self._pivot_for_pred

  def AddValue(self, val):
    result = val
    new_value = val.name not in self._values
    # Don't treat ops in this context as new values. Usually all known values
    # are in self._values, except when we're importing a while loop inside this
    # WhileContext. Since there's a cycle in this case, `val` may be part of the
    # imported while loop but not yet processed by this context and added to
    # self._values in _AddOpInternal. We only want to process external input
    # tensors to the while loop here.
    new_value &= val.op._control_flow_context is not self  
    if new_value:
      self._values.add(val.name)

      # If we are in a grad context and val is from its forward context,
      # use GetRealValue(), which adds the logic to save the history of
      # val in forward.
      grad_ctxt = ops.get_default_graph()._get_control_flow_context()

      if self._outer_context is not None:
        result = self._outer_context.AddValue(val)
      # Create an Enter to make `result` known to this loop context.
      with ops.control_dependencies(None):
        enter = _Enter(
            result,
            self._name,
            is_constant=True,
            parallel_iterations=self._parallel_iterations)
        enter.graph.prevent_feeding(enter)
        if self._outer_context:
          self._outer_context.AddInnerOp(enter.op)
      # Fix the control inputs and control flow context of these enter ops.
      self._FixControlInputsAndContext([enter])

      # Add `enter` in this context.
      self._values.add(enter.name)
      self._external_values[val.name] = enter
      result = enter
    else:
      actual_val = self._external_values.get(val.name)
      if actual_val is not None:
        result = actual_val
    return result

  def AddOp(self, op):
    if op.type in {"Shape", "Size", "Rank"}:
      grad_ctxt = ops.get_default_graph()._get_control_flow_context()
      if grad_ctxt:
        grad_ctxt = grad_ctxt.GetWhileContext()
        if grad_ctxt.grad_state:
          op_input_forward_ctxt = _GetWhileContext(op.inputs[0].op)
          if op_input_forward_ctxt == grad_ctxt.grad_state.forward_context:
            op_input_ctxt = op.inputs[0].op._get_control_flow_context()
            op._set_control_flow_context(op_input_ctxt)
            op_input_ctxt._AddOpInternal(op)
            return
    self._AddOpInternal(op)

  def _AddOpInternal(self, op):
    if not op.inputs:
      # Remove any external control dependency on this op
      control_inputs, external_inputs = self._RemoveExternalControlEdges(op)
      # Add a control edge from the control pivot to this op.
      if not control_inputs:
        
        op._add_control_input(self.GetControlPivot().op)
        
      for x in op.outputs:
        self._values.add(x.name)
    else:
      for index in range(len(op.inputs)):
        x = op.inputs[index]
        real_x = self.AddValue(x)
        if real_x != x:
          op._update_input(index, real_x)  
      # Remove any external control dependency on this op.
      _, external_inputs = self._RemoveExternalControlEdges(op)
      # Add a control dependency to prevent loop invariants from
      # enabling ops that should not be executed.
      self._MaybeAddControlDependency(op)
      for x in op.outputs:
        self._values.add(x.name)
    if external_inputs:
      # Use an identity to pull control inputs as data inputs. Note that we
      # ignore ops which don't have outputs. TODO(apassos): fix that
      with ops.control_dependencies(None):
        self.Enter()
        external_inputs = [array_ops.identity(x.outputs[0]).op
                           for x in external_inputs if x.outputs]
        self.Exit()
      op._add_control_inputs(external_inputs)  
    if self._outer_context or not util.IsLoopExit(op):
      op.graph.prevent_fetching(op)
      for x in op.outputs:
        op.graph.prevent_feeding(x)

    if self._outer_context:
      self._outer_context.AddInnerOp(op)

  def _MaybeAddControlDependency(self, op):
    def _IsOpFree(op):
      if op.control_inputs:
        return False
      
      if op.graph._is_function(op.type) or op.type == "SymbolicGradient":
        return True
      
      for x in op.inputs:
        if not util.IsLoopConstantEnter(x.op):
          return False
      return True
    if _IsOpFree(op):
      
      op._add_control_input(self.GetControlPivot().op)
      

  def AddForwardLoopCounter(self, outer_grad_state):
    n = constant_op.constant(0, name="f_count")
    if outer_grad_state is not None:
      # Force the stack pushes of i-th execution of an inner loop to be ordered
      # before the pushes of (i+1)-th execution of the same inner loop.
      outer_add_op = outer_grad_state.forward_index.op.inputs[0].op
      n.op._add_control_input(outer_add_op)  

    self.Enter()
    self.AddName(n.name)
    enter_n = _Enter(
        n,
        self._name,
        is_constant=False,
        parallel_iterations=self._parallel_iterations,
        name="f_count")
    self.loop_enters.append(enter_n)

    merge_n = merge([enter_n, enter_n])[0]
    switch_n = switch(merge_n, self._pivot)

    index = math_ops.add(switch_n[1], 1)
    next_n = _NextIteration(index)
    merge_n.op._update_input(1, next_n)

    total_iterations = exit(switch_n[0], name="f_count")
    self.loop_exits.append(total_iterations)
    self.ExitResult([total_iterations])
    self.Exit()
    return total_iterations, next_n

  def AddBackpropLoopCounter(self, count, outer_grad_state):
    in_separate_functions = count.graph is not ops.get_default_graph()
    if in_separate_functions:
      # Brings the count into this graph
      count = array_ops.identity(count)
    else:
      # TODO(apassos) XLA expects this constant to be created outside the loop,
      # so doing that for now.
      one = constant_op.constant(1, name="b_count")

    self.Enter()
    self.AddName(count.name)
    enter_count = _Enter(
        count,
        self._name,
        is_constant=False,
        parallel_iterations=self._parallel_iterations,
        name="b_count")
    self.loop_enters.append(enter_count)

    merge_count = merge([enter_count, enter_count])[0]
    self._pivot_for_pred = merge_count

    if in_separate_functions:
      one = constant_op.constant(1, name="b_count")
    pred = math_ops.greater_equal(merge_count, one)
    self._pivot = loop_cond(pred, name="b_count")
    switch_count = switch(merge_count, self._pivot)

    index = math_ops.sub(switch_count[1], one)
    self._pivot_for_body = index
    next_count = _NextIteration(index)
    merge_count.op._update_input(1, next_count)

    final_zero = exit(switch_count[0], name="b_count")
    self.loop_exits.append(final_zero)
    if outer_grad_state is not None:
      # Force the stack pops of i-th execution of an inner loop to be ordered
      # before the pops of (i+1)-th execution of the same inner loop.
      
      outer_grad_state.grad_sync._add_control_input(final_zero.op)
      

    self.ExitResult([final_zero])
    self.Exit()
    return next_count

  def _InitializeValues(self, values):
    self._values = set()
    for x in values:
      if isinstance(x, ops.Tensor):
        self._values.add(x.name)
      else:
        self._values.add(x.values.name)
        self._values.add(x.indices.name)
        if isinstance(x, ops.IndexedSlices):
          dense_shape = x.dense_shape
        elif isinstance(x, sparse_tensor.SparseTensor):
          dense_shape = x.dense_shape
        else:
          raise TypeError("Type %s not supported" % type(x))
        if dense_shape is not None:
          self._values.add(dense_shape.name)

  def _BuildLoop(self, pred, body, original_loop_vars, loop_vars,
                 shape_invariants):
    flat_loop_vars = nest.flatten(original_loop_vars)

    # Let the context know the loop variables so the loop variables
    # would be added in the outer contexts properly.
    self._InitializeValues(loop_vars)
    real_vars = loop_vars
    if self._outer_context:
      real_vars = [self._outer_context.AddValue(x) for x in loop_vars]
    with ops.control_dependencies(None):
      enter_vars = [
          _Enter(
              x,
              self._name,
              is_constant=False,
              parallel_iterations=self._parallel_iterations,
              use_input_shape=(shape_invariants is None)) for x in real_vars
      ]
      for x in enter_vars:
        x.graph.prevent_feeding(x)
        if self._outer_context:
          self._outer_context.AddInnerOp(x.op)

    # Finds the closest enclosing non-None control pivot.
    outer_context = self._outer_context
    control_pivot = None
    while outer_context is not None and control_pivot is None:
      control_pivot = outer_context.GetControlPivot()
      
      outer_context = outer_context._outer_context
      

    if control_pivot is not None:
      for var in enter_vars:
        if util.IsLoopConstantEnter(var.op.inputs[0].op):
          
          var.op._add_control_input(control_pivot.op)
          
    _SetShapeInvariants(real_vars, enter_vars, shape_invariants)

    # Fix the control inputs and control flow context of these enter ops.
    self._FixControlInputsAndContext(enter_vars)
    self._InitializeValues(enter_vars)
    self._loop_enters = enter_vars

    merge_vars = [merge([x, x])[0] for x in enter_vars]
    self._pivot_for_pred = merge_vars[0]

    # Build the graph for pred.
    merge_vars_with_tensor_arrays = (
        _convert_flows_to_tensorarrays(flat_loop_vars, merge_vars))
    packed_vars = nest.pack_sequence_as(
        structure=original_loop_vars,
        flat_sequence=merge_vars_with_tensor_arrays)
    c = ops.convert_to_tensor(pred(*packed_vars))
    self._pivot = loop_cond(c, name="LoopCond")
    switch_vars = [_SwitchRefOrTensor(x, self._pivot) for x in merge_vars]

    # Build the graph for body.
    vars_for_body = [_Identity(x[1]) for x in switch_vars]
    self._pivot_for_body = vars_for_body[0]
    # Convert TensorArray flow variables inside the context back into
    # their associated TensorArrays for calling the body.
    vars_for_body_with_tensor_arrays = (
        _convert_flows_to_tensorarrays(flat_loop_vars, vars_for_body))
    packed_vars_for_body = nest.pack_sequence_as(
        structure=original_loop_vars,
        flat_sequence=vars_for_body_with_tensor_arrays)
    pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  
    body_result = body(*packed_vars_for_body)
    post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  
    if not nest.is_sequence(body_result):
      body_result = [body_result]
    if len(post_summaries) > len(pre_summaries):
      new_summaries = post_summaries[len(pre_summaries):]
      summary_ref = ops.get_collection_ref(ops.GraphKeys._SUMMARY_COLLECTION)  
      summary_ref[:] = pre_summaries
      with ops.control_dependencies(new_summaries):

        def map_fn(x):
          # TODO(apassos) figure out how to trigger with tensor arrays as well
          if isinstance(x, tensor_array_ops.TensorArray):
            return x
          return array_ops.identity(x)

        body_result = nest.map_structure(map_fn, body_result)

    # Compare the structure types of input and output of body.
    # For backwards compatibility, the first layer is forced to a list
    # during this comparison, because inputs are typically lists and
    # outputs of the body are typically tuples.
    nest.assert_same_structure(list(packed_vars_for_body), list(body_result))

    # Store body_result to keep track of TensorArrays returned by body
    original_body_result = body_result
    # Convert TensorArrays returned by body into their flow variables
    result = nest.map_structure(_convert_tensorarray_to_flow,
                                nest.flatten(body_result))
    result = ops.convert_n_to_tensor_or_indexed_slices(result)

    # Add NextIteration and the back edges to complete the loop.
    if len(merge_vars) != len(result):
      raise ValueError("Number of inputs and outputs of body must match "
                       "loop_vars: %d, %d" % (len(merge_vars), len(result)))
    next_vars = []
    for m, v in zip(merge_vars, result):
      next_vars.append(_AddNextAndBackEdge(m, v))

    # Add the exit ops.
    exit_vars = [exit(x[0]) for x in switch_vars]
    self._loop_exits = exit_vars

    # Exit the loop.
    self.ExitResult(exit_vars)

    return original_body_result, exit_vars

  def BuildLoop(self, pred, body, loop_vars, shape_invariants,
                return_same_structure):

    # Keep original_loop_vars to identify which are TensorArrays
    original_loop_vars = loop_vars
    # Convert TensorArrays to their flow variables
    loop_vars = nest.map_structure(_convert_tensorarray_to_flow,
                                   nest.flatten(loop_vars))
    loop_vars = ops.convert_n_to_tensor_or_indexed_slices(loop_vars)
    try:
      self.Enter()
      # _BuildLoop calls _update_input in several places. _mutation_lock()
      # ensures a Session.run call cannot occur between creating and mutating
      # new ops.
      with ops.get_default_graph()._mutation_lock():  
        original_body_result, exit_vars = self._BuildLoop(
            pred, body, original_loop_vars, loop_vars, shape_invariants)
    finally:
      self.Exit()

    flat_result = nest.flatten(original_body_result)
    # Convert TensorArray flow variables outside the context back into
    # their associated TensorArrays for returning to caller.
    exit_vars_with_tensor_arrays = (
        _convert_flows_to_tensorarrays(flat_result, exit_vars))
    packed_exit_vars = nest.pack_sequence_as(
        structure=original_body_result,
        flat_sequence=exit_vars_with_tensor_arrays)

    if return_same_structure:
      return packed_exit_vars
    else:
      return packed_exit_vars[0] if len(exit_vars) == 1 else packed_exit_vars

  def _FixControlInputsAndContext(self, enters):
    graph = ops.get_default_graph()
    
    for e in enters:
      if isinstance(e, ops.Tensor):
        xs = [e]
      else:
        if not isinstance(e, (ops.IndexedSlices, sparse_tensor.SparseTensor)):
          raise TypeError("Type %s not supported" % type(e))
        xs = [e.values, e.indices]
        shape = e.dense_shape
        if shape is not None:
          xs.append(shape)
      for x in xs:
        inp_op = x.op.inputs[0].op
        control_inputs = graph._control_dependencies_for_inputs([inp_op])
        outer_control_inputs = [
            op for op in control_inputs if self._IsInOuterContext(op)
        ]
        x.op._set_control_flow_context(self)
        x.op._add_control_inputs(outer_control_inputs)
        graph._record_op_seen_by_control_dependencies(x.op)
    
def _Enter(data,
           frame_name,
           is_constant=False,
           parallel_iterations=10,
           use_ref=True,
           use_input_shape=True,
           name=None):
  data = ops.internal_convert_to_tensor_or_indexed_slices(data, as_ref=True)
  if isinstance(data, ops.Tensor):
    if data.dtype._is_ref_dtype and use_ref:  
      result = gen_control_flow_ops.ref_enter(
          data, frame_name, is_constant, parallel_iterations, name=name)
    else:
      result = enter(
          data, frame_name, is_constant, parallel_iterations, name=name)
    if use_input_shape:
      result.set_shape(data.get_shape())
    return result
  else:
    if not isinstance(data, (ops.IndexedSlices, sparse_tensor.SparseTensor)):
      raise TypeError("Type %s not supported" % type(data))
    values = _Enter(
        data.values,
        frame_name,
        is_constant,
        parallel_iterations=parallel_iterations,
        use_input_shape=use_input_shape,
        name=name)
    indices = gen_control_flow_ops.enter(
        data.indices,
        frame_name,
        is_constant,
        parallel_iterations,
        name="indices")
    if use_input_shape:
      indices.set_shape(data.indices.get_shape())
    if isinstance(data, ops.IndexedSlices):
      dense_shape = data.dense_shape
      if dense_shape is not None:
        dense_shape = gen_control_flow_ops.enter(
            dense_shape,
            frame_name,
            is_constant,
            parallel_iterations,
            name="dense_shape")
        if use_input_shape:
          dense_shape.set_shape(data.dense_shape.get_shape())
      return ops.IndexedSlices(values, indices, dense_shape)
    else:
      dense_shape = gen_control_flow_ops.enter(
          data.dense_shape,
          frame_name,
          is_constant,
          parallel_iterations,
          name="dense_shape")
      if use_input_shape:
        dense_shape.set_shape(data.dense_shape.get_shape())
      return sparse_tensor.SparseTensor(indices, values, dense_shape)

def _SetShapeInvariants(input_vars, enter_vars, shapes):
  if shapes is None:
    return
  flat_shapes = nest.flatten(shapes)
  if not all([isinstance(s, tensor_shape.TensorShape) for s in flat_shapes]):
    raise ValueError("`shapes` must be a (possibly nested) list of shapes.")
  # Check that the shapes of the inputs are less than the shape invariants,
  # and set the shapes of `enter_vars` to the shape invariants.
  for inp, var, shape in zip(input_vars, enter_vars, flat_shapes):
    if isinstance(var, ops.Tensor):
      if not _ShapeLessThanOrEqual(inp.get_shape(), shape):
        raise ValueError(
            "The shape invariant specified for %s is not compatible with "
            "the initial shape of the loop variable. It enters the loop "
            "with shape %s, but the specified shape invariant is %s." %
            (inp.name, inp.get_shape(), shape))
      var.set_shape(shape)
    else:
      if not isinstance(var, (ops.IndexedSlices, sparse_tensor.SparseTensor)):
        raise TypeError("Type %s not supported" % type(var))
      if isinstance(var, ops.IndexedSlices):
        if not _ShapeLessThanOrEqual(inp.values.get_shape(), shape):
          raise ValueError(
              "The shape invariant specified for %s is not compatible with "
              "the initial shape of the values tensor of this IndexedSlices. "
              "It enters the loop with shape %s, but the specified shape "
              "invariant is %s." % (inp.values.name, inp.values.get_shape(),
                                    shape))
        var.values.set_shape(shape)
        var.indices.set_shape(tensor_shape.TensorShape([shape[0]]))
        if var.dense_shape is not None:
          var.dense_shape.set_shape(tensor_shape.TensorShape([shape.ndims]))
      else:
        if not _ShapeLessThanOrEqual(inp.dense_shape.get_shape(), shape):
          raise ValueError(
              "The shape invariant specified for %s is not compatible with "
              "the initial shape of the shape tensor of this SparseTensor. "
              "It enters the loop with shape %s, but the specified shape "
              "invariant is %s." % (inp.dense_shape.name,
                                    inp.dense_shape.get_shape(), shape))
        var.values.set_shape(tensor_shape.TensorShape([None]))
        var.indices.set_shape(tensor_shape.TensorShape([None, shape.ndims]))
        var.dense_shape.set_shape(shape)

def _make_tensor_array(ta, t_or_flow):
  
  new_ta = tensor_array_ops.TensorArray(
      dtype=ta.dtype,
      handle=ta.handle,
      flow=t_or_flow,
      infer_shape=ta._infer_shape,
      colocate_with_first_write_call=ta._colocate_with_first_write_call)
  new_ta._colocate_with = ta._colocate_with
  new_ta._element_shape = ta._element_shape
  
  return new_ta

def _Identity(data, name=None):
  data = ops.internal_convert_to_tensor_or_indexed_slices(data, as_ref=True)
  if isinstance(data, ops.Tensor):
    if data.dtype._is_ref_dtype:  
      return gen_array_ops.ref_identity(data, name=name)
    else:
      return array_ops.identity(data, name=name)
  else:
    if not isinstance(data, (ops.IndexedSlices, sparse_tensor.SparseTensor)):
      raise TypeError("Type %s not supported" % type(data))
    values = _Identity(data.values, name=name)
    indices = array_ops.identity(data.indices, name="indices")
    if isinstance(data, ops.IndexedSlices):
      dense_shape = data.dense_shape
      if dense_shape is not None:
        dense_shape = array_ops.identity(dense_shape, name="dense_shape")
      return ops.IndexedSlices(values, indices, dense_shape)
    else:
      dense_shape = array_ops.identity(data.dense_shape, name="dense_shape")
      return sparse_tensor.SparseTensor(indices, values, dense_shape)

def _AddNextAndBackEdge(m, v, enforce_shape_invariant=True):
  if isinstance(m, ops.Tensor):
    v = ops.convert_to_tensor(v)
    v = _NextIteration(v)
    if enforce_shape_invariant:
      # Make sure the shapes of loop outputs are correct. We do this before
      # calling _update_input, which will raise a less-helpful error message if
      # the types don't match.
      # TODO(skyewm): call this for other cases below (needs testing)
      _EnforceShapeInvariant(m, v)
    m.op._update_input(1, v)  
  elif isinstance(m, ops.IndexedSlices):
    
    v = math_ops._as_indexed_slices(v, optimize=False)
    v = _NextIteration(v)
    m.values.op._update_input(1, v.values)
    m.indices.op._update_input(1, v.indices)
    
    if m.dense_shape is not None:
      if v.dense_shape is None:
        raise ValueError("Must have dense shape: %s" % v.name)
      m.dense_shape.op._update_input(1, v.dense_shape)
  elif isinstance(m, sparse_tensor.SparseTensor):
    if not isinstance(v, sparse_tensor.SparseTensor):
      raise ValueError("Must be a sparse tensor: %s" % v.name)
    v = _NextIteration(v)
    
    m.values.op._update_input(1, v.values)
    m.indices.op._update_input(1, v.indices)
    m.dense_shape.op._update_input(1, v.dense_shape)
    
  else:
    raise TypeError("Type %s not supported" % type(m))
  return v

def _NextIteration(data, name=None):
  data = ops.internal_convert_to_tensor_or_indexed_slices(data, as_ref=True)
  if isinstance(data, ops.Tensor):
    if data.dtype._is_ref_dtype:  
      return ref_next_iteration(data, name=name)
    else:
      return next_iteration(data, name=name)
  else:
    if not isinstance(data, (ops.IndexedSlices, sparse_tensor.SparseTensor)):
      raise TypeError("Type %s not supported" % type(data))
    values = _NextIteration(data.values, name=name)
    indices = next_iteration(data.indices, name="indices")
    if isinstance(data, ops.IndexedSlices):
      dense_shape = data.dense_shape
      if dense_shape is not None:
        dense_shape = next_iteration(dense_shape, name="dense_shape")
      return ops.IndexedSlices(values, indices, dense_shape)
    else:
      dense_shape = next_iteration(data.dense_shape, name="dense_shape")
      return sparse_tensor.SparseTensor(indices, values, dense_shape)

def _EnforceShapeInvariant(merge_var, next_var):
  if isinstance(merge_var, ops.Tensor):
    m_shape = merge_var.get_shape()
    n_shape = next_var.get_shape()
    if not _ShapeLessThanOrEqual(n_shape, m_shape):
      enter = merge_var.op.inputs[0].op
      assert util.IsLoopEnter(enter)
      input_t = enter.inputs[0]
      raise ValueError(
          "Input tensor '%s' enters the loop with shape %s, but has shape %s "
          "after one iteration. To allow the shape to vary across iterations, "
          "use the `shape_invariants` argument of tf.while_loop to specify a "
          "less-specific shape." %
          (input_t.name, input_t.shape, n_shape))
  else:
    if not isinstance(merge_var,
                      (ops.IndexedSlices, sparse_tensor.SparseTensor)):
      raise TypeError("Type %s not supported" % type(merge_var))
    if isinstance(merge_var, ops.IndexedSlices):
      m_values_shape = merge_var.values.get_shape()
      m_indices_shape = merge_var.indices.get_shape()
      m_shape_shape = tensor_shape.TensorShape(None)
      if merge_var.dense_shape is not None:
        m_shape_shape = merge_var.dense_shape.get_shape()
      n_values_shape = next_var.values.get_shape()
      n_indices_shape = next_var.indices.get_shape()
      n_shape_shape = tensor_shape.TensorShape(None)
      if next_var.dense_shape is not None:
        n_shape_shape = next_var.dense_shape.get_shape()
      if (not _ShapeLessThanOrEqual(n_values_shape, m_values_shape) or
          not _ShapeLessThanOrEqual(n_indices_shape, m_indices_shape)):
        if not _ShapeLessThanOrEqual(n_values_shape, m_values_shape):
          raise ValueError(
              "The shape for %s is not an invariant for the loop. It enters "
              "the loop with shape (%s, %s, %s), but has shape (%s, %s, %s) "
              "after one iteration. Provide shape invariants using either the "
              "`shape_invariants` argument of tf.while_loop or set_shape() "
              "on the loop variables." %
              (merge_var.name, m_values_shape, m_indices_shape, m_shape_shape,
               n_values_shape, n_indices_shape, n_shape_shape))
    else:
      m_values_shape = merge_var.values.get_shape()
      m_indices_shape = merge_var.indices.get_shape()
      m_shape_shape = merge_var.dense_shape.get_shape()
      n_values_shape = next_var.values.get_shape()
      n_indices_shape = next_var.indices.get_shape()
      n_shape_shape = next_var.dense_shape.get_shape()
      if (not _ShapeLessThanOrEqual(n_values_shape, m_values_shape) or
          not _ShapeLessThanOrEqual(n_indices_shape, m_indices_shape) or
          not _ShapeLessThanOrEqual(n_shape_shape, m_shape_shape)):
        raise ValueError(
            "The shape for %s is not an invariant for the loop. It enters "
            "the loop with shape (%s, %s, %s), but has shape (%s, %s, %s) "
            "after one iteration. Provide shape invariants using either "
            "the `shape_invariants` argument of tf.while_loop or set_shape() "
            "on the loop variables." %
            (merge_var.name, m_values_shape, m_indices_shape, m_shape_shape,
             n_values_shape, n_indices_shape, n_shape_shape))

def _ShapeLessThanOrEqual(shape1, shape2):
  if shape2.dims is None:
    return True
  if shape1.ndims != shape2.ndims:
    return False
  for dim1, dim2 in zip(shape1.dims, shape2.dims):
    if dim2.value is not None and dim1.value != dim2.value:
      return False
  return True

def MaybeCreateControlFlowState(between_op_list, between_ops,
                                colocate_gradients_with_ops):
  loop_state = None
  for op in between_op_list:
    if util.IsLoopExit(op):
      if loop_state is None:
        loop_state = ControlFlowState()
      if colocate_gradients_with_ops:
        with ops.colocate_with(op):
          loop_state.AddWhileContext(op, between_op_list, between_ops)
      else:
        loop_state.AddWhileContext(op, between_op_list, between_ops)
  return loop_state

class ControlFlowState(object):
  def __init__(self):
    self._map = {}  # maps forward loop context to GradLoopState

  def GetGradState(self, op, before):
    if before and util.IsLoopExit(op):
      forward_ctxt = op._get_control_flow_context()
      forward_ctxt = forward_ctxt.outer_context
      if forward_ctxt:
        forward_ctxt = forward_ctxt.GetWhileContext()
    else:
      forward_ctxt = _GetWhileContext(op)
    if forward_ctxt:
      return self._map.get(forward_ctxt)
    return None

  def ProcessUnusedLoopExits(self, pending_count, to_ops_set):
    loop_exits = []
    for grad_state in self._map.values():
      for y in grad_state.forward_loop_exits:
        if pending_count[y.op] == 0:
          grad_state.pending_exits_count -= 1
          if y.op not in to_ops_set:
            grad_state.unused_exits.append(y)
          if grad_state.pending_exits_count == 0:
            loop_exits.extend(grad_state.unused_exits)
      # Need to include Enters in backprop for higher-order gradients.
      for y in grad_state.forward_context.loop_enters:
        if pending_count[y.op] == 0:
          pending_count[y.op] = 1
    return loop_exits

  def EnterGradWhileContext(self, op, before):
    grad_state = self.GetGradState(op, before)
    if grad_state:
      grad_state.grad_context.Enter()

  def ExitGradWhileContext(self, op, before):
    grad_state = self.GetGradState(op, before)
    if grad_state:
      grad_state.grad_context.Exit()

  def AddWhileContext(self, op, between_op_list, between_ops):
    forward_ctxt = _GetWhileContext(op)
    grad_state = self._map.get(forward_ctxt)
    if grad_state is None:
      # This is a new while loop so create a grad state for it.
      outer_forward_ctxt = forward_ctxt.outer_context
      if outer_forward_ctxt:
        outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
      outer_grad_state = None
      if outer_forward_ctxt:
        outer_grad_state = self._map.get(outer_forward_ctxt)
      grad_state = GradLoopState(forward_ctxt, outer_grad_state)
      self._map[forward_ctxt] = grad_state

      # We need to include all exits of a loop for backprop.
      for loop_exit in grad_state.forward_loop_exits:
        if loop_exit.op not in between_ops:
          between_ops.add(loop_exit.op)
          between_op_list.append(loop_exit.op)

  def ZerosLikeForExit(self, val):
    val_shape = val.get_shape()
    forward_ctxt = val.op._get_control_flow_context()
    outer_forward_ctxt = forward_ctxt.outer_context
    if outer_forward_ctxt:
      outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
    outer_grad_state = None
    if outer_forward_ctxt:
      outer_grad_state = self._map.get(outer_forward_ctxt)
    if outer_grad_state:
      # This is a nested loop.
      if val_shape.is_fully_defined():
        # If the shape is known statically, just create a zero tensor
        # with the right shape in the right context.
        outer_grad_state.grad_context.Enter()
        result = array_ops.zeros(val_shape.dims, val.dtype)
        outer_grad_state.grad_context.Exit()
      else:
        # Only the shape of value is needed for backprop.
        forward_ctxt.outer_context.Enter()
        shape = array_ops.shape_internal(val, optimize=False)
        forward_ctxt.outer_context.Exit()
        # Save the shape to a stack.
        history_shape = outer_grad_state.AddForwardAccumulator(shape)
        # Get the shape back from the stack.
        outer_grad_ctxt = outer_grad_state.grad_context
        outer_grad_ctxt.Enter()
        real_shape = outer_grad_state.AddBackpropAccumulatedValue(
            history_shape, shape)
        result = array_ops.zeros(real_shape, val.dtype)
        outer_grad_ctxt.Exit()
    else:
      # This is not a nested loop.
      if val_shape.is_fully_defined():
        # If the shape is known statically, just create a zero tensor
        # with the right shape.
        result = array_ops.zeros(val_shape.dims, val.dtype)
      else:
        result = array_ops.zeros_like(val, optimize=False)
    return result

  def ZerosLike(self, op, index):
    if util.IsLoopSwitch(op):
      return None
    dead_branch = util.IsSwitch(op)
    forward_ctxt = _GetWhileContext(op)
    grad_state = self._map.get(forward_ctxt)
    if grad_state is None:
      # op is not in a while loop that is part of gradients().
      return ZerosLikeOutsideLoop(op, index)
    op_ctxt = op._get_control_flow_context()
    val = ops.convert_to_tensor(op.outputs[index], name="tensor")
    shape = val.get_shape()
    if shape.is_fully_defined():
      # If the shape is known statically, just create a zero tensor with
      # the right shape in the grad loop context.
      result = constant_op.constant(0, shape=shape.dims, dtype=val.dtype)
      if dead_branch:
        # op is a cond switch. Guard the zero tensor with a switch.
        pred = grad_state.history_map.get(op_ctxt.pred.name)
        branch = op_ctxt.branch
        result = _SwitchRefOrTensor(result, pred)[1 - branch]
    else:
      # Unknown shape so keep a history of the shape at runtime.
      if dead_branch:
        # Need to add a special switch to guard the value.
        pred = op_ctxt.pred
        branch = op_ctxt.branch
        op_ctxt.outer_context.Enter()
        val = _SwitchRefOrTensor(op.inputs[0], pred)[1 - branch]
        zeros_shape = array_ops.shape_internal(val, optimize=False)
        op_ctxt.outer_context.Exit()
        val.op._set_control_flow_context(op_ctxt)
        zeros_shape.op._set_control_flow_context(op_ctxt)
      else:
        op_ctxt.Enter()
        zeros_shape = array_ops.shape_internal(val, optimize=False)
        op_ctxt.Exit()

      # Add forward accumulator for shape.
      grad_state.grad_context.Exit()
      history_zeros_shape = grad_state.AddForwardAccumulator(
          zeros_shape, dead_branch=dead_branch)
      grad_state.grad_context.Enter()

      # Create a zero tensor with the right shape.
      shape = grad_state.AddBackpropAccumulatedValue(history_zeros_shape,
                                                     zeros_shape, dead_branch)
      result = array_ops.zeros(shape, val.dtype)
    return result

  def PostProcessing(self):
    for _, grad_state in self._map.items():
      for _, b_merge in grad_state.switch_map.items():
        if b_merge.op.inputs[0] == b_merge.op.inputs[1]:
          # The value of this loop variable at iteration i+1 doesn't
          # depend on its value at iteration i. So use zeros as the
          # gradients for all iterations > 0.
          dtype = b_merge.op.inputs[0].dtype
          shape = b_merge.op.inputs[0].get_shape()
          
          if shape.is_fully_defined():
            grad_state.grad_context.Enter()
            # Create a zeros and use it for iterations > 0.
            grad_val = constant_op.constant(0, dtype=dtype, shape=shape)
            next_grad_val = _NextIteration(grad_val)
            grad_state.grad_context.Exit()
          else:
            # Create a zeros in the outer grad context.
            outer_grad_ctxt = grad_state.grad_context.outer_context
            if outer_grad_ctxt:
              outer_grad_ctxt.Enter()
            enter_grad_op = b_merge.op.inputs[0].op
            enter_grad = enter_grad_op.inputs[0]
            grad_shape = array_ops.shape_internal(enter_grad, optimize=False)
            grad_val = array_ops.zeros(grad_shape)
            if outer_grad_ctxt:
              outer_grad_ctxt.Exit()
            # Use the zeros for iterations > 0.
            grad_state.grad_context.Enter()
            next_grad_val = _NextIteration(grad_val)
            grad_state.grad_context.Exit()
          b_merge.op._update_input(1, next_grad_val)

def _GetWhileContext(op):
  ctxt = op._get_control_flow_context()
  if ctxt:
    ctxt = ctxt.GetWhileContext()
  return ctxt

class GradLoopState(object):
  def __init__(self, forward_ctxt, outer_grad_state):
    # The grad loop state for the outer while loop.
    self._outer_grad_state = None

    # The while loop context for forward.
    self._forward_context = None

    # The loop counter added by AddForwardLoopCounter. It is the value
    # of the loop counter for the next iteration.
    self._forward_index = None

    # A sync op for forward.
    self._forward_sync = None

    # The while loop context for backprop.
    self._grad_context = None

    # The loop counter added by AddBackpropLoopCounter. It is the value
    # of the loop counter for the current iteration.
    self._grad_index = None

    # A sync op for backprop.
    self._grad_sync = None

    # Information needed by backprop.
    self._history_map = {}
    self._switch_map = {}
    self._unused_exits = []
    self._deferred_exits = []
    self._forward_loop_exits = list(forward_ctxt.loop_exits)
    self._pending_exits_count = len(forward_ctxt.loop_exits)

    self._outer_grad_state = outer_grad_state
    if outer_grad_state:
      outer_forward_ctxt = outer_grad_state.forward_context
    else:
      if not hasattr(forward_ctxt, "outer_context"):
        raise ValueError("Failed to call gradients on a while loop without"
                         "properly serializing graph via MetaGraphDef")
      outer_forward_ctxt = forward_ctxt.outer_context

    # Add the forward loop counter.
    with forward_ctxt._graph.as_default():  
      if outer_forward_ctxt:
        outer_forward_ctxt.Enter()
      cnt, forward_index = forward_ctxt.AddForwardLoopCounter(outer_grad_state)
      if outer_forward_ctxt:
        outer_forward_ctxt.Exit()
    self._forward_context = forward_ctxt
    self._forward_index = forward_index

    # Add the backprop WhileContext, and the backprop loop counter.
    if outer_grad_state:
      # This is a nested loop. Remember the iteration counts for each
      # execution of this inner loop.
      outer_forward_ctxt.AddName(cnt.name)
      history_cnt = outer_grad_state.AddForwardAccumulator(cnt)

      outer_grad_ctxt = outer_grad_state.grad_context
      outer_grad_ctxt.Enter()
      self._grad_context = WhileContext(
          maximum_iterations=forward_ctxt.maximum_iterations,
          parallel_iterations=forward_ctxt.parallel_iterations,
          back_prop=forward_ctxt.back_prop,
          swap_memory=forward_ctxt.swap_memory,
          name=forward_ctxt.name,
          grad_state=self)
      real_cnt = outer_grad_state.AddBackpropAccumulatedValue(history_cnt, cnt)
      self._grad_index = self._grad_context.AddBackpropLoopCounter(
          real_cnt, outer_grad_state)
      outer_grad_ctxt.Exit()
    else:
      if outer_forward_ctxt:
        outer_forward_ctxt.Enter()
      self._grad_context = WhileContext(
          maximum_iterations=forward_ctxt.maximum_iterations,
          parallel_iterations=forward_ctxt.parallel_iterations,
          back_prop=forward_ctxt.back_prop,
          swap_memory=forward_ctxt.swap_memory,
          name=forward_ctxt.name,
          grad_state=self)
      self._grad_index = self._grad_context.AddBackpropLoopCounter(
          cnt, outer_grad_state)
      if outer_forward_ctxt:
        outer_forward_ctxt.Exit()

  @property
  def outer_grad_state(self):
    return self._outer_grad_state

  @property
  def forward_context(self):
    return self._forward_context

  @property
  def forward_index(self):
    return self._forward_index

  @property
  def forward_sync(self):
    if self._forward_sync is None:
      with ops.control_dependencies(None):
        self._forward_sync = control_trigger(name="f_sync")
      self._forward_sync._set_control_flow_context(self._forward_context)
      self._forward_index.op._add_control_input(self._forward_sync)
    return self._forward_sync

  @property
  def grad_context(self):
    return self._grad_context

  @property
  def grad_index(self):
    return self._grad_index

  @property
  def grad_sync(self):
    if self._grad_sync is None:
      with ops.control_dependencies(None):
        self._grad_sync = control_trigger(name="b_sync")
      self._grad_sync._set_control_flow_context(self._grad_context)
      self._grad_index.op._add_control_input(self._grad_sync)
      if self._grad_context.outer_context:
        self._grad_context.outer_context.AddInnerOp(self._grad_sync)
    return self._grad_sync

  @property
  def history_map(self):
    return self._history_map

  @property
  def switch_map(self):
    return self._switch_map

  @property
  def unused_exits(self):
    return self._unused_exits

  @property
  def deferred_exits(self):
    return self._deferred_exits

  @property
  def forward_loop_exits(self):
    return self._forward_loop_exits

  @property
  def pending_exits_count(self):
    return self._pending_exits_count

  @pending_exits_count.setter
  def pending_exits_count(self, cnt):
    self._pending_exits_count = cnt

  def AddForwardAccumulator(self, value, dead_branch=False):
    # curr_ctxt is the context that tf.gradients was called in.
    with self._forward_index.graph.as_default():
      curr_ctxt = ops.get_default_graph()._get_control_flow_context()  
      with ops.control_dependencies(None):
        if curr_ctxt:
          curr_ctxt.Enter()
        with ops.colocate_with(value):
          # We only need to pass maximum_iterations to the stack if
          # we're inside an XLA context.
          if not util.IsInXLAContext(value.op):
            max_size = constant_op.constant(-1, dtypes.int32)
          else:
            max_size = GetMaxSizeFromNestedMaximumIterations(
                value, self.forward_context)
          acc = gen_data_flow_ops.stack_v2(
              max_size=max_size, elem_type=value.dtype.base_dtype, name="f_acc")
        if curr_ctxt:
          curr_ctxt.Exit()

        # Make acc available in the forward context.
        enter_acc = self.forward_context.AddValue(acc)

        # Add the stack_push op in the context of value.op.
        swap_enabled = self.forward_context.swap_memory
        value_ctxt = util.GetOutputContext(value.op)
        if value_ctxt == self.forward_context:
          # value is not nested in the forward context.
          self.forward_context.Enter()
          push = gen_data_flow_ops.stack_push_v2(
              enter_acc, value, swap_memory=swap_enabled)
          self.forward_context.Exit()
          # Protect stack push and order it before forward_index.
          self.forward_index.op._add_control_input(push.op)
        else:
          # value is in a cond context within the forward context.
          if not isinstance(value_ctxt, CondContext):
            raise TypeError("value_ctxt is not a CondContext: %s" % value_ctxt)
          if dead_branch:
            # The special case for creating a zero tensor for a dead
            # branch of a switch. See ControlFlowState.ZerosLike().
            value_ctxt.outer_context.Enter()
            push = gen_data_flow_ops.stack_push_v2(
                enter_acc, value, swap_memory=swap_enabled)
            value_ctxt.outer_context.Exit()
            push.op._set_control_flow_context(value_ctxt)
          else:
            value_ctxt.Enter()
            push = gen_data_flow_ops.stack_push_v2(
                enter_acc, value, swap_memory=swap_enabled)
            value_ctxt.Exit()
          # Protect stack push and order it before forward_sync.
          self.forward_sync._add_control_input(push.op)
        # Order stack push after the successor of forward_index
        add_op = self.forward_index.op.inputs[0].op
        push.op._add_control_input(add_op)
        return acc

  def AddBackpropAccumulatedValue(self, history_value, value,
                                  dead_branch=False):
    history_ctxt = history_value.op._get_control_flow_context()
    # Find the cond context that controls history_value if any.
    cond_ctxt = None
    value_ctxt = value.op._get_control_flow_context()
    while value_ctxt and value_ctxt != history_ctxt:
      if isinstance(value_ctxt, CondContext):
        cond_ctxt = value_ctxt
        break
      value_ctxt = value_ctxt.outer_context
    with ops.control_dependencies(None):
      self.grad_context.Enter()
      if cond_ctxt:
        # Guard stack pop with a switch if it is controlled by a cond.
        grad_state = self
        pred = None
        while pred is None and grad_state:
          pred = grad_state.history_map.get(cond_ctxt.pred.name)
          grad_state = grad_state.outer_grad_state
        if pred is None:
          pred = cond_ctxt.pred
        branch = (1 - cond_ctxt.branch) if dead_branch else cond_ctxt.branch
        history_value = _SwitchRefOrTensor(history_value, pred)[branch]
      pop = gen_data_flow_ops.stack_pop_v2(history_value,
                                           value.dtype.base_dtype)
      pop.set_shape(value.get_shape())
      self.grad_context.Exit()
    parallel_iterations = self.grad_context.parallel_iterations
    if parallel_iterations > 1:
      # All pops are ordered after pivot_for_body and before grad_sync.
      self.grad_sync._add_control_input(pop.op)
    return pop

  def GetRealValue(self, value):
    assert value.op.type not in ["Variable", "VariableV2"]
    real_value = self._history_map.get(value.name)
    if real_value is None:
      cur_value = value
      cur_grad_state = self
      while True:
        enter_op = util.GetLoopConstantEnter(cur_value)
        if enter_op:
          # Special case: cur_value comes from a constant Enter node.
          cur_value = enter_op.inputs[0]
          cur_grad_state = cur_grad_state.outer_grad_state
          if cur_grad_state is None:
            # We are now outside all nested loops for this gradient(),
            # so `value` is a loop invariant and there is no need to
            # save the history of value. Just make cur_value to enter
            # the right control flow context.
            real_value = self._grad_context.AddValue(cur_value)
            break
        elif constant_op.is_constant(cur_value):
          # If the value to be forwarded is a constant, clone the constant in
          # the gradient loop rather than using a stack.
          # TODO(phawkins): consider hoisting the constant out of the loop
          # instead.
          real_value = constant_op.constant(
              tensor_util.constant_value(cur_value), dtype=cur_value.dtype)
          break
        else:
          # Record the history of this value in forward_ctxt.
          self._grad_context.Exit()
          history_value = cur_grad_state.AddForwardAccumulator(cur_value)
          self._grad_context.Enter()
          break

      if real_value is None:
        # Add the stack pop op in the grad context.
        real_value = cur_grad_state.AddBackpropAccumulatedValue(
            history_value, cur_value)
        if cur_grad_state != self:
          real_value = self._grad_context.AddValue(real_value)
      self._history_map[value.name] = real_value
    return real_value

def ZerosLikeOutsideLoop(op, index):
  val = op.outputs[index]
  if not util.IsSwitch(op):
    if val.dtype == dtypes.resource:
      return array_ops.zeros(gen_resource_variable_ops.variable_shape(val))
    return array_ops.zeros_like(val, optimize=False)
  else:
    op_ctxt = op._get_control_flow_context()
    if op_ctxt:
      # We are in a cond context. Use a switch to create zeros only when needed.
      pred = op_ctxt.pred
      branch = op_ctxt.branch
      switch_val = switch(op.inputs[0], pred)[1 - branch]
      # A op is created along the branch taken as control dependencies are on
      # the whole op and not on the tensor output.
      pivot = array_ops.identity(switch_val)
      if val.dtype == dtypes.resource:
        with ops.control_dependencies([pivot]):
          return array_ops.zeros(
              gen_resource_variable_ops.variable_shape(switch_val))
      zeros_shape = array_ops.shape_internal(switch_val, optimize=False)
      # Ensure ops created within array_ops.zeros are dominated by switch in
      # cond context.
      with ops.control_dependencies([pivot]):
        return array_ops.zeros(zeros_shape, dtype=val.dtype)
    else:
      return array_ops.zeros_like(val, optimize=False)

#edit

def switch(data, pred, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Switch", data=data, pred=pred, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    execute.record_gradient(
      "Switch", _inputs_flat, _attrs, _result, name)
    _result = _SwitchOutput._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Switch", name,
        _ctx._post_execution_callbacks, data, pred)
      _result = _SwitchOutput._make(_result)
      return _result
    except _core._FallbackException:
      return switch_eager_fallback(
          data, pred, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

_switch_outputs = ["output_false", "output_true"]
_SwitchOutput = collections.namedtuple(
    "Switch", _switch_outputs)

def merge(inputs, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(inputs, (list, _basetuple)):
      raise TypeError(
          "Expected list for 'inputs' argument to "
          "'merge' Op, not %r." % inputs)
    _attr_N = len(inputs)
    _, _, _op = _op_def_lib._apply_op_helper(
        "Merge", inputs=inputs, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "N", _op.get_attr("N"))
    execute.record_gradient(
      "Merge", _inputs_flat, _attrs, _result, name)
    _result = _MergeOutput._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Merge", name,
        _ctx._post_execution_callbacks, inputs)
      _result = _MergeOutput._make(_result)
      return _result
    except _core._FallbackException:
      return merge_eager_fallback(
          inputs, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

_merge_outputs = ["output", "value_index"]
_MergeOutput = collections.namedtuple(
    "Merge", _merge_outputs)

def enter(data, frame_name, is_constant=False, parallel_iterations=10, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    frame_name = execute.make_str(frame_name, "frame_name")
    if is_constant is None:
      is_constant = False
    is_constant = execute.make_bool(is_constant, "is_constant")
    if parallel_iterations is None:
      parallel_iterations = 10
    parallel_iterations = execute.make_int(parallel_iterations, "parallel_iterations")
    _, _, _op = _op_def_lib._apply_op_helper(
        "Enter", data=data, frame_name=frame_name, is_constant=is_constant,
        parallel_iterations=parallel_iterations, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "frame_name",
              _op.get_attr("frame_name"), "is_constant",
              _op.get_attr("is_constant"), "parallel_iterations",
              _op.get_attr("parallel_iterations"))
    execute.record_gradient(
      "Enter", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Enter", name,
        _ctx._post_execution_callbacks, data, "frame_name", frame_name,
        "is_constant", is_constant, "parallel_iterations",
        parallel_iterations)
      return _result
    except _core._FallbackException:
      return enter_eager_fallback(
          data, frame_name=frame_name, is_constant=is_constant,
          parallel_iterations=parallel_iterations, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def loop_cond(input, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "LoopCond", input=input, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
    execute.record_gradient(
      "LoopCond", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "LoopCond",
        name, _ctx._post_execution_callbacks, input)
      return _result
    except _core._FallbackException:
      return loop_cond_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def next_iteration(data, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "NextIteration", data=data, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    execute.record_gradient(
      "NextIteration", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "NextIteration", name, _ctx._post_execution_callbacks, data)
      return _result
    except _core._FallbackException:
      return next_iteration_eager_fallback(
          data, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def no_op(name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "NoOp", name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "NoOp", name,
        _ctx._post_execution_callbacks)
      return _result
    except _core._FallbackException:
      return no_op_eager_fallback(
          name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def control_trigger(name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "ControlTrigger", name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "ControlTrigger", name, _ctx._post_execution_callbacks)
      return _result
    except _core._FallbackException:
      return control_trigger_eager_fallback(
          name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def with_dependencies(dependencies, output_tensor, name=None):
  if context.executing_eagerly():
    return output_tensor
  with ops.name_scope(name, "control_dependency",
                      list(dependencies) + [output_tensor]) as name:
    with ops.colocate_with(output_tensor):
      with ops.control_dependencies(dependencies):
        output_tensor = ops.convert_to_tensor_or_indexed_slices(output_tensor)
        if isinstance(output_tensor, ops.Tensor):
          return _Identity(output_tensor, name=name)
        else:
          return ops.IndexedSlices(
              _Identity(output_tensor.values, name=name), output_tensor.indices,
              output_tensor.dense_shape)

def _GroupControlDeps(dev, deps, name=None):
  with ops.control_dependencies(deps):
    if dev is None:
      return no_op(name=name)
    else:
      with ops.device(dev):
        return no_op(name=name)

def group(*inputs, **kwargs):
  if context.executing_eagerly():
    return None
  name = kwargs.pop("name", None)
  if kwargs:
    raise ValueError("Unknown keyword arguments: " + ", ".join(kwargs.keys()))
  with ops.name_scope(name, "group_deps", inputs) as name:
    # Grouping no inputs means do nothing
    if not inputs:
      return no_op(name=name)

    # Sorts *inputs according to their devices.
    ops_on_device = {}  # device -> operations specified on the device.
    for inp in nest.flatten(inputs):
      if not hasattr(inp, "device"):
        raise TypeError("Expected tf.group() expected Tensor arguments not "
                        "'%s' with type '%s'" % (inp, type(inp)))
      dev = inp.device
      if dev in ops_on_device:
        ops_on_device[dev].append(inp)
      else:
        ops_on_device[dev] = [inp]
    if len(ops_on_device) == 1:
      # 1-level tree. The root node is the returned NoOp node.
      (dev, deps), = ops_on_device.items()
      return _GroupControlDeps(dev, deps, name=name)

    # 2-level tree. The root node is the returned NoOp node.
    # deps contains 1 NoOp node for each device.
    deps = []

    def device_key(dev):
      return "" if dev is None else dev

    for dev in sorted(six.iterkeys(ops_on_device), key=device_key):
      deps.append(_GroupControlDeps(dev, ops_on_device[dev]))

    with ops.control_dependencies(deps):
      return no_op(name=name)

def tuple(tensors, name=None, control_inputs=None):  
  if context.executing_eagerly():
    return tensors
  with ops.name_scope(name, "tuple", tensors) as name:
    tensors = [t if (isinstance(t, ops.Operation)
                     or tensor_util.is_tensor(t)
                     or t is None)
               else ops.convert_to_tensor(t) for t in tensors]
    gating_ops = [t if isinstance(t, ops.Operation) else t.op for t in tensors
                  if t is not None]
    if control_inputs:
      for c in control_inputs:
        if isinstance(c, ops.Tensor):
          c = c.op
        elif not isinstance(c, ops.Operation):
          raise TypeError("Control input must be Operation or Tensor: %s" % c)
        gating_ops.append(c)
    # Note that in order to ensure ordering in the pbtxt, we must take care to
    # ensure the order here.
    gating_ops = sorted(set(gating_ops), key=lambda op: op._id)  # Uniquify ops.
    if not gating_ops:
      raise ValueError("Must have at least one Tensor: %s" % tensors)
    gate = group(*gating_ops)
    tpl = []
    for t in tensors:
      if tensor_util.is_tensor(t):
        tpl.append(with_dependencies([gate], t))
      elif isinstance(t, ops.Operation):
        with ops.control_dependencies([gate]):
          tpl.append(group(t))
      else:
        tpl.append(None)
    return tpl

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib

_op_def_lib = _InitOpDefLibrary(b"\n@\n\005Abort\"\027\n\terror_msg\022\006string\032\002\022\000\"\036\n\022exit_without_error\022\004bool\032\002(\000\n\020\n\016ControlTrigger\ny\n\005Enter\022\t\n\004data\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\024\n\nframe_name\022\006string\"\027\n\013is_constant\022\004bool\032\002(\000\"\036\n\023parallel_iterations\022\003int\032\002\030\n\n)\n\004Exit\022\t\n\004data\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n!\n\010LoopCond\022\t\n\005input\030\n\032\n\n\006output\030\n\nN\n\005Merge\022\016\n\006inputs\"\001T*\001N\032\013\n\006output\"\001T\032\017\n\013value_index\030\003\"\t\n\001T\022\004type\"\014\n\001N\022\003int(\0010\001\n2\n\rNextIteration\022\t\n\004data\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n\006\n\004NoOp\n\202\001\n\010RefEnter\022\014\n\004data\"\001T\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\"\024\n\nframe_name\022\006string\"\027\n\013is_constant\022\004bool\032\002(\000\"\036\n\023parallel_iterations\022\003int\032\002\030\n\n2\n\007RefExit\022\014\n\004data\"\001T\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\nW\n\010RefMerge\022\021\n\006inputs\"\001T*\001N\200\001\001\032\016\n\006output\"\001T\200\001\001\032\017\n\013value_index\030\003\"\t\n\001T\022\004type\"\014\n\001N\022\003int(\0010\001\n;\n\020RefNextIteration\022\014\n\004data\"\001T\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\nR\n\tRefSelect\022\t\n\005index\030\003\022\021\n\006inputs\"\001T*\001N\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\"\014\n\001N\022\003int(\0010\001\n\\\n\tRefSwitch\022\014\n\004data\"\001T\200\001\001\022\010\n\004pred\030\n\032\024\n\014output_false\"\001T\200\001\001\032\023\n\013output_true\"\001T\200\001\001\"\t\n\001T\022\004type\230\001\001\nM\n\006Switch\022\t\n\004data\"\001T\022\010\n\004pred\030\n\032\021\n\014output_false\"\001T\032\020\n\013output_true\"\001T\"\t\n\001T\022\004type")
