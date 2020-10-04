from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import sys
import warnings

import numpy as np
import six
from six.moves import xrange

from backend import context
from backend.ops import control_flow_util
from backend.ops import array_ops
from backend.ops import control_flow_ops
from backend.framework import dtypes
from backend.framework import constant_op
from backend.framework import function as framework_function
from backend.framework import ops

def gradients(ys,
              xs,
              grad_ys=None,
              name="gradients",
              colocate_gradients_with_ops=False,
              gate_gradients=False,
              aggregation_method=None,
              stop_gradients=None):

  with ops.get_default_graph()._mutation_lock():  
    return _GradientsHelper(ys, xs, grad_ys, name, colocate_gradients_with_ops,
                            gate_gradients, aggregation_method, stop_gradients)

def _GradientsHelper(ys,
                     xs,
                     grad_ys=None,
                     name="gradients",
                     colocate_gradients_with_ops=False,
                     gate_gradients=False,
                     aggregation_method=None,
                     stop_gradients=None,
                     src_graph=None):
  if context.executing_eagerly():
    raise RuntimeError("tf.gradients is not supported when eager execution "
                       "is enabled. Use tf.GradientTape instead.")
  if src_graph is None:
    src_graph = ops.get_default_graph()

  func_graphs = []
  curr_graph = src_graph

  ys = _AsList(ys)
  xs = _AsList(xs)
  stop_gradients = [] if stop_gradients is None else _AsList(stop_gradients)
  if grad_ys is None:
    grad_ys = [None] * len(ys)
  else:
    grad_ys = _AsList(grad_ys)

  with ops.name_scope(
      name, "gradients",
      list(ys) + list(xs) + list(stop_gradients) + list(grad_ys)) as grad_scope:
    gradient_uid = ops.get_default_graph().unique_name("uid")
    ys = ops.convert_n_to_tensor_or_indexed_slices(ys, name="y")
    grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops,
                             gradient_uid)

    to_ops = [t.op for t in ys]
    from_ops = [t.op for t in xs]
    stop_gradient_ops = [t.op for t in stop_gradients]
    reachable_to_ops, pending_count, loop_state = _PendingCount(
        to_ops, from_ops, colocate_gradients_with_ops, func_graphs, xs)

    grads = {}

    for y, grad_y in zip(ys, grad_ys):
      _SetGrad(grads, y, grad_y)

    queue = collections.deque()
    to_ops_set = set()
    for op in to_ops:
      ready = (pending_count[op] == 0)
      if ready and op not in to_ops_set and op in reachable_to_ops:
        to_ops_set.add(op)
        queue.append(op)

    if loop_state:
      loop_exits = loop_state.ProcessUnusedLoopExits(pending_count, to_ops_set)
      for y in loop_exits:
        if _IsTrainable(y):
          _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
          queue.append(y.op)

    stop_ops = _StopOps(from_ops, stop_gradient_ops, pending_count, xs)
    while queue:
      op = queue.popleft()
      with _maybe_colocate_with(op, gradient_uid, colocate_gradients_with_ops):
        if loop_state:
          loop_state.EnterGradWhileContext(op, before=True)
        out_grads = _AggregatedGrads(grads, op, gradient_uid, loop_state,
                                     aggregation_method)
        if loop_state:
          loop_state.ExitGradWhileContext(op, before=True)

        grad_fn = None
        func_call = None
        is_partitioned_call = _IsPartitionedCall(op)
        
        is_func_call = (
            src_graph._is_function(op.type) or is_partitioned_call)
        
        has_out_grads = any(isinstance(g, ops.Tensor) or g for g in out_grads)
        if has_out_grads and (op not in stop_ops):
          if is_func_call:
            if is_partitioned_call:
              func_call = src_graph._get_function(  
                  compat.as_bytes(op.get_attr("f").name))
            else:
              func_call = src_graph._get_function(op.type)  
            func_call = getattr(op, "__defun", func_call)
            grad_fn = func_call.python_grad_func
          else:
            try:
              grad_fn = ops.get_gradient_function(op)
            except LookupError:
              raise LookupError(
                  "No gradient defined for operation '%s' (op type: %s)" %
                  (op.name, op.type))
        if loop_state:
          loop_state.EnterGradWhileContext(op, before=False)
        
        if (control_flow_util.IsSwitch(op) and
            op._control_flow_context is not None and
            op._control_flow_context.IsWhileContext() and
            op._control_flow_context ==
            ops.get_default_graph()._get_control_flow_context()):
          _RaiseNoGradWrtInitialLoopValError(op, from_ops, xs)
        

        if (grad_fn or is_func_call) and has_out_grads:
          for i, out_grad in enumerate(out_grads):
            if (not isinstance(out_grad, ops.Tensor) and not out_grad) and (
                (not grad_fn and is_func_call) or _IsTrainable(op.outputs[i])):
              if loop_state:
                out_grads[i] = loop_state.ZerosLike(op, i)
              else:
                out_grads[i] = control_flow_ops.ZerosLikeOutsideLoop(op, i)
          with ops.name_scope(op.name + "_grad"):
            
            with src_graph._original_op(op):
              
              if grad_fn:
                in_grads = _MaybeCompile(grad_scope, op, func_call,
                                         lambda: grad_fn(op, *out_grads))
              else:
                in_grads = _MaybeCompile(grad_scope, op, func_call,
                                         lambda: _SymGrad(op, out_grads))
              in_grads = _AsList(in_grads)
              _VerifyGeneratedGradients(in_grads, op)
              if gate_gradients and len([x for x in in_grads
                                         if x is not None]) > 1:
                with ops.device(None):
                  with ops._colocate_with_for_gradient(  
                      None,
                      gradient_uid,
                      ignore_existing=True):
                    in_grads = control_flow_ops.tuple(in_grads)
          _LogOpGradients(op, out_grads, in_grads)
        else:
          in_grads = [None] * len(_NonEagerInputs(op, xs))
        for i, (t_in, in_grad) in enumerate(zip(_NonEagerInputs(op, xs),
                                                in_grads)):
          if in_grad is not None:
            if (isinstance(in_grad, ops.Tensor) and
                t_in.dtype != dtypes.resource):
              try:
                in_grad.set_shape(t_in.get_shape())
              except ValueError:
                raise ValueError(
                    "Incompatible shapes between op input and calculated "
                    "input gradient.  Forward operation: %s.  Input index: %d. "
                    "Original input shape: %s.  "
                    "Calculated input gradient shape: %s" %
                    (op.name, i, t_in.shape, in_grad.shape))
            _SetGrad(grads, t_in, in_grad)
        if loop_state:
          loop_state.ExitGradWhileContext(op, before=False)
      _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state,
                                    xs)

  if loop_state:
    loop_state.PostProcessing()
  return [_GetGrad(grads, x) for x in xs]

def _IsFunction(graph):
  return (isinstance(graph, framework_function._FuncGraph))  

def _AsList(x):
  return x if isinstance(x, (list, tuple)) else [x]

def _DefaultGradYs(grad_ys,
                   ys,
                   colocate_gradients_with_ops,
                   gradient_uid="__unsupported__"):
  if len(grad_ys) != len(ys):
    raise ValueError("Passed %d grad_ys for %d ys" % (len(grad_ys), len(ys)))
  grad_ys = ops.convert_n_to_tensor_or_indexed_slices(grad_ys, name="grad_y")
  new_grad_ys = []
  for i in xrange(len(grad_ys)):
    grad_y = grad_ys[i]
    y = ys[i]
    with _maybe_colocate_with(y.op, gradient_uid, colocate_gradients_with_ops):
      if grad_y is None:
        if y.dtype.is_complex:
          raise TypeError(
              "Gradients of complex tensors must set grad_ys (y.dtype = %r)" %
              y.dtype)
        new_grad_ys.append(
            array_ops.fill(
                array_ops.shape(y),
                constant_op.constant(1, dtype=y.dtype, name="grad_ys_%d" % i)))
        continue
      if y.dtype.is_floating or y.dtype.is_integer:
        if not grad_y.dtype.is_floating and not grad_y.dtype.is_integer:
          raise TypeError(
              "Gradient type %s generated for real or "
              "integer-valued tensor %s with type %s must be "
              "real or integer" % (dtypes.as_dtype(grad_y.dtype).name, y,
                                   dtypes.as_dtype(y.dtype).name))
      elif y.dtype.is_complex:
        if not grad_y.dtype.is_complex:
          raise TypeError(
              "Gradient type %s generated for complex-valued "
              "tensor %s with type %s must be real" % (dtypes.as_dtype(
                  grad_y.dtype).name, y, dtypes.as_dtype(y.dtype).name))
      elif y.dtype == dtypes.variant:
        if grad_y.dtype != dtypes.variant:
          raise TypeError(
              "Gradient type %s generated for variant "
              "tensor %s with type %s must be variant" % (dtypes.as_dtype(
                  grad_y.dtype).name, y, dtypes.as_dtype(y.dtype).name))
      else:
        raise TypeError(
            "Tensor %s with type %s must be numeric "
            "to obtain a default gradient" % (y, dtypes.as_dtype(y.dtype).name))
      if isinstance(grad_y, ops.IndexedSlices):
        new_grad_ys.append(
            ops.IndexedSlices(
                indices=(array_ops.identity(
                    grad_y.indices, name="grad_ys_%d_indices" % i)
                         if isinstance(grad_y.indices, ops.Tensor) else
                         grad_y.indices),
                values=(array_ops.identity(
                    grad_y.values, name="grad_ys_%d_values" % i) if isinstance(
                        grad_y.values, ops.Tensor) else grad_y.values),
                dense_shape=(array_ops.identity(
                    grad_y.dense_shape, name="grad_ys_%d_shape" % i)
                             if isinstance(grad_y.dense_shape, ops.Tensor) else
                             grad_y.dense_shape)))
      else:
        new_grad_ys.append(array_ops.identity(grad_y, name="grad_ys_%d" % i))

  return new_grad_ys

@contextlib.contextmanager
def _maybe_colocate_with(op, gradient_uid, colocate_gradients_with_ops):
  if colocate_gradients_with_ops:
    with ops._colocate_with_for_gradient(op, gradient_uid):  
      yield
  else:
    yield

def _PendingCount(to_ops, from_ops, colocate_gradients_with_ops, func_graphs,
                  xs):
  reached_ops = set()
  _MarkReachedOps(from_ops, reached_ops, func_graphs)

  reachable_to_ops = set(op for op in to_ops if op in reached_ops)

  between_ops = set()
  between_op_list = []
  queue = collections.deque()
  queue.extend(to_ops)
  while queue:
    op = queue.popleft()
    if op in reached_ops:
      between_ops.add(op)
      between_op_list.append(op)
      reached_ops.remove(op)
  loop_state = control_flow_ops.MaybeCreateControlFlowState(
      between_op_list, between_ops, colocate_gradients_with_ops)

  pending_count = collections.defaultdict(int)
  for op in between_op_list:
    for x in _NonEagerInputs(op, xs):
      if x.op in between_ops:
        pending_count[x.op] += 1

  return reachable_to_ops, pending_count, loop_state

def _MarkReachedOps(from_ops, reached_ops, func_graphs):
  queue = collections.deque()
  queue.extend(from_ops)
  while queue:
    op = queue.popleft()
    if op not in reached_ops:
      reached_ops.add(op)
      for output in op.outputs:
        if _IsBackpropagatable(output):
          queue.extend(_Consumers(output, func_graphs))

def _IsBackpropagatable(tensor):
  if _IsTrainable(tensor):
    return True
  dtype = dtypes.as_dtype(tensor.dtype)
  return dtype.base_dtype in (dtypes.bfloat16, dtypes.variant)

def _IsTrainable(tensor):
  dtype = dtypes.as_dtype(tensor.dtype)
  return dtype.base_dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                              dtypes.complex64, dtypes.complex128,
                              dtypes.resource)

def _Consumers(t, func_graphs):
  consumers = t.consumers()
  for func in func_graphs:
    for input_t, placeholder in _Captures(func).items():
      if input_t == t:
        consumers.extend(_Consumers(placeholder, func_graphs))
  return consumers

def _NonEagerInputs(op, xs):
  if _IsFunction(op.graph):  
    inputs = []
    for t in op.inputs:
      if t not in xs:
        t = _MaybeCaptured(t)
        if isinstance(t, ops.EagerTensor): continue
      inputs.append(t)
    return inputs
  else:
    return op.inputs

def _SetGrad(grads, t, grad):
  op = t.op
  op_grads = grads.get(op)
  if not op_grads:
    op_grads = [[] for _ in xrange(len(op.outputs))]
    grads[op] = op_grads
  t_grads = op_grads[t.value_index]
  if isinstance(t_grads, list):
    t_grads.append(grad)
  else:
    assert control_flow_util.IsLoopSwitch(op)
    op_grads[t.value_index] = grad

def _StopOps(from_ops, stop_gradient_ops, pending_count, xs):

  stop_ops = set()
  for op in from_ops:
    is_stop_op = True
    for inp in _NonEagerInputs(op, xs):
      if pending_count[inp.op] > 0:
        is_stop_op = False
        break
    if is_stop_op:
      stop_ops.add(op)
  stop_ops.update(op for op in stop_gradient_ops)
  return stop_ops

def _AggregatedGrads(grads,
                     op,
                     gradient_uid,
                     loop_state,
                     aggregation_method=None):
  if aggregation_method is None:
    aggregation_method = AggregationMethod.DEFAULT
  if aggregation_method not in [
      AggregationMethod.ADD_N, AggregationMethod.EXPERIMENTAL_TREE,
      AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
  ]:
    raise ValueError(
        "Invalid aggregation_method specified %s." % aggregation_method)
  out_grads = _GetGrads(grads, op)
  for i, out_grad in enumerate(out_grads):
    if loop_state:
      if isinstance(out_grad, (ops.Tensor, ops.IndexedSlices)):
        assert control_flow_util.IsLoopSwitch(op)
        continue
    if (isinstance(out_grad, collections.Sequence) and not all([
        isinstance(g, (ops.Tensor, ops.IndexedSlices))
        for g in out_grad
        if g is not None
    ])):
      raise TypeError("gradients have to be either all Tensors "
                      "or all IndexedSlices")
    if out_grad:
      if len(out_grad) < 2:
        used = "nop"
        out_grads[i] = out_grad[0]
      elif all([isinstance(g, ops.Tensor) for g in out_grad if g is not None]):
        tensor_shape = _AccumulatorShape(out_grad)
        if (aggregation_method == AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
            and len(out_grad) > 2 and tensor_shape.is_fully_defined()):
          used = "accumulate_n"
          out_grads[i] = math_ops.accumulate_n(out_grad)
        elif aggregation_method in [
            AggregationMethod.EXPERIMENTAL_TREE,
            AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
        ]:
          used = "tree"
          with ops.name_scope(op.name + "_gradient_sum"):
            running_sum = out_grad[0]
            for grad in out_grad[1:]:
              running_sum = math_ops.add_n([running_sum, grad])
            out_grads[i] = running_sum
        else:
          used = "add_n"
          out_grads[i] = _MultiDeviceAddN(out_grad, gradient_uid)
      else:
        out_grads[i] = _AggregateIndexedSlicesGradients(out_grad)
    else:
      out_grads[i] = None
  return out_grads

class AggregationMethod(object):
  ADD_N = 0
  DEFAULT = ADD_N
  EXPERIMENTAL_TREE = 1
  EXPERIMENTAL_ACCUMULATE_N = 2

def _GetGrads(grads, op):
  if op in grads:
    return grads[op]
  else:
    return [[] for _ in xrange(len(op.outputs))]

def _IsPartitionedCall(op):
  return op.type == "PartitionedCall" or op.type == "StatefulPartitionedCall"

def _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state,
                                  xs):
  for x in _NonEagerInputs(op, xs):
    pending_count[x.op] -= 1
    ready = (pending_count[x.op] == 0)
    if loop_state and not ready:
      ready = pending_count[x.op] > 0 and control_flow_util.IsLoopSwitch(x.op)
    if ready:
      if control_flow_util.IsLoopExit(x.op):
        grad_state = loop_state.GetGradState(x.op, before=False)
        grad_state.deferred_exits.append(x)
        grad_state.pending_exits_count -= 1
        if grad_state.pending_exits_count == 0:
          has_not_none_grad = False
          for y in grad_state.deferred_exits:
            if _HasAnyNotNoneGrads(grads, y.op):
              has_not_none_grad = True
              queue.append(y.op)
            else:
              grad_state.unused_exits.append(y)
          if has_not_none_grad:
            for y in grad_state.unused_exits:
              if _IsTrainable(y):
                _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
              queue.append(y.op)
          else:
            for y in grad_state.unused_exits:
              queue.append(y.op)
      else:
        queue.append(x.op)

def _HasAnyNotNoneGrads(grads, op):
  out_grads = _GetGrads(grads, op)
  for out_grad in out_grads:
    if isinstance(out_grad, (ops.Tensor, ops.IndexedSlices)):
      return True
    if out_grad and isinstance(out_grad, collections.Sequence):
      if any([g is not None for g in out_grad]):
        return True
  return False

def _GetGrad(grads, t):
  op = t.op
  op_grads = grads.get(op)
  if not op_grads:
    return None
  t_grad = op_grads[t.value_index]
  assert not isinstance(
      t_grad, list), ("gradients list should have been aggregated by now.")
  return t_grad
