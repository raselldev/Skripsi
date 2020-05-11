from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import sys
import warnings

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin


from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import context
#from tensorflow.python.framework import function as constant_op
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_grad
from tensorflow.python.ops import math_grad
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_grad
#from tensorflow.python import tf_logging as logging
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops


# This is to avoid a circular dependency (eager.function depends on
# gradients_impl). This is set in eager/function.py.
_function = None

# This is to avoid a circular dependency with cond_v2_impl.


# Warn the user if we convert a sparse representation to dense with at
# least this number of elements.
_LARGE_SPARSE_NUM_ELEMENTS = 100000000


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


def _PendingCount(to_ops, from_ops, colocate_gradients_with_ops, func_graphs,
                  xs):
  reached_ops = set()
  _MarkReachedOps(from_ops, reached_ops, func_graphs)
  # X in reached_ops iff X is reachable from from_ops by a path of zero or more
  # backpropagatable tensors.

  reachable_to_ops = set(op for op in to_ops if op in reached_ops)

  # Mark between ops.
  between_ops = set()
  between_op_list = []
  queue = collections.deque()
  queue.extend(to_ops)
  while queue:
    op = queue.popleft()
    # We are interested in this op.
    if op in reached_ops:
      between_ops.add(op)
      between_op_list.append(op)
      # Clear the boolean so we won't add the inputs again.
      reached_ops.remove(op)
      for inp in _NonEagerInputs(op, xs):
        queue.append(inp.op)
  # X in between_ops iff X is on a path of zero or more backpropagatable tensors
  # between from_ops and to_ops

  # 'loop_state' is None if there are no while loops.
  loop_state = control_flow_ops.MaybeCreateControlFlowState(
      between_op_list, between_ops, colocate_gradients_with_ops)

  # Initialize pending count for between ops.
  pending_count = collections.defaultdict(int)
  for op in between_op_list:
    for x in _NonEagerInputs(op, xs):
      if x.op in between_ops:
        pending_count[x.op] += 1

  return reachable_to_ops, pending_count, loop_state


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
      # Create a grad_y tensor in the name scope of the gradient.
      # Required for TensorArrays to identify which gradient call a
      # grad_y value is coming from.
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


def _IsTrainable(tensor):
  dtype = dtypes.as_dtype(tensor.dtype)
  return dtype.base_dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                              dtypes.complex64, dtypes.complex128,
                              dtypes.resource)


def _IsBackpropagatable(tensor):
  if _IsTrainable(tensor):
    return True
  dtype = dtypes.as_dtype(tensor.dtype)
  return dtype.base_dtype in (dtypes.bfloat16, dtypes.variant)


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


@contextlib.contextmanager
def _maybe_colocate_with(op, gradient_uid, colocate_gradients_with_ops):  # pylint: disable=invalid-name
  if colocate_gradients_with_ops:
    with ops._colocate_with_for_gradient(op, gradient_uid):  # pylint: disable=protected-access
      yield
  else:
    yield


def _IsPartitionedCall(op):
  return op.type == "PartitionedCall" or op.type == "StatefulPartitionedCall"


def _IsFunction(graph):
  return (isinstance(graph, _function.FuncGraph) or
          isinstance(graph, constant_op._FuncGraph))  # pylint: disable=protected-access


def _NonEagerInputs(op, xs):
  if _IsFunction(op.graph):  # pylint: disable=protected-access
    inputs = []
    for t in op.inputs:
      if t not in xs:
        t = _MaybeCaptured(t)
        # Skip captured eager inputs.
        if isinstance(t, ops.EagerTensor): continue
      inputs.append(t)
    return inputs
  else:
    return op.inputs


def _Consumers(t, func_graphs):
  consumers = t.consumers()
  for func in func_graphs:
    for input_t, placeholder in _Captures(func).items():
      if input_t == t:
        consumers.extend(_Consumers(placeholder, func_graphs))
  return consumers


def gradients(ys,
              xs,
              grad_ys=None,
              name="gradients",
              colocate_gradients_with_ops=False,
              gate_gradients=False,
              aggregation_method=None,
              stop_gradients=None):
  with ops.get_default_graph()._mutation_lock():  # pylint: disable=protected-access
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

  # If src_graph is a _FuncGraph (i.e. a function body), gather it and all
  # ancestor graphs. This is necessary for correctly handling captured values.
  func_graphs = []
  curr_graph = src_graph
  while _IsFunction(curr_graph):
    func_graphs.append(curr_graph)
    if isinstance(curr_graph, _function.FuncGraph):
      curr_graph = curr_graph.outer_graph
    else:
      assert isinstance(curr_graph, constant_op._FuncGraph)  # pylint: disable=protected-access
      curr_graph = curr_graph._outer_graph  # pylint: disable=protected-access

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
    # Get a uid for this call to gradients that can be used to help
    # cluster ops for compilation.
    gradient_uid = ops.get_default_graph().unique_name("uid")
    ys = ops.convert_n_to_tensor_or_indexed_slices(ys, name="y")
    xs = [
        x.handle if resource_variable_ops.is_resource_variable(x) else x
        for x in xs
    ]
    xs = ops.internal_convert_n_to_tensor_or_indexed_slices(
        xs, name="x", as_ref=True)
    grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops,
                             gradient_uid)

    # The approach we take here is as follows: Create a list of all ops in the
    # subgraph between the ys and xs.  Visit these ops in reverse order of ids
    # to ensure that when we visit an op the gradients w.r.t its outputs have
    # been collected.  Then aggregate these gradients if needed, call the op's
    # gradient function, and add the generated gradients to the gradients for
    # its input.

    # Initialize the pending count for ops in the connected subgraph from ys
    # to the xs.
    to_ops = [t.op for t in ys]
    from_ops = [t.op for t in xs]
    stop_gradient_ops = [t.op for t in stop_gradients]
    reachable_to_ops, pending_count, loop_state = _PendingCount(
        to_ops, from_ops, colocate_gradients_with_ops, func_graphs, xs)

    # Iterate over the collected ops.
    #
    # grads: op => list of gradients received on each output endpoint of the
    # op.  The gradients for each endpoint are initially collected as a list.
    # When it is time to call the op's gradient function, for each endpoint we
    # aggregate the list of received gradients into a Add() Operation if there
    # is more than one.
    grads = {}

    # Add the initial gradients for the ys.
    for y, grad_y in zip(ys, grad_ys):
      _SetGrad(grads, y, grad_y)

    # Initialize queue with to_ops.
    queue = collections.deque()
    # Add the ops in 'to_ops' into the queue.
    to_ops_set = set()
    for op in to_ops:
      # 'ready' handles the case where one output gradient relies on
      # another output's gradient.
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
      # generate gradient subgraph for op.
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
        # pylint: disable=protected-access
        is_func_call = (
            src_graph._is_function(op.type) or is_partitioned_call)
        # pylint: enable=protected-access
        has_out_grads = any(isinstance(g, ops.Tensor) or g for g in out_grads)
        if has_out_grads and (op not in stop_ops):
          if is_func_call:
            if is_partitioned_call:
              func_call = src_graph._get_function(  # pylint: disable=protected-access
                  compat.as_bytes(op.get_attr("f").name))
            else:
              func_call = src_graph._get_function(op.type)  # pylint: disable=protected-access
            # Note that __defun is not set if the graph is
            # imported. If it's set, we prefer to access the original
            # defun.
            func_call = getattr(op, "__defun", func_call)
            grad_fn = func_call.python_grad_func
          
        if loop_state:
          loop_state.EnterGradWhileContext(op, before=False)

        # NOTE(skyewm): We don't support computing gradients wrt a loop variable
        # unless it's within the context of a single iteration (i.e. the
        # gradient is wrt to the loop parameter in the body function, not wrt or
        # through the initial value). This means if we're in a while loop
        # context, we should never see a switch node from this context.
        # pylint: disable=protected-access
        if (control_flow_util.IsSwitch(op) and
            op._control_flow_context is not None and
            op._control_flow_context.IsWhileContext() and
            op._control_flow_context ==
            ops.get_default_graph()._get_control_flow_context()):
          _RaiseNoGradWrtInitialLoopValError(op, from_ops, xs)
        # pylint: enable=protected-access

        if (grad_fn or is_func_call) and has_out_grads:
          # NOTE: If _AggregatedGrads didn't compute a value for the i'th
          # output, it means that the cost does not depend on output[i],
          # therefore dC/doutput[i] is 0.
          for i, out_grad in enumerate(out_grads):
            if (not isinstance(out_grad, ops.Tensor) and not out_grad) and (
                (not grad_fn and is_func_call) or _IsTrainable(op.outputs[i])):
              # Only trainable outputs or outputs for a function call that
              # will use SymbolicGradient get a zero gradient. Gradient
              # functions should ignore the gradient for other outputs.
              # TODO(apassos) gradients of resource handles might be an
              # issue here because of zeros.
              if loop_state:
                out_grads[i] = loop_state.ZerosLike(op, i)
              else:
                out_grads[i] = control_flow_ops.ZerosLikeOutsideLoop(op, i)
          with ops.name_scope(op.name + "_grad"):
            # pylint: disable=protected-access
            with src_graph._original_op(op):
              # pylint: enable=protected-access
              if grad_fn:
                # If grad_fn was found, do not use SymbolicGradient even for
                # functions.
                in_grads = _MaybeCompile(grad_scope, op, func_call,
                                         lambda: grad_fn(op, *out_grads))
              else:
                # For function call ops, we add a 'SymbolicGradient'
                # node to the graph to compute gradients.
                in_grads = _MaybeCompile(grad_scope, op, func_call,
                                         lambda: _SymGrad(op, out_grads))
              in_grads = _AsList(in_grads)
              _VerifyGeneratedGradients(in_grads, op)
              if gate_gradients and len([x for x in in_grads
                                         if x is not None]) > 1:
                with ops.device(None):
                  with ops._colocate_with_for_gradient(  # pylint: disable=protected-access
                      None,
                      gradient_uid,
                      ignore_existing=True):
                    in_grads = control_flow_ops.tuple(in_grads)
          _LogOpGradients(op, out_grads, in_grads)
        else:
          # If no grad_fn is defined or none of out_grads is available,
          # just propagate a list of None backwards.
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

      # Update pending count for the inputs of op and enqueue ready ops.
      _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state,
                                    xs)

  if loop_state:
    loop_state.PostProcessing()
  return [_GetGrad(grads, x) for x in xs]


def _HasAnyNotNoneGrads(grads, op):
  out_grads = _GetGrads(grads, op)
  for out_grad in out_grads:
    if isinstance(out_grad, (ops.Tensor, ops.IndexedSlices)):
      return True
    if out_grad and isinstance(out_grad, collections.Sequence):
      if any([g is not None for g in out_grad]):
        return True
  return False


def _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state,
                                  xs):
  for x in _NonEagerInputs(op, xs):
    pending_count[x.op] -= 1
    ready = (pending_count[x.op] == 0)
    if loop_state and not ready:
      ready = pending_count[x.op] > 0 and control_flow_util.IsLoopSwitch(x.op)
    if ready:
      if control_flow_util.IsLoopExit(x.op):
        # if x is an exit without real gradient, defer processing them.
        grad_state = loop_state.GetGradState(x.op, before=False)
        grad_state.deferred_exits.append(x)
        grad_state.pending_exits_count -= 1
        if grad_state.pending_exits_count == 0:
          # We now have all the exits so process them.
          has_not_none_grad = False
          for y in grad_state.deferred_exits:
            if _HasAnyNotNoneGrads(grads, y.op):
              has_not_none_grad = True
              queue.append(y.op)
            else:
              grad_state.unused_exits.append(y)
          if has_not_none_grad:
            # For an unused exit, if it has trainable outputs, backprop
            # a zero gradient. Otherwise, just ignore it.
            for y in grad_state.unused_exits:
              if _IsTrainable(y):
                _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
              queue.append(y.op)
          else:
            # All exits are "unused" so use None as gradient.
            for y in grad_state.unused_exits:
              queue.append(y.op)
      else:
        queue.append(x.op)


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


def _GetGrad(grads, t):
  op = t.op
  op_grads = grads.get(op)
  if not op_grads:
    return None
  t_grad = op_grads[t.value_index]
  assert not isinstance(
      t_grad, list), ("gradients list should have been aggregated by now.")
  return t_grad

def _GetGrads(grads, op):
  if op in grads:
    return grads[op]
  else:
    return [[] for _ in xrange(len(op.outputs))]

@tf_export("AggregationMethod")
class AggregationMethod(object):
  ADD_N = 0
  DEFAULT = ADD_N
  # The following are experimental and may not be supported in future releases.
  EXPERIMENTAL_TREE = 1
  EXPERIMENTAL_ACCUMULATE_N = 2

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
    # Aggregate multiple gradients, and convert [] to None.
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
        logging.vlog(2, "  _AggregatedGrads %d x %s using %s", len(out_grad),
                     tensor_shape, used)
      else:
        out_grads[i] = _AggregateIndexedSlicesGradients(out_grad)
    else:
      out_grads[i] = None
  return out_grads


