from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import re
import sys
import threading
import weakref

import numpy as np
import six

from tensorflow.python.framework import ops
from tensorflow.python.ops import cond_v2_impl
from tensorflow.python.ops import gradients_impl


gradients_impl._function = sys.modules[__name__]  # pylint: disable=protected-access


class FuncGraph(ops.Graph):
  """Graph representing a function body.

  Attributes:
    name: The name of the function.
    inputs: Placeholder tensors representing the inputs to this function. The
      tensors are in this FuncGraph. This represents "regular" inputs as well as
      captured inputs (i.e. the values of self.captures), with the regular
      inputs coming first.
    outputs: Tensors that will be returned by this function. The tensors are in
      this FuncGraph.
    structured_outputs: A possibly-nested python object which will be returned
      by this function. The Tensors in this structure are the same as those of
      self.outputs. Note that this structure might contain Python `None`s.
    variables: Variables that should be watched during function execution.
    outer_graph: The graph this function is defined in. May be another FuncGraph
      or the global default Graph.
    captures: Maps external tensor -> internal tensor (i.e. input placeholder).
      The entries are in the order they were captured.
    seed: The graph-level random seed.
  """

  def __init__(self, name):
    """Construct a new FuncGraph.

    The graph will inherit its graph key, collections, seed, device stack, and
    distribution strategy stack from the current context or graph.

    Args:
      name: the name of the function.
    """
    super(FuncGraph, self).__init__()

    self.name = name
    self.inputs = []
    self.outputs = []
    self.structured_outputs = None
    self._weak_variables = []
    self.outer_graph = ops.get_default_graph()
    self.captures = collections.OrderedDict()

    self._building_function = True
    # Map from resource tensor name to last op (in program order) which uses
    # this tensor. Used to enforce that execution order matches program order
    # for resource tensors.
    self._last_op_using_resource_tensor = {}

    graph = self.outer_graph

    if context.executing_eagerly():
      self.seed = context.global_seed()
      self._xla_compile = (context.context().device_spec.device_type == "TPU")
      self._add_device_to_stack(context.context().device_name)
    else:
      self.seed = graph.seed
      self._xla_compile = getattr(graph, "_xla_compile", False)
      self._device_function_stack = graph._device_function_stack.copy()  # pylint: disable=protected-access
      self._colocation_stack = graph._colocation_stack.copy()  # pylint: disable=protected-access

    # TODO(b/112165328, b/112906995): summaries depend on inheriting collections
    # from the default graph even in eager mode. It'd be nice to not have a
    # default graph with eager execution, so hopefully this will go away when we
    # remove collections.
    # pylint: disable=protected-access
    self._collections = graph._collections
    # TODO(b/112906995): distribution strategy depends on inheriting this stack
    # from the default graph even in eager mode. Maybe it should be part of the
    # eager context?
    self._distribution_strategy_stack = graph._distribution_strategy_stack
    # Inherit the graph key, since this is used for matching variables in
    # optimizers.
    self._graph_key = graph._graph_key
    # pylint: enable=protected-access

  @property
  def variables(self):
    """A list of variables accessed by this FuncGraph.

    Note that functions keep only weak references to variables. Calling the
    function after a variable it accesses has been deleted is an error.

    Yields:
      Strong references to variables accessed by this FuncGraph.
    """
    for weak_v in self._weak_variables:
      v = weak_v()
      if v is None:
        raise AssertionError(
            "Called a function referencing variables which have been deleted. "
            "This likely means that function-local variables were created and "
            "not referenced elsewhere in the program. This is generally a "
            "mistake; consider storing variables in an object attribute on "
            "first call.")
      yield v

  @variables.setter
  def variables(self, var_list):
    self._weak_variables = [weakref.ref(v) for v in var_list]

  def control_dependencies(self, control_inputs):
    # Drop control dependencies to outside of the graph. TODO(b/117109273)
    # unclear how to capture an op, not a tensor.
    if not control_inputs:
      return super(FuncGraph, self).control_dependencies(control_inputs)
    return super(FuncGraph, self).control_dependencies(
        [c for c in control_inputs
         if getattr(c, "graph", None) is self])

  def create_op(
      self,
      op_type,
      inputs,
      dtypes,
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_shapes=True,
      compute_device=True):
    """Like Graph.create_op, except handles external input tensors.

    This overload adds functionality to create_op to "capture" any external
    input tensors, i.e. tensors from the eager context or outer function graphs
    if this is a nested function. See `capture` for more information.

    Args:
      op_type: The `Operation` type to create. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.
      dtypes: A list of `DType` objects that will be the types of the tensors
        that the operation produces.
      input_types: (Optional.) A list of `DType`s that will be the types of
        the tensors that the operation consumes. By default, uses the base
        `DType` of each input in `inputs`. Operations that expect
        reference-typed inputs must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_shapes: (Optional.) Deprecated. Has no effect (shapes are always
        computed).
      compute_device: (Optional.) If True, device functions will be executed
        to compute the device property of the Operation.

    Returns:
      An `Operation` object.
    """
    # This capturing logic interacts poorly with control flow contexts which
    # want to replace inputs of ops far too late in the process. This can lead
    # the context to get confused and try to create an Enter for an Enter. We
    # can detect this here and skip the additional Enter which can confuse loop
    # validation logic.
    if op_type == "Enter" and inputs[0].op.type == "Enter":
      if inputs[0].op.get_attr("frame_name") == attrs["frame_name"].s:
        return inputs[0].op
    # Calling AddValue on the control flow contexts to force creation of the
    # backward accumulators in the original graph before we create placeholders
    # to capture the inputs.
    ctxt = ops.get_default_graph()._control_flow_context  # pylint: disable=protected-access
    for i, inp in enumerate(inputs):
      # TPU Estimator defines a control flow context with no AddValue method.
      if ctxt is not None and hasattr(ctxt, "AddValue"):
        inp = ctxt.AddValue(inp)
      inp = self.capture(inp)
      inputs[i] = inp
    return super(FuncGraph, self).create_op(
        op_type, inputs, dtypes, input_types, name, attrs, op_def,
        compute_device=compute_device)

  def capture(self, tensor, name=None):
    """Captures `tensor` if it's external to this graph.

    If `tensor` is from a different graph, returns a placeholder for it.
    `tensor` and the placeholder will appear in self.captures, and the
    placeholder will appear in self.inputs.  Multiple calls to this method with
    the same `tensor` argument will return the same placeholder. If `tensor` is
    from this graph, returns `tensor`.

    Args:
      tensor: Tensor. May be from this FuncGraph or a different graph.
      name: Optional name if a placeholder is created.

    Returns:
      Tensor from this FuncGraph.
    """
    if isinstance(tensor, ops.EagerTensor):
      if name is None:
        name = str(ops.uid())
      return self._capture_helper(tensor, name)
    if tensor.graph is not self:
      if name is None:
        name = tensor.op.name
      return self._capture_helper(tensor, name)
    return tensor

  def _capture_helper(self, tensor, name):
    captured_tensor = self.captures.get(tensor, None)
    if captured_tensor is None:
      captured_tensor = _create_substitute_placeholder(tensor, name=name,
                                                       dtype=tensor.dtype)
      self.captures[tensor] = captured_tensor
      self.inputs.append(captured_tensor)
    tape.record_operation("captured_value", [captured_tensor], [tensor],
                          lambda x: [x])
    return captured_tensor

  @property
  def external_captures(self):
    """External tensors captured by this function."""
    return list(self.captures.keys())

  @property
  def internal_captures(self):
    """Placeholders in this function corresponding captured tensors."""
    return list(self.captures.values())

