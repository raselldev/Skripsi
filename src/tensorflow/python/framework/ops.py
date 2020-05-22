from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import re
import sys
import threading
import contextlib
import numpy as np
import six
from six.moves import xrange  
#from tensorflow.python.framework import device as pydev
from tensorflow.python.util import function_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_util
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import errors_impl as errors
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.util import tf_stack
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python import pywrap_tensorflow_internal as pywrap_tensorflow
from tensorflow.python.util import compat
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import traceable_stack
from tensorflow.python.util import lock_util
from tensorflow.python import context
from tensorflow.python.util import deprecation
from tensorflow.python.util import decorator_utils
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util import tf_contextlib
from tensorflow.python import pywrap_tensorflow_internal as c_api
from tensorflow.python.pywrap_tensorflow_internal import TFE_Py_UID as uid
from tensorflow.python.util.tf_export import tf_export

_USE_C_API = True
_USE_C_SHAPES = True
_TENSOR_LIKE_TYPES = tuple()
_VALID_OP_NAME_REGEX = re.compile("^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$")
_VALID_SCOPE_NAME_REGEX = re.compile("^[A-Za-z0-9_.\\-/]*$")
_MUTATION_LOCK_GROUP = 0
_SESSION_RUN_LOCK_GROUP = 1


class _UserDeviceSpec(object):
  def __init__(self, device_name_or_function):
    self._device_name_or_function = device_name_or_function

    self.display_name = str(self._device_name_or_function)
    if callable(self._device_name_or_function):
      dev_func = self._device_name_or_function
      func_name = function_utils.get_func_name(dev_func)
      func_code = function_utils.get_func_code(dev_func)
      if func_code:
        fname = func_code.co_filename
        lineno = func_code.co_firstlineno
      else:
        fname = "unknown"
        lineno = -1
      self.display_name = "%s<%s, %d>" % (func_name, fname, lineno)

    self.function = self._device_name_or_function

class Tensor():
  OVERLOADABLE_OPERATORS = {
      # Binary.
      "__add__",
      "__radd__",
      "__sub__",
      "__rsub__",
      "__mul__",
      "__rmul__",
      "__div__",
      "__rdiv__",
      "__truediv__",
      "__rtruediv__",
      "__floordiv__",
      "__rfloordiv__",
      "__mod__",
      "__rmod__",
      "__lt__",
      "__le__",
      "__gt__",
      "__ge__",
      "__and__",
      "__rand__",
      "__or__",
      "__ror__",
      "__xor__",
      "__rxor__",
      "__getitem__",
      "__pow__",
      "__rpow__",
      # Unary.
      "__invert__",
      "__neg__",
      "__abs__",
      "__matmul__",
      "__rmatmul__"
  }

  def __init__(self, op, value_index, dtype):
    self._op = op
    self._value_index = value_index
    self._dtype = dtypes.as_dtype(dtype)
    self._shape_val = None
    self._consumers = []
    self._id = uid()

  @property
  def op(self):
    return self._op

  @property
  def dtype(self):
    return self._dtype

  @property
  def graph(self):
    return self._op.graph

  @property
  def name(self):
    if not self._op.name:
      raise ValueError("Operation was not named: %s" % self._op)
    return "%s:%d" % (self._op.name, self._value_index)

  @property
  def device(self):
    return self._op.device

  @property
  def shape(self):
    if self._shape_val is None:
      if _USE_C_SHAPES:
        self._shape_val = self._c_api_shape()
      else:
        
        need_shapes = self._get_input_ops_without_shapes(self.op)
        need_shapes.sort(key=lambda op: op._id)
        for op in need_shapes:
          set_shape_and_handle_data_for_outputs(op)
    return self._shape_val

  def _get_input_ops_without_shapes(self, target_op):
    result = []
    stack = [self._op]
    visited = set()
    while stack:
      op = stack.pop()
      if op in visited: continue
      result.append(op)
      stack.extend(t.op for t in op.inputs if t._shape_val is None)
      visited.add(op)
    return result

  def _c_api_shape(self):
    c_graph = self._op._graph._c_graph
    shape_vector, unknown_shape = c_api.TF_GraphGetTensorShapeHelper(
        c_graph, self._as_tf_output())
    if unknown_shape:
      return tensor_shape.unknown_shape()
    else:
      shape_vector = [None if d == -1 else d for d in shape_vector]
      return tensor_shape.TensorShape(shape_vector)

  @property
  def _shape(self):
    logging.warning("Tensor._shape is private, use Tensor.shape "
                    "instead. Tensor._shape will eventually be removed.")
    return self.shape

  @_shape.setter
  def _shape(self, value):
    raise ValueError(
        "Tensor._shape cannot be assigned, use Tensor.set_shape instead.")

  def __iter__(self):
    if not context.executing_eagerly():
      raise TypeError(
          "Tensor objects are only iterable when eager execution is "
          "enabled. To iterate over this tensor use tf.map_fn.")
    shape = self._shape_tuple()
    if shape is None:
      raise TypeError("Cannot iterate over a tensor with unknown shape.")
    if not shape:
      raise TypeError("Cannot iterate over a scalar tensor.")
    if shape[0] is None:
      raise TypeError(
          "Cannot iterate over a tensor with unknown first dimension.")
    for i in xrange(shape[0]):
      yield self[i]

  def _shape_as_list(self):
    if self.shape.ndims is not None:
      return [dim.value for dim in self.shape.dims]
    else:
      return None

  def _shape_tuple(self):
    shape = self._shape_as_list()
    if shape is None:
      return None
    return tuple(shape)

  def _rank(self):
    return self.shape.ndims

  def get_shape(self):
    return self.shape

  def set_shape(self, shape):
    if _USE_C_SHAPES:
      # Reset cached shape.
      self._shape_val = None
    else:
      self._shape_val = self.shape.merge_with(shape)

    # Update C shape even if _USE_C_SHAPES = False, since we still want
    # set_shape to be reflected in the C API graph for when we run it.
    if not isinstance(shape, tensor_shape.TensorShape):
      shape = tensor_shape.TensorShape(shape)
    dim_list = []
    if shape.dims is None:
      unknown_shape = True
    else:
      unknown_shape = False
      for dim in shape.dims:
        if dim.value is None:
          dim_list.append(-1)
        else:
          dim_list.append(dim.value)
    try:
      c_api.TF_GraphSetTensorShape_wrapper(
          self._op._graph._c_graph,
          self._as_tf_output(),
          dim_list,
          unknown_shape)
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(str(e))

  @property
  def value_index(self):
    return self._value_index

  def consumers(self):
    consumer_names = c_api.TF_OperationOutputConsumers_wrapper(
        self._as_tf_output())
  
    return [
        self.graph._get_operation_by_name_unsafe(name)
        for name in consumer_names
    ]
    

  def _as_node_def_input(self):
    if not self._op.name:
      raise ValueError("Operation was not named: %s" % self._op)
    if self._value_index == 0:
      return self._op.name
    else:
      return "%s:%d" % (self._op.name, self._value_index)

  def _as_tf_output(self):
  
    return tf_output(self.op._c_op, self.value_index)
    
  __array_priority__ = 100

  @staticmethod
  def _override_operator(operator, func):
    _override_helper(Tensor, operator, func)
  
class IndexedSlices():

  def __init__(self, values, indices, dense_shape=None):
    _get_graph_from_inputs([values, indices, dense_shape])
    self._values = values
    self._indices = indices
    self._dense_shape = dense_shape

  @property
  def values(self):
    return self._values

  @property
  def indices(self):
    return self._indices

  @property
  def dense_shape(self):
    return self._dense_shape

  @property
  def name(self):
    return self.values.name

  @property
  def device(self):
    return self.values.device

  @property
  def op(self):
    return self.values.op

  @property
  def dtype(self):
    return self.values.dtype

  @property
  def graph(self):
    return self._values.graph

  def __str__(self):
    return "IndexedSlices(indices=%s, values=%s%s)" % (
        self._indices, self._values, (", dense_shape=%s" % self._dense_shape)
        if self._dense_shape is not None else "")

  def __neg__(self):
    return IndexedSlices(-self.values, self.indices, self.dense_shape)

class Operation(object):

  def __init__(self,
               node_def,
               g,
               inputs=None,
               output_types=None,
               control_inputs=None,
               input_types=None,
               original_op=None,
               op_def=None):

    if isinstance(node_def, node_def_pb2.NodeDef):
      if node_def.ByteSize() >= (1 << 31) or node_def.ByteSize() < 0:
        raise ValueError(
            "Cannot create a tensor proto whose content is larger than 2GB.")
      if not _VALID_OP_NAME_REGEX.match(node_def.name):
        raise ValueError("'%s' is not a valid node name" % node_def.name)
      c_op = None
    elif type(node_def).__name__ == "SwigPyObject":
      assert inputs is None
      assert output_types is None
      assert control_inputs is None
      assert input_types is None
      assert original_op is None
      assert op_def is None
      c_op = node_def
    else:
      raise TypeError("node_def needs to be a NodeDef: %s" % node_def)

    if not isinstance(g, Graph):
      raise TypeError("g needs to be a Graph: %s" % g)
    self._graph = g

    if inputs is None:
      inputs = []
    elif not isinstance(inputs, list):
      raise TypeError("inputs needs to be a list of Tensors: %s" % inputs)
    for a in inputs:
      if not isinstance(a, Tensor):
        raise TypeError("input needs to be a Tensor: %s" % a)
    if input_types is None:
      input_types = [i.dtype.base_dtype for i in inputs]
    else:
      if not all(
          x.is_compatible_with(i.dtype)
          for i, x in zip(inputs, input_types)):
        raise TypeError("In op '%s', input types (%s) are not compatible "
                        "with expected types (%s)" %
                        (node_def.name, [i.dtype for i in inputs],
                         input_types))

    # Build the list of control inputs.
    control_input_ops = []
    if control_inputs:
      for c in control_inputs:
        control_op = None
        if isinstance(c, Operation):
          control_op = c
        elif isinstance(c, (Tensor, IndexedSlices)):
          control_op = c.op
        else:
          raise TypeError("Control input must be an Operation, "
                          "a Tensor, or IndexedSlices: %s" % c)
        control_input_ops.append(control_op)

    # This will be set by self.inputs.
    self._inputs_val = None

  
    self._id_value = self._graph._next_id()
    self._original_op = original_op

    self._device_code_locations = None
    # Dict mapping op name to file and line information for op colocation
    # context managers.
    self._colocation_code_locations = None
    self._control_flow_context = self.graph._get_control_flow_context()
    

    # Initialize self._c_op.
    if c_op:
      self._c_op = c_op
    else:
      if op_def is None:
        op_def = self._graph._get_op_def(node_def.op)
      # TODO(skyewm): op_def_library.apply_op() flattens the incoming inputs.
      # Refactor so we don't have to do this here.
      grouped_inputs = self._reconstruct_sequence_inputs(
          op_def, inputs, node_def.attr)
      self._c_op = _create_c_op(self._graph, node_def, grouped_inputs,
                                control_input_ops)

    # Initialize self._outputs.
    num_outputs = c_api.TF_OperationNumOutputs(self._c_op)
    output_types = [
        c_api.TF_OperationOutputType(tf_output(self._c_op, i))
        for i in range(num_outputs)]
    self._outputs = [
        Tensor(self, i, output_type)
        for i, output_type in enumerate(output_types)
    ]

    self._graph._add_op(self)

    if not c_op:
      self._control_flow_post_processing()

  def _control_flow_post_processing(self):
    for input_tensor in self.inputs:
      control_flow_util.CheckInputFromValidContext(self, input_tensor.op)
    if self._control_flow_context is not None:
      self._control_flow_context.AddOp(self)

  def _reconstruct_sequence_inputs(self, op_def, inputs, attrs):
    grouped_inputs = []
    i = 0
    for input_arg in op_def.input_arg:
      if input_arg.number_attr:
        input_len = attrs[input_arg.number_attr].i
        is_sequence = True
      elif input_arg.type_list_attr:
        input_len = len(attrs[input_arg.type_list_attr].list.type)
        is_sequence = True
      else:
        input_len = 1
        is_sequence = False

      if is_sequence:
        grouped_inputs.append(inputs[i:i + input_len])
      else:
        grouped_inputs.append(inputs[i])
      i += input_len

    assert i == len(inputs)
    return grouped_inputs

  def colocation_groups(self):
    default_colocation_group = [
        compat.as_bytes("loc:@%s" % self.name)
    ]
    try:
      class_attr = self.get_attr("_class")
    except ValueError:
      # This op has no explicit colocation group, so it is itself its
      # own root of a colocation group.
      return default_colocation_group

    attr_groups = [
        class_name for class_name in class_attr
        if class_name.startswith(b"loc:@")
    ]

    # If there are no colocation groups in the explicit _class field,
    # return the default colocation group.
    return attr_groups if attr_groups else default_colocation_group

  def values(self):
    return tuple(self.outputs)

  def _get_control_flow_context(self):
    return self._control_flow_context

  def _set_control_flow_context(self, ctx):
    self._control_flow_context = ctx

  @property
  def name(self):
    return c_api.TF_OperationName(self._c_op)

  @property
  def _id(self):
    return self._id_value

  @property
  def device(self):
    return c_api.TF_OperationDevice(self._c_op)

  @property
  def _device_assignments(self):
    return self._device_code_locations or []

  @property
  def _colocation_dict(self):
    locations_dict = self._colocation_code_locations or {}
    return locations_dict.copy()

  @property
  def _output_types(self):
    num_outputs = c_api.TF_OperationNumOutputs(self._c_op)
    output_types = [
        c_api.TF_OperationOutputType(self._tf_output(i))
        for i in xrange(num_outputs)
    ]
    if output_types:
      assert isinstance(output_types[0], int)
    return output_types

  def _tf_output(self, output_idx):
    tf_output = c_api.TF_Output()
    tf_output.oper = self._c_op
    tf_output.index = output_idx
    return tf_output

  def _tf_input(self, input_idx):
    tf_input = c_api.TF_Input()
    tf_input.oper = self._c_op
    tf_input.index = input_idx
    return tf_input

  def _set_device(self, device):  
    c_api.SetRequestedDevice(
        self._graph._c_graph,
        self._c_op,
        compat.as_str(_device_string(device)))

  def _update_input(self, index, tensor):
    if not isinstance(tensor, Tensor):
      raise TypeError("tensor must be a Tensor: %s" % tensor)
    _assert_same_graph(self, tensor)

    # Make sure output shapes are already computed for this op in case we create
    # a cycle (we cannot compute shapes for cycles). Usually shapes are computed
    # lazily upon request.
    if not _USE_C_SHAPES:
      set_shape_and_handle_data_for_outputs(self)

    # Reset cached inputs.
    self._inputs_val = None
    c_api.UpdateEdge(
        self._graph._c_graph,
        tensor._as_tf_output(),
        self._tf_input(index))

  def _add_control_inputs(self, ops):
    for op in ops:
      if not isinstance(op, Operation):
        raise TypeError("op must be an Operation: %s" % op)
      c_api.AddControlInput(self._graph._c_graph, self._c_op, op._c_op)

  def _add_control_input(self, op):
    if not isinstance(op, Operation):
      raise TypeError("op must be an Operation: %s" % op)
    c_api.AddControlInput(self._graph._c_graph, self._c_op, op._c_op)

  def _remove_all_control_inputs(self):
    c_api.RemoveAllControlInputs(self._graph._c_graph, self._c_op)

  @property
  def outputs(self):
    return self._outputs



  class _InputList(object):

    def __init__(self, inputs):
      self._inputs = inputs

    def __iter__(self):
      return iter(self._inputs)

    def __len__(self):
      return len(self._inputs)

    def __bool__(self):
      return bool(self._inputs)

    __nonzero__ = __bool__

    def __getitem__(self, i):
      return self._inputs[i]



  @property
  def inputs(self):
    if self._inputs_val is None:
      tf_outputs = c_api.GetOperationInputs(self._c_op)
    
      retval = [
          self.graph._get_tensor_by_tf_output(tf_output)
          for tf_output in tf_outputs
      ]
      
      self._inputs_val = Operation._InputList(retval)
    return self._inputs_val

  @property
  def _inputs(self):
    logging.warning("Operation._inputs is private, use Operation.inputs "
                    "instead. Operation._inputs will eventually be removed.")
    return self.inputs

  @_inputs.setter
  def _inputs(self, value):
    raise ValueError("Cannot assign _inputs")

  @property
  def _input_types(self):
    num_inputs = c_api.TF_OperationNumInputs(self._c_op)
    input_types = [
        dtypes.as_dtype(c_api.TF_OperationInputType(self._tf_input(i)))
        for i in xrange(num_inputs)
    ]
    return input_types


  @property
  def control_inputs(self):
    control_c_ops = c_api.TF_OperationGetControlInputs_wrapper(self._c_op)
  
    return [
        self.graph._get_operation_by_name_unsafe(
            c_api.TF_OperationName(c_op)) for c_op in control_c_ops
    ]
    

  @property
  def _control_outputs(self):
    control_c_ops = c_api.TF_OperationGetControlOutputs_wrapper(self._c_op)
  
    return [
        self.graph._get_operation_by_name_unsafe(
            c_api.TF_OperationName(c_op)) for c_op in control_c_ops
    ]
    

  @property
  def _control_inputs(self):
    logging.warning("Operation._control_inputs is private, use "
                    "Operation.control_inputs instead. "
                    "Operation._control_inputs will eventually be removed.")
    return self.control_inputs

  @_control_inputs.setter
  def _control_inputs(self, value):
    logging.warning("Operation._control_inputs is private, use "
                    "Operation.control_inputs instead. "
                    "Operation._control_inputs will eventually be removed.")
    value = copy.copy(value)
    self._remove_all_control_inputs()
    self._add_control_inputs(value)

  @property
  def type(self):
    return c_api.TF_OperationOpType(self._c_op)

  @property
  def graph(self):
    return self._graph

  @property
  def node_def(self):
    
    with tf_buffer() as buf:
      c_api.TF_OperationToNodeDef(self._c_op, buf)
      data = c_api.TF_GetBuffer(buf)
    node_def = node_def_pb2.NodeDef()
    node_def.ParseFromString(compat.as_bytes(data))
    return node_def

  @property
  def _node_def(self):
    logging.warning("Operation._node_def is private, use Operation.node_def "
                    "instead. Operation._node_def will eventually be removed.")
    return self.node_def

  @property
  def op_def(self):
    return self._graph._get_op_def(self.type)

  @property
  def _op_def(self):
    logging.warning("Operation._op_def is private, use Operation.op_def "
                    "instead. Operation._op_def will eventually be removed.")
    return self.op_def

  @property
  def traceback(self):
    return tf_stack.convert_stack(self._traceback)

  @property
  def traceback_with_start_lines(self):
    return tf_stack.convert_stack(self._traceback,
                                  include_func_start_lineno=True)

  def _set_attr(self, attr_name, attr_value):
    buf = c_api.TF_NewBufferFromString(
        compat.as_bytes(attr_value.SerializeToString()))
    try:
    
      c_api.SetAttr(self._graph._c_graph, self._c_op, attr_name, buf)
      
    finally:
      c_api.TF_DeleteBuffer(buf)

  def get_attr(self, name):
    fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]
    try:
      with tf_buffer() as buf:
        c_api.TF_OperationGetAttrValueProto(self._c_op, name, buf)
        data = c_api.TF_GetBuffer(buf)
    except errors.InvalidArgumentError as e:
      raise ValueError(str(e))
    x = attr_value_pb2.AttrValue()
    x.ParseFromString(data)

    if not x.WhichOneof("value"):
      return []
    if x.HasField("list"):
      for f in fields:
        if getattr(x.list, f):
          if f == "type":
            return [dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
          else:
            return list(getattr(x.list, f))
      return []
    else:
      for f in fields:
        if x.HasField(f):
          if f == "type":
            return dtypes.as_dtype(getattr(x, f))
          else:
            return getattr(x, f)
      assert False, "Unsupported field type in " + str(x)

class RegisterGradient(object):
  def __init__(self, op_type):
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string")
    self._op_type = op_type

  def __call__(self, f):
    return f

class RegisterShape(object):
  def __init__(self, op_type):
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string")
    self._op_type = op_type

  def __call__(self, f):
    if f is None:
      assert _call_cpp_shape_fn
    return f

class RegisterStatistics(object):
  def __init__(self, op_type, statistic_type):
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string.")
    if "," in op_type:
      raise TypeError("op_type must not contain a comma.")
    self._op_type = op_type
    if not isinstance(statistic_type, six.string_types):
      raise TypeError("statistic_type must be a string.")
    if "," in statistic_type:
      raise TypeError("statistic_type must not contain a comma.")
    self._statistic_type = statistic_type

  def __call__(self, f):
    return f

class Graph(object):
  def __init__(self):
    self._lock = threading.RLock()
    self._group_lock = lock_util.GroupLock(num_groups=2)
    self._nodes_by_id = dict()  
    self._next_id_counter = 0  
    self._nodes_by_name = dict()  
    self._version = 0  
    self._names_in_use = {}
    self._stack_state_is_thread_local = False
    self._thread_local = threading.local()
    self._graph_device_function_stack = traceable_stack.TraceableStack()
    self._default_original_op = None
    self._control_flow_context = None
    self._graph_control_dependencies_stack = []
    self._collections = {}
    self._seed = None
    self._attr_scope_map = {}
    self._op_to_kernel_label_map = {}
    self._gradient_override_map = {}
    self._finalized = False
    self._functions = collections.OrderedDict()
    self._building_function = False
    self._graph_colocation_stack = traceable_stack.TraceableStack()
    self._unfeedable_tensors = set()
    self._unfetchable_ops = set()
    self._handle_feeders = {}
    self._handle_readers = {}
    self._handle_movers = {}
    self._handle_deleters = {}
    self._graph_key = "grap-key-%d/" % (uid(),)
    # A string with the last reduction method passed to
    # losses.compute_weighted_loss(), or None.
    self._last_loss_reduction = None
    self._container = ""
    self._registered_ops = op_def_registry.get_registered_ops()

    # TODO(skyewm): fold as much of the above as possible into the C
    # implementation
    #self._scoped_c_graph = ScopedTFGraph()
    self._scoped_c_graph = c_api.TF_NewGraph()
    # The C API requires all ops to have shape functions. Disable this
    # requirement (many custom ops do not have shape functions, and we don't
    # want to break these existing cases).
    c_api.SetRequireShapeInferenceFns(self._c_graph, False)

  # Note: this method is private because the API of tf.Graph() is public and
  # frozen, and this functionality is still not ready for public visibility.
  @tf_contextlib.contextmanager
  def _variable_creator_scope(self, creator):
    # This step makes a copy of the existing stack, and it also initializes
    # self._thread_local._variable_creator_stack if it doesn't exist yet.
    old = list(self._variable_creator_stack)
    self._thread_local._variable_creator_stack.append(creator)
    try:
      yield
    finally:
      self._thread_local._variable_creator_stack = old

  # Note: this method is private because the API of tf.Graph() is public and
  # frozen, and this functionality is still not ready for public visibility.
  @property
  def _variable_creator_stack(self):
    if not hasattr(self._thread_local, "_variable_creator_stack"):
      self._thread_local._variable_creator_stack = []
    return list(self._thread_local._variable_creator_stack)

  @_variable_creator_stack.setter
  def _variable_creator_stack(self, variable_creator_stack):
    self._thread_local._variable_creator_stack = variable_creator_stack

  def _check_not_finalized(self):
    if self._finalized:
      raise RuntimeError("Graph is finalized and cannot be modified.")

  def _add_op(self, op):
    self._check_not_finalized()
    if not isinstance(op, (Tensor, Operation)):
      raise TypeError("op must be a Tensor or Operation: %s" % op)
    with self._lock:
    
      if op._id in self._nodes_by_id:
        raise ValueError("cannot add an op with id %d as it already "
                         "exists in the graph" % op._id)
      if op.name in self._nodes_by_name:
        raise ValueError("cannot add op with name %s as that name "
                         "is already used" % op.name)
      self._nodes_by_id[op._id] = op
      self._nodes_by_name[op.name] = op
      self._version = max(self._version, op._id)
      

  @property
  def _c_graph(self):
    if self._scoped_c_graph:
      return self._scoped_c_graph
    return None

  @property
  def version(self):
    if self._finalized:
      return self._version

    with self._lock:
      return self._version

  @property
  def graph_def_versions(self):
    with tf_buffer() as buf:
      c_api.TF_GraphVersions(self._c_graph, buf)
      data = c_api.TF_GetBuffer(buf)
    version_def = versions_pb2.VersionDef()
    version_def.ParseFromString(compat.as_bytes(data))
    return version_def

  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, seed):
    self._seed = seed

  @property
  def finalized(self):
    return self._finalized

  def finalize(self):
    self._finalized = True

  def _unsafe_unfinalize(self):
    self._finalized = False

  def _get_control_flow_context(self):
    return self._control_flow_context

  def _set_control_flow_context(self, ctx):
    self._control_flow_context = ctx

  def _copy_functions_to_graph_def(self, graph_def, starting_bytesize):
    bytesize = starting_bytesize
    for f in self._functions.values():
      bytesize += f.definition.ByteSize()
      if bytesize >= (1 << 31) or bytesize < 0:
        raise ValueError("GraphDef cannot be larger than 2GB.")
      graph_def.library.function.extend([f.definition])
      if f.grad_func_name:
        grad_def = function_pb2.GradientDef()
        grad_def.function_name = f.name
        grad_def.gradient_func = f.grad_func_name
        graph_def.library.gradient.extend([grad_def])

  def _as_graph_def(self, from_version=None, add_shapes=False):    
    with self._lock:
      with tf_buffer() as buf:
        c_api.TF_GraphToGraphDef(self._c_graph, buf)
        data = c_api.TF_GetBuffer(buf)
      graph = graph_pb2.GraphDef()
      graph.ParseFromString(compat.as_bytes(data))
      # Strip the experimental library field iff it's empty.
      if not graph.library.function:
        graph.ClearField("library")

      if add_shapes:
        for node in graph.node:
          op = self._nodes_by_name[node.name]
          if op.outputs:
            node.attr["_output_shapes"].list.shape.extend(
                [output.get_shape().as_proto() for output in op.outputs])
    return graph, self._version

  def as_graph_def(self, from_version=None, add_shapes=False):
    result, _ = self._as_graph_def(from_version, add_shapes)
    return result

  def _is_function(self, name):
    return compat.as_str(name) in self._functions

  def _get_function(self, name):
    return self._functions.get(compat.as_str(name), None)

  def _add_function(self, function):
    name = function.name
    if (function.grad_func_name is not None) and (function.python_grad_func is
                                                  not None):
      raise ValueError("Gradient defined twice for function %s" % name)
    if not function._c_func:
      serialized = function.definition.SerializeToString()
      c_func = c_api.TF_FunctionImportFunctionDef(serialized)
      function._c_func = ScopedTFFunction(c_func)
    gradient = (function._grad_func._c_func.func if function._grad_func
                else None)
    c_api.TF_GraphCopyFunction(self._c_graph, function._c_func.func, gradient)
    

    self._functions[compat.as_str(name)] = function
    if self._graph_def_versions.min_consumer < 12:
      self._graph_def_versions.min_consumer = 12

  @property
  def building_function(self):
    return self._building_function

  # Helper functions to create operations.
  @deprecated_args(None,
                   "Shapes are always computed; don't use the compute_shapes "
                   "as it has no effect.", "compute_shapes")
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
    
    del compute_shapes

    self._check_not_finalized()
    for idx, a in enumerate(inputs):
      if not isinstance(a, Tensor):
        raise TypeError("Input #%d is not a tensor: %s" % (idx, a))
    if name is None:
      name = op_type
    # If a names ends with a '/' it is a "name scope" and we use it as-is,
    # after removing the trailing '/'.
    if name and name[-1] == "/":
      name = _name_from_scope_name(name)
    else:
      name = self.unique_name(name)

    node_def = _NodeDef(op_type, name, device=None, attrs=attrs)

    input_ops = set([t.op for t in inputs])
    control_inputs = self._control_dependencies_for_inputs(input_ops)
    # _create_op_helper mutates the new Operation. `_mutation_lock` ensures a
    # Session.run call cannot occur between creating and mutating the op.
    with self._mutation_lock():
      ret = Operation(
          node_def,
          self,
          inputs=inputs,
          output_types=dtypes,
          control_inputs=control_inputs,
          input_types=input_types,
          original_op=self._default_original_op,
          op_def=op_def)
      self._create_op_helper(ret, compute_device=compute_device)
    return ret

  def _create_op_from_tf_operation(self, c_op, compute_device=True):
    self._check_not_finalized()
    ret = Operation(c_op, self)
    name_key = ret.name.lower()
    if name_key not in self._names_in_use:
      self._names_in_use[name_key] = 1
    self._create_op_helper(ret, compute_device=compute_device)
    return ret

  def _make_colocation_conflict_message(self, op, colocation_op):
    op_info = error_interpolation.compute_field_dict(op)
    coloc_op_info = error_interpolation.compute_field_dict(colocation_op)
    msg = ("Tried to colocate op '{op_name}'{op_loc} having device '{op_dev}' "
           "with op '{coloc_op_name}'{coloc_op_loc} which had an incompatible "
           "device '{coloc_op_dev}'.\n\n{op_summary}\n\n{coloc_op_summary}"
           .format(op_name=op.name,
                   op_loc=op_info["defined_at"],
                   op_dev=op.device,
                   op_summary=op_info["devs_and_colocs"],
                   coloc_op_name=colocation_op.name,
                   coloc_op_loc=coloc_op_info["defined_at"],
                   coloc_op_dev=colocation_op.device,
                   coloc_op_summary=coloc_op_info["devs_and_colocs"]))
    return msg

  def _create_op_helper(self, op, compute_device=True):
    for key, value in self._attr_scope_map.items():
      try:
        op.get_attr(key)
      except ValueError:
        if callable(value):
          value = value(op.node_def)
          if not isinstance(value, (type(None), attr_value_pb2.AttrValue)):
            raise TypeError(
                "Callable for scope map key '%s' must return either None or "
                "an AttrValue protocol buffer; but it returned: %s" % (key,
                                                                       value))
        if value:
          op._set_attr(key, value)

    # Apply a kernel label if one has been specified for this op type.
    try:
      kernel_label = self._op_to_kernel_label_map[op.type]
      op._set_attr("_kernel",
                   attr_value_pb2.AttrValue(s=compat.as_bytes(kernel_label)))
    except KeyError:
      pass

    # Apply the overriding op type for gradients if one has been specified for
    # this op type.
    try:
      mapped_op_type = self._gradient_override_map[op.type]
      op._set_attr("_gradient_op_type",
                   attr_value_pb2.AttrValue(s=compat.as_bytes(mapped_op_type)))
    except KeyError:
      pass

    self._record_op_seen_by_control_dependencies(op)

    

    if self._colocation_stack:
      all_colocation_groups = []
      for colocation_op in self._colocation_stack.peek_objs():
        all_colocation_groups.extend(colocation_op.colocation_groups())
        if colocation_op.device:
          if (op.device and pydev.canonical_name(op.device) !=
              pydev.canonical_name(colocation_op.device)):
            msg = self._make_colocation_conflict_message(op, colocation_op)
            logging.warning(msg)
          else:
            op._set_device(colocation_op.device)

      all_colocation_groups = sorted(set(all_colocation_groups))
    
      op._set_attr("_class", attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(s=all_colocation_groups)))
      

    # Sets "container" attribute if
    # (1) self._container is not None
    # (2) "is_stateful" is set in OpDef
    # (3) "container" attribute is in OpDef
    # (4) "container" attribute is None
    if self._container and op.op_def.is_stateful:
      try:
        container_attr = op.get_attr("container")
      except ValueError:
        # "container" attribute is not in OpDef
        pass
      else:
        if not container_attr:
          op._set_attr("container", attr_value_pb2.AttrValue(
              s=compat.as_bytes(self._container)))

  def _add_new_tf_operations(self, compute_devices=True):
    new_ops = [
        self._create_op_from_tf_operation(c_op, compute_device=compute_devices)
        for c_op in new_tf_operations(self)
    ]

  
    for op in new_ops:
      if not _USE_C_SHAPES:
        _set_shape_and_handle_data_for_outputs_c_api(op)
      new_control_inputs = self._control_dependencies_for_inputs(op.inputs)
      op._add_control_inputs(new_control_inputs)
      op._control_flow_post_processing()
    return new_ops

  def as_graph_element(self, obj, allow_tensor=True, allow_operation=True):
    if self._finalized:
      return self._as_graph_element_locked(obj, allow_tensor, allow_operation)

    with self._lock:
      return self._as_graph_element_locked(obj, allow_tensor, allow_operation)

  def _as_graph_element_locked(self, obj, allow_tensor, allow_operation):
    if allow_tensor and allow_operation:
      types_str = "Tensor or Operation"
    elif allow_tensor:
      types_str = "Tensor"
    elif allow_operation:
      types_str = "Operation"
    else:
      raise ValueError("allow_tensor and allow_operation can't both be False.")

    temp_obj = _as_graph_element(obj)
    if temp_obj is not None:
      obj = temp_obj

    if isinstance(obj, compat.bytes_or_text_types):
      name = compat.as_str(obj)

      if ":" in name and allow_tensor:
        try:
          op_name, out_n = name.split(":")
          out_n = int(out_n)
        except:
          raise ValueError("The name %s looks a like a Tensor name, but is "
                           "not a valid one. Tensor names must be of the "
                           "form \"<op_name>:<output_index>\"." % repr(name))
        if op_name in self._nodes_by_name:
          op = self._nodes_by_name[op_name]
        else:
          raise KeyError("The name %s refers to a Tensor which does not "
                         "exist. The operation, %s, does not exist in the "
                         "graph." % (repr(name), repr(op_name)))
        try:
          return op.outputs[out_n]
        except:
          raise KeyError("The name %s refers to a Tensor which does not "
                         "exist. The operation, %s, exists but only has "
                         "%s outputs." % (repr(name), repr(op_name),
                                          len(op.outputs)))

      elif ":" in name and not allow_tensor:
        # Looks like a Tensor name but can't be a Tensor.
        raise ValueError("Name %s appears to refer to a Tensor, not a %s." %
                         (repr(name), types_str))

      elif ":" not in name and allow_operation:
        # Looks like an Operation name and can be an Operation.
        if name not in self._nodes_by_name:
          raise KeyError("The name %s refers to an Operation not in the "
                         "graph." % repr(name))
        return self._nodes_by_name[name]

      elif ":" not in name and not allow_operation:
        # Looks like an Operation name but can't be an Operation.
        if name in self._nodes_by_name:
          # Yep, it's an Operation name
          err_msg = ("The name %s refers to an Operation, not a %s." %
                     (repr(name), types_str))
        else:
          err_msg = ("The name %s looks like an (invalid) Operation name, "
                     "not a %s." % (repr(name), types_str))
        err_msg += (" Tensor names must be of the form "
                    "\"<op_name>:<output_index>\".")
        raise ValueError(err_msg)

    elif isinstance(obj, Tensor) and allow_tensor:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Tensor %s is not an element of this graph." % obj)
      return obj
    elif isinstance(obj, Operation) and allow_operation:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Operation %s is not an element of this graph." % obj)
      return obj
    else:
      # We give up!
      raise TypeError("Can not convert a %s into a %s." % (type(obj).__name__,
                                                           types_str))

  def get_operations(self):
    if self._finalized:
      return list(self._nodes_by_id.values())

    with self._lock:
      return list(self._nodes_by_id.values())

  def get_operation_by_name(self, name):
    if not isinstance(name, six.string_types):
      raise TypeError("Operation names are strings (or similar), not %s." %
                      type(name).__name__)
    return self.as_graph_element(name, allow_tensor=False, allow_operation=True)

  def _get_operation_by_name_unsafe(self, name):
    if self._finalized:
      return self._nodes_by_name[name]

    with self._lock:
      return self._nodes_by_name[name]

  def _get_operation_by_tf_operation(self, tf_oper):
    op_name = c_api.TF_OperationName(tf_oper)
    return self._get_operation_by_name_unsafe(op_name)

  def get_tensor_by_name(self, name):
    if not isinstance(name, six.string_types):
      raise TypeError("Tensor names are strings (or similar), not %s." %
                      type(name).__name__)
    return self.as_graph_element(name, allow_tensor=True, allow_operation=False)

  def _get_tensor_by_tf_output(self, tf_output):
    op = self._get_operation_by_tf_operation(tf_output.oper)
    return op.outputs[tf_output.index]

  def _next_id(self):
    self._check_not_finalized()
    with self._lock:
      self._next_id_counter += 1
      return self._next_id_counter

  @property
  def _last_id(self):
    return self._next_id_counter

  def _get_op_def(self, type):  
    with tf_buffer() as buf:
    
      c_api.TF_GraphGetOpDef(self._c_graph, compat.as_bytes(type), buf)
      
      data = c_api.TF_GetBuffer(buf)
    op_def = op_def_pb2.OpDef()
    op_def.ParseFromString(compat.as_bytes(data))
    return op_def

  def as_default(self):
    return _default_graph_stack.get_controller(self)

  @property
  def collections(self):
    return list(self._collections)

  def add_to_collection(self, name, value):
    self._check_not_finalized()
    with self._lock:
      if name not in self._collections:
        self._collections[name] = [value]
      else:
        self._collections[name].append(value)

  def add_to_collections(self, names, value):
    names = (names,) if isinstance(names, six.string_types) else set(names)
    for name in names:
      self.add_to_collection(name, value)

  def get_collection_ref(self, name):
    with self._lock:
      coll_list = self._collections.get(name, None)
      if coll_list is None:
        coll_list = []
        self._collections[name] = coll_list
      return coll_list

  def get_collection(self, name, scope=None):
    with self._lock:
      collection = self._collections.get(name, None)
      if collection is None:
        return []
      if scope is None:
        return list(collection)
      else:
        c = []
        regex = re.compile(scope)
        for item in collection:
          if hasattr(item, "name") and regex.match(item.name):
            c.append(item)
        return c

  def get_all_collection_keys(self):
    with self._lock:
      return [x for x in self._collections if isinstance(x, six.string_types)]

  def clear_collection(self, name):
    self._check_not_finalized()
    with self._lock:
      if name in self._collections:
        del self._collections[name]

  @property
  def _name_stack(self):
    # This may be called from a thread where name_stack doesn't yet exist.
    if not hasattr(self._thread_local, "_name_stack"):
      self._thread_local._name_stack = ""
    return self._thread_local._name_stack

  @_name_stack.setter
  def _name_stack(self, name_stack):
    self._thread_local._name_stack = name_stack

  @tf_contextlib.contextmanager
  def name_scope(self, name):
    if name:
      if isinstance(name, compat.bytes_or_text_types):
        name = compat.as_str(name)

      if self._name_stack:
        # Scopes created in a nested scope may have initial characters
        # that are illegal as the initial character of an op name
        # (viz. '-', '\', '/', and '_').
        if not _VALID_SCOPE_NAME_REGEX.match(name):
          raise ValueError("'%s' is not a valid scope name" % name)
      else:
        # Scopes created in the root must match the more restrictive
        # op name regex, which constrains the initial character.
        if not _VALID_OP_NAME_REGEX.match(name):
          raise ValueError("'%s' is not a valid scope name" % name)
    old_stack = self._name_stack
    if not name:  # Both for name=None and name="" we re-set to empty scope.
      new_stack = None
    elif name[-1] == "/":
      new_stack = _name_from_scope_name(name)
    else:
      new_stack = self.unique_name(name)
    self._name_stack = new_stack
    try:
      yield "" if new_stack is None else new_stack + "/"
    finally:
      self._name_stack = old_stack

  def unique_name(self, name, mark_as_used=True):
    if self._name_stack:
      name = self._name_stack + "/" + name

    # For the sake of checking for names in use, we treat names as case
    # insensitive (e.g. foo = Foo).
    name_key = name.lower()
    i = self._names_in_use.get(name_key, 0)
    # Increment the number for "name_key".
    if mark_as_used:
      self._names_in_use[name_key] = i + 1
    if i > 0:
      base_name_key = name_key
      # Make sure the composed name key is not already used.
      while name_key in self._names_in_use:
        name_key = "%s_%d" % (base_name_key, i)
        i += 1
      # Mark the composed name_key as used in case someone wants
      # to call unique_name("name_1").
      if mark_as_used:
        self._names_in_use[name_key] = 1

      # Return the new name with the original capitalization of the given name.
      name = "%s_%d" % (name, i-1)
    return name

  def get_name_scope(self):
    return self._name_stack

  @tf_contextlib.contextmanager
  def _colocate_with_for_gradient(self, op, gradient_uid,
                                  ignore_existing=False):
    with self.colocate_with(op, ignore_existing):
      if gradient_uid is not None and self._control_flow_context is not None:
        self._control_flow_context.EnterGradientColocation(op, gradient_uid)
        try:
          yield
        finally:
          self._control_flow_context.ExitGradientColocation(op, gradient_uid)
      else:
        yield

  @tf_contextlib.contextmanager
  def colocate_with(self, op, ignore_existing=False):
    if op is None and not ignore_existing:
      raise ValueError("Trying to reset colocation (op is None) but "
                       "ignore_existing is not True")

    if op is not None and not isinstance(op, Operation):
      # We always want to colocate with the reference op.
      op = internal_convert_to_tensor_or_indexed_slices(op, as_ref=True).op

    device_fn_tmp = self._device_function_stack
    self._device_function_stack = traceable_stack.TraceableStack()

    if ignore_existing:
      current_stack = self._colocation_stack
      self._colocation_stack = traceable_stack.TraceableStack()

    if op is not None:
      self._colocation_stack.push_obj(op, offset=4)

    try:
      yield
    finally:
      # Restore device function stack
      self._device_function_stack = device_fn_tmp
      if op is not None:
        self._colocation_stack.pop_obj()

      # Reset the colocation stack if requested.
      if ignore_existing:
        self._colocation_stack = current_stack

  def _add_device_to_stack(self, device_name_or_function, offset=0):
    total_offset = 1 + offset
    spec = _UserDeviceSpec(device_name_or_function)
    self._device_function_stack.push_obj(spec, offset=total_offset)
    return spec

  @tf_contextlib.contextmanager
  def device(self, device_name_or_function):
    self._add_device_to_stack(device_name_or_function, offset=2)
    try:
      yield
    finally:
      self._device_function_stack.pop_obj()

  def _apply_device_functions(self, op):
    for device_spec in self._device_function_stack.peek_objs():
      if device_spec.function is None:
        break
      op._set_device(device_spec.function(op))
    op._device_code_locations = self._snapshot_device_function_stack_metadata()
    
  @tf_contextlib.contextmanager
  def container(self, container_name):
    original_container = self._container
    self._container = container_name
    try:
      yield self._container
    finally:
      self._container = original_container

  class _ControlDependenciesController(object):
    def __init__(self, graph, control_inputs):
      self._graph = graph
      if control_inputs is None:
        self._control_inputs_val = []
        self._new_stack = True
      else:
        self._control_inputs_val = control_inputs
        self._new_stack = False
      self._seen_nodes = set()
      self._old_stack = None
      self._old_control_flow_context = None

    def __enter__(self):
      if self._new_stack:
        # Clear the control_dependencies graph.
        self._old_stack = self._graph._control_dependencies_stack
        self._graph._control_dependencies_stack = []
        # Clear the control_flow_context too.
        self._old_control_flow_context = self._graph._get_control_flow_context()
        self._graph._set_control_flow_context(None)
      self._graph._push_control_dependencies_controller(self)

    def __exit__(self, unused_type, unused_value, unused_traceback):
      self._graph._pop_control_dependencies_controller(self)
      if self._new_stack:
        self._graph._control_dependencies_stack = self._old_stack
        self._graph._set_control_flow_context(self._old_control_flow_context)

    @property
    def control_inputs(self):
      return self._control_inputs_val

    def add_op(self, op):
      self._seen_nodes.add(op)

    def op_in_group(self, op):
      return op in self._seen_nodes

  def _push_control_dependencies_controller(self, controller):
    self._control_dependencies_stack.append(controller)

  def _pop_control_dependencies_controller(self, controller):
    assert self._control_dependencies_stack[-1] is controller
    self._control_dependencies_stack.pop()

  def _current_control_dependencies(self):
    ret = set()
    for controller in self._control_dependencies_stack:
      for op in controller.control_inputs:
        ret.add(op)
    return ret

  def _control_dependencies_for_inputs(self, input_ops):
    ret = []
    for controller in self._control_dependencies_stack:
      # If any of the input_ops already depends on the inputs from controller,
      # we say that the new op is dominated (by that input), and we therefore
      # do not need to add control dependencies for this controller's inputs.
      dominated = False
      for op in input_ops:
        if controller.op_in_group(op):
          dominated = True
          break
      if not dominated:
        # Don't add a control input if we already have a data dependency on i.
        # NOTE(mrry): We do not currently track transitive data dependencies,
        #   so we may add redundant control inputs.
        ret.extend([c for c in controller.control_inputs if c not in input_ops])
    return ret

  def _record_op_seen_by_control_dependencies(self, op):
    for controller in self._control_dependencies_stack:
      controller.add_op(op)

  def control_dependencies(self, control_inputs):
    if control_inputs is None:
      return self._ControlDependenciesController(self, None)
    # First convert the inputs to ops, and deduplicate them.
    # NOTE(mrry): Other than deduplication, we do not currently track direct
    #   or indirect dependencies between control_inputs, which may result in
    #   redundant control inputs.
    control_ops = []
    current = self._current_control_dependencies()
    for c in control_inputs:
      if isinstance(c, IndexedSlices):
        c = c.op
      c = self.as_graph_element(c)
      if isinstance(c, Tensor):
        c = c.op
      elif not isinstance(c, Operation):
        raise TypeError("Control input must be Operation or Tensor: %s" % c)
      if c not in current:
        control_ops.append(c)
        current.add(c)
    return self._ControlDependenciesController(self, control_ops)
  
  @tf_contextlib.contextmanager
  def _attr_scope(self, attr_map):
    if not isinstance(attr_map, dict):
      raise TypeError("attr_map must be a dictionary mapping "
                      "strings to AttrValue protocol buffers")
    # The saved_attrs dictionary stores any currently-set labels that
    # will be overridden by this context manager.
    saved_attrs = {}
    # Install the given attribute
    for name, attr in attr_map.items():
      if not (isinstance(name, six.string_types) and
              (isinstance(attr, (type(None), attr_value_pb2.AttrValue)) or
               callable(attr))):
        raise TypeError("attr_map must be a dictionary mapping "
                        "strings to AttrValue protocol buffers or "
                        "callables that emit AttrValue protocol buffers")
      try:
        saved_attrs[name] = self._attr_scope_map[name]
      except KeyError:
        pass
      if attr is None:
        del self._attr_scope_map[name]
      else:
        self._attr_scope_map[name] = attr
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the attributes set for this context, and restore any saved
      # attributes.
      for name, attr in attr_map.items():
        try:
          self._attr_scope_map[name] = saved_attrs[name]
        except KeyError:
          del self._attr_scope_map[name]

  @tf_contextlib.contextmanager
  def _kernel_label_map(self, op_to_kernel_label_map):
    if not isinstance(op_to_kernel_label_map, dict):
      raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_labels dictionary stores any currently-set labels that
    # will be overridden by this context manager.
    saved_labels = {}
    # Install the given label
    for op_type, label in op_to_kernel_label_map.items():
      if not (isinstance(op_type, six.string_types) and
              isinstance(label, six.string_types)):
        raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_labels[op_type] = self._op_to_kernel_label_map[op_type]
      except KeyError:
        pass
      self._op_to_kernel_label_map[op_type] = label
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, label in op_to_kernel_label_map.items():
        try:
          self._op_to_kernel_label_map[op_type] = saved_labels[op_type]
        except KeyError:
          del self._op_to_kernel_label_map[op_type]
  
  @tf_contextlib.contextmanager
  def gradient_override_map(self, op_type_map):
    if not isinstance(op_type_map, dict):
      raise TypeError("op_type_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_mappings dictionary stores any currently-set mappings that
    # will be overridden by this context manager.
    saved_mappings = {}
    # Install the given label
    for op_type, mapped_op_type in op_type_map.items():
      if not (isinstance(op_type, six.string_types) and
              isinstance(mapped_op_type, six.string_types)):
        raise TypeError("op_type_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_mappings[op_type] = self._gradient_override_map[op_type]
      except KeyError:
        pass
      self._gradient_override_map[op_type] = mapped_op_type
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, mapped_op_type in op_type_map.items():
        try:
          self._gradient_override_map[op_type] = saved_mappings[op_type]
        except KeyError:
          del self._gradient_override_map[op_type]

  def prevent_feeding(self, tensor):
    self._unfeedable_tensors.add(tensor)

  def is_feedable(self, tensor):
    return tensor not in self._unfeedable_tensors

  def prevent_fetching(self, op):
    self._unfetchable_ops.add(op)

  def is_fetchable(self, tensor_or_op):
    if isinstance(tensor_or_op, Tensor):
      return tensor_or_op.op not in self._unfetchable_ops
    else:
      return tensor_or_op not in self._unfetchable_ops

  def switch_to_thread_local(self):
    if not self._stack_state_is_thread_local:
      self._stack_state_is_thread_local = True

  @property
  def _device_function_stack(self):
    if self._stack_state_is_thread_local:
      # This may be called from a thread where device_function_stack doesn't yet
      # exist.
    
      if not hasattr(self._thread_local, "_device_function_stack"):
        stack_copy_for_this_thread = self._graph_device_function_stack.copy()
        self._thread_local._device_function_stack = stack_copy_for_this_thread
      return self._thread_local._device_function_stack
      
    else:
      return self._graph_device_function_stack

  @property
  def _device_functions_outer_to_inner(self):
    user_device_specs = self._device_function_stack.peek_objs()
    device_functions = [spec.function for spec in user_device_specs]
    device_functions_outer_to_inner = list(reversed(device_functions))
    return device_functions_outer_to_inner

  def _snapshot_device_function_stack_metadata(self):
    traceable_objects = self._device_function_stack.peek_traceable_objs()
    snapshot = []
    for obj in traceable_objects:
      obj_copy = obj.copy_metadata()
      obj_copy.obj = obj.obj.display_name
      snapshot.append(obj_copy)
    return snapshot

  @_device_function_stack.setter
  def _device_function_stack(self, device_function_stack):
    if self._stack_state_is_thread_local:
    
      self._thread_local._device_function_stack = device_function_stack
      
    else:
      self._graph_device_function_stack = device_function_stack

  @property
  def _colocation_stack(self):
    if self._stack_state_is_thread_local:
      # This may be called from a thread where colocation_stack doesn't yet
      # exist.
    
      if not hasattr(self._thread_local, "_colocation_stack"):
        stack_copy_for_this_thread = self._graph_colocation_stack.copy()
        self._thread_local._colocation_stack = stack_copy_for_this_thread
      return self._thread_local._colocation_stack
      
    else:
      return self._graph_colocation_stack

  def _snapshot_colocation_stack_metadata(self):
    traceable_objects = self._colocation_stack.peek_traceable_objs()
    return {obj.obj.name: obj.copy_metadata() for obj in traceable_objects}

  @_colocation_stack.setter
  def _colocation_stack(self, colocation_stack):
    if self._stack_state_is_thread_local:
    
      self._thread_local._colocation_stack = colocation_stack
      
    else:
      self._graph_colocation_stack = colocation_stack

  @property
  def _control_dependencies_stack(self):
    if self._stack_state_is_thread_local:
      # This may be called from a thread where control_dependencies_stack
      # doesn't yet exist.
      if not hasattr(self._thread_local, "_control_dependencies_stack"):
        self._thread_local._control_dependencies_stack = (
            self._graph_control_dependencies_stack[:])
      return self._thread_local._control_dependencies_stack
    else:
      return self._graph_control_dependencies_stack

  @_control_dependencies_stack.setter
  def _control_dependencies_stack(self, control_dependencies):
    if self._stack_state_is_thread_local:
      self._thread_local._control_dependencies_stack = control_dependencies
    else:
      self._graph_control_dependencies_stack = control_dependencies

  @property
  def _distribution_strategy_stack(self):
    if not hasattr(self._thread_local, "_distribution_strategy_stack"):
      self._thread_local._distribution_strategy_stack = []
    return self._thread_local._distribution_strategy_stack

  @_distribution_strategy_stack.setter
  def _distribution_strategy_stack(self, _distribution_strategy_stack):
    self._thread_local._distribution_strategy_stack = (
        _distribution_strategy_stack)

  def _mutation_lock(self):
    return self._group_lock.group(_MUTATION_LOCK_GROUP)

  def _session_run_lock(self):
    return self._group_lock.group(_SESSION_RUN_LOCK_GROUP)

class _DefaultStack(threading.local):
  def __init__(self):
    super(_DefaultStack, self).__init__()
    self._enforce_nesting = True
    self.stack = []

  def get_default(self):
    return self.stack[-1] if len(self.stack) >= 1 else None

  def reset(self):
    self.stack = []

  def is_cleared(self):
    return not self.stack

  @property
  def enforce_nesting(self):
    return self._enforce_nesting

  @enforce_nesting.setter
  def enforce_nesting(self, value):
    self._enforce_nesting = value

  @tf_contextlib.contextmanager
  def get_controller(self, default):
    self.stack.append(default)
    try:
      yield default
    finally:
      # stack may be empty if reset() was called
      if self.stack:
        if self._enforce_nesting:
          if self.stack[-1] is not default:
            raise AssertionError(
                "Nesting violated for default stack of %s objects" %
                type(default))
          self.stack.pop()
        else:
          self.stack.remove(default)

class _DefaultGraphStack(_DefaultStack):

  def __init__(self):
    super(_DefaultGraphStack, self).__init__()
    self._global_default_graph = None

  def get_default(self):
    ret = super(_DefaultGraphStack, self).get_default()
    if ret is None:
      ret = self._GetGlobalDefaultGraph()
    return ret

  def _GetGlobalDefaultGraph(self):
    if self._global_default_graph is None:
      # TODO(mrry): Perhaps log that the default graph is being used, or set
      #   provide some other feedback to prevent confusion when a mixture of
      #   the global default graph and an explicit graph are combined in the
      #   same process.
      self._global_default_graph = Graph()
    return self._global_default_graph

  def reset(self):
    super(_DefaultGraphStack, self).reset()
    self._global_default_graph = None

  @tf_contextlib.contextmanager
  def get_controller(self, default):
    context.context().context_switches.push(
        default.building_function, default.as_default)
    try:
      with super(_DefaultGraphStack, self).get_controller(
          default) as g, context.graph_mode():
        yield g
    finally:
      # If an exception is raised here it may be hiding a related exception in
      # the try-block (just above).
      context.context().context_switches.pop()

class GraphKeys(object):
  # Key to collect Variable objects that are global (shared across machines).
  # Default collection for all variables, except local ones.
  GLOBAL_VARIABLES = "variables"
  # Key to collect local variables that are local to the machine and are not
  # saved/restored.
  LOCAL_VARIABLES = "local_variables"
  # Key to collect local variables which are used to accumulate interal state
  # to be used in tf.metrics.*.
  METRIC_VARIABLES = "metric_variables"
  # Key to collect model variables defined by layers.
  MODEL_VARIABLES = "model_variables"
  # Key to collect Variable objects that will be trained by the
  # optimizers.
  TRAINABLE_VARIABLES = "trainable_variables"
  # Key to collect summaries.
  SUMMARIES = "summaries"
  # Key to collect QueueRunners.
  QUEUE_RUNNERS = "queue_runners"
  # Key to collect table initializers.
  TABLE_INITIALIZERS = "table_initializer"
  # Key to collect asset filepaths. An asset represents an external resource
  # like a vocabulary file.
  ASSET_FILEPATHS = "asset_filepaths"
  # Key to collect Variable objects that keep moving averages.
  MOVING_AVERAGE_VARIABLES = "moving_average_variables"
  # Key to collect regularization losses at graph construction.
  REGULARIZATION_LOSSES = "regularization_losses"
  # Key to collect concatenated sharded variables.
  CONCATENATED_VARIABLES = "concatenated_variables"
  # Key to collect savers.
  SAVERS = "savers"
  # Key to collect weights
  WEIGHTS = "weights"
  # Key to collect biases
  BIASES = "biases"
  # Key to collect activations
  ACTIVATIONS = "activations"
  # Key to collect update_ops
  UPDATE_OPS = "update_ops"
  # Key to collect losses
  LOSSES = "losses"
  # Key to collect BaseSaverBuilder.SaveableObject instances for checkpointing.
  SAVEABLE_OBJECTS = "saveable_objects"
  # Key to collect all shared resources used by the graph which need to be
  # initialized once per cluster.
  RESOURCES = "resources"
  # Key to collect all shared resources used in this graph which need to be
  # initialized once per session.
  LOCAL_RESOURCES = "local_resources"
  # Trainable resource-style variables.
  TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables"

  # Key to indicate various ops.
  INIT_OP = "init_op"
  LOCAL_INIT_OP = "local_init_op"
  READY_OP = "ready_op"
  READY_FOR_LOCAL_INIT_OP = "ready_for_local_init_op"
  SUMMARY_OP = "summary_op"
  GLOBAL_STEP = "global_step"

  EVAL_STEP = "eval_step"
  TRAIN_OP = "train_op"

  # Key for control flow context.
  COND_CONTEXT = "cond_context"
  WHILE_CONTEXT = "while_context"

  # Used to store v2 summary names.
  _SUMMARY_COLLECTION = "_SUMMARY_V2"

  # List of all collections that keep track of variables.
  _VARIABLE_COLLECTIONS = [
      GLOBAL_VARIABLES,
      LOCAL_VARIABLES,
      METRIC_VARIABLES,
      MODEL_VARIABLES,
      TRAINABLE_VARIABLES,
      MOVING_AVERAGE_VARIABLES,
      CONCATENATED_VARIABLES,
      TRAINABLE_RESOURCE_VARIABLES,
  ]

  _STREAMING_MODEL_PORTS = "streaming_model_ports"

class name_scope(object):
  def name(self):
    return self._name

  def __init__(self, name, default_name=None, values=None):
    self._name = default_name if name is None else name
    self._default_name = default_name
    self._values = values
    self._ctx = context.context()
    self._in_eager_mode = self._ctx.executing_eagerly()

  def __enter__(self):
    if self._in_eager_mode:
      self._old_name = self._ctx.scope_name
      if not self._name:
        scope_name = ""
      else:
        cache_key = self._name, self._old_name, self._default_name
        if cache_key in name_scope_cache:
          self._ctx.scope_name = name_scope_cache[cache_key]
          return self._ctx.scope_name
        elif self._name[-1] == "/":
          # A trailing slash breaks out of nested name scopes, indicating a
          # fully specified scope name, for compatibility with Graph.name_scope.
          scope_name = self._name
        else:
          name_with_trailing_slash = self._name + "/"
          scope_name = (
              self._old_name + name_with_trailing_slash
              if self._old_name else name_with_trailing_slash)
        name_scope_cache[cache_key] = scope_name
      self._ctx.scope_name = scope_name
      return scope_name
    else:
      if self._name is None and self._values is not None:
        raise ValueError(
            "At least one of name (%s) and default_name (%s) must be provided."
            % (self._name, self._default_name))
      if self._values is None:
        self._values = []
      g = _get_graph_from_inputs(self._values)
      self._g_manager = g.as_default()
      self._g_manager.__enter__()
      try:
        self._name_scope = g.name_scope(self._name)
        return self._name_scope.__enter__()
      except:
        self._g_manager.__exit__(*sys.exc_info())
        raise

  def __exit__(self, type_arg, value_arg, traceback_arg):
    self._name_scope.__exit__(type_arg, value_arg, traceback_arg)
    self._g_manager.__exit__(type_arg, value_arg, traceback_arg)
    return False

@tf_contextlib.contextmanager
def tf_buffer(data=None):
  if data:
    buf = c_api.TF_NewBufferFromString(compat.as_bytes(data))
  else:
    buf = c_api.TF_NewBuffer()
  try:
    yield buf
  finally:
    c_api.TF_DeleteBuffer(buf)

def tf_output(c_op, index):
  ret = c_api.TF_Output()
  ret.oper = c_op
  ret.index = index
  return ret

@contextlib.contextmanager
def stop_recording():
  try:
    c_api.TFE_Py_TapeSetStopOnThread()
    yield
  finally:
    c_api.TFE_Py_TapeSetRestartOnThread()

def _override_helper(clazz_object, operator, func):
  existing = getattr(clazz_object, operator, None)
  if operator not in Tensor.OVERLOADABLE_OPERATORS:
    raise ValueError("Overriding %s is disallowed" % operator)
  setattr(clazz_object, operator, func)

def _as_graph_element(obj):
  conv_fn = getattr(obj, "_as_graph_element", None)

def register_dense_tensor_like_type(tensor_type):
  try:
    if not isinstance(tensor_type.name, property):
      raise TypeError("Type %s does not define a `name` property" %
                      tensor_type.__name__)
  except AttributeError:
    raise TypeError("Type %s does not define a `name` property" %
                    tensor_type.__name__)
  try:
    if not isinstance(tensor_type.dtype, property):
      raise TypeError("Type %s does not define a `dtype` property" %
                      tensor_type.__name__)
  except AttributeError:
    raise TypeError("Type %s does not define a `dtype` property" %
                    tensor_type.__name__)
  # We expect this list to be small, so choose quadratic complexity
  # for registration, so that we have a tuple that can be used for
  # more efficient `isinstance` checks later.
  global _TENSOR_LIKE_TYPES
  _TENSOR_LIKE_TYPES = tuple(list(_TENSOR_LIKE_TYPES) + [tensor_type])

def _TensorTensorConversionFunction(t, dtype=None, name=None, as_ref=False):
  _ = name, as_ref
  if dtype and not dtype.is_compatible_with(t.dtype):
    raise ValueError(
        "Tensor conversion requested dtype %s for Tensor with dtype %s: %r" %
        (dtype.name, t.dtype.name, str(t)))
  return t

def convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None):
    return internal_convert_to_tensor(
      value=value,
      dtype=dtype,
      name=name,
      preferred_dtype=preferred_dtype,
      as_ref=False)

def internal_convert_to_tensor(value,
                               dtype=None,
                               name=None,
                               as_ref=False,
                               preferred_dtype=None,
                               ctx=None):
  
  if ctx is None: ctx = context.context()
  
  if dtype is not None:
    dtype = dtypes.as_dtype(dtype)
  unwrapped_type = type(value)
  conversion_func_list = _tensor_conversion_func_cache.get(unwrapped_type, None)
  if conversion_func_list is None:
    with _tensor_conversion_func_lock:
      conversion_func_list = []
      for _, funcs_at_priority in sorted(
          _tensor_conversion_func_registry.items()):
        for base_type, conversion_func in funcs_at_priority:
          if isinstance(value, base_type):
            conversion_func_list.append((base_type, conversion_func))
      _tensor_conversion_func_cache[unwrapped_type] = conversion_func_list

  for base_type, conversion_func in conversion_func_list:
    # If dtype is None but preferred_dtype is not None, we try to
    # cast to preferred_dtype first.
    ret = None
    if dtype is None and preferred_dtype is not None:
      try:
        ret = conversion_func(
            value, dtype=preferred_dtype, name=name, as_ref=as_ref)
      except (TypeError, ValueError, errors.UnimplementedError,
              errors.InvalidArgumentError):
        # Could not coerce the conversion to use the preferred dtype.
        ret = None

      if ret is not None and ret is not NotImplemented:
        if (ret.dtype.base_dtype !=
            dtypes.as_dtype(preferred_dtype).base_dtype):
          raise TypeError("convert_to_tensor did not convert to "
                          "the preferred dtype: %s vs %s " %
                          (ret.dtype.base_dtype,
                           dtypes.as_dtype(preferred_dtype).base_dtype))

    if ret is None:
      ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)

    if ret is NotImplemented:
      continue

    if not isinstance(ret, Tensor):
      raise RuntimeError(" ")
    return ret

def internal_convert_n_to_tensor(values,
                                 dtype=None,
                                 name=None,
                                 as_ref=False,
                                 preferred_dtype=None,
                                 ctx=None):
  if not isinstance(values, collections.Sequence):
    raise TypeError("values must be a list.")
  ret = []
  if ctx is None: ctx = context.context()
  for i, value in enumerate(values):
    n = None if name is None else "%s_%d" % (name, i)
    ret.append(
        internal_convert_to_tensor(
            value,
            dtype=dtype,
            name=n,
            as_ref=as_ref,
            preferred_dtype=preferred_dtype,
            ctx=ctx))
  return ret

def convert_n_to_tensor(values, dtype=None, name=None, preferred_dtype=None):
  return internal_convert_n_to_tensor(
      values=values,
      dtype=dtype,
      name=name,
      preferred_dtype=preferred_dtype,
      as_ref=False)

def convert_to_tensor_or_indexed_slices(value, dtype=None, name=None):
  return internal_convert_to_tensor_or_indexed_slices(
      value=value, dtype=dtype, name=name, as_ref=False)

def internal_convert_to_tensor_or_indexed_slices(value,
                                                 dtype=None,
                                                 name=None,
                                                 as_ref=False):
  return internal_convert_to_tensor(
        value, dtype=dtype, name=name, as_ref=as_ref)

def internal_convert_n_to_tensor_or_indexed_slices(values,
                                                   dtype=None,
                                                   name=None,
                                                   as_ref=False):
  if not isinstance(values, collections.Sequence):
    raise TypeError("values must be a list.")
  ret = []
  for i, value in enumerate(values):
    if value is None:
      ret.append(value)
    else:
      n = None if name is None else "%s_%d" % (name, i)
      ret.append(
          internal_convert_to_tensor_or_indexed_slices(
              value, dtype=dtype, name=n, as_ref=as_ref))
  return ret

def convert_n_to_tensor_or_indexed_slices(values, dtype=None, name=None):
  return internal_convert_n_to_tensor_or_indexed_slices(
      values=values, dtype=dtype, name=name, as_ref=False)

def register_tensor_conversion_function(base_type,
                                        conversion_func,
                                        priority=100):
  global _tensor_conversion_func_cache
  with _tensor_conversion_func_lock:
    if not (isinstance(base_type, type) or
            (isinstance(base_type, tuple) and
             all(isinstance(x, type) for x in base_type))):
      raise TypeError("base_type must be a type or a tuple of types.")
    if not callable(conversion_func):
      raise TypeError("conversion_func must be callable.")

    if context._context is not None and context.executing_eagerly(
    ) and isinstance(base_type, six.integer_types + (
        float,
        np.ndarray,
    )):
      # TODO(nareshmodi): consider setting a context variable which disables the
      # fastpath instead.
      raise TypeError(
          "Cannot register conversions for numpy arrays, python number types "
          "when executing eagerly.")

    try:
      funcs_at_priority = _tensor_conversion_func_registry[priority]
    except KeyError:
      funcs_at_priority = []
      _tensor_conversion_func_registry[priority] = funcs_at_priority
    funcs_at_priority.append((base_type, conversion_func))
    _tensor_conversion_func_cache = {}

def _device_string(dev_spec):
  if isinstance(dev_spec, pydev.DeviceSpec):
    return dev_spec.to_string()
  else:
    return dev_spec

def _NodeDef(op_type, name, device=None, attrs=None):  
  node_def = node_def_pb2.NodeDef()
  node_def.op = compat.as_bytes(op_type)
  node_def.name = compat.as_bytes(name)
  if attrs is not None:
    for k, v in six.iteritems(attrs):
      node_def.attr[k].CopyFrom(v)
  if device is not None:
    if callable(device):
      node_def.device = device(node_def)
    else:
      node_def.device = _device_string(device)
  return node_def

def _create_c_op(graph, node_def, inputs, control_inputs):

  op_desc = c_api.TF_NewOperation(graph._c_graph,
                                  compat.as_str(node_def.op),
                                  compat.as_str(node_def.name))
  # Add inputs
  for op_input in inputs:
    if isinstance(op_input, (list, tuple)):
      c_api.TF_AddInputList(op_desc, [t._as_tf_output() for t in op_input])
    else:
      c_api.TF_AddInput(op_desc, op_input._as_tf_output())

  # Add control inputs
  for control_input in control_inputs:
    c_api.TF_AddControlInput(op_desc, control_input._c_op)
  

  # Add attrs
  for name, attr_value in node_def.attr.items():
    serialized = attr_value.SerializeToString()
    # TODO(skyewm): this creates and deletes a new TF_Status for every attr.
    # It might be worth creating a convenient way to re-use the same status.
    c_api.TF_SetAttrValueProto(op_desc, compat.as_str(name), serialized)

  try:
    c_op = c_api.TF_FinishOperation(op_desc)
  except errors.InvalidArgumentError as e:
    # Convert to ValueError for backwards compatibility.
    raise ValueError(str(e))

  return c_op

def NotDifferentiable(op_type):
  if not isinstance(op_type, six.string_types):
    raise TypeError("op_type must be a string")

def get_gradient_function(op):
  if not op.inputs:
    return None
  try:
    op_type = op.get_attr("_gradient_op_type")
  except ValueError:
    op_type = op.type
  return lookup(op_type)

def _set_call_cpp_shape_fn(call_cpp_shape_fn):
  global _call_cpp_shape_fn, _call_cpp_shape_fn_and_require_op
  if _call_cpp_shape_fn:
    return

  def call_without_requiring(op):
    return call_cpp_shape_fn(op, require_shape_fn=False)

  _call_cpp_shape_fn = call_without_requiring

  def call_with_requiring(op):
    return call_cpp_shape_fn(op, require_shape_fn=True)

  _call_cpp_shape_fn_and_require_op = call_with_requiring

def _set_shape_and_handle_data_for_outputs_c_api(op):
  assert not _USE_C_SHAPES
  for output in op.outputs:
    output._shape_val = output._c_api_shape()
    serialized = c_api.GetHandleShapeAndType(op._graph._c_graph,
                                             output._as_tf_output())
    if serialized:
      output._handle_data = (
          cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData
          .FromString(compat.as_bytes(serialized)))
    else:
      output._handle_data = None

def set_shape_and_handle_data_for_outputs(op):
  if _USE_C_SHAPES: return

  if op.graph._is_function(op.type):
    for output in op.outputs:
      output._shape_val = tensor_shape.unknown_shape()
    return

  try:
    shape_func = lookup(op.type)
  except LookupError:
    try:
      shape_func = lookup(op.type)
    except LookupError:
      shape_func = _call_cpp_shape_fn_and_require_op

  shapes = shape_func(op)
  if shapes is None:
    raise RuntimeError(
        "Shape function for op %s did not return any shapes" % op)
  elif isinstance(shapes, dict):
    shapes_dict = shapes
    shapes = shapes_dict["shapes"]
    handle_datas = shapes_dict["handle_data"]
    for output, handle_data in zip(op.outputs, handle_datas):
      if output._handle_data is None:
        output._handle_data = handle_data
      

  if len(op.outputs) != len(shapes):
    raise RuntimeError(
        "Shape function for op %s returned %d shapes but expected %d %s %s" %
        (op, len(shapes), len(op.outputs), shape_func.__name__, str(shapes)))
  for output, s in zip(op.outputs, shapes):
    output._shape_val = tensor_shape.unknown_shape()
    output._shape_val = output._shape_val.merge_with(s)

def get_stats_for_node_def(graph, node, statistic_type):
  try:
    stats_func = lookup(node.op + "," + statistic_type)
    result = stats_func(graph, node)
  except LookupError:
    result = OpStats(statistic_type)
  return result

def _name_from_scope_name(name):
  return name[:-1] if (name and name[-1] == "/") else name

def device(device_name_or_function):
  return get_default_graph().device(device_name_or_function)

def _colocate_with_for_gradient(op, gradient_uid, ignore_existing=False):
  if context.executing_eagerly():
    if op is not None:
      return device(op.device)
    else:
      return _NullContextmanager()
  else:
    default_graph = get_default_graph()
    
    return default_graph._colocate_with_for_gradient(
        op, gradient_uid=gradient_uid, ignore_existing=ignore_existing)

def colocate_with(op, ignore_existing=False):
  return _colocate_with_for_gradient(op, None, ignore_existing=ignore_existing)

def control_dependencies(control_inputs):
  if context.executing_eagerly():
    if control_inputs:
      # Excute any pending callables.
      for control in control_inputs:
        if callable(control):
          control()
    return _NullContextmanager()
  else:
    return get_default_graph().control_dependencies(control_inputs)

def _eval_using_default_session(tensors, feed_dict, graph, session=None):
  if session is None:
    session = get_default_session()
    if session is None:
      raise ValueError("Cannot evaluate tensor using `eval()`: No default "
                       "session is registered. Use `with "
                       "sess.as_default()` or pass an explicit session to "
                       "`eval(session=sess)`")
    if session.graph is not graph:
      raise ValueError("Cannot use the default session to evaluate tensor: "
                       "the tensor's graph is different from the session's "
                       "graph. Pass an explicit session to "
                       "`eval(session=sess)`.")
  else:
    if session.graph is not graph:
      raise ValueError("Cannot use the given session to evaluate tensor: "
                       "the tensor's graph is different from the session's "
                       "graph.")
  return session.run(tensors, feed_dict)

@tf_contextlib.contextmanager
def init_scope():
  if context.executing_eagerly():
    # Fastpath.
    with stop_recording():
      yield
  else:
    default_graph = get_default_graph()
    scope = default_graph.get_name_scope()
    if scope and scope[-1] != "/":
      scope = scope + "/"
    inner_device_stack = default_graph._device_function_stack

    outer_context = None
    if not _default_graph_stack.stack:
      if default_graph.building_function:
        raise RuntimeError("The global graph is building a function.")
      outer_context = default_graph.as_default
    else:
      for stack_entry in reversed(context.context().context_switches.stack):
        if not stack_entry.is_building_function:
          outer_context = stack_entry.enter_context_fn
          break

      if outer_context is None:
        outer_context = _default_graph_stack._GetGlobalDefaultGraph().as_default

    if outer_context is None:
      raise RuntimeError("All graphs are building functions, and no "
                         "eager context was previously active.")

    outer_graph = None
    outer_device_stack = None
    try:
      with outer_context(), name_scope(scope), control_dependencies(
          None), stop_recording():
        if not context.executing_eagerly():
          outer_graph = get_default_graph()
          outer_device_stack = outer_graph._device_function_stack
          outer_graph._device_function_stack = inner_device_stack
        yield
    finally:
      if outer_graph is not None:
        outer_graph._device_function_stack = outer_device_stack

def get_default_graph():
  return _default_graph_stack.get_default()

def get_name_scope():
  if context.executing_eagerly():
    return context.context().scope_name.rstrip("/")
  return get_default_graph().get_name_scope()

def _assert_same_graph(original_item, item):
  if original_item.graph is not item.graph:
    raise ValueError("%s must be from the same graph as %s." % (item,
                                                                original_item))

def _get_graph_from_inputs(op_input_list, graph=None):
  op_input_list = tuple(op_input_list)  # Handle generators correctly
  original_graph_element = None
  for op_input in op_input_list:
    graph_element = None
    if (isinstance(op_input, (Operation)) and
        ((not isinstance(op_input, Tensor)) or type(op_input) == Tensor)):  # pylint: disable=unidiomatic-typecheck
      graph_element = op_input
    else:
      graph_element = _as_graph_element(op_input)

    if graph_element is not None:
      if not graph:
        original_graph_element = graph_element
        graph = graph_element.graph
      elif original_graph_element is not None:
        _assert_same_graph(original_graph_element, graph_element)
      elif graph_element.graph is not graph:
        raise ValueError("%s is not from the passed-in graph." % graph_element)
  return graph or get_default_graph()

def add_to_collection(name, value):
  get_default_graph().add_to_collection(name, value)

def add_to_collections(names, value):
  get_default_graph().add_to_collections(names, value)

def get_collection_ref(key):
  return get_default_graph().get_collection_ref(key)

def get_collection(key, scope=None):
  return get_default_graph().get_collection(key, scope)


IndexedSlicesValue = collections.namedtuple(
    "IndexedSlicesValue", ["values", "indices", "dense_shape"])

EagerTensor = c_api.TFE_Py_InitEagerTensor(Tensor)
NoGradient = NotDifferentiable


_tensor_conversion_func_registry = {
    0: [(Tensor, _TensorTensorConversionFunction)]
}
_tensor_conversion_func_cache = {}
_tensor_conversion_func_lock = threading.Lock()
register_dense_tensor_like_type(Tensor)
#_gradient_registry = Registry("gradient")
#_shape_registry = Registry("shape functions")
#_default_shape_function_registry = Registry("default shape functions")

_call_cpp_shape_fn = None
_call_cpp_shape_fn_and_require_op = None
#_stats_registry = Registry("statistical functions")
_default_session_stack = _DefaultStack()
_default_graph_stack = _DefaultGraphStack()