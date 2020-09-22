from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
from backend.python import pywrap_backend
from backend.python.framework import ops
from backend.core import variable_pb2
from backend.python.ops import array_ops
#from backend.python.ops import math_ops
from backend.python.ops import variables




class ResourceVariable(variables.RefVariable):
  pass

class _UnreadVariable(ResourceVariable):
  pass


class _MixedPrecisionVariable(ResourceVariable):
  pass




#ops.register_tensor_conversion_function(ResourceVariable, _dense_var_to_tensor)
ops.register_tensor_conversion_function(
    variables.Variable, variables.Variable._TensorConversionFunction)  # pylint: disable=protected-access

# pylint: disable=protected-access
#ResourceVariable._OverloadAllOperators()
#ops.register_dense_tensor_like_type(ResourceVariable)


@ops.RegisterGradient("ReadVariableOp")
def _ReadGrad(_, grad):
  """Gradient for read op."""
  return grad


@ops.RegisterGradient("ResourceGather")
def _GatherGrad(op, grad):
  """Gradient for gather op."""
  # Build appropriately shaped IndexedSlices
  handle = op.inputs[0]
  indices = op.inputs[1]
  params_shape = gen_resource_variable_ops.variable_shape(handle)
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[1:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return (ops.IndexedSlices(values, indices, params_shape), None)


def _to_proto_fn(v, export_scope=None):
  """Converts Variable and ResourceVariable to VariableDef for collections."""
  return v.to_proto(export_scope=export_scope)


def _from_proto_fn(v, import_scope=None):
  """Creates Variable or ResourceVariable from VariableDef as needed."""
  if v.is_resource:
    return ResourceVariable.from_proto(v, import_scope=import_scope)
  return variables.Variable.from_proto(v, import_scope=import_scope)




def is_resource_variable(var):
  """"Returns True if `var` is to be considered a ResourceVariable."""
  return isinstance(var, ResourceVariable) or hasattr(
      var, "_should_act_as_resource_variable")
