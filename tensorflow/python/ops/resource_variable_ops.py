# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ops to use variables as resources."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import variable_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables

class ResourceVariable(variables.RefVariable):
  def __init__(self,
               initial_value=None,
               trainable=True,
               collections=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               dtype=None,
               variable_def=None,
               import_scope=None,
               constraint=None):
    
    """"""
  
 
class _MixedPrecisionVariable(ResourceVariable):
  def op(self):
    return self._var.op
ops.register_tensor_conversion_function(
    variables.Variable, variables.Variable._TensorConversionFunction)

def _to_proto_fn(v, export_scope=None):
  return v.to_proto(export_scope=export_scope)

def _from_proto_fn(v, import_scope=None):
  return variables.Variable.from_proto(v, import_scope=import_scope)

def is_resource_variable(var):
  return isinstance(var, ResourceVariable) or hasattr(
      var, "_should_act_as_resource_variable")
