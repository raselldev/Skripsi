# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""smart_cond and related utilties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from backend import pywrap_backend as c_api
from backend.python.framework import ops
from backend.python.framework import tensor_util
from backend.python.ops import control_flow_ops


def smart_cond(pred, true_fn=None, false_fn=None, name=None):
  pred_value = smart_constant_value(pred)
  if pred_value is not None:
    if pred_value:
      return true_fn()
    else:
      return false_fn()
  else:
    return control_flow_ops.cond(pred, true_fn=true_fn, false_fn=false_fn,
                                 name=name)


def smart_constant_value(pred):
  if pred in {0, 1}:  # Accept 1/0 as valid boolean values
    pred_value = bool(pred)
  elif isinstance(pred, bool):
    pred_value = pred
  elif isinstance(pred, ops.Tensor):
    pred_value = tensor_util.constant_value(pred)
    # TODO(skyewm): consider folding this into tensor_util.constant_value.
    # pylint: disable=protected-access
    if pred_value is None:
      pred_value = c_api.TF_TryEvaluateConstant_wrapper(pred.graph._c_graph,
                                                        pred._as_tf_output())
  return pred_value


