from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow_internal as c_api
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops


def smart_cond(pred, true_fn=None, false_fn=None, name=None):
  if not callable(true_fn):
    raise TypeError("`true_fn` must be callable.")
  if not callable(false_fn):
    raise TypeError("`false_fn` must be callable.")

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
    # pylint: enable=protected-access

  else:
    raise TypeError("`pred` must be a Tensor, or a Python bool, or 1 or 0. "
                    "Found instead: %s" % pred)
  return pred_value


def smart_case(pred_fn_pairs, default=None, exclusive=False, name="smart_case"):
  return control_flow_ops._case_helper(  # pylint: disable=protected-access
      smart_cond, pred_fn_pairs, default, exclusive, name,
      allow_python_preds=True)
