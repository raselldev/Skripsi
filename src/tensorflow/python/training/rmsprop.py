from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training import optimizer
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
#from tensorflow.python.training import gen_training_ops as training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("train.RMSPropOptimizer")
class RMSPropOptimizer(optimizer.Optimizer):

  def __init__(self,
               learning_rate,
               decay=0.9,
               momentum=0.0,
               epsilon=1e-10,
               use_locking=False,
               centered=False,
               name="RMSProp"):
    super(RMSPropOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._decay = decay
    self._momentum = momentum
    self._epsilon = epsilon
    self._centered = centered

    # Tensors for learning rate and momentum.  Created in _prepare.
    self._learning_rate_tensor = None
    self._decay_tensor = None
    self._momentum_tensor = None
    self._epsilon_tensor = None

  def _create_slots(self, var_list):
    for v in var_list:
      if v.get_shape().is_fully_defined():
        init_rms = init_ops.ones_initializer(dtype=v.dtype.base_dtype)
      else:
        init_rms = array_ops.ones_like(v)
      self._get_or_make_slot_with_initializer(v, init_rms, v.get_shape(),
                                              v.dtype.base_dtype, "rms",
                                              self._name)
      if self._centered:
        self._zeros_slot(v, "mg", self._name)
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._learning_rate)
    decay = self._call_if_callable(self._decay)
    momentum = self._call_if_callable(self._momentum)
    epsilon = self._call_if_callable(self._epsilon)

    self._learning_rate_tensor = ops.convert_to_tensor(lr, name="learning_rate")
    self._decay_tensor = ops.convert_to_tensor(decay, name="decay")
    self._momentum_tensor = ops.convert_to_tensor(momentum, name="momentum")
    self._epsilon_tensor = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return training_ops.apply_centered_rms_prop(
          var,
          mg,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad,
          use_locking=self._use_locking).op
    else:
      return training_ops.apply_rms_prop(
          var,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad,
          use_locking=self._use_locking).op
