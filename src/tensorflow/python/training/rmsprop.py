from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training import optimizer
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export

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
