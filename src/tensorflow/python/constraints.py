
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six


class Constraint(object):

  def __call__(self, w):
    return w

  def get_config(self):
    return {}


class MaxNorm(Constraint):

  def __init__(self, max_value=2, axis=0):
    self.max_value = max_value
    self.axis = axis

  def get_config(self):
    return {'max_value': self.max_value, 'axis': self.axis}


class NonNeg(Constraint):
  """Constrains the weights to be non-negative.
  """

  def __call__(self, w):
    return w * math_ops.cast(math_ops.greater_equal(w, 0.), K.floatx())


class UnitNorm(Constraint):

  def __init__(self, axis=0):
    self.axis = axis

  def __call__(self, w):
    return w / (
        K.epsilon() + K.sqrt(
            math_ops.reduce_sum(
                math_ops.square(w), axis=self.axis, keepdims=True)))

  def get_config(self):
    return {'axis': self.axis}


class MinMaxNorm(Constraint):

  def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
    self.min_value = min_value
    self.max_value = max_value
    self.rate = rate
    self.axis = axis

  def __call__(self, w):
    norms = K.sqrt(
        math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True))
    desired = (
        self.rate * K.clip(norms, self.min_value, self.max_value) +
        (1 - self.rate) * norms)
    return w * (desired / (K.epsilon() + norms))

  def get_config(self):
    return {
        'min_value': self.min_value,
        'max_value': self.max_value,
        'rate': self.rate,
        'axis': self.axis
    }


# Aliases.

max_norm = MaxNorm
non_neg = NonNeg
unit_norm = UnitNorm
min_max_norm = MinMaxNorm

# Legacy aliases.
maxnorm = max_norm
nonneg = non_neg
unitnorm = unit_norm

def serialize(constraint):
  return serialize_keras_object(constraint)


def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='constraint')


def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret constraint identifier: ' +
                     str(identifier))
