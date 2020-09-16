# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Built-in regularizers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six


#from tensorflow.python.util.tf_export import tf_export


#@tf_export('keras.regularizers.Regularizer')
class Regularizer(object):
  """Regularizer base class.
  """

  def __call__(self, x):
    return 0.

  @classmethod
  def from_config(cls, config):
    return cls(**config)



# Aliases.


#@tf_export('keras.regularizers.l1')
def l1(l=0.01):
  return L1L2(l1=l)


#@tf_export('keras.regularizers.l2')
def l2(l=0.01):
  return L1L2(l2=l)


#@tf_export('keras.regularizers.l1_l2')
def l1_l2(l1=0.01, l2=0.01):  # pylint: disable=redefined-outer-name
  return L1L2(l1=l1, l2=l2)


#@tf_export('keras.regularizers.serialize')
def serialize(regularizer):
  return serialize_keras_object(regularizer)


#@tf_export('keras.regularizers.deserialize')
def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='regularizer')


#@tf_export('keras.regularizers.get')
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
