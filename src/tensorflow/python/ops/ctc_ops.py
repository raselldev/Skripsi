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

"""CTC (Connectionist Temporal Classification) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.util.tf_export import tf_export


@tf_export("nn.ctc_loss")
def ctc_loss(labels, inputs, sequence_length,
             preprocess_collapse_repeated=False,
             ctc_merge_repeated=True,
             ignore_longer_outputs_than_inputs=False, time_major=True):
  if not isinstance(labels, sparse_tensor.SparseTensor):
    raise TypeError("Expected labels (first argument) to be a SparseTensor")
  if not time_major:
    inputs = array_ops.transpose(inputs, [1, 0, 2])  # (B,T,N) => (T,B,N)
  loss, _ = gen_ctc_ops.ctc_loss(
      inputs,
      labels.indices,
      labels.values,
      sequence_length,
      preprocess_collapse_repeated=preprocess_collapse_repeated,
      ctc_merge_repeated=ctc_merge_repeated,
      ignore_longer_outputs_than_inputs=ignore_longer_outputs_than_inputs)

  return loss

@ops.RegisterGradient("CTCLoss")
def _CTCLossGrad(op, grad_loss, _):
  """The derivative provided by CTC Loss.

  Args:
     op: the CTCLoss op.
     grad_loss: The backprop for cost.

  Returns:
     The CTC Loss gradient.
  """
  # Outputs are: loss, grad
  #
  # Currently there is no way to take the second derivative of this op
  # due to the fused implementation's interaction with tf.gradients(),
  # so we make sure we prevent silently incorrect results by raising
  # an error if the second derivative is requested via prevent_gradient.
  grad_without_gradient = array_ops.prevent_gradient(
      op.outputs[1], message="Currently there is no way to take the second "
      " derivative of ctc_loss due to the fused implementation's interaction "
      " with tf.gradients()")
  # Return gradient for inputs and None for
  # labels_indices, labels_values and sequence_length
  return [_BroadcastMul(grad_loss, grad_without_gradient), None, None, None]


@tf_export("nn.ctc_greedy_decoder")
def ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True):
  outputs = gen_ctc_ops.ctc_greedy_decoder(
      inputs, sequence_length, merge_repeated=merge_repeated)
  (decoded_ix, decoded_val, decoded_shape, log_probabilities) = outputs
  return ([sparse_tensor.SparseTensor(decoded_ix, decoded_val, decoded_shape)],
          log_probabilities)