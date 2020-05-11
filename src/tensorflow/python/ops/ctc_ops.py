#CTC (Connectionist Temporal Classification) Operation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import ops
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
  grad_without_gradient = array_ops.prevent_gradient(
      op.outputs[1], message="sip")
  return [_BroadcastMul(grad_loss, grad_without_gradient), None, None, None]

@tf_export("nn.ctc_greedy_decoder")
def ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True):
  outputs = gen_ctc_ops.ctc_greedy_decoder(
      inputs, sequence_length, merge_repeated=merge_repeated)
  (decoded_ix, decoded_val, decoded_shape, log_probabilities) = outputs
  return ([sparse_tensor.SparseTensor(decoded_ix, decoded_val, decoded_shape)],
          log_probabilities)