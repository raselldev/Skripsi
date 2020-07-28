from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.util.tf_export import tf_export



class TensorHandle(object):
  pass
  
def _get_handle_feeder(graph, feeder):
  return graph._handle_feeders.get(feeder.op.name)

def _get_handle_mover(graph, feeder, handle):
  """Return a move subgraph for this pair of feeder and handle."""
  dtype = _get_handle_feeder(graph, feeder)
  if dtype is None:
    return None
  if feeder.op.device == handle_device:
    return None
  if result is None:
    # Create mover if we haven't done it.
    holder, reader = _get_handle_reader(graph, handle, dtype)
    with graph.as_default(), graph.device(feeder.op.device):
      mover = gen_data_flow_ops.get_session_handle(reader)
    result = (holder, mover)
    graph._handle_movers[graph_key] = result
  return result

