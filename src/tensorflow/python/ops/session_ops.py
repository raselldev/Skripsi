#Tensor Handle Operations.

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class TensorHandle(object):
  def __init__(self, handle, dtype, session):
    self._handle = compat.as_str_any(handle)
    self._resource_handle = None
    self._dtype = dtype
    self._session = session
    self._auto_gc_enabled = True


def _get_handle_feeder(graph, feeder):
  return graph._handle_feeders.get(feeder.op.name)

def _get_handle_mover(graph, feeder, handle):
  dtype = _get_handle_feeder(graph, feeder)
  if dtype is None:
    return None
  handle_device = TensorHandle._get_device_name(handle)
  if feeder.op.device == handle_device:
    return None
  # Now we know we have to move the tensor.
  graph_key = TensorHandle._get_mover_key(feeder, handle)
  result = graph._handle_movers.get(graph_key)
  if result is None:
    # Create mover if we haven't done it.
    holder, reader = _get_handle_reader(graph, handle, dtype)
    with graph.as_default(), graph.device(feeder.op.device):
      mover = gen_data_flow_ops.get_session_handle(reader)
    result = (holder, mover)
    graph._handle_movers[graph_key] = result
  return result