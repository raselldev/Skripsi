from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from backend.python import pywrap_backend as c_api
from backend.python.util import tf_contextlib

class ScopedTFStatus(object):
  """Wrapper around TF_Status that handles deletion."""

  def __init__(self):
    self.status = c_api.TF_NewStatus()

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api is not None and c_api.TF_DeleteStatus is not None:
      c_api.TF_DeleteStatus(self.status)


class ScopedTFGraph(object):
  """Wrapper around TF_Graph that handles deletion."""

  def __init__(self):
    self.graph = c_api.TF_NewGraph()

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api is not None and c_api.TF_DeleteGraph is not None:
      pass
      #c_api.TF_DeleteGraph(self.graph)


@tf_contextlib.contextmanager
def tf_buffer(data=None):
  if data:
    buf = c_api.TF_NewBufferFromString(compat.as_bytes(data))
  else:
    buf = c_api.TF_NewBuffer()
  try:
    yield buf
  finally:
    c_api.TF_DeleteBuffer(buf)


def tf_output(c_op, index):
  ret = c_api.TF_Output()
  ret.oper = c_op
  ret.index = index
  return ret
