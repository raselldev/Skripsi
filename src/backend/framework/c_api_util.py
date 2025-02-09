from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from backend import pywrap_backend as c_api
from backend.util import tf_contextlib

class ScopedTFStatus(object):
  def __init__(self):
    self.status = c_api.TF_NewStatus()

  def __del__(self):
    if c_api is not None and c_api.TF_DeleteStatus is not None:
      c_api.TF_DeleteStatus(self.status)


class ScopedTFGraph(object):
  def __init__(self):
    self.graph = c_api.TF_NewGraph()



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
