from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback
import warnings

from backend.core import error_codes_pb2
from backend import pywrap_backend as c_api
from backend.framework import c_api_util


class OpError(Exception):

  def __init__(self, node_def, op, message, error_code):
    super(OpError, self).__init__()
    self._node_def = node_def
    self._op = op
    self._message = message
    self._error_code = error_code

  def __reduce__(self):
    # Allow the subclasses to accept less arguments in their __init__.
    init_argspec = tf_inspect.getargspec(self.__class__.__init__)
    args = tuple(getattr(self, arg) for arg in init_argspec.args[1:])
    return self.__class__, args

  @property
  def message(self):
    return self._message




OK = error_codes_pb2.OK
#tf_export("errors.OK").export_constant(__name__, "OK")
CANCELLED = error_codes_pb2.CANCELLED
#tf_export("errors.CANCELLED").export_constant(__name__, "CANCELLED")
UNKNOWN = error_codes_pb2.UNKNOWN
#tf_export("errors.UNKNOWN").export_constant(__name__, "UNKNOWN")
INVALID_ARGUMENT = error_codes_pb2.INVALID_ARGUMENT
#tf_export("errors.INVALID_ARGUMENT").export_constant(__name__,                                                     "INVALID_ARGUMENT")
DEADLINE_EXCEEDED = error_codes_pb2.DEADLINE_EXCEEDED
#tf_export("errors.DEADLINE_EXCEEDED").export_constant(__name__,                                                    "DEADLINE_EXCEEDED")
NOT_FOUND = error_codes_pb2.NOT_FOUND


class CancelledError(OpError):
  def __init__(self, node_def, op, message):
    super(CancelledError, self).__init__(node_def, op, message, CANCELLED)


class UnknownError(OpError):
  def __init__(self, node_def, op, message, error_code=UNKNOWN):
    super(UnknownError, self).__init__(node_def, op, message, error_code)


class InvalidArgumentError(OpError):
  def __init__(self, node_def, op, message):
    super(InvalidArgumentError, self).__init__(node_def, op, message,
                                               INVALID_ARGUMENT)

class DeadlineExceededError(OpError):
  def __init__(self, node_def, op, message):
    super(DeadlineExceededError, self).__init__(node_def, op, message,
                                                DEADLINE_EXCEEDED)


class NotFoundError(OpError):
  def __init__(self, node_def, op, message):
    super(NotFoundError, self).__init__(node_def, op, message, NOT_FOUND)





_CODE_TO_EXCEPTION_CLASS = {
    CANCELLED: CancelledError,
    UNKNOWN: UnknownError,
    INVALID_ARGUMENT: InvalidArgumentError,
    DEADLINE_EXCEEDED: DeadlineExceededError,
    NOT_FOUND: NotFoundError,
}

c_api.PyExceptionRegistry_Init(_CODE_TO_EXCEPTION_CLASS)




class raise_exception_on_not_ok_status(object):
  def __enter__(self):
    self.status = c_api_util.ScopedTFStatus()
    return self.status.status

  def __exit__(self, type_arg, value_arg, traceback_arg):
    try:
      if c_api.TF_GetCode(self.status.status) != 0:
        raise _make_specific_exception(
            None, None,
            compat.as_text(c_api.TF_Message(self.status.status)),
            c_api.TF_GetCode(self.status.status))

    finally:
      del self.status
    return False