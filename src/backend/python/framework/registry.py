from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback


from backend.util import compat

_LOCATION_TAG = "location"
_TYPE_TAG = "type"


class Registry(object):
  def __init__(self, name):
    """Creates a new registry."""
    self._name = name
    self._registry = dict()

  def register(self, candidate, name=None):
    if not name:
      name = candidate.__name__
    if name in self._registry:
      (filename, line_number, function_name, _) = (
          self._registry[name][_LOCATION_TAG])
      raise KeyError("Registering two %s with name '%s' !"
                     "(Previous registration was in %s %s:%d)" %
                     (self._name, name, function_name, filename, line_number))
    stack = traceback.extract_stack()
    self._registry[name] = {_TYPE_TAG: candidate, _LOCATION_TAG: stack[2]}

