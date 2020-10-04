from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib


from backend.framework import ops


class _FuncGraph(ops.Graph):
  def __init__(self, name, capture_by_value, *args, **kwargs):
    super(_FuncGraph, self).__init__(*args, **kwargs)
    self._capture_by_value = capture_by_value
    self._building_function = True
    self._outer_graph = ops.get_default_graph()
    self._vscope = vs.get_variable_scope()
    self._old_custom_getter = self._vscope.custom_getter
    self.name = name
    self.inputs = []
    self.outputs = []
    self._captured = {}
    self.extra_inputs = []
    self.extra_args = []
    self.extra_vars = []

