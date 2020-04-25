from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops


_XLA_SCOPE_KEY = ("__xla_scope",)


class _XlaScope(object):
  """Keeps track of previous XLA scope calls, and depth of current call."""

  def __init__(self, count, depth):
    self.count = count
    self.depth = depth


@contextlib.contextmanager
def experimental_jit_scope(compile_ops=True, separate_compiled_gradients=False):
  if callable(compile_ops):
    def xla_compile(node_def):
      return attr_value_pb2.AttrValue(b=compile_ops(node_def))
  else:
    xla_compile = attr_value_pb2.AttrValue(b=compile_ops)

  attrs = {
      "_XlaCompile":
          xla_compile,
      "_XlaSeparateCompiledGradients":
          attr_value_pb2.AttrValue(b=bool(separate_compiled_gradients))
  }

  # Find the singleton counter for the current scoped graph.  If it
  # doesn't exist, create one.
  xla_scope_counter = ops.get_collection(_XLA_SCOPE_KEY)
  if not xla_scope_counter:
    xla_scope_counter = _XlaScope(0, 0)
    ops.add_to_collection(_XLA_SCOPE_KEY, xla_scope_counter)
  else:
    xla_scope_counter = xla_scope_counter[0]

  if xla_scope_counter.depth == 0:
    # If we're at the root xla scope, we can increase the counter so
    # future calls to jit_scope use a different scope value.
    # If we're already within a scope, we'll be fusing using the scope
    # controlled by the parent.
    attrs["_XlaScope"] = attr_value_pb2.AttrValue(
        s=("jit_scope_%d" % xla_scope_counter.count).encode())
    xla_scope_counter.count += 1

  xla_scope_counter.depth += 1

  # pylint: disable=protected-access
  with ops.get_default_graph()._attr_scope(attrs):
    yield
  # pylint: enable=protected-access

  xla_scope_counter.depth -= 1
