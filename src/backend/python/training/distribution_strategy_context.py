from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from backend.python.framework import ops
from backend.python.util.lazy_loader import LazyLoader


distribute_lib = LazyLoader(
    "distribute_lib", globals(),
    "backend.python.training.distribute")



class _ThreadMode(object):

  def __init__(self, dist, cross, tower):
    self.distribution_strategy = dist
    self.cross_tower_context = cross
    self.tower_context = tower


class _CrossTowerThreadMode(_ThreadMode):

  def __init__(self, distribution_strategy):
    _ThreadMode.__init__(
        self, distribution_strategy, distribution_strategy, None)


class _InTowerThreadMode(_ThreadMode):

  def __init__(self, tower_ctx):
    _ThreadMode.__init__(
        self, tower_ctx.distribution_strategy, None, tower_ctx)


def _push_per_thread_mode(context):
  ops.get_default_graph()._distribution_strategy_stack.append(context)  


def _pop_per_thread_mode():
  ops.get_default_graph()._distribution_strategy_stack.pop(-1)  


class _DefaultTowerThreadMode(_ThreadMode):
  """Type of default value returned by `_get_per_thread_mode()`.

  Used when the thread-local stack is empty.
  """

  def __init__(self):
    _ThreadMode.__init__(self, _get_default_distribution_strategy(), None,
                         _get_default_tower_context())


def _get_per_thread_mode():
  try:
    return ops.get_default_graph()._distribution_strategy_stack[-1]  
  except (AttributeError, IndexError):
    return _get_default_tower_mode()


def get_tower_context():
  return _get_per_thread_mode().tower_context


def get_cross_tower_context():
  return _get_per_thread_mode().cross_tower_context


def get_distribution_strategy():
  return _get_per_thread_mode().distribution_strategy


def has_distribution_strategy():
  return get_distribution_strategy() is not _get_default_distribution_strategy()



_defaults = {
    "distribution_strategy": None,
    "tower_context": None,
    "tower_mode": None
}


def _get_default_distribution_strategy():
  if _defaults["distribution_strategy"] is None:
    _defaults["distribution_strategy"] = (
        distribute_lib._DefaultDistributionStrategy())  
  return _defaults["distribution_strategy"]


def _get_default_tower_context():
  if _defaults["tower_context"] is None:
    _defaults["tower_context"] = distribute_lib.TowerContext(
        _get_default_distribution_strategy(), tower_id=0)
  return _defaults["tower_context"]


def _get_default_tower_mode():
  if _defaults["tower_mode"] is None:
    _defaults["tower_mode"] = _DefaultTowerThreadMode()
  return _defaults["tower_mode"]
