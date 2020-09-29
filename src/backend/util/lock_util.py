from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading


class GroupLock(object):
  def __init__(self, num_groups=2):
    if num_groups < 1:
      raise ValueError("num_groups must be a positive integer, got {}".format(
          num_groups))
    self._ready = threading.Condition(threading.Lock())
    self._num_groups = num_groups
    self._group_member_counts = [0] * self._num_groups

  def group(self, group_id):
    #self._validate_group_id(group_id)
    return self._Context(self, group_id)

  def acquire(self, group_id):
    #self._validate_group_id(group_id)

    self._ready.acquire()
    while self._another_group_active(group_id):
      self._ready.wait()
    self._group_member_counts[group_id] += 1
    self._ready.release()

  def release(self, group_id):
    #self._validate_group_id(group_id)

    self._ready.acquire()
    self._group_member_counts[group_id] -= 1
    if self._group_member_counts[group_id] == 0:
      self._ready.notifyAll()
    self._ready.release()

  def _another_group_active(self, group_id):
    return any(
        c > 0 for g, c in enumerate(self._group_member_counts) if g != group_id)



  class _Context(object):
    def __init__(self, lock, group_id):
      self._lock = lock
      self._group_id = group_id

    def __enter__(self):
      self._lock.acquire(self._group_id)

    def __exit__(self, type_arg, value_arg, traceback_arg):
      del type_arg, value_arg, traceback_arg
      self._lock.release(self._group_id)
