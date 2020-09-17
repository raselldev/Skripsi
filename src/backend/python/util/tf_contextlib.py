from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib as _contextlib

from backend.python.util import tf_decorator


def contextmanager(target):
  context_manager = _contextlib.contextmanager(target)
  return tf_decorator.make_decorator(target, context_manager, 'contextmanager')
