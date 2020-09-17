from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools as _functools
import traceback as _traceback


def make_decorator(target,
                   decorator_func,
                   decorator_name=None,
                   decorator_doc='',
                   decorator_argspec=None):
  if decorator_name is None:
    frame = _traceback.extract_stack(limit=2)[0]
    # frame name is tuple[2] in python2, and object.name in python3
    decorator_name = getattr(frame, 'name', frame[2])  # Caller's name
  decorator = TFDecorator(decorator_name, target, decorator_doc,
                          decorator_argspec)
  setattr(decorator_func, '_tf_decorator', decorator)
  # Objects that are callables (e.g., a functools.partial object) may not have
  # the following attributes.
  if hasattr(target, '__name__'):
    decorator_func.__name__ = target.__name__
  if hasattr(target, '__module__'):
    decorator_func.__module__ = target.__module__
  if hasattr(target, '__doc__'):
    decorator_func.__doc__ = decorator.__doc__
  decorator_func.__wrapped__ = target
  return decorator_func





class TFDecorator(object):
  def __init__(self,
               decorator_name,
               target,
               decorator_doc='',
               decorator_argspec=None):
    self._decorated_target = target
    self._decorator_name = decorator_name
    self._decorator_doc = decorator_doc
    self._decorator_argspec = decorator_argspec
    if hasattr(target, '__name__'):
      self.__name__ = target.__name__
    if self._decorator_doc:
      self.__doc__ = self._decorator_doc
    elif hasattr(target, '__doc__') and target.__doc__:
      self.__doc__ = target.__doc__
    else:
      self.__doc__ = ''

  def __get__(self, obj, objtype):
    return _functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs):
    return self._decorated_target(*args, **kwargs)

  @property
  def decorated_target(self):
    return self._decorated_target

  @property
  def decorator_name(self):
    return self._decorator_name

  @property
  def decorator_doc(self):
    return self._decorator_doc

  @property
  def decorator_argspec(self):
    return self._decorator_argspec
