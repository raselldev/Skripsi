from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import six
import inspect as _inspect

from backend.util.tf_decorator import TFDecorator
from backend.util import tf_inspect


def fn_args(fn):
  if isinstance(fn, functools.partial):
    args = fn_args(fn.func)
    args = [a for a in args[len(fn.args):] if a not in (fn.keywords or [])]
  else:
    args = tf_inspect.getfullargspec(fn).args
  return tuple(args)

def has_kwargs(fn):
  if isinstance(fn, functools.partial):
    fn = fn.func
  return tf_inspect.getfullargspec(fn).varkw is not None

def unwrap(maybe_tf_decorator):
  decorators = []
  cur = maybe_tf_decorator
  while True:
    if isinstance(cur, TFDecorator):
      decorators.append(cur)
    elif hasattr(cur, '_tf_decorator'):
      decorators.append(getattr(cur, '_tf_decorator'))
    else:
      break
    cur = decorators[-1].decorated_target
  return decorators, cur

def isfunction(object):  
  return _inspect.isfunction(unwrap(object)[1])

def ismethod(object):  
  return _inspect.ismethod(unwrap(object)[1])

def get_func_name(func):
  _, func = unwrap(func)
  if callable(func):
    if isfunction(func):
      return func.__name__
    elif ismethod(func):
      return '%s.%s' % (six.get_method_self(func).__class__.__name__,
                        six.get_method_function(func).__name__)
    else:
      return str(type(func))

def get_func_code(func):
  _, func = unwrap(func)

