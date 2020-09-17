from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools
import inspect as _inspect

import six

from backend.python.util import tf_decorator
from backend.python.util import function_utils

ArgSpec = _inspect.ArgSpec


if hasattr(_inspect, 'FullArgSpec'):
  FullArgSpec = _inspect.FullArgSpec  
else:
  FullArgSpec = namedtuple('FullArgSpec', [
      'args', 'varargs', 'varkw', 'defaults', 'kwonlyargs', 'kwonlydefaults',
      'annotations'
  ])


def currentframe():
  return _inspect.stack()[1][0]


def getargspec(obj):
  if isinstance(obj, functools.partial):
    return _get_argspec_for_partial(obj)

  decorators, target = function_utils.unwrap(obj)

  spec = next((d.decorator_argspec
               for d in decorators
               if d.decorator_argspec is not None), None)
  if spec:
    return spec

  try:
    return _inspect.getargspec(target)
  except TypeError:
    pass

  if isinstance(target, type):
    try:
      return _inspect.getargspec(target.__init__)
    except TypeError:
      pass

    try:
      return _inspect.getargspec(target.__new__)
    except TypeError:
      pass

  return _inspect.getargspec(type(target).__call__)


def _get_argspec_for_partial(obj):
  n_prune_args = len(obj.args)
  partial_keywords = obj.keywords or {}

  args, varargs, keywords, defaults = getargspec(obj.func)

  args = args[n_prune_args:]

  all_defaults = [None] * len(args)
  if defaults:
    all_defaults[-len(defaults):] = defaults

  for kw, default in six.iteritems(partial_keywords):
    idx = args.index(kw)
    all_defaults[idx] = default

  first_default = next((idx for idx, x in enumerate(all_defaults) if x), None)

  if first_default is None:
    return ArgSpec(args, varargs, keywords, None)

  invalid_default_values = [
      args[i] for i, j in enumerate(all_defaults) if not j and i > first_default
  ]

  if invalid_default_values:
    raise ValueError('Some arguments %s do not have default value, but they '
                     'are positioned after those with default values. This can '
                     'not be expressed with ArgSpec.' % invalid_default_values)

  return ArgSpec(args, varargs, keywords, tuple(all_defaults[first_default:]))


if hasattr(_inspect, 'getfullargspec'):
  _getfullargspec = _inspect.getfullargspec
else:

  def _getfullargspec(target):
    argspecs = getargspec(target)
    fullargspecs = FullArgSpec(
        args=argspecs.args,
        varargs=argspecs.varargs,
        varkw=argspecs.keywords,
        defaults=argspecs.defaults,
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})
    return fullargspecs


def getfullargspec(obj):
  decorators, target = function_utils.unwrap(obj)
  return next((d.decorator_argspec
               for d in decorators
               if d.decorator_argspec is not None), _getfullargspec(target))


def getcallargs(func, *positional, **named):
  argspec = getfullargspec(func)
  call_args = named.copy()
  this = getattr(func, 'im_self', None) or getattr(func, '__self__', None)

  remaining_positionals = [arg for arg in argspec.args if arg not in call_args]
  call_args.update(dict(zip(remaining_positionals, positional)))
  default_count = 0 if not argspec.defaults else len(argspec.defaults)
  if default_count:
    for arg, value in zip(argspec.args[-default_count:], argspec.defaults):
      if arg not in call_args:
        call_args[arg] = value
  return call_args






