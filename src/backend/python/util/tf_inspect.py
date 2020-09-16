from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools
import inspect as _inspect

import six

from backend.python.util import tf_decorator

ArgSpec = _inspect.ArgSpec


if hasattr(_inspect, 'FullArgSpec'):
  FullArgSpec = _inspect.FullArgSpec  # pylint: disable=invalid-name
else:
  FullArgSpec = namedtuple('FullArgSpec', [
      'args', 'varargs', 'varkw', 'defaults', 'kwonlyargs', 'kwonlydefaults',
      'annotations'
  ])


def currentframe():
  """TFDecorator-aware replacement for inspect.currentframe."""
  return _inspect.stack()[1][0]


def getargspec(obj):
  if isinstance(obj, functools.partial):
    return _get_argspec_for_partial(obj)

  decorators, target = tf_decorator.unwrap(obj)

  spec = next((d.decorator_argspec
               for d in decorators
               if d.decorator_argspec is not None), None)
  if spec:
    return spec

  try:
    # Python3 will handle most callables here (not partial).
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

  # The `type(target)` ensures that if a class is received we don't return
  # the signature of it's __call__ method.
  return _inspect.getargspec(type(target).__call__)


def _get_argspec_for_partial(obj):
  n_prune_args = len(obj.args)
  partial_keywords = obj.keywords or {}

  args, varargs, keywords, defaults = getargspec(obj.func)

  # Pruning first n_prune_args arguments.
  args = args[n_prune_args:]

  # Partial function may give default value to any argument, therefore length
  # of default value list must be len(args) to allow each argument to
  # potentially be given a default value.
  all_defaults = [None] * len(args)
  if defaults:
    all_defaults[-len(defaults):] = defaults

  # Fill in default values provided by partial function in all_defaults.
  for kw, default in six.iteritems(partial_keywords):
    idx = args.index(kw)
    all_defaults[idx] = default

  # Find first argument with default value set.
  first_default = next((idx for idx, x in enumerate(all_defaults) if x), None)

  # If no default values are found, return ArgSpec with defaults=None.
  if first_default is None:
    return ArgSpec(args, varargs, keywords, None)

  # Checks if all arguments have default value set after first one.
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
    """A python2 version of getfullargspec.

    Args:
      target: the target object to inspect.
    Returns:
      A FullArgSpec with empty kwonlyargs, kwonlydefaults and annotations.
    """
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
  decorators, target = tf_decorator.unwrap(obj)
  return next((d.decorator_argspec
               for d in decorators
               if d.decorator_argspec is not None), _getfullargspec(target))


def getcallargs(func, *positional, **named):
  argspec = getfullargspec(func)
  call_args = named.copy()
  this = getattr(func, 'im_self', None) or getattr(func, '__self__', None)
  if ismethod(func) and this:
    positional = (this,) + positional
  remaining_positionals = [arg for arg in argspec.args if arg not in call_args]
  call_args.update(dict(zip(remaining_positionals, positional)))
  default_count = 0 if not argspec.defaults else len(argspec.defaults)
  if default_count:
    for arg, value in zip(argspec.args[-default_count:], argspec.defaults):
      if arg not in call_args:
        call_args[arg] = value
  return call_args



def isfunction(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.isfunction."""
  return _inspect.isfunction(tf_decorator.unwrap(object)[1])


def ismethod(object):  # pylint: disable=redefined-builtin
  """TFDecorator-aware replacement for inspect.ismethod."""
  return _inspect.ismethod(tf_decorator.unwrap(object)[1])
