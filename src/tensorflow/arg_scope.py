from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import tf_contextlib

__all__ = [
    'arg_scope', 'add_arg_scope', 'current_arg_scope', 'has_arg_scope',
    'arg_scoped_arguments', 'arg_scope_func_key'
]

_ARGSTACK = [{}]

_DECORATED_OPS = {}


def _get_arg_stack():
  if _ARGSTACK:
    return _ARGSTACK
  else:
    _ARGSTACK.append({})
    return _ARGSTACK


def current_arg_scope():
  stack = _get_arg_stack()
  return stack[-1]


def arg_scope_func_key(op):
  return getattr(op, '_key_op', str(op))


def _name_op(op):
  return (op.__module__, op.__name__)


def _kwarg_names(func):
  kwargs_length = len(func.__defaults__) if func.__defaults__ else 0
  return func.__code__.co_varnames[-kwargs_length:func.__code__.co_argcount]


def _add_op(op):
  key_op = arg_scope_func_key(op)
  _DECORATED_OPS[key_op] = _kwarg_names(op)


@tf_contextlib.contextmanager
def arg_scope(list_ops_or_scope, **kwargs):
  if isinstance(list_ops_or_scope, dict):
    # Assumes that list_ops_or_scope is a scope that is being reused.
    if kwargs:
      raise ValueError('When attempting to re-use a scope by suppling a'
                       'dictionary, kwargs must be empty.')
    current_scope = list_ops_or_scope.copy()
    try:
      _get_arg_stack().append(current_scope)
      yield current_scope
    finally:
      _get_arg_stack().pop()
  else:
    # Assumes that list_ops_or_scope is a list/tuple of ops with kwargs.
    if not isinstance(list_ops_or_scope, (list, tuple)):
      raise TypeError('list_ops_or_scope must either be a list/tuple or reused '
                      'scope (i.e. dict)')
    try:
      current_scope = current_arg_scope().copy()
      for op in list_ops_or_scope:
        key = arg_scope_func_key(op)
        if not has_arg_scope(op):
          raise ValueError('%s is not decorated with @add_arg_scope',
                           _name_op(op))
        if key in current_scope:
          current_kwargs = current_scope[key].copy()
          current_kwargs.update(kwargs)
          current_scope[key] = current_kwargs
        else:
          current_scope[key] = kwargs.copy()
      _get_arg_stack().append(current_scope)
      yield current_scope
    finally:
      _get_arg_stack().pop()


def add_arg_scope(func):
  def func_with_args(*args, **kwargs):
    current_scope = current_arg_scope()
    current_args = kwargs
    key_func = arg_scope_func_key(func)
    if key_func in current_scope:
      current_args = current_scope[key_func].copy()
      current_args.update(kwargs)
    return func(*args, **current_args)

  _add_op(func)
  setattr(func_with_args, '_key_op', arg_scope_func_key(func))
  return tf_decorator.make_decorator(func, func_with_args)


def has_arg_scope(func):
  return arg_scope_func_key(func) in _DECORATED_OPS


def arg_scoped_arguments(func):
  assert has_arg_scope(func)
  return _DECORATED_OPS[arg_scope_func_key(func)]
