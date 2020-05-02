# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tensor utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import re

from tensorflow.python.util import tf_inspect
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import tf_contextlib

# Allow deprecation warnings to be silenced temporarily with a context manager.
_PRINT_DEPRECATION_WARNINGS = True
IS_IN_GRAPH_MODE = lambda: True

# Remember which deprecation warnings have been printed already.
_PRINTED_WARNING = {}

def _add_deprecated_function_notice_to_docstring(doc, date, instructions):
  """Adds a deprecation notice to a docstring for deprecated functions."""
  main_text = ['THIS FUNCTION IS DEPRECATED. It will be removed %s.' %
               ('in a future version' if date is None else ('after %s' % date))]
  if instructions:
    main_text.append('Instructions for updating:')
  return decorator_utils.add_notice_to_docstring(
      doc, instructions,
      'DEPRECATED FUNCTION',
      '(deprecated)', main_text)


def _add_deprecated_arg_notice_to_docstring(doc, date, instructions):
  """Adds a deprecation notice to a docstring for deprecated arguments."""
  return decorator_utils.add_notice_to_docstring(
      doc, instructions,
      'DEPRECATED FUNCTION ARGUMENTS',
      '(deprecated arguments)', [
          'SOME ARGUMENTS ARE DEPRECATED. '
          'They will be removed %s.' % (
              'in a future version' if date is None else ('after %s' % date)),
          'Instructions for updating:'])


def _validate_deprecation_args(date, instructions):
  if date is not None and not re.match(r'20\d\d-[01]\d-[0123]\d', date):
    raise ValueError('Date must be YYYY-MM-DD.')
  if not instructions:
    raise ValueError('Don\'t deprecate things without conversion instructions!')


def deprecated_endpoints(*args):
  def deprecated_wrapper(func):
    # pylint: disable=protected-access
    if '_tf_deprecated_api_names' in func.__dict__:
      raise DeprecatedNamesAlreadySet(
          'Cannot set deprecated names for %s to %s. '
          'Deprecated names are already set to %s.' % (
              func.__name__, str(args), str(func._tf_deprecated_api_names)))
    func._tf_deprecated_api_names = args
    # pylint: disable=protected-access
    return func
  return deprecated_wrapper


def deprecated(date, instructions, warn_once=True):
  _validate_deprecation_args(date, instructions)

  def deprecated_wrapper(func):
    """Deprecation wrapper."""
    decorator_utils.validate_callable(func, 'deprecated')
    @functools.wraps(func)
    def new_func(*args, **kwargs):  # pylint: disable=missing-docstring
      if _PRINT_DEPRECATION_WARNINGS:
        if func not in _PRINTED_WARNING:
          if warn_once:
            _PRINTED_WARNING[func] = True
          logging.warning(
              'From %s: %s (from %s) is deprecated and will be removed %s.\n'
              'Instructions for updating:\n%s',
              _call_location(), decorator_utils.get_qualified_name(func),
              func.__module__,
              'in a future version' if date is None else ('after %s' % date),
              instructions)
      return func(*args, **kwargs)
    return tf_decorator.make_decorator(
        func, new_func, 'deprecated',
        _add_deprecated_function_notice_to_docstring(func.__doc__, date,
                                                     instructions))
  return deprecated_wrapper


DeprecatedArgSpec = collections.namedtuple(
    'DeprecatedArgSpec', ['position', 'has_ok_value', 'ok_value'])


def deprecated_args(date, instructions, *deprecated_arg_names_or_tuples,
                    **kwargs):
  _validate_deprecation_args(date, instructions)
  if not deprecated_arg_names_or_tuples:
    raise ValueError('Specify which argument is deprecated.')
  if kwargs and list(kwargs.keys()) != ['warn_once']:
    kwargs.pop('warn_once', None)
    raise ValueError('Illegal argument to deprecated_args: %s' % kwargs)
  warn_once = kwargs.get('warn_once', True)

  def _get_arg_names_to_ok_vals():
    """Returns a dict mapping arg_name to DeprecatedArgSpec w/o position."""
    d = {}
    for name_or_tuple in deprecated_arg_names_or_tuples:
      if isinstance(name_or_tuple, tuple):
        d[name_or_tuple[0]] = DeprecatedArgSpec(-1, True, name_or_tuple[1])
      else:
        d[name_or_tuple] = DeprecatedArgSpec(-1, False, None)
    return d

  def _get_deprecated_positional_arguments(names_to_ok_vals, arg_spec):
    """Builds a dictionary from deprecated arguments to their spec.

    Returned dict is keyed by argument name.
    Each value is a DeprecatedArgSpec with the following fields:
       position: The zero-based argument position of the argument
         within the signature.  None if the argument isn't found in
         the signature.
       ok_values:  Values of this argument for which warning will be
         suppressed.

    Args:
      names_to_ok_vals: dict from string arg_name to a list of values,
        possibly empty, which should not elicit a warning.
      arg_spec: Output from tf_inspect.getfullargspec on the called function.

    Returns:
      Dictionary from arg_name to DeprecatedArgSpec.
    """
    arg_name_to_pos = {
        name: pos for pos, name in enumerate(arg_spec.args)}
    deprecated_positional_args = {}
    for arg_name, spec in iter(names_to_ok_vals.items()):
      if arg_name in arg_name_to_pos:
        pos = arg_name_to_pos[arg_name]
        deprecated_positional_args[arg_name] = DeprecatedArgSpec(
            pos, spec.has_ok_value, spec.ok_value)
    return deprecated_positional_args

  def deprecated_wrapper(func):
    """Deprecation decorator."""
    decorator_utils.validate_callable(func, 'deprecated_args')
    deprecated_arg_names = _get_arg_names_to_ok_vals()

    arg_spec = tf_inspect.getfullargspec(func)
    deprecated_positions = _get_deprecated_positional_arguments(
        deprecated_arg_names, arg_spec)

    is_varargs_deprecated = arg_spec.varargs in deprecated_arg_names
    is_kwargs_deprecated = arg_spec.varkw in deprecated_arg_names

    if (len(deprecated_positions) + is_varargs_deprecated + is_kwargs_deprecated
        != len(deprecated_arg_names_or_tuples)):
      known_args = arg_spec.args + [arg_spec.varargs, arg_spec.varkw]
      missing_args = [arg_name for arg_name in deprecated_arg_names
                      if arg_name not in known_args]
      raise ValueError('The following deprecated arguments are not present '
                       'in the function signature: %s. '
                       'Found next arguments: %s.' % (missing_args, known_args))

    def _same_value(a, b):
      """A comparison operation that works for multiple object types.

      Returns True for two empty lists, two numeric values with the
      same value, etc.

      Returns False for (pd.DataFrame, None), and other pairs which
      should not be considered equivalent.

      Args:
        a: value one of the comparison.
        b: value two of the comparison.

      Returns:
        A boolean indicating whether the two inputs are the same value
        for the purposes of deprecation.
      """
      if a is b:
        return True
      try:
        equality = a == b
        if isinstance(equality, bool):
          return equality
      except TypeError:
        return False
      return False

    @functools.wraps(func)
    def new_func(*args, **kwargs):
      """Deprecation wrapper."""
      # TODO(apassos) figure out a way to have reasonable performance with
      # deprecation warnings and eager mode.
      if IS_IN_GRAPH_MODE() and _PRINT_DEPRECATION_WARNINGS:
        invalid_args = []
        named_args = tf_inspect.getcallargs(func, *args, **kwargs)
        for arg_name, spec in iter(deprecated_positions.items()):
          if (spec.position < len(args) and
              not (spec.has_ok_value and
                   _same_value(named_args[arg_name], spec.ok_value))):
            invalid_args.append(arg_name)
        if is_varargs_deprecated and len(args) > len(arg_spec.args):
          invalid_args.append(arg_spec.varargs)
        if is_kwargs_deprecated and kwargs:
          invalid_args.append(arg_spec.varkw)
        for arg_name in deprecated_arg_names:
          if (arg_name in kwargs and
              not (deprecated_positions[arg_name].has_ok_value and
                   _same_value(named_args[arg_name],
                               deprecated_positions[arg_name].ok_value))):
            invalid_args.append(arg_name)
        for arg_name in invalid_args:
          if (func, arg_name) not in _PRINTED_WARNING:
            if warn_once:
              _PRINTED_WARNING[(func, arg_name)] = True
            logging.warning(
                'From %s: calling %s (from %s) with %s is deprecated and will '
                'be removed %s.\nInstructions for updating:\n%s',
                _call_location(), decorator_utils.get_qualified_name(func),
                func.__module__, arg_name,
                'in a future version' if date is None else ('after %s' % date),
                instructions)
      return func(*args, **kwargs)
    return tf_decorator.make_decorator(func, new_func, 'deprecated',
                                       _add_deprecated_arg_notice_to_docstring(
                                           func.__doc__, date, instructions))
  return deprecated_wrapper


def deprecated_arg_values(date, instructions, warn_once=True,
                          **deprecated_kwargs):
  _validate_deprecation_args(date, instructions)
  if not deprecated_kwargs:
    raise ValueError('Specify which argument values are deprecated.')

  def deprecated_wrapper(func):
    """Deprecation decorator."""
    decorator_utils.validate_callable(func, 'deprecated_arg_values')
    @functools.wraps(func)
    def new_func(*args, **kwargs):
      """Deprecation wrapper."""
      if _PRINT_DEPRECATION_WARNINGS:
        named_args = tf_inspect.getcallargs(func, *args, **kwargs)
        for arg_name, arg_value in deprecated_kwargs.items():
          if arg_name in named_args and named_args[arg_name] == arg_value:
            if (func, arg_name) not in _PRINTED_WARNING:
              if warn_once:
                _PRINTED_WARNING[(func, arg_name)] = True
              logging.warning(
                  'From %s: calling %s (from %s) with %s=%s is deprecated and '
                  'will be removed %s.\nInstructions for updating:\n%s',
                  _call_location(), decorator_utils.get_qualified_name(func),
                  func.__module__, arg_name, arg_value, 'in a future version'
                  if date is None else ('after %s' % date), instructions)
      return func(*args, **kwargs)
    return tf_decorator.make_decorator(func, new_func, 'deprecated',
                                       _add_deprecated_arg_notice_to_docstring(
                                           func.__doc__, date, instructions))
  return deprecated_wrapper


def deprecated_argument_lookup(new_name, new_value, old_name, old_value):
 
  if old_value is not None:
    if new_value is not None:
      raise ValueError("Cannot specify both '%s' and '%s'" %
                       (old_name, new_name))
    return old_value
  return new_value


@tf_contextlib.contextmanager
def silence():
  """Temporarily silence deprecation warnings."""
  global _PRINT_DEPRECATION_WARNINGS
  print_deprecation_warnings = _PRINT_DEPRECATION_WARNINGS
  _PRINT_DEPRECATION_WARNINGS = False
  yield
  _PRINT_DEPRECATION_WARNINGS = print_deprecation_warnings
