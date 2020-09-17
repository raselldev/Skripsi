from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as _collections

import six as _six

from backend.python import pywrap_backend as _pywrap_tensorflow


def _is_namedtuple(instance, strict=False):
  return _pywrap_tensorflow.IsNamedtuple(instance, strict)

_is_mapping = _pywrap_tensorflow.IsMapping
_is_attrs = _pywrap_tensorflow.IsAttrs

def _sequence_like(instance, args):
  if _is_mapping(instance):
    result = dict(zip(_sorted(instance), args))
    return type(instance)((key, result[key]) for key in _six.iterkeys(instance))
  elif _is_namedtuple(instance) or _is_attrs(instance):
    return type(instance)(*args)
  else:
    # Not a namedtuple
    return type(instance)(args)

def _yield_value(iterable):
  if _is_mapping(iterable):
    for key in _sorted(iterable):
      yield iterable[key]
  elif _is_attrs(iterable):
    for value in _get_attrs_values(iterable):
      yield value
  else:
    for value in iterable:
      yield value

is_sequence = _pywrap_tensorflow.IsSequence

flatten = _pywrap_tensorflow.Flatten

_same_namedtuples = _pywrap_tensorflow.SameNamedtuples

def assert_same_structure(nest1, nest2, check_types=True):
  try:
    _pywrap_tensorflow.AssertSameStructure(nest1, nest2, check_types)
  except (ValueError, TypeError) as e:
    str1 = str(map_structure(lambda _: _DOT, nest1))
    str2 = str(map_structure(lambda _: _DOT, nest2))
    raise type(e)("%s\n"
                  "Entire first structure:\n%s\n"
                  "Entire second structure:\n%s"
                  % (str(e), str1, str2))

def flatten_dict_items(dictionary):
  if not isinstance(dictionary, (dict, _collections.Mapping)):
    raise TypeError("input must be a dictionary")
  flat_dictionary = {}
  for i, v in _six.iteritems(dictionary):
    if not is_sequence(i):
      if i in flat_dictionary:
        raise ValueError(
            "Could not flatten dictionary: key %s is not unique." % i)
      flat_dictionary[i] = v
    else:
      flat_i = flatten(i)
      flat_v = flatten(v)
      if len(flat_i) != len(flat_v):
        raise ValueError(
            "Could not flatten dictionary. Key had %d elements, but value had "
            "%d elements. Key: %s, value: %s."
            % (len(flat_i), len(flat_v), flat_i, flat_v))
      for new_i, new_v in zip(flat_i, flat_v):
        if new_i in flat_dictionary:
          raise ValueError(
              "Could not flatten dictionary: key %s is not unique."
              % (new_i))
        flat_dictionary[new_i] = new_v
  return flat_dictionary

def _packed_nest_with_indices(structure, flat, index):
  packed = []
  for s in _yield_value(structure):
    if is_sequence(s):
      new_index, child = _packed_nest_with_indices(s, flat, index)
      packed.append(_sequence_like(s, child))
      index = new_index
    else:
      packed.append(flat[index])
      index += 1
  return index, packed

def pack_sequence_as(structure, flat_sequence):
  if not is_sequence(flat_sequence):
    raise TypeError("flat_sequence must be a sequence")

  if not is_sequence(structure):
    if len(flat_sequence) != 1:
      raise ValueError("Structure is a scalar but len(flat_sequence) == %d > 1"
                       % len(flat_sequence))
    return flat_sequence[0]

  try:
    final_index, packed = _packed_nest_with_indices(structure, flat_sequence, 0)
    if final_index < len(flat_sequence):
      raise IndexError
  except IndexError:
    flat_structure = flatten(structure)
    if len(flat_structure) != len(flat_sequence):
      raise ValueError(
          "Could not pack sequence. Structure had %d elements, but "
          "flat_sequence had %d elements.  Structure: %s, flat_sequence: %s." %
          (len(flat_structure), len(flat_sequence), structure, flat_sequence))
  return _sequence_like(structure, packed)

def map_structure(func, *structure, **check_types_dict):
  if not callable(func):
    raise TypeError("func must be callable, got: %s" % func)

  if not structure:
    raise ValueError("Must provide at least one structure")

  if check_types_dict:
    if "check_types" not in check_types_dict or len(check_types_dict) > 1:
      raise ValueError("Only valid keyword argument is check_types")
    check_types = check_types_dict["check_types"]
  else:
    check_types = True

  for other in structure[1:]:
    assert_same_structure(structure[0], other, check_types=check_types)

  flat_structure = [flatten(s) for s in structure]
  entries = zip(*flat_structure)

  return pack_sequence_as(
      structure[0], [func(*x) for x in entries])

_pywrap_tensorflow.RegisterType("Mapping", _collections.Mapping)
_pywrap_tensorflow.RegisterType("Sequence", _collections.Sequence)
