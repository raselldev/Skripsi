from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from backend.core.framework import tensor_shape_pb2
from backend.python.framework import dtypes
from backend.python.util import compat
from backend.python.util.tf_export import tf_export


@tf_export("Dimension")
class Dimension(object):
  def __init__(self, value):
    if value is None:
      self._value = None
    elif isinstance(value, dtypes.DType):
      raise TypeError("Cannot convert %s to Dimension" % value)
    else:
      self._value = int(value)
      if (not isinstance(value, compat.bytes_or_text_types) and
          self._value != value):
        raise ValueError("Ambiguous dimension: %s" % value)
      if self._value < 0:
        raise ValueError("Dimension %d must be >= 0" % self._value)

  def __repr__(self):
    return "Dimension(%s)" % repr(self._value)

  def __str__(self):
    value = self._value
    return "?" if value is None else str(value)

  def __eq__(self, other):
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return None
    return self._value == other.value

  def __ne__(self, other):
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return None
    return self._value != other.value

  def __int__(self):
    return self._value

  def __long__(self):
    return self._value

  def __index__(self):
    # Allow use in Python 3 range
    return self._value

  @property
  def value(self):
    return self._value

  def is_compatible_with(self, other):
    other = as_dimension(other)
    return (self._value is None or other.value is None or
            self._value == other.value)

  def assert_is_compatible_with(self, other):
    if not self.is_compatible_with(other):
      raise ValueError("Dimensions %s and %s are not compatible" % (self,
                                                                    other))

  def merge_with(self, other):
    other = as_dimension(other)
    self.assert_is_compatible_with(other)
    if self._value is None:
      return Dimension(other.value)
    else:
      return Dimension(self._value)

  def __add__(self, other):
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value + other.value)

  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value - other.value)

  def __rsub__(self, other):
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(other.value - self._value)

  def __mul__(self, other):
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented

    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value * other.value)

  def __rmul__(self, other):
    return self * other

  def __floordiv__(self, other):
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value // other.value)

  def __rfloordiv__(self, other):
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(other.value // self._value)

  def __div__(self, other):
    return self // other

  def __mod__(self, other):
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value % other.value)

  def __rmod__(self, other):
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    return other % self

  def __lt__(self, other):
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value < other.value

  def __le__(self, other):
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value <= other.value

  def __gt__(self, other):
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value > other.value

  def __ge__(self, other):
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value >= other.value

  def __reduce__(self):
    return Dimension, (self._value,)


def as_dimension(value):
  """Converts the given value to a Dimension.

  A Dimension input will be returned unmodified.
  An input of `None` will be converted to an unknown Dimension.
  An integer input will be converted to a Dimension with that value.

  Args:
    value: The value to be converted.

  Returns:
    A Dimension corresponding to the given value.
  """
  if isinstance(value, Dimension):
    return value
  else:
    return Dimension(value)


@tf_export("TensorShape")
class TensorShape(object):
  """Represents the shape of a `Tensor`.

  A `TensorShape` represents a possibly-partial shape specification for a
  `Tensor`. It may be one of the following:

  * *Fully-known shape:* has a known number of dimensions and a known size
    for each dimension. e.g. `TensorShape([16, 256])`
  * *Partially-known shape:* has a known number of dimensions, and an unknown
    size for one or more dimension. e.g. `TensorShape([None, 256])`
  * *Unknown shape:* has an unknown number of dimensions, and an unknown
    size in all dimensions. e.g. `TensorShape(None)`

  If a tensor is produced by an operation of type `"Foo"`, its shape
  may be inferred if there is a registered shape function for
  `"Foo"`. See [Shape
  functions](https://backend.org/extend/adding_an_op#shape_functions_in_c)
  for details of shape functions and how to register them. Alternatively,
  the shape may be set explicitly using `tf.Tensor.set_shape`.
  """

  def __init__(self, dims):
    """Creates a new TensorShape with the given dimensions.

    Args:
      dims: A list of Dimensions, or None if the shape is unspecified.
        DEPRECATED: A single integer is treated as a singleton list.

    Raises:
      TypeError: If dims cannot be converted to a list of dimensions.
    """
    # TODO(irving): Eliminate the single integer special case.
    if dims is None:
      self._dims = None
    elif isinstance(dims, compat.bytes_or_text_types):
      raise TypeError("A string has ambiguous TensorShape, please wrap in a "
                      "list or convert to an int: %s" % dims)
    elif isinstance(dims, tensor_shape_pb2.TensorShapeProto):
      if dims.unknown_rank:
        self._dims = None
      else:
        self._dims = [
            # Protos store variable-size dimensions as -1
            as_dimension(dim.size if dim.size != -1 else None)
            for dim in dims.dim
        ]
    elif isinstance(dims, TensorShape):
      self._dims = dims.dims
    else:
      try:
        dims_iter = iter(dims)
      except TypeError:
        # Treat as a singleton dimension
        self._dims = [as_dimension(dims)]
      else:
        # Got a list of dimensions
        self._dims = [as_dimension(d) for d in dims_iter]
    self._ndims = None

  def __repr__(self):
    return "TensorShape(%r)" % self._dims

  def __str__(self):
    if self.ndims is None:
      return "<unknown>"
    elif self.ndims == 1:
      return "(%s,)" % self._dims[0]
    else:
      return "(%s)" % ", ".join(str(d) for d in self._dims)

  @property
  def dims(self):
    """Returns a list of Dimensions, or None if the shape is unspecified."""
    return self._dims

  @dims.setter
  def dims(self, dims):
    self._dims = dims
    self._ndims = None

  @property
  def ndims(self):
    """Returns the rank of this shape, or None if it is unspecified."""
    if self._dims is None:
      return None
    else:
      if self._ndims is None:
        self._ndims = len(self._dims)
      return self._ndims

  def __len__(self):
    """Returns the rank of this shape, or raises ValueError if unspecified."""
    if self._dims is None:
      raise ValueError("Cannot take the length of Shape with unknown rank.")
    return self.ndims

  def __bool__(self):
    """Returns True if this shape contains non-zero information."""
    return self._dims is not None

  # Python 3 wants __bool__, Python 2.7 wants __nonzero__
  __nonzero__ = __bool__

  def __iter__(self):
    """Returns `self.dims` if the rank is known, otherwise raises ValueError."""
    if self._dims is None:
      raise ValueError("Cannot iterate over a shape with unknown rank.")
    else:
      return iter(self._dims)

  def __getitem__(self, key):
    """Returns the value of a dimension or a shape, depending on the key.

    Args:
      key: If `key` is an integer, returns the dimension at that index;
        otherwise if `key` is a slice, returns a TensorShape whose
        dimensions are those selected by the slice from `self`.

    Returns:
      A dimension if `key` is an integer, or a `TensorShape` if `key` is a
      slice.

    Raises:
      ValueError: If `key` is a slice and `self` is completely unknown and
        the step is set.
    """
    if self._dims is not None:
      if isinstance(key, slice):
        return TensorShape(self._dims[key])
      else:
        return self._dims[key]
    else:
      if isinstance(key, slice):
        start = key.start if key.start is not None else 0
        stop = key.stop

        if key.step is not None:
          # TODO(mrry): Handle these maybe.
          raise ValueError("Steps are not yet handled")
        if stop is None:
          # NOTE(mrry): This implies that TensorShape(None) is compatible with
          # TensorShape(None)[1:], which is obviously not true. It would be
          # possible to track the number of dimensions symbolically,
          # and perhaps we should do that.
          return unknown_shape()
        elif start < 0 or stop < 0:
          # TODO(mrry): Handle this better, as it will be useful for handling
          # suffixes of otherwise unknown shapes.
          return unknown_shape()
        else:
          return unknown_shape(ndims=stop - start)
      else:
        return Dimension(None)

  def num_elements(self):
    """Returns the total number of elements, or none for incomplete shapes."""
    if self.is_fully_defined():
      size = 1
      for dim in self._dims:
        size *= dim.value
      return size
    else:
      return None

  def merge_with(self, other):
    """Returns a `TensorShape` combining the information in `self` and `other`.

    The dimensions in `self` and `other` are merged elementwise,
    according to the rules defined for `Dimension.merge_with()`.

    Args:
      other: Another `TensorShape`.

    Returns:
      A `TensorShape` containing the combined information of `self` and
      `other`.

    Raises:
      ValueError: If `self` and `other` are not compatible.
    """
    other = as_shape(other)
    if self._dims is None:
      return other
    else:
      try:
        self.assert_same_rank(other)
        new_dims = []
        for i, dim in enumerate(self._dims):
          new_dims.append(dim.merge_with(other[i]))
        return TensorShape(new_dims)
      except ValueError:
        raise ValueError("Shapes %s and %s are not compatible" % (self, other))

  def concatenate(self, other):
    other = as_shape(other)
    if self._dims is None or other.dims is None:
      return unknown_shape()
    else:
      return TensorShape(self._dims + other.dims)

  def assert_same_rank(self, other):
    other = as_shape(other)
    if self.ndims is not None and other.ndims is not None:
      if self.ndims != other.ndims:
        raise ValueError("Shapes %s and %s must have the same rank" % (self,
                                                                       other))

  def assert_has_rank(self, rank):
    if self.ndims not in (None, rank):
      raise ValueError("Shape %s must have rank %d" % (self, rank))

  def with_rank(self, rank):
    try:
      return self.merge_with(unknown_shape(ndims=rank))
    except ValueError:
      raise ValueError("Shape %s must have rank %d" % (self, rank))

  def with_rank_at_least(self, rank):
    if self.ndims is not None and self.ndims < rank:
      raise ValueError("Shape %s must have rank at least %d" % (self, rank))
    else:
      return self

  def with_rank_at_most(self, rank):
    if self.ndims is not None and self.ndims > rank:
      raise ValueError("Shape %s must have rank at most %d" % (self, rank))
    else:
      return self

  def is_compatible_with(self, other):
    other = as_shape(other)
    if self._dims is not None and other.dims is not None:
      if self.ndims != other.ndims:
        return False
      for x_dim, y_dim in zip(self._dims, other.dims):
        if not x_dim.is_compatible_with(y_dim):
          return False
    return True

  def assert_is_compatible_with(self, other):
    if not self.is_compatible_with(other):
      raise ValueError("Shapes %s and %s are incompatible" % (self, other))

  def most_specific_compatible_shape(self, other):
    other = as_shape(other)
    if self._dims is None or other.dims is None or self.ndims != other.ndims:
      return unknown_shape()

    dims = [(Dimension(None))] * self.ndims
    for i, (d1, d2) in enumerate(zip(self._dims, other.dims)):
      if d1 is not None and d2 is not None and d1 == d2:
        dims[i] = d1
    return TensorShape(dims)

  def is_fully_defined(self):
    return (self._dims is not None and all(dim.value is not None
                                           for dim in self._dims))

  def assert_is_fully_defined(self):
    if not self.is_fully_defined():
      raise ValueError("Shape %s is not fully defined" % self)

  def as_list(self):
    if self._dims is None:
      raise ValueError("as_list() is not defined on an unknown TensorShape.")
    return [dim.value for dim in self._dims]

  def as_proto(self):
    if self._dims is None:
      return tensor_shape_pb2.TensorShapeProto(unknown_rank=True)
    else:
      return tensor_shape_pb2.TensorShapeProto(dim=[
          tensor_shape_pb2.TensorShapeProto.Dim(size=-1
                                                if d.value is None else d.value)
          for d in self._dims
      ])

  def __eq__(self, other):
    try:
      other = as_shape(other)
    except TypeError:
      return NotImplemented
    return self._dims == other.dims

  def __ne__(self, other):
    try:
      other = as_shape(other)
    except TypeError:
      return NotImplemented
    if self.ndims is None or other.ndims is None:
      raise ValueError("The inequality of unknown TensorShapes is undefined.")
    if self.ndims != other.ndims:
      return True
    return self._dims != other.dims

  def __reduce__(self):
    return TensorShape, (self._dims,)


def as_shape(shape):
  if isinstance(shape, TensorShape):
    return shape
  else:
    return TensorShape(shape)

def unknown_shape(ndims=None):
  if ndims is None:
    return TensorShape(None)
  else:
    return TensorShape([Dimension(None)] * ndims)

_SCALAR_SHAPE = TensorShape([])

def scalar():
  return _SCALAR_SHAPE

def vector(length):
  return TensorShape([length])

def matrix(rows, cols):
  return TensorShape([rows, cols])
