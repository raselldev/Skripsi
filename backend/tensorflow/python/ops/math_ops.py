from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.util import deprecation
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op


# This is set by resource_variable_ops.py. It is included in this way since
# there is a circular dependency between math_ops and resource_variable_ops
_resource_variable_type = None


def _set_doc(doc):
  def _decorator(func):
    func.__doc__ = doc
    return func

  return _decorator


# pylint: disable=redefined-builtin
def _bucketize(input, boundaries, name=None):
  return gen_math_ops.bucketize(input=input, boundaries=boundaries, name=name)


# pylint: enable=redefined-builtin


class DivideDelegateWithName(object):
  """Use Python2/Python3 division delegation to implement divide for tensors."""

  def __init__(self, x, name):
    """Construct DivideDelegateWithName.

    Args:
      x: Tensor to use as left operand in operator overloads
      name: The name that is preferred for the op created.
    """
    self.x = x
    self.name = name

  def __truediv__(self, y):
    return _truediv_python3(self.x, y, self.name)

  def __floordiv__(self, y):
    return floordiv(self.x, y, self.name)

  def __div__(self, y):
    return _div_python2(self.x, y, self.name)


@tf_export("math.divide", "divide")
def divide(x, y, name=None):
  """Computes Python style division of `x` by `y`."""

  if name is not None:
    # Cannot use tensors operator overload, because it has no way to track
    # override names. Use a dummy class to track the runtime division behavior
    return DivideDelegateWithName(x, name) / y
  else:
    return x / y


@tf_export("math.multiply", "multiply")
def multiply(x, y, name=None):
  return gen_math_ops.mul(x, y, name)


multiply.__doc__ = gen_math_ops.mul.__doc__.replace("Multiply", "`tf.multiply`")


# TODO(aselle): put deprecation in after another round of global code changes
@deprecation.deprecated(
    "2016-12-30",
    "`tf.mul(x, y)` is deprecated, please use `tf.multiply(x, y)` or `x * y`")
def _mul(x, y, name=None):
  return gen_math_ops.mul(x, y, name)


_mul.__doc__ = (
    gen_math_ops.mul.__doc__ + ("" if _mul.__doc__ is None else _mul.__doc__))


@tf_export("math.subtract", "subtract")
def subtract(x, y, name=None):
  return gen_math_ops.sub(x, y, name)


subtract.__doc__ = gen_math_ops.sub.__doc__.replace("`Sub`", "`tf.subtract`")


# TODO(aselle): put deprecation in after another round of global code changes
@deprecation.deprecated(
    "2016-12-30",
    "`tf.sub(x, y)` is deprecated, please use `tf.subtract(x, y)` or `x - y`")
def _sub(x, y, name=None):
  return gen_math_ops.sub(x, y, name)


_sub.__doc__ = (
    gen_math_ops.sub.__doc__ + ("" if _sub.__doc__ is None else _sub.__doc__))


# pylint: disable=g-docstring-has-escape
@tf_export("math.negative", "negative")
def negative(x, name=None):
  """Computes numerical negative value element-wise.

  I.e., \\(y = -x\\).

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
  """
  with ops.name_scope(name, "Neg", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      x_neg = gen_math_ops.neg(x.values, name=name)
      return sparse_tensor.SparseTensor(
          indices=x.indices, values=x_neg, dense_shape=x.dense_shape)
    else:
      return gen_math_ops.neg(x, name=name)


# pylint: enable=g-docstring-has-escape


# pylint: disable=g-docstring-has-escape
@deprecation.deprecated(
    "2016-12-30",
    "`tf.neg(x)` is deprecated, please use `tf.negative(x)` or `-x`")
def _neg(x, name=None):
  """Computes numerical negative value element-wise.

  I.e., \\(y = -x\\).

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
  """
  return negative(x, name)


# pylint: enable=g-docstring-has-escape


@tf_export("math.sign", "sign")
def sign(x, name=None):
  """Returns an element-wise indication of the sign of a number.

  `y = sign(x) = -1` if `x < 0`; 0 if `x == 0` or `tf.is_nan(x)`; 1 if `x > 0`.

  Zero is returned for NaN inputs.

  For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(numpy)
  Equivalent to numpy.sign except for the behavior for input values of NaN.
  @end_compatibility
  """
  with ops.name_scope(name, "Sign", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      x_sign = gen_math_ops.sign(x.values, name=name)
      return sparse_tensor.SparseTensor(
          indices=x.indices, values=x_sign, dense_shape=x.dense_shape)
    else:
      return gen_math_ops.sign(x, name=name)


@tf_export("math.square", "square")
def square(x, name=None):
  r"""Computes square of x element-wise.

  I.e., \\(y = x * x = x^2\\).

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`. Has the same type as `x`.
  """
  with ops.name_scope(name, "Square", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      x_square = gen_math_ops.square(x.values, name=name)
      return sparse_tensor.SparseTensor(
          indices=x.indices, values=x_square, dense_shape=x.dense_shape)
    else:
      return gen_math_ops.square(x, name=name)


@tf_export("math.sqrt", "sqrt")
def sqrt(x, name=None):
  r"""Computes square root of x element-wise.

  I.e., \\(y = \sqrt{x} = x^{1/2}\\).

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
  """
  with ops.name_scope(name, "Sqrt", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      x_sqrt = gen_math_ops.sqrt(x.values, name=name)
      return sparse_tensor.SparseTensor(
          indices=x.indices, values=x_sqrt, dense_shape=x.dense_shape)
    else:
      return gen_math_ops.sqrt(x, name=name)


@tf_export("math.erf", "erf")
@deprecation.deprecated_endpoints("erf")
def erf(x, name=None):
  """Computes the Gauss error function of `x` element-wise.

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
  """
  with ops.name_scope(name, "Erf", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      x_erf = gen_math_ops.erf(x.values, name=name)
      return sparse_tensor.SparseTensor(
          indices=x.indices, values=x_erf, dense_shape=x.dense_shape)
    else:
      return gen_math_ops.erf(x, name=name)


@tf_export("math.scalar_mul", "scalar_mul")
def scalar_mul(scalar, x):
  """Multiplies a scalar times a `Tensor` or `IndexedSlices` object.

  Intended for use in gradient code which might deal with `IndexedSlices`
  objects, which are easy to multiply by a scalar but more expensive to
  multiply with arbitrary tensors.

  Args:
    scalar: A 0-D scalar `Tensor`. Must have known shape.
    x: A `Tensor` or `IndexedSlices` to be scaled.

  Returns:
    `scalar * x` of the same type (`Tensor` or `IndexedSlices`) as `x`.

  Raises:
    ValueError: if scalar is not a 0-D `scalar`.
  """
  scalar = ops.convert_to_tensor(
      scalar, dtype=x.dtype.base_dtype, name="scalar")
  shape = scalar.get_shape()
  if shape.ndims == 0:
    if isinstance(x, ops.IndexedSlices):
      return ops.IndexedSlices(scalar * x.values, x.indices, x.dense_shape)
    else:
      return scalar * x
  else:
    raise ValueError("Only scalar multiply works, got shape %s" % shape)


@tf_export("math.pow", "pow")
def pow(x, y, name=None):  # pylint: disable=redefined-builtin
  r"""Computes the power of one value to another.

  Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
  corresponding elements in `x` and `y`. For example:

  ```python
  x = tf.constant([[2, 2], [3, 3]])
  y = tf.constant([[8, 16], [2, 3]])
  tf.pow(x, y)  # [[256, 65536], [9, 27]]
  ```

  Args:
    x: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
     `complex64`, or `complex128`.
    y: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
     `complex64`, or `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`.
  """
  with ops.name_scope(name, "Pow", [x]) as name:
    return gen_math_ops._pow(x, y, name=name)


# pylint: disable=redefined-builtin,redefined-outer-name
@tf_export("dtypes.complex", "complex")
def complex(real, imag, name=None):
  r"""Converts two real numbers to a complex number.

  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation returns complex numbers elementwise of the form \\(a + bj\\), where
  *a* represents the `real` part and *b* represents the `imag` part.

  The input tensors `real` and `imag` must have the same shape.

  For example:

  ```python
  real = tf.constant([2.25, 3.25])
  imag = tf.constant([4.75, 5.75])
  tf.complex(real, imag)  # [[2.25 + 4.75j], [3.25 + 5.75j]]
  ```

  Args:
    real: A `Tensor`. Must be one of the following types: `float32`,
      `float64`.
    imag: A `Tensor`. Must have the same type as `real`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64` or `complex128`.
  """
  real = ops.convert_to_tensor(real, name="real")
  imag = ops.convert_to_tensor(imag, name="imag")
  with ops.name_scope(name, "Complex", [real, imag]) as name:
    input_types = (real.dtype, imag.dtype)
    if input_types == (dtypes.float64, dtypes.float64):
      Tout = dtypes.complex128
    elif input_types == (dtypes.float32, dtypes.float32):
      Tout = dtypes.complex64
    else:
      raise TypeError("real and imag have incorrect types: "
                      "{} {}".format(real.dtype.name, imag.dtype.name))
    return gen_math_ops._complex(real, imag, Tout=Tout, name=name)


@tf_export("math.real", "real")
@deprecation.deprecated_endpoints("real")
def real(input, name=None):
  r"""Returns the real part of a complex (or real) tensor.

  Given a tensor `input`, this operation returns a tensor of type `float` that
  is the real part of each element in `input` considered as a complex number.

  For example:

  ```python
  x = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j])
  tf.real(x)  # [-2.25, 3.25]
  ```

  If `input` is already real, it is returned unchanged.

  Args:
    input: A `Tensor`. Must have numeric type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Real", [input]) as name:
    if input.dtype.is_complex:
      real_dtype = input.dtype.real_dtype
      return gen_math_ops.real(input, Tout=real_dtype, name=name)
    else:
      return input


@tf_export("math.imag", "imag")
@deprecation.deprecated_endpoints("imag")
def imag(input, name=None):
  r"""Returns the imaginary part of a complex (or real) tensor.

  Given a tensor `input`, this operation returns a tensor of type `float` that
  is the imaginary part of each element in `input` considered as a complex
  number. If `input` is real, a tensor of all zeros is returned.

  For example:

  ```python
  x = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j])
  tf.imag(x)  # [4.75, 5.75]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float`, `double`,
      `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Imag", [input]) as name:
    if input.dtype.is_complex:
      return gen_math_ops.imag(input, Tout=input.dtype.real_dtype, name=name)
    else:
      return array_ops.zeros_like(input)


@tf_export("math.angle", "angle")
@deprecation.deprecated_endpoints("angle")
def angle(input, name=None):
  r"""Returns the element-wise argument of a complex (or real) tensor.

  Given a tensor `input`, this operation returns a tensor of type `float` that
  is the argument of each element in `input` considered as a complex number.

  The elements in `input` are considered to be complex numbers of the form
  \\(a + bj\\), where *a* is the real part and *b* is the imaginary part.
  If `input` is real then *b* is zero by definition.

  The argument returned by this function is of the form \\(atan2(b, a)\\).
  If `input` is real, a tensor of all zeros is returned.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.angle(input) ==> [2.0132, 1.056]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float`, `double`,
      `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Angle", [input]) as name:
    if input.dtype.is_complex:
      return gen_math_ops.angle(input, Tout=input.dtype.real_dtype, name=name)
    else:
      return array_ops.zeros_like(input)


# pylint: enable=redefined-outer-name,redefined-builtin


@tf_export("math.round", "round")
def round(x, name=None):  # pylint: disable=redefined-builtin
  """Rounds the values of a tensor to the nearest integer, element-wise.

  Rounds half to even.  Also known as bankers rounding. If you want to round
  according to the current system rounding mode use tf::cint.
  For example:

  ```python
  x = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
  tf.round(x)  # [ 1.0, 2.0, 2.0, 2.0, -4.0 ]
  ```

  Args:
    x: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, or `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as `x`.
  """
  x = ops.convert_to_tensor(x, name="x")
  if x.dtype.is_integer:
    return x
  else:
    return gen_math_ops.round(x, name=name)


@tf_export("dtypes.cast", "cast")
def cast(x, dtype, name=None):
  base_type = dtypes.as_dtype(dtype).base_dtype
  if isinstance(x,
                (ops.Tensor, _resource_variable_type)) and base_type == x.dtype:
    return x
  with ops.name_scope(name, "Cast", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      values_cast = cast(x.values, base_type, name=name)
      x = sparse_tensor.SparseTensor(x.indices, values_cast, x.dense_shape)
    elif isinstance(x, ops.IndexedSlices):
      values_cast = cast(x.values, base_type, name=name)
      x = ops.IndexedSlices(values_cast, x.indices, x.dense_shape)
    else:
      # TODO(josh11b): If x is not already a Tensor, we could return
      # ops.convert_to_tensor(x, dtype=dtype, ...)  here, but that
      # allows some conversions that cast() can't do, e.g. casting numbers to
      # strings.
      x = ops.convert_to_tensor(x, name="x")
      if x.dtype.base_dtype != base_type:
        x = gen_math_ops.cast(x, base_type, name=name)
    if x.dtype.is_complex and base_type.is_floating:
      logging.warn("Casting complex to real discards imaginary part.")
    return x


@tf_export("dtypes.saturate_cast", "saturate_cast")
def saturate_cast(value, dtype, name=None):
  """Performs a safe saturating cast of `value` to `dtype`.

  This function casts the input to `dtype` without applying any scaling.  If
  there is a danger that values would over or underflow in the cast, this op
  applies the appropriate clamping before the cast.

  Args:
    value: A `Tensor`.
    dtype: The desired output `DType`.
    name: A name for the operation (optional).

  Returns:
    `value` safely cast to `dtype`.
  """
  # When casting to a type with smaller representable range, clamp.
  # Note that this covers casting to unsigned types as well.
  with ops.name_scope(name, "saturate_cast", [value]) as name:
    value = ops.convert_to_tensor(value, name="value")
    dtype = dtypes.as_dtype(dtype).base_dtype
    if value.dtype.min < dtype.min:
      value = gen_math_ops.maximum(value,
                                   ops.convert_to_tensor(
                                       dtype.min, dtype=value.dtype,
                                       name="min"))
    if value.dtype.max > dtype.max:
      value = gen_math_ops.minimum(value,
                                   ops.convert_to_tensor(
                                       dtype.max, dtype=value.dtype,
                                       name="max"))
    return cast(value, dtype, name=name)


@tf_export("to_float")
def to_float(x, name="ToFloat"):
  """Casts a tensor to type `float32`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `float32`.

  Raises:
    TypeError: If `x` cannot be cast to the `float32`.
  """
  return cast(x, dtypes.float32, name=name)


@tf_export("to_double")
def to_double(x, name="ToDouble"):
  """Casts a tensor to type `float64`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `float64`.

  Raises:
    TypeError: If `x` cannot be cast to the `float64`.
  """
  return cast(x, dtypes.float64, name=name)


@tf_export("to_int32")
def to_int32(x, name="ToInt32"):
  """Casts a tensor to type `int32`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `int32`.

  Raises:
    TypeError: If `x` cannot be cast to the `int32`.
  """
  return cast(x, dtypes.int32, name=name)


@tf_export("to_int64")
def to_int64(x, name="ToInt64"):
  """Casts a tensor to type `int64`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `int64`.

  Raises:
    TypeError: If `x` cannot be cast to the `int64`.
  """
  return cast(x, dtypes.int64, name=name)


@tf_export("to_bfloat16")
def to_bfloat16(x, name="ToBFloat16"):
  """Casts a tensor to type `bfloat16`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `bfloat16`.

  Raises:
    TypeError: If `x` cannot be cast to the `bfloat16`.
  """
  return cast(x, dtypes.bfloat16, name=name)


@tf_export("to_complex64")
def to_complex64(x, name="ToComplex64"):
  """Casts a tensor to type `complex64`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `complex64`.

  Raises:
    TypeError: If `x` cannot be cast to the `complex64`.
  """
  return cast(x, dtypes.complex64, name=name)


@tf_export("to_complex128")
def to_complex128(x, name="ToComplex128"):
  """Casts a tensor to type `complex128`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `complex128`.

  Raises:
    TypeError: If `x` cannot be cast to the `complex128`.
  """
  return cast(x, dtypes.complex128, name=name)


#ops.Tensor._override_operator("__neg__", gen_math_ops.neg)
#ops.Tensor._override_operator("__abs__", abs)
# __invert__ corresponds to the ~ operator.  Here we follow the numpy convention
# ~ marks an elementwise bit-wise inverse.  This is only implemented for boolean
# tensors and will throw a TypeError if used on nonboolean arrays
#ops.Tensor._override_operator("__invert__", gen_math_ops.logical_not)


def _OverrideBinaryOperatorHelper(func, op_name, clazz_object=ops.Tensor):
  """Register operators with different tensor and scalar versions.

  If `clazz_object` is `SparseTensor`, assumes `func` takes `(sp_indices,
  sp_values, sp_shape, dense)` and outputs `(new_sp_values)`.

  Args:
    func: the operator
    op_name: name of the operator being overridden
    clazz_object: class to override for.  Either `Tensor` or `SparseTensor`.
  """

  def binary_op_wrapper(x, y):
    with ops.name_scope(None, op_name, [x, y]) as name:
      if isinstance(x, ops.Tensor) and isinstance(y, ops.Tensor):
        return func(x, y, name=name)
      elif not isinstance(y, sparse_tensor.SparseTensor):
        try:
          y = ops.convert_to_tensor(y, dtype=x.dtype.base_dtype, name="y")
        except TypeError:
          # If the RHS is not a tensor, it might be a tensor aware object
          # that can implement the operator with knowledge of itself
          # and the tensor.
          if hasattr(type(y), "__r%s__" % op_name):
            return NotImplemented
          else:
            raise
      return func(x, y, name=name)

  def binary_op_wrapper_sparse(sp_x, y):
    with ops.name_scope(None, op_name, [sp_x, y]) as name:
      y = ops.convert_to_tensor(y, dtype=sp_x.dtype.base_dtype, name="y")
      return sparse_tensor.SparseTensor(sp_x.indices,
                                        func(
                                            sp_x.indices,
                                            sp_x.values,
                                            sp_x.dense_shape,
                                            y,
                                            name=name), sp_x.dense_shape)

  def r_binary_op_wrapper(y, x):
    with ops.name_scope(None, op_name, [x, y]) as name:
      x = ops.convert_to_tensor(x, dtype=y.dtype.base_dtype, name="x")
      return func(x, y, name=name)

  # Propagate func.__doc__ to the wrappers
  try:
    doc = func.__doc__
  except AttributeError:
    doc = None
  binary_op_wrapper.__doc__ = doc
  r_binary_op_wrapper.__doc__ = doc
  binary_op_wrapper_sparse.__doc__ = doc

  if clazz_object is ops.Tensor:
    clazz_object._override_operator("__%s__" % op_name, binary_op_wrapper)
    del binary_op_wrapper
    clazz_object._override_operator("__r%s__" % op_name, r_binary_op_wrapper)
    del r_binary_op_wrapper
  else:
    clazz_object._override_operator("__%s__" % op_name,
                                    binary_op_wrapper_sparse)
    del binary_op_wrapper_sparse


# Conversion table for __truediv__.  None entries mean no conversion required.
_TRUEDIV_TABLE = {
    dtypes.uint8: dtypes.float32,
    dtypes.int8: dtypes.float32,
    dtypes.uint16: dtypes.float32,
    dtypes.int16: dtypes.float32,
    dtypes.int32: dtypes.float64,
    dtypes.int64: dtypes.float64,
    dtypes.bfloat16: None,
    dtypes.float16: None,
    dtypes.float32: None,
    dtypes.float64: None,
    dtypes.complex64: None,
    dtypes.complex128: None,
}


# NOTE: the support of "sparse (true)div dense" is currently not baked in into
# "tf.(true_)div()".  Until such an API decision is made, the supported usage is
# to explicitly use the "/" operator to invoke either truediv or div.
def _sparse_dense_truediv(sp_indices, sp_values, sp_shape, y, name=None):
  """Internal helper function for 'sp_t / dense_t'."""
  with ops.name_scope(name, "truediv",
                      [sp_indices, sp_values, sp_shape, y]) as name:
    sp_values = ops.convert_to_tensor(sp_values, name="sp_values")
    y = ops.convert_to_tensor(y, name="y")
    x_dtype = sp_values.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError("x and y must have the same dtype, got %r != %r" %
                      (x_dtype, y_dtype))
    try:
      dtype = _TRUEDIV_TABLE[x_dtype]
    except KeyError:
      raise TypeError("Invalid dtype %r in __truediv__" % x_dtype)
    if dtype is not None:
      sp_values = cast(sp_values, dtype)
      y = cast(y, dtype)
    return gen_sparse_ops.sparse_dense_cwise_div(
        sp_indices, sp_values, sp_shape, y, name=name)


def _truediv_python3(x, y, name=None):
  with ops.name_scope(name, "truediv", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, name="y")
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError("x and y must have the same dtype, got %r != %r" %
                      (x_dtype, y_dtype))
    try:
      dtype = _TRUEDIV_TABLE[x_dtype]
    except KeyError:
      raise TypeError("Invalid dtype %r in __truediv__" % x_dtype)
    if dtype is not None:
      x = cast(x, dtype)
      y = cast(y, dtype)
    return gen_math_ops.real_div(x, y, name=name)


def _div_python2(x, y, name=None):
  """Divide two values using Python 2 semantics. Used for Tensor.__div__.

  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).
  Returns:
    `x / y` returns the quotient of x and y.
  """

  with ops.name_scope(name, "div", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, name="y", dtype=x.dtype.base_dtype)
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError("x and y must have the same dtype, got %r != %r" %
                      (x_dtype, y_dtype))
    if x_dtype.is_floating or x_dtype.is_complex:
      return gen_math_ops.real_div(x, y, name=name)
    else:
      return gen_math_ops.floor_div(x, y, name=name)


@tf_export("math.truediv", "truediv")
def truediv(x, y, name=None):
  """Divides x / y elementwise (using Python 3 division operator semantics).

  NOTE: Prefer using the Tensor operator or tf.divide which obey Python
  division operator semantics.

  This function forces Python 3 division operator semantics where all integer
  arguments are cast to floating types first.   This op is generated by normal
  `x / y` division in Python 3 and in Python 2.7 with
  `from __future__ import division`.  If you want integer division that rounds
  down, use `x // y` or `tf.math.floordiv`.

  `x` and `y` must have the same numeric type.  If the inputs are floating
  point, the output will have the same type.  If the inputs are integral, the
  inputs are cast to `float32` for `int8` and `int16` and `float64` for `int32`
  and `int64` (matching the behavior of Numpy).

  Args:
    x: `Tensor` numerator of numeric type.
    y: `Tensor` denominator of numeric type.
    name: A name for the operation (optional).

  Returns:
    `x / y` evaluated in floating point.

  Raises:
    TypeError: If `x` and `y` have different dtypes.
  """
  return _truediv_python3(x, y, name)


@tf_export("div")
def div(x, y, name=None):
  """Divides x / y elementwise (using Python 2 division operator semantics).

  NOTE: Prefer using the Tensor division operator or tf.divide which obey Python
  division operator semantics.

  This function divides `x` and `y`, forcing Python 2.7 semantics. That is,
  if one of `x` or `y` is a float, then the result will be a float.
  Otherwise, the output will be an integer type. Flooring semantics are used
  for integer division.

  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).
  Returns:
    `x / y` returns the quotient of x and y.
  """
  return _div_python2(x, y, name)


@tf_export("div_no_nan")
def div_no_nan(x, y, name=None):
  """Computes an unsafe divide which returns 0 if the y is zero.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    y: A `Tensor` whose dtype is compatible with `x`.
    name: A name for the operation (optional).
  Returns:
    The element-wise value of the x divided by y.
  """

  with ops.name_scope(name, "div_no_nan", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, name="y", dtype=x.dtype.base_dtype)
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError("x and y must have the same dtype, got %r != %r" %
                      (x_dtype, y_dtype))
    return gen_math_ops.div_no_nan(x, y, name=name)


# TODO(aselle): This should be removed
mod = gen_math_ops.floor_mod


# TODO(aselle): Deprecate this once all internal functionality uses
# tf.truncatediv
@tf_export("math.floordiv", "floordiv")
@deprecation.deprecated_endpoints("floordiv")
def floordiv(x, y, name=None):
  """Divides `x / y` elementwise, rounding toward the most negative integer.

  The same as `tf.div(x,y)` for integers, but uses `tf.floor(tf.div(x,y))` for
  floating point arguments so that the result is always an integer (though
  possibly an integer represented as floating point).  This op is generated by
  `x // y` floor division in Python 3 and in Python 2.7 with
  `from __future__ import division`.

  `x` and `y` must have the same type, and the result will have the same type
  as well.

  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).

  Returns:
    `x / y` rounded down.

  Raises:
    TypeError: If the inputs are complex.
  """
  with ops.name_scope(name, "floordiv", [x, y]) as name:
    return gen_math_ops.floor_div(x, y, name=name)


realdiv = gen_math_ops.real_div
tf_export("realdiv")(realdiv)
truncatediv = gen_math_ops.truncate_div
tf_export("truncatediv")(truncatediv)
# TODO(aselle): Rename this to floordiv when we can.
floor_div = gen_math_ops.floor_div
tf_export("floor_div")(floor_div)
truncatemod = gen_math_ops.truncate_mod
tf_export("truncatemod")(truncatemod)
floormod = gen_math_ops.floor_mod
tf_export("floormod", "mod")(floormod)


def _mul_dispatch(x, y, name=None):
  """Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse"."""
  is_tensor_y = isinstance(y, ops.Tensor)
  if is_tensor_y:
    return gen_math_ops.mul(x, y, name=name)
  else:
    assert isinstance(y, sparse_tensor.SparseTensor)  # Case: Dense * Sparse.
    new_vals = gen_sparse_ops.sparse_dense_cwise_mul(y.indices, y.values,
                                                     y.dense_shape, x, name)
    return sparse_tensor.SparseTensor(y.indices, new_vals, y.dense_shape)


# NOTE(aselle): When integer division is added for sparse_dense_cwise,
# div, truediv, and floordiv should be delegated appropriately for
# Python sematnics, analogous to dense cwise tensor operations.
_OverrideBinaryOperatorHelper(gen_sparse_ops.sparse_dense_cwise_div, "div",
                              sparse_tensor.SparseTensor)
_OverrideBinaryOperatorHelper(_sparse_dense_truediv, "truediv",
                              sparse_tensor.SparseTensor)
_OverrideBinaryOperatorHelper(gen_sparse_ops.sparse_dense_cwise_mul, "mul",
                              sparse_tensor.SparseTensor)

_OverrideBinaryOperatorHelper(gen_math_ops.add, "add")
_OverrideBinaryOperatorHelper(gen_math_ops.sub, "sub")
_OverrideBinaryOperatorHelper(_mul_dispatch, "mul")
_OverrideBinaryOperatorHelper(_div_python2, "div")
_OverrideBinaryOperatorHelper(_truediv_python3, "truediv")
_OverrideBinaryOperatorHelper(floordiv, "floordiv")
_OverrideBinaryOperatorHelper(gen_math_ops.floor_mod, "mod")
_OverrideBinaryOperatorHelper(pow, "pow")


@tf_export("math.logical_xor", "logical_xor")
@deprecation.deprecated_endpoints("logical_xor")
def logical_xor(x, y, name="LogicalXor"):
  """x ^ y = (x | y) & ~(x & y)."""
  # TODO(alemi) Make this a cwise op if people end up relying on it.
  return gen_math_ops.logical_and(
      gen_math_ops.logical_or(x, y),
      gen_math_ops.logical_not(gen_math_ops.logical_and(x, y)),
      name=name)


_OverrideBinaryOperatorHelper(gen_math_ops.logical_and, "and")
_OverrideBinaryOperatorHelper(gen_math_ops.logical_or, "or")
_OverrideBinaryOperatorHelper(logical_xor, "xor")

ops.Tensor._override_operator("__lt__", gen_math_ops.less)
ops.Tensor._override_operator("__le__", gen_math_ops.less_equal)
ops.Tensor._override_operator("__gt__", gen_math_ops.greater)
ops.Tensor._override_operator("__ge__", gen_math_ops.greater_equal)


@tf_export("range")
def range(start, limit=None, delta=1, dtype=None, name="range"):  # pylint: disable=redefined-builtin
  if limit is None:
    start, limit = 0, start

  with ops.name_scope(name, "Range", [start, limit, delta]) as name:
    start = ops.convert_to_tensor(start, dtype=dtype, name="start")
    limit = ops.convert_to_tensor(limit, dtype=dtype, name="limit")
    delta = ops.convert_to_tensor(delta, dtype=dtype, name="delta")

    # infer dtype if not explicitly provided
    if dtype is None:
      dtype_hierarchy = [
          dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
      ]
      assert all(arg.dtype in dtype_hierarchy for arg in [start, limit, delta])
      inferred_dtype = max(
          [arg.dtype for arg in [start, limit, delta]],
          key=dtype_hierarchy.index)

      start = cast(start, inferred_dtype)
      limit = cast(limit, inferred_dtype)
      delta = cast(delta, inferred_dtype)

    return gen_math_ops._range(start, limit, delta, name=name)

# Reduction operations
def _ReductionDims(x, axis, reduction_indices):
  """Returns range(0, rank(x)) if reduction_indices is None."""
  # TODO(aselle): Remove this after deprecation
  if reduction_indices is not None:
    if axis is not None:
      raise ValueError("Can't specify both axis' and 'reduction_indices'.")
    axis = reduction_indices
  if axis is not None:
    return axis
  else:
    # Fast path: avoid creating Rank and Range ops if ndims is known.
    rank = common_shapes.rank(x)
    if rank is not None:
      return constant_op.constant(np.arange(rank), dtype=dtypes.int32)
    if (isinstance(x, sparse_tensor.SparseTensor) and
        x.dense_shape.get_shape().is_fully_defined()):
      rank = x.dense_shape.get_shape()[0].value  # sparse.dense_shape is 1-D.
      return constant_op.constant(np.arange(rank), dtype=dtypes.int32)

    # Otherwise, we rely on Range and Rank to do the right thing at run-time.
    return range(0, array_ops.rank(x))

def _may_reduce_to_scalar(keepdims, axis, reduction_indices, output):
  """Set a reduction's output shape to be a scalar if we are certain."""
  if not common_shapes.has_fully_defined_shape(output) and (not keepdims) and (
      axis is None) and (reduction_indices is None):
    output.set_shape(())
  return output

def reduce_mean(input_tensor,
                axis=None,
                keepdims=None,
                name=None,
                reduction_indices=None,
                keep_dims=None):
  
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)

  if keepdims is None:
    keepdims = False
  return _may_reduce_to_scalar(keepdims, axis, reduction_indices,
                               gen_math_ops.mean(
                                   input_tensor,
                                   _ReductionDims(input_tensor, axis,
                                                  reduction_indices),
                                   keepdims,
                                   name=name))

def matmul(a,
           b,
           transpose_a=False,
           transpose_b=False,
           adjoint_a=False,
           adjoint_b=False,
           a_is_sparse=False,
           b_is_sparse=False,
           name=None):
  with ops.name_scope(name, "MatMul", [a, b]) as name:
    if transpose_a and adjoint_a:
      raise ValueError("Only one of transpose_a and adjoint_a can be True.")
    if transpose_b and adjoint_b:
      raise ValueError("Only one of transpose_b and adjoint_b can be True.")

    if context.executing_eagerly():
      if not isinstance(a, (ops.EagerTensor, _resource_variable_type)):
        a = ops.convert_to_tensor(a, name="a")
      if not isinstance(b, (ops.EagerTensor, _resource_variable_type)):
        b = ops.convert_to_tensor(b, name="b")
    else:
      a = ops.convert_to_tensor(a, name="a")
      b = ops.convert_to_tensor(b, name="b")

    # TODO(apassos) remove _shape_tuple here when it is not needed.
    a_shape = a._shape_tuple()  # pylint: disable=protected-access
    b_shape = b._shape_tuple()  # pylint: disable=protected-access
    if (not a_is_sparse and
        not b_is_sparse) and ((a_shape is None or len(a_shape) > 2) and
                              (b_shape is None or len(b_shape) > 2)):
      # BatchMatmul does not support transpose, so we conjugate the matrix and
      # use adjoint instead. Conj() is a noop for real matrices.
      if transpose_a:
        a = conj(a)
        adjoint_a = True
      if transpose_b:
        b = conj(b)
        adjoint_b = True
      return gen_math_ops.batch_mat_mul(
          a, b, adj_x=adjoint_a, adj_y=adjoint_b, name=name)

    # Neither matmul nor sparse_matmul support adjoint, so we conjugate
    # the matrix and use transpose instead. Conj() is a noop for real
    # matrices.
    if adjoint_a:
      a = conj(a)
      transpose_a = True
    if adjoint_b:
      b = conj(b)
      transpose_b = True

    use_sparse_matmul = False
    if a_is_sparse or b_is_sparse:
      sparse_matmul_types = [dtypes.bfloat16, dtypes.float32]
      use_sparse_matmul = (
          a.dtype in sparse_matmul_types and b.dtype in sparse_matmul_types)
    if ((a.dtype == dtypes.bfloat16 or b.dtype == dtypes.bfloat16) and
        a.dtype != b.dtype):
      # matmul currently doesn't handle mixed-precision inputs.
      use_sparse_matmul = True
    if use_sparse_matmul:
      ret = sparse_matmul(
          a,
          b,
          transpose_a=transpose_a,
          transpose_b=transpose_b,
          a_is_sparse=a_is_sparse,
          b_is_sparse=b_is_sparse,
          name=name)
      # sparse_matmul always returns float32, even with
      # bfloat16 inputs. This prevents us from configuring bfloat16 training.
      # casting to bfloat16 also matches non-sparse matmul behavior better.
      if a.dtype == dtypes.bfloat16 and b.dtype == dtypes.bfloat16:
        ret = cast(ret, dtypes.bfloat16)
      return ret
    else:
      return gen_math_ops.mat_mul(
          a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)

_OverrideBinaryOperatorHelper(matmul, "matmul")