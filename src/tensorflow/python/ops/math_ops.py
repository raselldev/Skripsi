from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  
import collections as _collections
import six as _six

from tensorflow.python import context
from tensorflow.python import execute
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor

from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.util import deprecation
from tensorflow.core import op_def_pb2
from tensorflow.python.framework import op_def_library
from tensorflow.python.framework import op_def_registry

#linspace = gen_math_ops.lin_space

_resource_variable_type = None

def subtract(x, y, name=None):
  return gen_math_ops.sub(x, y, name)

def negative(x, name=None):
  with ops.name_scope(name, "Neg", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      x_neg = gen_math_ops.neg(x.values, name=name)
      return sparse_tensor.SparseTensor(
          indices=x.indices, values=x_neg, dense_shape=x.dense_shape)
    else:
      return gen_math_ops.neg(x, name=name)

def _neg(x, name=None):
  return negative(x, name)

def sign(x, name=None):
  with ops.name_scope(name, "Sign", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      x_sign = gen_math_ops.sign(x.values, name=name)
      return sparse_tensor.SparseTensor(
          indices=x.indices, values=x_sign, dense_shape=x.dense_shape)
    else:
      return gen_math_ops.sign(x, name=name)

def square(x, name=None):
  with ops.name_scope(name, "Square", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      x_square = gen_math_ops.square(x.values, name=name)
      return sparse_tensor.SparseTensor(
          indices=x.indices, values=x_square, dense_shape=x.dense_shape)
    else:
      return gen_math_ops.square(x, name=name)

def sqrt(x, name=None):
  with ops.name_scope(name, "Sqrt", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      x_sqrt = gen_math_ops.sqrt(x.values, name=name)
      return sparse_tensor.SparseTensor(
          indices=x.indices, values=x_sqrt, dense_shape=x.dense_shape)
    else:
      return gen_math_ops.sqrt(x, name=name)

def erf(x, name=None):
  with ops.name_scope(name, "Erf", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      x_erf = gen_math_ops.erf(x.values, name=name)
      return sparse_tensor.SparseTensor(
          indices=x.indices, values=x_erf, dense_shape=x.dense_shape)
    else:
      return gen_math_ops.erf(x, name=name)

def scalar_mul(scalar, x):
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

def pow(x, y, name=None):  
  with ops.name_scope(name, "Pow", [x]) as name:
    return gen_math_ops._pow(x, y, name=name)

def complex(real, imag, name=None):
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

def real(input, name=None):
  with ops.name_scope(name, "Real", [input]) as name:
    if input.dtype.is_complex:
      real_dtype = input.dtype.real_dtype
      return gen_math_ops.real(input, Tout=real_dtype, name=name)
    else:
      return input

def imag(input, name=None):
  with ops.name_scope(name, "Imag", [input]) as name:
    if input.dtype.is_complex:
      return gen_math_ops.imag(input, Tout=input.dtype.real_dtype, name=name)
    else:
      return array_ops.zeros_like(input)

def angle(input, name=None):
  with ops.name_scope(name, "Angle", [input]) as name:
    if input.dtype.is_complex:
      return gen_math_ops.angle(input, Tout=input.dtype.real_dtype, name=name)
    else:
      return array_ops.zeros_like(input)

def round(x, name=None):  
  x = ops.convert_to_tensor(x, name="x")
  if x.dtype.is_integer:
    return x
  else:
    return gen_math_ops.round(x, name=name)

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
      x = ops.convert_to_tensor(x, name="x")
      if x.dtype.base_dtype != base_type:
        x = gen_math_ops.cast(x, base_type, name=name)
    if x.dtype.is_complex and base_type.is_floating:
      logging.warn("Casting complex to real discards imaginary part.")
    return x

def saturate_cast(value, dtype, name=None):
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

def to_float(x, name="ToFloat"):
  return cast(x, dtypes.float32, name=name)

def to_double(x, name="ToDouble"):
  return cast(x, dtypes.float64, name=name)

def to_int32(x, name="ToInt32"):
  return cast(x, dtypes.int32, name=name)

def to_int64(x, name="ToInt64"):
  return cast(x, dtypes.int64, name=name)

def to_bfloat16(x, name="ToBFloat16"):
  return cast(x, dtypes.bfloat16, name=name)

def to_complex64(x, name="ToComplex64"):
  return cast(x, dtypes.complex64, name=name)

def to_complex128(x, name="ToComplex128"):
  return cast(x, dtypes.complex128, name=name)

ops.Tensor._override_operator("__neg__", gen_math_ops.neg)
ops.Tensor._override_operator("__abs__", abs)
ops.Tensor._override_operator("__invert__", gen_math_ops.logical_not)

def _OverrideBinaryOperatorHelper(func, op_name, clazz_object=ops.Tensor):
  def binary_op_wrapper(x, y):
    with ops.name_scope(None, op_name, [x, y]) as name:
      if isinstance(x, ops.Tensor) and isinstance(y, ops.Tensor):
        return func(x, y, name=name)
      elif not isinstance(y, sparse_tensor.SparseTensor):
        try:
          y = ops.convert_to_tensor(y, dtype=x.dtype.base_dtype, name="y")
        except TypeError:
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
  

  if clazz_object is ops.Tensor:
    clazz_object._override_operator("__%s__" % op_name, binary_op_wrapper)
    del binary_op_wrapper
    clazz_object._override_operator("__r%s__" % op_name, r_binary_op_wrapper)
    del r_binary_op_wrapper
  else:
    clazz_object._override_operator("__%s__" % op_name,
                                    binary_op_wrapper_sparse)
    del binary_op_wrapper_sparse

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

def _sparse_dense_truediv(sp_indices, sp_values, sp_shape, y, name=None):
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

def truediv(x, y, name=None):
  return _truediv_python3(x, y, name)

def div(x, y, name=None):
  return _div_python2(x, y, name)

def div_no_nan(x, y, name=None):
  with ops.name_scope(name, "div_no_nan", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, name="y", dtype=x.dtype.base_dtype)
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError("x and y must have the same dtype, got %r != %r" %
                      (x_dtype, y_dtype))
    return gen_math_ops.div_no_nan(x, y, name=name)

mod = gen_math_ops.floor_mod

def floordiv(x, y, name=None):
  with ops.name_scope(name, "floordiv", [x, y]) as name:
    return gen_math_ops.floor_div(x, y, name=name)

realdiv = gen_math_ops.real_div
truncatediv = gen_math_ops.truncate_div
floor_div = gen_math_ops.floor_div
truncatemod = gen_math_ops.truncate_mod
floormod = gen_math_ops.floor_mod

def _mul_dispatch(x, y, name=None):
  is_tensor_y = isinstance(y, ops.Tensor)
  if is_tensor_y:
    return mul(x, y, name=name)
  else:
    assert isinstance(y, sparse_tensor.SparseTensor)
    new_vals = gen_sparse_ops.sparse_dense_cwise_mul(y.indices, y.values,
                                                     y.dense_shape, x, name)
    return sparse_tensor.SparseTensor(y.indices, new_vals, y.dense_shape)



def sparse_dense_cwise_div(sp_indices, sp_values, sp_shape, dense, name=None):
  r"""Component-wise divides a SparseTensor by a dense Tensor.

  *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
  the other direction.

  Args:
    sp_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    sp_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `sp_indices`.
    sp_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    dense: A `Tensor`. Must have the same type as `sp_values`.
      `R`-D.  The dense Tensor operand.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sp_values`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "SparseDenseCwiseDiv", sp_indices=sp_indices, sp_values=sp_values,
        sp_shape=sp_shape, dense=dense, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "SparseDenseCwiseDiv", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "SparseDenseCwiseDiv", name, _ctx._post_execution_callbacks,
        sp_indices, sp_values, sp_shape, dense)
      return _result
    except _core._FallbackException:
      return sparse_dense_cwise_div_eager_fallback(
          sp_indices, sp_values, sp_shape, dense, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def sparse_dense_cwise_mul(sp_indices, sp_values, sp_shape, dense, name=None):
  r"""Component-wise multiplies a SparseTensor by a dense Tensor.

  The output locations corresponding to the implicitly zero elements in the sparse
  tensor will be zero (i.e., will not take up storage space), regardless of the
  contents of the dense tensor (even if it's +/-INF and that INF*0 == NaN).

  *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
  the other direction.

  Args:
    sp_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    sp_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `sp_indices`.
    sp_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    dense: A `Tensor`. Must have the same type as `sp_values`.
      `R`-D.  The dense Tensor operand.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sp_values`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "SparseDenseCwiseMul", sp_indices=sp_indices, sp_values=sp_values,
        sp_shape=sp_shape, dense=dense, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "SparseDenseCwiseMul", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "SparseDenseCwiseMul", name, _ctx._post_execution_callbacks,
        sp_indices, sp_values, sp_shape, dense)
      return _result
    except _core._FallbackException:
      return sparse_dense_cwise_mul_eager_fallback(
          sp_indices, sp_values, sp_shape, dense, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

_OverrideBinaryOperatorHelper(sparse_dense_cwise_div, "div",
                              sparse_tensor.SparseTensor)
_OverrideBinaryOperatorHelper(_sparse_dense_truediv, "truediv",
                              sparse_tensor.SparseTensor)
_OverrideBinaryOperatorHelper(sparse_dense_cwise_mul, "mul",
                              sparse_tensor.SparseTensor)

_OverrideBinaryOperatorHelper(gen_math_ops.add, "add")
_OverrideBinaryOperatorHelper(gen_math_ops.sub, "sub")
_OverrideBinaryOperatorHelper(_mul_dispatch, "mul")
_OverrideBinaryOperatorHelper(_div_python2, "div")
_OverrideBinaryOperatorHelper(_truediv_python3, "truediv")
_OverrideBinaryOperatorHelper(floordiv, "floordiv")
_OverrideBinaryOperatorHelper(gen_math_ops.floor_mod, "mod")
_OverrideBinaryOperatorHelper(pow, "pow")

def logical_xor(x, y, name="LogicalXor"):
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

def range(start, limit=None, delta=1, dtype=None, name="range"):
  if limit is None:
    start, limit = 0, start

  with ops.name_scope(name, "Range", [start, limit, delta]) as name:
    start = ops.convert_to_tensor(start, dtype=dtype, name="start")
    limit = ops.convert_to_tensor(limit, dtype=dtype, name="limit")
    delta = ops.convert_to_tensor(delta, dtype=dtype, name="delta")

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

def _ReductionDims(x, axis, reduction_indices):
  if reduction_indices is not None:
    if axis is not None:
      raise ValueError("Can't specify both axis' and 'reduction_indices'.")
    axis = reduction_indices
  if axis is not None:
    return axis
  else:
    rank = rank1(x)
    if rank is not None:
      return constant_op.constant(np.arange(rank), dtype=dtypes.int32)
    if (isinstance(x, sparse_tensor.SparseTensor) and
        x.dense_shape.get_shape().is_fully_defined()):
      rank = x.dense_shape.get_shape()[0].value
      return constant_op.constant(np.arange(rank), dtype=dtypes.int32)
    return range(0, array_ops.rank(x))

def rank1(tensor):
  if isinstance(tensor, ops.Tensor):
    return tensor._rank()
  return None

def has_fully_defined_shape(tensor):
  return isinstance(tensor, ops.EagerTensor) or tensor.shape.is_fully_defined()

def _may_reduce_to_scalar(keepdims, axis, reduction_indices, output):
  if not has_fully_defined_shape(output) and (not keepdims) and (
      axis is None) and (reduction_indices is None):
    output.set_shape(())
  return output

def reduce_sum(input_tensor,
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
                               gen_math_ops._sum(
                                   input_tensor,
                                   _ReductionDims(input_tensor, axis,
                                                  reduction_indices),
                                   keepdims,
                                   name=name))

def count_nonzero(input_tensor,
                  axis=None,
                  keepdims=None,
                  dtype=dtypes.int64,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)
  if keepdims is None:
    keepdims = False

  with ops.name_scope(name, "count_nonzero", [input_tensor]):
    input_tensor = ops.convert_to_tensor(input_tensor, name="input_tensor")
    zero = array_ops.zeros([], dtype=input_tensor.dtype)
    return cast(
        reduce_sum(
            to_int64(gen_math_ops.not_equal(input_tensor, zero)),
            axis=axis,
            keepdims=keepdims,
            reduction_indices=reduction_indices),
        dtype=dtype)

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

def reduce_prod(input_tensor,
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
                               gen_math_ops.prod(
                                   input_tensor,
                                   _ReductionDims(input_tensor, axis,
                                                  reduction_indices),
                                   keepdims,
                                   name=name))

def reduce_min(input_tensor,
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
                               gen_math_ops._min(
                                   input_tensor,
                                   _ReductionDims(input_tensor, axis,
                                                  reduction_indices),
                                   keepdims,
                                   name=name))

def reduce_max(input_tensor,
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
                               gen_math_ops._max(
                                   input_tensor,
                                   _ReductionDims(input_tensor, axis,
                                                  reduction_indices),
                                   keepdims,
                                   name=name))

def reduce_all(input_tensor,
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
                               gen_math_ops._all(
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

    a_shape = a._shape_tuple()
    b_shape = b._shape_tuple()
    if (not a_is_sparse and
        not b_is_sparse) and ((a_shape is None or len(a_shape) > 2) and
                              (b_shape is None or len(b_shape) > 2)):
      if transpose_a:
        a = conj(a)
        adjoint_a = True
      if transpose_b:
        b = conj(b)
        adjoint_b = True
      return gen_math_ops.batch_mat_mul(
          a, b, adj_x=adjoint_a, adj_y=adjoint_b, name=name)

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
      if a.dtype == dtypes.bfloat16 and b.dtype == dtypes.bfloat16:
        ret = cast(ret, dtypes.bfloat16)
      return ret
    else:
      return gen_math_ops.mat_mul(
          a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)

_OverrideBinaryOperatorHelper(matmul, "matmul")

sparse_matmul = gen_math_ops.sparse_mat_mul

def conj(x, name=None):
  if isinstance(x, ops.Tensor):
    dt = x.dtype
    if dt.is_floating or dt.is_integer:
      return x
  with ops.name_scope(name, "Conj", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if x.dtype.is_complex or x.dtype == dtypes.variant:
      return gen_math_ops.conj(x, name=name)
    elif x.dtype.is_floating or x.dtype.is_integer:
      return x
    else:
      raise TypeError(
          "Expected numeric or variant tensor, got dtype %r" % x.dtype)

def lin_space(start, stop, num, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "LinSpace", start=start, stop=stop, num=num, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "Tidx", _op.get_attr("Tidx"))
    _execute.record_gradient(
      "LinSpace", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "LinSpace",
        name, _ctx._post_execution_callbacks, start, stop, num)
      return _result
    except _core._FallbackException:
      return lin_space_eager_fallback(
          start, stop, num, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def mul(x, y, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Mul", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    execute.record_gradient(
      "Mul", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Mul", name,
        _ctx._post_execution_callbacks, x, y)
      return _result
    except _core._FallbackException:
      return mul_eager_fallback(
          x, y, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)






def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib

_op_def_lib = _InitOpDefLibrary(b"\n,\n\003Abs\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\003\t\no\n\rAccumulateNV2\022\016\n\006inputs\"\001T*\001N\032\010\n\003sum\"\001T\"\014\n\001N\022\003int(\0010\001\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\016\n\005shape\022\005shape\200\001\001\220\001\001\n/\n\004Acos\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n.\n\005Acosh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n:\n\003Add\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\005\003\t\010\022\007\nW\n\004AddN\022\016\n\006inputs\"\001T*\001N\032\010\n\003sum\"\001T\"\014\n\001N\022\003int(\0010\001\"!\n\001T\022\004type:\026\n\0242\022\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\025\200\001\001\220\001\001\nA\n\005AddV2\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\032\n\001T\022\004type:\017\n\r2\013\016\023\001\002\004\006\005\003\t\010\022\200\001\001\220\001\001\nh\n\003All\022\t\n\005input\030\n\022\031\n\021reduction_indices\"\004Tidx\032\n\n\006output\030\n\"\025\n\tkeep_dims\022\004bool\032\002(\000\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\nT\n\005Angle\022\n\n\005input\"\001T\032\016\n\006output\"\004Tout\"\025\n\001T\022\004type\032\0020\010:\006\n\0042\002\010\022\"\030\n\004Tout\022\004type\032\0020\001:\006\n\0042\002\001\002\nh\n\003Any\022\t\n\005input\030\n\022\031\n\021reduction_indices\"\004Tidx\032\n\n\006output\030\n\"\025\n\tkeep_dims\022\004bool\032\002(\000\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\ni\n\020ApproximateEqual\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\031\n\ttolerance\022\005float\032\005%\254\305\'7\220\001\001\n\233\001\n\006ArgMax\022\n\n\005input\"\001T\022\021\n\tdimension\"\004Tidx\032\025\n\006output\"\013output_type\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\"\037\n\013output_type\022\004type\032\0020\t:\006\n\0042\002\003\t\n\233\001\n\006ArgMin\022\n\n\005input\"\001T\022\021\n\tdimension\"\004Tidx\032\025\n\006output\"\013output_type\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\"\037\n\013output_type\022\004type\032\0020\t:\006\n\0042\002\003\t\n/\n\004Asin\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n.\n\005Asinh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n/\n\004Atan\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n4\n\005Atan2\022\006\n\001y\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n.\n\005Atanh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\nh\n\013BatchMatMul\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\013\n\006output\"\001T\"\026\n\001T\022\004type:\013\n\t2\007\016\023\001\002\003\010\022\"\021\n\005adj_x\022\004bool\032\002(\000\"\021\n\005adj_y\022\004bool\032\002(\000\n0\n\tBesselI0e\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n0\n\tBesselI1e\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n<\n\007Betainc\022\006\n\001a\"\001T\022\006\n\001b\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\nK\n\010Bincount\022\007\n\003arr\030\003\022\010\n\004size\030\003\022\014\n\007weights\"\001T\032\t\n\004bins\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\003\t\001\002\nS\n\tBucketize\022\n\n\005input\"\001T\032\n\n\006output\030\003\"\023\n\001T\022\004type:\010\n\0062\004\003\t\001\002\"\031\n\nboundaries\022\013list(float)\nN\n\004Cast\022\t\n\001x\"\004SrcT\032\t\n\001y\"\004DstT\"\014\n\004SrcT\022\004type\"\014\n\004DstT\022\004type\"\024\n\010Truncate\022\004bool\032\002(\000\n+\n\004Ceil\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\nn\n\013ClipByValue\022\006\n\001t\"\001T\022\023\n\016clip_value_min\"\001T\022\023\n\016clip_value_max\"\001T\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\nT\n\021CompareAndBitpack\022\n\n\005input\"\001T\022\016\n\tthreshold\"\001T\032\n\n\006output\030\004\"\027\n\001T\022\004type:\014\n\n2\010\n\023\001\002\006\005\003\t\n]\n\007Complex\022\t\n\004real\"\001T\022\t\n\004imag\"\001T\032\013\n\003out\"\004Tout\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\030\n\004Tout\022\004type\032\0020\010:\006\n\0042\002\010\022\nP\n\nComplexAbs\022\006\n\001x\"\001T\032\t\n\001y\"\004Tout\"\025\n\001T\022\004type\032\0020\010:\006\n\0042\002\010\022\"\030\n\004Tout\022\004type\032\0020\001:\006\n\0042\002\001\002\n7\n\004Conj\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\026\n\001T\022\004type\032\0020\010:\007\n\0052\003\010\022\025\n,\n\003Cos\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n-\n\004Cosh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\nB\n\005Cross\022\006\n\001a\"\001T\022\006\n\001b\"\001T\032\014\n\007product\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n\221\001\n\007Cumprod\022\006\n\001x\"\001T\022\014\n\004axis\"\004Tidx\032\010\n\003out\"\001T\"\025\n\texclusive\022\004bool\032\002(\000\"\023\n\007reverse\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\220\001\n\006Cumsum\022\006\n\001x\"\001T\022\014\n\004axis\"\004Tidx\032\010\n\003out\"\001T\"\025\n\texclusive\022\004bool\032\002(\000\"\023\n\007reverse\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n.\n\007Digamma\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n:\n\003Div\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\n5\n\010DivNoNan\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\nB\n\005Equal\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\037\n\001T\022\004type:\024\n\0222\020\016\023\001\002\004\006\005\003\t\010\014\013\r\007\n\022\220\001\001\n*\n\003Erf\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n+\n\004Erfc\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n,\n\003Exp\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n.\n\005Expm1\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n,\n\005Floor\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n?\n\010FloorDiv\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\n9\n\010FloorMod\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\003\t\016\023\001\002\n=\n\007Greater\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\nB\n\014GreaterEqual\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n}\n\023HistogramFixedWidth\022\013\n\006values\"\001T\022\020\n\013value_range\"\001T\022\t\n\005nbins\030\003\032\014\n\003out\"\005dtype\"\023\n\001T\022\004type:\010\n\0062\004\003\t\001\002\"\031\n\005dtype\022\004type\032\0020\003:\006\n\0042\002\003\t\n3\n\006Igamma\022\006\n\001a\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\n8\n\013IgammaGradA\022\006\n\001a\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\n4\n\007Igammac\022\006\n\001a\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\nS\n\004Imag\022\n\n\005input\"\001T\032\016\n\006output\"\004Tout\"\025\n\001T\022\004type\032\0020\010:\006\n\0042\002\010\022\"\030\n\004Tout\022\004type\032\0020\001:\006\n\0042\002\001\002\n.\n\003Inv\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n9\n\007InvGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n.\n\010IsFinite\022\006\n\001x\"\001T\032\005\n\001y\030\n\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n+\n\005IsInf\022\006\n\001x\"\001T\032\005\n\001y\030\n\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n+\n\005IsNan\022\006\n\001x\"\001T\032\005\n\001y\030\n\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n:\n\004Less\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n?\n\tLessEqual\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\n-\n\006Lgamma\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\ni\n\010LinSpace\022\n\n\005start\"\001T\022\t\n\004stop\"\001T\022\013\n\003num\"\004Tidx\032\013\n\006output\"\001T\"\022\n\001T\022\004type:\007\n\0052\003\016\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n,\n\003Log\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n.\n\005Log1p\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n$\n\nLogicalAnd\022\005\n\001x\030\n\022\005\n\001y\030\n\032\005\n\001z\030\n\220\001\001\n\032\n\nLogicalNot\022\005\n\001x\030\n\032\005\n\001y\030\n\n#\n\tLogicalOr\022\005\n\001x\030\n\022\005\n\001y\030\n\032\005\n\001z\030\n\220\001\001\np\n\006MatMul\022\006\n\001a\"\001T\022\006\n\001b\"\001T\032\014\n\007product\"\001T\"\027\n\013transpose_a\022\004bool\032\002(\000\"\027\n\013transpose_b\022\004bool\032\002(\000\"\026\n\001T\022\004type:\013\n\t2\007\016\023\001\002\003\010\022\n\214\001\n\003Max\022\n\n\005input\"\001T\022\031\n\021reduction_indices\"\004Tidx\032\013\n\006output\"\001T\"\025\n\tkeep_dims\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n;\n\007Maximum\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\003\t\220\001\001\n\215\001\n\004Mean\022\n\n\005input\"\001T\022\031\n\021reduction_indices\"\004Tidx\032\013\n\006output\"\001T\"\025\n\tkeep_dims\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\214\001\n\003Min\022\n\n\005input\"\001T\022\031\n\021reduction_indices\"\004Tidx\032\013\n\006output\"\001T\"\025\n\tkeep_dims\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n;\n\007Minimum\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\003\t\220\001\001\n5\n\003Mod\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\026\n\001T\022\004type:\013\n\t2\007\003\t\023\023\016\001\002\n=\n\003Mul\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\220\001\001\n.\n\003Neg\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\nE\n\010NotEqual\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\005\n\001z\030\n\"\037\n\001T\022\004type:\024\n\0222\020\016\023\001\002\004\006\005\003\t\010\014\013\r\007\n\022\220\001\001\n6\n\tPolygamma\022\006\n\001a\"\001T\022\006\n\001x\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\n6\n\003Pow\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\001\023\002\003\t\010\022\n\215\001\n\004Prod\022\n\n\005input\"\001T\022\031\n\021reduction_indices\"\004Tidx\032\013\n\006output\"\001T\"\025\n\tkeep_dims\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\267\001\n\032QuantizeDownAndShrinkRange\022\017\n\005input\"\006Tinput\022\r\n\tinput_min\030\001\022\r\n\tinput_max\030\001\032\022\n\006output\"\010out_type\032\016\n\noutput_min\030\001\032\016\n\noutput_max\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\033\n\010out_type\022\004type:\t\n\0072\005\013\014\r\017\020\n\301\001\n\014QuantizedAdd\022\007\n\001x\"\002T1\022\007\n\001y\"\002T2\022\t\n\005min_x\030\001\022\t\n\005max_x\030\001\022\t\n\005min_y\030\001\022\t\n\005max_y\030\001\032\014\n\001z\"\007Toutput\032\t\n\005min_z\030\001\032\t\n\005max_z\030\001\"\025\n\002T1\022\004type:\t\n\0072\005\013\014\r\017\020\"\025\n\002T2\022\004type:\t\n\0072\005\013\014\r\017\020\"\036\n\007Toutput\022\004type\032\0020\r:\t\n\0072\005\013\014\r\017\020\220\001\001\n\235\002\n\017QuantizedMatMul\022\007\n\001a\"\002T1\022\007\n\001b\"\002T2\022\t\n\005min_a\030\001\022\t\n\005max_a\030\001\022\t\n\005min_b\030\001\022\t\n\005max_b\030\001\032\016\n\003out\"\007Toutput\032\013\n\007min_out\030\001\032\013\n\007max_out\030\001\"\025\n\002T1\022\004type:\t\n\0072\005\013\014\r\017\020\"\025\n\002T2\022\004type:\t\n\0072\005\013\014\r\017\020\"\036\n\007Toutput\022\004type\032\0020\r:\t\n\0072\005\013\014\r\017\020\"\027\n\013transpose_a\022\004bool\032\002(\000\"\027\n\013transpose_b\022\004bool\032\002(\000\"\"\n\013Tactivation\022\004type\032\0020\014:\t\n\0072\005\013\014\r\017\020\n\301\001\n\014QuantizedMul\022\007\n\001x\"\002T1\022\007\n\001y\"\002T2\022\t\n\005min_x\030\001\022\t\n\005max_x\030\001\022\t\n\005min_y\030\001\022\t\n\005max_y\030\001\032\014\n\001z\"\007Toutput\032\t\n\005min_z\030\001\032\t\n\005max_z\030\001\"\025\n\002T1\022\004type:\t\n\0072\005\013\014\r\017\020\"\025\n\002T2\022\004type:\t\n\0072\005\013\014\r\017\020\"\036\n\007Toutput\022\004type\032\0020\r:\t\n\0072\005\013\014\r\017\020\220\001\001\na\n\005Range\022\r\n\005start\"\004Tidx\022\r\n\005limit\"\004Tidx\022\r\n\005delta\"\004Tidx\032\016\n\006output\"\004Tidx\"\033\n\004Tidx\022\004type\032\0020\003:\t\n\0072\005\016\001\002\003\t\nS\n\004Real\022\n\n\005input\"\001T\032\016\n\006output\"\004Tout\"\025\n\001T\022\004type\032\0020\010:\006\n\0042\002\010\022\"\030\n\004Tout\022\004type\032\0020\001:\006\n\0042\002\001\002\n>\n\007RealDiv\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\n5\n\nReciprocal\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n@\n\016ReciprocalGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n\177\n\023RequantizationRange\022\017\n\005input\"\006Tinput\022\r\n\tinput_min\030\001\022\r\n\tinput_max\030\001\032\016\n\noutput_min\030\001\032\016\n\noutput_max\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\n\333\001\n\nRequantize\022\017\n\005input\"\006Tinput\022\r\n\tinput_min\030\001\022\r\n\tinput_max\030\001\022\030\n\024requested_output_min\030\001\022\030\n\024requested_output_max\030\001\032\022\n\006output\"\010out_type\032\016\n\noutput_min\030\001\032\016\n\noutput_max\030\001\"\031\n\006Tinput\022\004type:\t\n\0072\005\013\014\r\017\020\"\033\n\010out_type\022\004type:\t\n\0072\005\013\014\r\017\020\n+\n\004Rint\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\023\n\001T\022\004type:\010\n\0062\004\016\023\001\002\n0\n\005Round\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n.\n\005Rsqrt\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n;\n\tRsqrtGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\nt\n\nSegmentMax\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\nz\n\013SegmentMean\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\nt\n\nSegmentMin\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\nz\n\013SegmentProd\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\ny\n\nSegmentSum\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\n?\n\006Select\022\r\n\tcondition\030\n\022\006\n\001t\"\001T\022\006\n\001e\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n0\n\007Sigmoid\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n=\n\013SigmoidGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n/\n\004Sign\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n,\n\003Sin\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n-\n\004Sinh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n\301\001\n\014SparseMatMul\022\007\n\001a\"\002Ta\022\007\n\001b\"\002Tb\032\013\n\007product\030\001\"\027\n\013transpose_a\022\004bool\032\002(\000\"\027\n\013transpose_b\022\004bool\032\002(\000\"\027\n\013a_is_sparse\022\004bool\032\002(\000\"\027\n\013b_is_sparse\022\004bool\032\002(\000\"\026\n\002Ta\022\004type\032\0020\001:\006\n\0042\002\001\016\"\026\n\002Tb\022\004type\032\0020\001:\006\n\0042\002\001\016\nz\n\021SparseSegmentMean\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\217\001\n\025SparseSegmentMeanGrad\022\t\n\004grad\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\022\017\n\013output_dim0\030\003\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\311\001\n SparseSegmentMeanWithNumSegments\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n{\n\022SparseSegmentSqrtN\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\220\001\n\026SparseSegmentSqrtNGrad\022\t\n\004grad\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\022\017\n\013output_dim0\030\003\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\312\001\n!SparseSegmentSqrtNWithNumSegments\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n\203\001\n\020SparseSegmentSum\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n\322\001\n\037SparseSegmentSumWithNumSegments\022\t\n\004data\"\001T\022\017\n\007indices\"\004Tidx\022\017\n\013segment_ids\030\003\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n-\n\004Sqrt\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n:\n\010SqrtGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n1\n\006Square\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\nG\n\021SquaredDifference\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\220\001\001\n:\n\003Sub\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\n\214\001\n\003Sum\022\n\n\005input\"\001T\022\031\n\021reduction_indices\"\004Tidx\032\013\n\006output\"\001T\"\025\n\tkeep_dims\022\004bool\032\002(\000\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\004Tidx\022\004type\032\0020\003:\006\n\0042\002\003\t\n.\n\003Tan\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\027\n\001T\022\004type:\014\n\n2\010\016\023\001\002\003\t\010\022\n-\n\004Tanh\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\n:\n\010TanhGrad\022\006\n\001y\"\001T\022\007\n\002dy\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\016\023\001\002\010\022\nB\n\013TruncateDiv\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\016\023\001\002\004\006\021\005\003\t\010\022\n<\n\013TruncateMod\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\003\t\016\023\001\002\n\274\001\n\022UnsortedSegmentMax\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n\274\001\n\022UnsortedSegmentMin\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n\302\001\n\023UnsortedSegmentProd\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n\301\001\n\022UnsortedSegmentSum\022\t\n\004data\"\001T\022\027\n\013segment_ids\"\010Tindices\022\034\n\014num_segments\"\014Tnumsegments\032\013\n\006output\"\001T\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\" \n\014Tnumsegments\022\004type\032\0020\003:\006\n\0042\002\003\t\n5\n\005Xdivy\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\024\n\001T\022\004type:\t\n\0072\005\023\001\002\010\022\n5\n\005Xlogy\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\024\n\001T\022\004type:\t\n\0072\005\023\001\002\010\022\n1\n\004Zeta\022\006\n\001x\"\001T\022\006\n\001q\"\001T\032\006\n\001z\"\001T\"\021\n\001T\022\004type:\006\n\0042\002\001\002")
