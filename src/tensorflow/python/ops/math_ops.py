from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  

from tensorflow.python import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.util import deprecation

linspace = gen_math_ops.lin_space

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
    return gen_math_ops.mul(x, y, name=name)
  else:
    assert isinstance(y, sparse_tensor.SparseTensor)
    new_vals = gen_sparse_ops.sparse_dense_cwise_mul(y.indices, y.values,
                                                     y.dense_shape, x, name)
    return sparse_tensor.SparseTensor(y.indices, new_vals, y.dense_shape)

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