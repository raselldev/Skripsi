from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from tensorflow.python import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.util.tf_export import tf_export


newaxis = None
tf_export("newaxis").export_constant(__name__, "newaxis")

_BaseSlice = slice


def shape_internal(input, name=None, optimize=True, out_type=dtypes.int32):
  with ops.name_scope(name, "Shape", [input]) as name:
    if isinstance(input, (sparse_tensor.SparseTensor,
                          sparse_tensor.SparseTensorValue)):
      return gen_math_ops.cast(input.dense_shape, out_type)
    else:
      if not context.executing_eagerly():
        input_tensor = ops.convert_to_tensor(input)
        input_shape = input_tensor.get_shape()
        if optimize and input_shape.is_fully_defined():
          return constant(input_shape.as_list(), out_type, name=name)
      return gen_array_ops.shape(input, name=name, out_type=out_type)


def _slice_helper(tensor, slice_spec, var=None):
  if not isinstance(slice_spec, (list, tuple)):
    slice_spec = [slice_spec]

  begin, end, strides = [], [], []
  index = 0

  new_axis_mask, shrink_axis_mask = 0, 0
  begin_mask, end_mask = 0, 0
  ellipsis_mask = 0
  for s in slice_spec:
    if isinstance(s, _BaseSlice):
      if s.start is not None and s.start is not sys.maxsize:
        begin.append(s.start)
      else:
        begin.append(0)
        begin_mask |= (1 << index)
      if s.stop is not None and s.stop != sys.maxsize:
        end.append(s.stop)
      else:
        end.append(0)
        end_mask |= (1 << index)
      if s.step is not None:
        strides.append(s.step)
      else:
        strides.append(1)
    elif s is Ellipsis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      ellipsis_mask |= (1 << index)
    elif s is newaxis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      new_axis_mask |= (1 << index)
    else:
      begin.append(s)
      end.append(s + 1)
      strides.append(1)
      shrink_axis_mask |= (1 << index)
    index += 1

  with ops.name_scope(None, "strided_slice",
                      [tensor] + begin + end + strides) as name:
    if begin:
      packed_begin, packed_end, packed_strides = (stack(begin), stack(end),
                                                  stack(strides))
      if (packed_begin.dtype == dtypes.int64 or
          packed_end.dtype == dtypes.int64 or
          packed_strides.dtype == dtypes.int64):
        if packed_begin.dtype != dtypes.int64:
          packed_begin = gen_math_ops.cast(packed_begin, dtypes.int64)
        if packed_end.dtype != dtypes.int64:
          packed_end = gen_math_ops.cast(packed_end, dtypes.int64)
        if packed_strides.dtype != dtypes.int64:
          packed_strides = gen_math_ops.cast(packed_strides, dtypes.int64)
    else:
      var_empty = constant([], dtype=dtypes.int32)
      packed_begin = packed_end = packed_strides = var_empty
    return strided_slice(
        tensor,
        packed_begin,
        packed_end,
        packed_strides,
        begin_mask=begin_mask,
        end_mask=end_mask,
        shrink_axis_mask=shrink_axis_mask,
        new_axis_mask=new_axis_mask,
        ellipsis_mask=ellipsis_mask,
        var=var,
        name=name)


@tf_export("slice")
def slice(input_, begin, size, name=None):
  return gen_array_ops._slice(input_, begin, size, name=name)


@tf_export("strided_slice")
def strided_slice(input_,
                  begin,
                  end,
                  strides=None,
                  begin_mask=0,
                  end_mask=0,
                  ellipsis_mask=0,
                  new_axis_mask=0,
                  shrink_axis_mask=0,
                  var=None,
                  name=None):

  if strides is None:
    strides = ones_like(begin)

  op = gen_array_ops.strided_slice(
      input=input_,
      begin=begin,
      end=end,
      strides=strides,
      name=name,
      begin_mask=begin_mask,
      end_mask=end_mask,
      ellipsis_mask=ellipsis_mask,
      new_axis_mask=new_axis_mask,
      shrink_axis_mask=shrink_axis_mask)

  parent_name = name

  if not (var is None and isinstance(op, ops.EagerTensor)):
    def assign(val, name=None):

      if var is None:
        raise ValueError("Sliced assignment is only supported for variables")

      if name is None:
        name = parent_name + "_assign"

      return var._strided_slice_assign(
          begin=begin,
          end=end,
          strides=strides,
          value=val,
          name=name,
          begin_mask=begin_mask,
          end_mask=end_mask,
          ellipsis_mask=ellipsis_mask,
          new_axis_mask=new_axis_mask,
          shrink_axis_mask=shrink_axis_mask)

    op.assign = assign
  return op


def _SliceHelperVar(var, slice_spec):
  return _slice_helper(var._AsTensor(), slice_spec, var)


ops.Tensor._override_operator("__getitem__", _slice_helper)



@tf_export("stack")
def stack(values, axis=0, name="stack"):
  if axis == 0:
    try:
      return ops.convert_to_tensor(values, name=name)
    except (TypeError, ValueError):
      pass 

  value_shape = ops.convert_to_tensor(values[0], name=name)._shape_tuple()
  if value_shape is not None:
    expanded_num_dims = len(value_shape) + 1
    if axis < -expanded_num_dims or axis >= expanded_num_dims:
      raise ValueError("axis = %d not in [%d, %d)" % (axis, -expanded_num_dims,
                                                      expanded_num_dims))

  return gen_array_ops.pack(values, axis=axis, name=name)


def _get_dtype_from_nested_lists(list_or_tuple):
  for elem in list_or_tuple:
    if ops.is_dense_tensor_like(elem):
      return elem.dtype.base_dtype
    elif isinstance(elem, (list, tuple)):
      maybe_dtype = _get_dtype_from_nested_lists(elem)
      if maybe_dtype is not None:
        return maybe_dtype
  return None


def _autopacking_conversion_function(v, dtype=None, name=None, as_ref=False):
  if as_ref:
    return NotImplemented
  inferred_dtype = _get_dtype_from_nested_lists(v)
  if inferred_dtype is None:
    return NotImplemented
  if dtype is None:
    dtype = inferred_dtype
  elif dtype != inferred_dtype:
    v = nest.map_structure(_cast_nested_seqs_to_dtype(dtype), v)
  return _autopacking_helper(v, dtype, name or "packed")


ops.register_tensor_conversion_function((list, tuple),
                                        _autopacking_conversion_function, 99)


@tf_export("concat")
def concat(values, axis, name="concat"):
  
  if not isinstance(values, (list, tuple)):
    values = [values]
  if len(values) == 1: 
    with ops.name_scope(name) as scope:
      ops.convert_to_tensor(
          axis, name="concat_dim",
          dtype=dtypes.int32).get_shape().assert_is_compatible_with(
              tensor_shape.scalar())
      return identity(values[0], name=scope)
  return gen_array_ops.concat_v2(values=values, axis=axis, name=name)


@tf_export("split")
def split(value, num_or_size_splits, axis=0, num=None, name="split"):
  
  size_splits = ops.convert_to_tensor(num_or_size_splits)
  if size_splits._rank() == 0 and size_splits.dtype.is_integer:
    return gen_array_ops.split(
        axis=axis, num_split=num_or_size_splits, value=value, name=name)

  if num is None:
    size_splits_shape = size_splits._shape_tuple()
    if size_splits_shape:
      num = size_splits_shape[0]
    if num is None:
      raise ValueError("Cannot infer num from shape %s" % num_or_size_splits)

  return gen_array_ops.split_v(
      value=value, size_splits=size_splits, axis=axis, num_split=num, name=name)


def _constant_if_small(value, shape, dtype, name):
  try:
    if np.prod(shape) < 1000:
      return constant(value, shape=shape, dtype=dtype, name=name)
  except TypeError:
    pass
  return None


@tf_export("zeros")
def zeros(shape, dtype=dtypes.float32, name=None):
  dtype = dtypes.as_dtype(dtype).base_dtype
  with ops.name_scope(name, "zeros", [shape]) as name:
    if dtype == dtypes.bool:
      zero = False
    elif dtype == dtypes.string:
      zero = ""
    else:
      zero = 0

    if not isinstance(shape, ops.Tensor):
      try:
        
        output = _constant_if_small(zero, shape, dtype, name)
        if output is not None:
          return output

        shape = constant_op._tensor_shape_tensor_conversion_function(
            tensor_shape.TensorShape(shape))
      except (TypeError, ValueError):
        shape = ops.convert_to_tensor(shape, dtype=dtypes.int32)
    if not shape._shape_tuple():
      shape = reshape(shape, [-1])
    output = fill(shape, constant(zero, dtype=dtype), name=name)
  assert output.dtype.base_dtype == dtype
  return output


@tf_export("zeros_like")
def zeros_like(tensor, dtype=None, name=None, optimize=True):
  with ops.name_scope(name, "zeros_like", [tensor]) as name:
    tensor = ops.convert_to_tensor(tensor, name="tensor")

    if context.executing_eagerly():
      if dtype is not None and dtype != tensor.dtype:
        return zeros(
            shape_internal(tensor, optimize=optimize), dtype=dtype, name=name)
      with ops.device(tensor.device):
        return gen_array_ops.zeros_like(tensor, name=name)

    if (optimize and tensor.shape.is_fully_defined() and
        tensor.dtype != dtypes.variant):
      return zeros(tensor.shape, dtype=dtype or tensor.dtype, name=name)

    if dtype is not None and dtype != tensor.dtype and dtype != dtypes.variant:
      return zeros(
          shape_internal(tensor, optimize=optimize), dtype=dtype, name=name)
    else:
      return gen_array_ops.zeros_like(tensor, name=name)


@tf_export("ones")
def ones(shape, dtype=dtypes.float32, name=None):
  dtype = dtypes.as_dtype(dtype).base_dtype
  with ops.name_scope(name, "ones", [shape]) as name:
    one = True if dtype == dtypes.bool else 1
    if not isinstance(shape, ops.Tensor):
      try:
        output = _constant_if_small(one, shape, dtype, name)
        if output is not None:
          return output

        shape = constant_op._tensor_shape_tensor_conversion_function(
            tensor_shape.TensorShape(shape))
      except (TypeError, ValueError):

        shape = ops.convert_to_tensor(shape, dtype=dtypes.int32)
    if not shape._shape_tuple():
      shape = reshape(shape, [-1]) 
    output = fill(shape, constant(one, dtype=dtype), name=name)
  assert output.dtype.base_dtype == dtype
  return output
