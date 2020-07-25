from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops


@ops.RegisterGradient("Transpose")
def _TransposeGrad(op, grad):
  """Returns unshuffle(grad)."""
  p = op.inputs[1]
  return [array_ops.transpose(grad, array_ops.invert_permutation(p)), None]

@ops.RegisterGradient("Squeeze")
def _SqueezeGrad(op, grad):
  return _ReshapeToInput(op, grad)

def _ReshapeToInput(op, grad):
  """Reshapes the gradient to the shape of the original input."""
  return array_ops.reshape(grad, array_ops.shape(op.inputs[0]))

@ops.RegisterGradient("ExpandDims")
def _ExpandDimsGrad(op, grad):
  return [_ReshapeToInput(op, grad), None]

@ops.RegisterGradient("PlaceholderWithDefault")
@ops.RegisterGradient("Identity")
def _IdGrad(_, grad):
  return grad

@ops.RegisterGradient("ReverseV2")
def _ReverseV2Grad(op, grad):
  axis = op.inputs[1]
  return array_ops.reverse_v2(grad, axis), None

@ops.RegisterGradient("Split")
def _SplitGrad(op, *grads):
  return None, array_ops.concat(list(grads), op.inputs[0])


@ops.RegisterGradient("ConcatV2")
def _ConcatGradV2(op, grad):
  return _ConcatGradHelper(
      op, grad, start_value_index=0, end_value_index=-1, dim_index=-1)


def _ConcatGradHelper(op, grad, start_value_index, end_value_index, dim_index):
  def _CreateDenseMaskAndBegin(sizes, concat_dim):
    """Create variables for iteratively slicing a dense gradients tensor."""
    # Since shape is 1-D, shape_of_shape = [rank-of-inputs]
    shape_of_shape = array_ops.shape(sizes[0])
    # Make a vector of length equal to the input's dimensions,
    # with 0's everywhere and 1 in the concat dim position.
    # Note: Can't use sparse_to_dense since it isn't GPU-capable (for now)
    mask = array_ops.concat([
        array_ops.fill(array_ops.expand_dims(concat_dim, 0), 0), [1],
        array_ops.fill(shape_of_shape - concat_dim - 1, 0)
    ], 0)
    begin = array_ops.fill(shape_of_shape, 0)
    return mask, begin

  def _ExtractInputShapes(inputs):
    """Extract the shapes of a set of input tensors."""
    if context.executing_eagerly():
      return array_ops.shape_n(inputs)
    sizes = []
    fully_known = True
    for x in inputs:
      input_shape = array_ops.shape(x)
      if not isinstance(input_shape,
                        ops.Tensor) or input_shape.op.type != "Const":
        fully_known = False
        break
      sizes.append(input_shape)

    if fully_known:
      return sizes
    else:
      return array_ops.shape_n(inputs)

  # Degenerate concatenation, just return grad.
  if len(op.inputs) == 2:
    return grad + [None] if end_value_index <= dim_index else [None] + grad

  concat_dim = op.inputs[dim_index]
  input_values = op.inputs[start_value_index:end_value_index]

  out_grads = []
  if isinstance(grad, ops.Tensor):
    if context.executing_eagerly():
      # Using mod here for convenience since concat_dim is already verified
      # in concat implementation to be within the allowed [-rank, rank) range.
      non_neg_concat_dim = (
          concat_dim._numpy().item(0) % input_values[0]._rank())  # pylint: disable=protected-access
      # All inputs are guaranteed to be EagerTensors in eager mode
      sizes = pywrap_tensorflow.TFE_Py_TensorShapeSlice(input_values,
                                                        non_neg_concat_dim)
      out_grads = array_ops.split(grad, sizes, non_neg_concat_dim)
    else:
      if constant_op.is_constant(concat_dim):
        # If concat_dim is a constant defined in a different context,
        # then we duplicate it in the current context to avoid passing it
        # through an Enter node.
        # This is a small optimization in general, but it is required when
        # compiling with XLA, as XLA needs the concat input to be folded into a
        # constant.
        grad_context = control_flow_util.GetOutputContext(grad.op)
        dim_context = control_flow_util.GetOutputContext(concat_dim.op)
        if dim_context != grad_context:
          value = tensor_util.constant_value(concat_dim)
          concat_dim = constant_op.constant(value=value, dtype=concat_dim.dtype)

      # Using mod here for convenience since concat_dim is already verified
      # in concat implementation to be within the allowed [-rank, rank) range.
      non_neg_concat_dim = concat_dim % array_ops.rank(input_values[0])

      # Get the inputs' tensor shapes
      sizes = _ExtractInputShapes(input_values)
      # The magic number of 16 was found through benchmarking a range of sizes
      # on CPUs and a Maxwell TitanX.  A speedup was seen in a large majority of
      # cases when switching implementations at N=16, but it is possible that
      # there will be a small number of performance regressions.
      if len(sizes) > 16:
        # extract the size of each input along the concat dimension
        sizes = array_ops.squeeze(
            array_ops.slice(
                array_ops.stack(sizes, axis=1), [non_neg_concat_dim, 0],
                [1, -1]))
        out_grads = array_ops.split(grad, sizes, non_neg_concat_dim)
      else:
        offset = gen_array_ops.concat_offset(non_neg_concat_dim, sizes)
        for (begin, size) in zip(offset, sizes):
          out_grads.append(array_ops.slice(grad, begin, size))
  elif isinstance(grad, ops.IndexedSlices):
    # Using mod here for convenience since concat_dim is already verified
    # in concat implementation to be within the allowed [-rank, rank) range.
    non_neg_concat_dim = concat_dim % array_ops.rank(input_values[0])
    concat_dim_static = tensor_util.constant_value(concat_dim)
    if concat_dim_static is None:
      raise ValueError("Can only compute IndexedSlices gradient with "
                       "statically-known concat_dim")
    if concat_dim_static < 0:
      rank = tensor_util.constant_value(array_ops.rank(input_values[0]))
      if rank is None:
        raise ValueError("Can only compute IndexedSlices gradient with "
                         "negative concat_dim when first value rank is "
                         "statically-known.")
      concat_dim_static %= rank
    # Get the inputs' tensor shapes
    sizes = [array_ops.shape(x) for x in input_values]
    if concat_dim_static > 0:
      # IndexedSlices, non_neg_concat_dim > 0. Each input gets IndexedSlices
      # gradients with all the indices, but with grad.values sliced accordingly.
      # This is like the Tensor case, except shape(grad.values)[0] is not equal
      # to shape(sizes[i])[0], since only a subset of the dim-0 values are
      # stored.
      mask, begin = _CreateDenseMaskAndBegin(sizes, non_neg_concat_dim)
      for size in sizes:
        new_values = array_ops.slice(
            grad.values, begin,
            array_ops.concat([[-1], array_ops.slice(size, [1], [-1])], 0))
        out_grads.append(ops.IndexedSlices(new_values, grad.indices, size))
        # Lint complains begin = begin + ...
        begin = math_ops.add(begin, size * mask)
    else:
      # IndexedSlices, concat_dim == 0. Each input gets IndexedSlices gradients
      # only for the relevant indices.
      start = constant_op.constant(0, dtype=grad.indices.dtype)
      for size in sizes:
        size_concat_dim = array_ops.gather(size, non_neg_concat_dim)
        if size_concat_dim.dtype != grad.indices.dtype:
          size_concat_dim = math_ops.cast(
              size_concat_dim, dtype=grad.indices.dtype)
        end = start + size_concat_dim
        # Compute the 1-D Tensor of indices relevant for this input.
        indices_to_select = array_ops.squeeze(
            array_ops.where(
                math_ops.logical_and(grad.indices >= start,
                                     grad.indices < end)),
            axis=[1])
        new_indices = array_ops.gather(grad.indices, indices_to_select) - start
        new_values = array_ops.gather(grad.values, indices_to_select)
        out_grads.append(ops.IndexedSlices(new_values, new_indices, size))
        start = end
  else:
    raise TypeError("Expected Tensor or IndexedSlices, got %s" % type(grad))

  return (out_grads + [None]
          if end_value_index <= dim_index else [None] + out_grads)

