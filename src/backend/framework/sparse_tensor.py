from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from backend import pywrap_backend
from backend.framework import dtypes
from backend.framework import ops
from backend.framework import tensor_util


class SparseTensor(object):
  @classmethod
  def from_value(cls, sparse_tensor_value):
    if not is_sparse(sparse_tensor_value):
      raise TypeError("Neither a SparseTensor nor SparseTensorValue: %s." %
                      sparse_tensor_value)
    return SparseTensor(
        indices=sparse_tensor_value.indices,
        values=sparse_tensor_value.values,
        dense_shape=sparse_tensor_value.dense_shape)

  def __init__(self, indices, values, dense_shape):
    with ops.name_scope(None, "SparseTensor",
                        [indices, values, dense_shape]):
      indices = ops.convert_to_tensor(
          indices, name="indices", dtype=dtypes.int64)
      values = ops.internal_convert_to_tensor(
          values, name="values", as_ref=True)
      dense_shape = ops.convert_to_tensor(
          dense_shape, name="dense_shape", dtype=dtypes.int64)
    self._indices = indices
    self._values = values
    self._dense_shape = dense_shape

    indices_shape = indices.get_shape().with_rank(2)
    values_shape = values.get_shape().with_rank(1)
    dense_shape_shape = dense_shape.get_shape().with_rank(1)
    indices_shape[0].merge_with(values_shape[0])
    indices_shape[1].merge_with(dense_shape_shape[0])

  def get_shape(self):
    return tensor_util.constant_value_as_shape(self._dense_shape)

  @property
  def indices(self):
    return self._indices

  @property
  def values(self):
    return self._values

  @property
  def op(self):
    return self.values.op

  @property
  def dtype(self):
    return self._values.dtype

  @property
  def dense_shape(self):
    return self._dense_shape

  @property
  def shape(self):
    return tensor_util.constant_value_as_shape(self._dense_shape)

  @property
  def graph(self):
    return self._indices.graph

  def consumers(self):
    values_consumers = set(self._values.consumers())
    indices_consumers = set(self._indices.consumers())
    dense_shape_consumers = set(self._dense_shape.consumers())
    return list(values_consumers \
                .union(indices_consumers, dense_shape_consumers))

  def __str__(self):
    return "SparseTensor(indices=%s, values=%s, dense_shape=%s)" % (
        self._indices, self._values, self._dense_shape)




SparseTensorValue = collections.namedtuple(
    "SparseTensorValue", ["indices", "values", "dense_shape"])
pywrap_backend.RegisterType("SparseTensorValue", SparseTensorValue)
