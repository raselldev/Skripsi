# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""For seeding individual ops based on a graph-level seed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python import context
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


DEFAULT_GRAPH_SEED = 87654321
_MAXINT32 = 2**31 - 1


def _truncate_seed(seed):
  return seed % _MAXINT32  # Truncate to fit into 32-bit integer


@tf_export('random.get_seed', 'get_seed')
@deprecation.deprecated_endpoints('get_seed')
def get_seed(op_seed):
  eager = context.executing_eagerly()

  if eager:
    global_seed = context.global_seed()
  else:
    global_seed = ops.get_default_graph().seed

  if global_seed is not None:
    if op_seed is None:
      # pylint: disable=protected-access
      if eager:
        op_seed = context.internal_operation_seed()
      else:
        op_seed = ops.get_default_graph()._last_id

    seeds = _truncate_seed(global_seed), _truncate_seed(op_seed)
  else:
    if op_seed is not None:
      seeds = DEFAULT_GRAPH_SEED, _truncate_seed(op_seed)
    else:
      seeds = None, None
  # Avoid (0, 0) as the C++ ops interpret it as nondeterminism, which would
  # be unexpected since Python docs say nondeterminism is (None, None).
  if seeds == (0, 0):
    return (0, _MAXINT32)
  return seeds


@tf_export('random.set_random_seed', 'set_random_seed')
def set_random_seed(seed):
  if context.executing_eagerly():
    context.set_global_seed(seed)
  else:
    ops.get_default_graph().seed = seed
