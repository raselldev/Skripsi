# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Code for backpropagation using the tape utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import six

from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export


@tf_export("GradientTape")
class GradientTape(object):
  """Record operations for automatic differentiation.

  Operations are recorded if they are executed within this context manager and
  at least one of their inputs is being "watched".

  Trainable variables (created by `tf.Variable` or `tf.get_variable`, where
  `trainable=True` is default in both cases) are automatically watched. Tensors
  can be manually watched by invoking the `watch` method on this context
  manager.

  For example, consider the function `y = x * x`. The gradient at `x = 3.0` can
  be computed as:

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape() as g:
    g.watch(x)
    y = x * x
  dy_dx = g.gradient(y, x) # Will compute to 6.0
  ```

  GradientTapes can be nested to compute higher-order derivatives. For example,

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
      gg.watch(x)
      y = x * x
    dy_dx = gg.gradient(y, x)     # Will compute to 6.0
  d2y_dx2 = g.gradient(dy_dx, x)  # Will compute to 2.0
  ```

  By default, the resources held by a GradientTape are released as soon as
  GradientTape.gradient() method is called. To compute multiple gradients over
  the same computation, create a persistent gradient tape. This allows multiple
  calls to the gradient() method as resources are released when the tape object
  is garbage collected. For example:

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = x * x
    z = y * y
  dz_dx = g.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
  dy_dx = g.gradient(y, x)  # 6.0
  del g  # Drop the reference to the tape
  ```

  By default GradientTape will automatically watch any trainable variables that
  are accessed inside the context. If you want fine grained control over which
  variables are watched you can disable automatic tracking by passing
  `watch_accessed_variables=False` to the tape constructor:

  ```python
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(variable_a)
    y = variable_a ** 2  # Gradients will be available for `variable_a`.
    z = variable_b ** 3  # No gradients will be avaialble since `variable_b` is
                         # not being watched.
  ```

  Note that when using models you should ensure that your variables exist when
  using `watch_accessed_variables=False`. Otherwise it's quite easy to make your
  first iteration not have any gradients:

  ```python
  a = tf.keras.layers.Dense(32)
  b = tf.keras.layers.Dense(32)

  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(a.variables)  # Since `a.build` has not been called at this point
                             # `a.variables` will return an empty list and the
                             # tape will not be watching anything.
    result = b(a(inputs))
    tape.gradient(result, a.variables)  # The result of this computation will be
                                        # a list of `None`s since a's variables
                                        # are not being watched.
  ```

  Note that only tensors with real or complex dtypes are differentiable.
  """

  def __init__(self, persistent=False, watch_accessed_variables=True):
    """Creates a new GradientTape.

    Args:
      persistent: Boolean controlling whether a persistent gradient tape
        is created. False by default, which means at most one call can
        be made to the gradient() method on this object.
      watch_accessed_variables: Boolean controlling whether the tape will
        automatically `watch` any (trainable) variables accessed while the tape
        is active. Defaults to True meaning gradients can be requested from any
        result computed in the tape derived from reading a trainable `Variable`.
        If False users must explicitly `watch` any `Variable`s they want to
        request gradients from.
    """
    self._tape = None
    self._persistent = persistent
    self._watch_accessed_variables = watch_accessed_variables
    self._recording = False
    self._created_eagerly = context.executing_eagerly()
    if self._created_eagerly:
      context.context().start_step()

  def __enter__(self):
    """Enters a context inside which operations are recorded on this tape."""
    self._push_tape()
    return self

  def __exit__(self, typ, value, traceback):
    """Exits the recording context, no further operations are traced."""
    if self._recording:
      self._pop_tape()

  def _push_tape(self):
    if self._recording:
      raise ValueError("Tape is already recording.")
    if self._tape is None:
      self._tape = tape.push_new_tape(
          persistent=self._persistent,
          watch_accessed_variables=self._watch_accessed_variables)
    else:
      tape.push_tape(self._tape)
    self._recording = True

  def _pop_tape(self):
    if not self._recording:
      raise ValueError("Tape is not recording.")
    tape.pop_tape(self._tape)
    self._recording = False

  def __del__(self):
    if self._created_eagerly:
      context.context().end_step()

  def watch(self, tensor):
    """Ensures that `tensor` is being traced by this tape.

    Args:
      tensor: a Tensor or list of Tensors.
    """
    for t in nest.flatten(tensor):
      if hasattr(t, "handle"):
        # There are many variable-like objects, all of them currently have
        # `handle` attribute that points to a tensor. If this changes, internals
        # of watch_variable need to change as well.
        tape.watch_variable(self._tape, t)
      else:
        tape.watch(self._tape, t)

  @tf_contextlib.contextmanager
  def stop_recording(self):
    """Temporarily stops recording operations on this tape.

    Operations executed while this context manager is active will not be
    recorded on the tape. This is useful for reducing the memory used by tracing
    all computations.

    For example:

    ```
      with tf.GradientTape(persistent=True) as t:
        loss = compute_loss(model)
        with t.stop_recording():
          # The gradient computation below is not traced, saving memory.
          grads = t.gradient(loss, model.variables)
    ```

    Yields:
      None
    Raises:
      RuntimeError: if the tape is not currently recording.
    """
    if self._tape is None:
      raise RuntimeError(
          "Trying to stop recording a tape which is not recording.")
    self._pop_tape()
    try:
      yield
    finally:
      self._push_tape()

  def reset(self):
    """Clears all information stored in this tape.

    Equivalent to exiting and reentering the tape context manager with a new
    tape. For example, the two following code blocks are equivalent:
    ```
    with tf.GradientTape() as t:
      loss = loss_fn()
    with tf.GradientTape() as t:
      loss += other_loss_fn()
    t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn


    # The following is equivalent to the above
    with tf.GradientTape() as t:
      loss = loss_fn()
      t.reset()
      loss += other_loss_fn()
    t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn
    ```

    This is useful if you don't want to exit the context manager for the tape,
    or can't because the desired reset point is inside a control flow construct:

    ```
    with tf.GradientTape() as t:
      loss = ...
      if loss > k:
        t.reset()
    ```
    """
    self._pop_tape()
    self._tape = None
    self._push_tape()

  def watched_variables(self):
    """Returns variables watched by this tape in order of construction."""
    return self._tape.watched_variables()

  def gradient(self, target, sources, output_gradients=None):
    """Computes the gradient using operations recorded in context of this tape.

    Args:
      target: Tensor (or list of tensors) to be differentiated.
      sources: a list or nested structure of Tensors or Variables. `target`
        will be differentiated against elements in `sources`.
      output_gradients: a list of gradients, one for each element of
        target. Defaults to None.

    Returns:
      a list or nested structure of Tensors (or IndexedSlices, or None),
      one for each element in `sources`. Returned structure is the same as
      the structure of `sources`.

    Raises:
      RuntimeError: if called inside the context of the tape, or if called more
       than once on a non-persistent tape.
    """
    if self._tape is None:
      raise RuntimeError("GradientTape.gradient can only be called once on "
                         "non-persistent tapes.")
    if self._recording:
      if not self._persistent:
        self._pop_tape()
      else:
        logging.log_first_n(logging.WARN,
                            "Calling GradientTape.gradient on a persistent "
                            "tape inside it's context is significantly less "
                            "efficient than calling it outside the context (it "
                            "causes the gradient ops to be recorded on the "
                            "tape, leading to increased CPU and memory usage). "
                            "Only call GradientTape.gradient inside the "
                            "context if you actually want to trace the "
                            "gradient in order to compute higher order "
                            "derrivatives.", 1)

    flat_sources = nest.flatten(sources)
    flat_sources = [_handle_or_self(x) for x in flat_sources]

    if output_gradients is not None:
      output_gradients = [None if x is None else ops.convert_to_tensor(x)
                          for x in nest.flatten(output_gradients)]

    flat_grad = imperative_grad.imperative_grad(
        self._tape,
        nest.flatten(target),
        flat_sources,
        output_gradients=output_gradients)

    if not self._persistent:
      self._tape = None

    grad = nest.pack_sequence_as(sources, flat_grad)
    return grad
