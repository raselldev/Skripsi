from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import functools
import json
import weakref

import six


from backend import context
from backend.python.framework import dtypes
from backend.python.framework import ops
from backend.util import tf_decorator

CheckpointableReference = collections.namedtuple(
    "CheckpointableReference",
    [
        # The local name for this dependency.
        "name",
        # The Checkpointable object being referenced.
        "ref"
    ])


def no_automatic_dependency_tracking(method):
  def _method_wrapper(self, *args, **kwargs):
    previous_value = getattr(self, "_setattr_tracking", True)
    self._setattr_tracking = False  # pylint: disable=protected-access
    try:
      method(self, *args, **kwargs)
    finally:
      self._setattr_tracking = previous_value  # pylint: disable=protected-access

  return tf_decorator.make_decorator(
      target=method, decorator_func=_method_wrapper)


class CheckpointInitialValue(ops.Tensor):
  def __init__(self, checkpoint_position, shape=None):
    self.wrapped_value = checkpoint_position.value_tensors()[
        VARIABLE_VALUE_KEY]
    if shape:
      self.wrapped_value.set_shape(shape)
    self._checkpoint_position = checkpoint_position

  def __getattr__(self, attr):
    try:
      return getattr(self.wrapped_value, attr)
    except AttributeError:
      return self.__getattribute__(attr)

  @property
  def checkpoint_position(self):
    return self._checkpoint_position


class CheckpointableBase(object):
  @no_automatic_dependency_tracking
  def _maybe_initialize_checkpointable(self):
    if hasattr(self, "_unconditional_checkpoint_dependencies"):
      return
    self._unconditional_checkpoint_dependencies = []
    self._unconditional_dependency_names = {}
    self._unconditional_deferred_dependencies = {}
    if hasattr(self, "_update_uid"):
      raise AssertionError(
          "Internal error: the object had an update UID set before its "
          "initialization code was run.")
    self._update_uid = -1
    self._name_based_restores = set()

  def _no_dependency(self, value):
    return value

  def _name_based_attribute_restore(self, checkpoint):
    self._name_based_restores.add(checkpoint)
    if self._update_uid < checkpoint.restore_uid:
      checkpoint.eager_restore(self)
      self._update_uid = checkpoint.restore_uid

  @property
  def _checkpoint_dependencies(self):
    return self._unconditional_checkpoint_dependencies

  @property
  def _deferred_dependencies(self):
    return self._unconditional_deferred_dependencies

  def _lookup_dependency(self, name):
    return self._unconditional_dependency_names.get(name, None)

  def _add_variable_with_custom_getter(
      self, name, shape=None, dtype=dtypes.float32,
      initializer=None, getter=None, overwrite=False,
      **kwargs_for_getter):
    self._maybe_initialize_checkpointable()
    with ops.init_scope():
      if context.executing_eagerly():
        checkpoint_initializer = self._preload_simple_restoration(
            name=name, shape=shape)
      else:
        checkpoint_initializer = None
      if (checkpoint_initializer is not None
          and not (
              isinstance(initializer, CheckpointInitialValue)
              and (initializer.restore_uid
                   > checkpoint_initializer.restore_uid))):
        initializer = checkpoint_initializer
        shape = None
    new_variable = getter(
        name=name, shape=shape, dtype=dtype, initializer=initializer,
        **kwargs_for_getter)

    if not overwrite or isinstance(new_variable, CheckpointableBase):
      return self._track_checkpointable(new_variable, name=name,
                                        overwrite=overwrite)
    else:
      return new_variable

  def _preload_simple_restoration(self, name, shape):
    deferred_dependencies_list = self._deferred_dependencies.get(name, ())
    if not deferred_dependencies_list:
      return
    for checkpoint_position in deferred_dependencies_list:
      if not checkpoint_position.is_simple_variable():
        return None
    checkpoint_position = max(
        deferred_dependencies_list,
        key=lambda restore: restore.checkpoint.restore_uid)
    return CheckpointInitialValue(
        checkpoint_position=checkpoint_position, shape=shape)

  def _track_checkpointable(self, checkpointable, name, overwrite=False):
    self._maybe_initialize_checkpointable()
    if not isinstance(checkpointable, CheckpointableBase):
      raise TypeError(
          ("Checkpointable._track_checkpointable() passed type %s, not a "
           "Checkpointable.") % (type(checkpointable),))
    new_reference = CheckpointableReference(name=name, ref=checkpointable)
    current_object = self._lookup_dependency(name)
    if (current_object is not None
        and current_object is not checkpointable):
      if not overwrite:
        raise ValueError(
            ("Called Checkpointable._track_checkpointable() with name='%s', "
             "but a Checkpointable with this name is already declared as a "
             "dependency. Names must be unique (or overwrite=True).") % (name,))
      for index, (old_name, _) in enumerate(
          self._unconditional_checkpoint_dependencies):
        if name == old_name:
          self._unconditional_checkpoint_dependencies[index] = new_reference
    elif current_object is None:
      self._unconditional_checkpoint_dependencies.append(new_reference)
      self._handle_deferred_dependencies(
          name=name, checkpointable=checkpointable)
    self._unconditional_dependency_names[name] = checkpointable
    return checkpointable

  def _handle_deferred_dependencies(self, name, checkpointable):
    self._maybe_initialize_checkpointable()
    checkpointable._maybe_initialize_checkpointable()  # pylint: disable=protected-access
    deferred_dependencies_list = self._deferred_dependencies.pop(name, ())
    for checkpoint_position in sorted(
        deferred_dependencies_list,
        key=lambda restore: restore.checkpoint.restore_uid,
        reverse=True):
      checkpoint_position.restore(checkpointable)

    # Pass on any name-based restores queued in this object.
    for name_based_restore in sorted(
        self._name_based_restores,
        key=lambda checkpoint: checkpoint.restore_uid,
        reverse=True):
      checkpointable._name_based_attribute_restore(name_based_restore)  # pylint: disable=protected-access

  def _restore_from_checkpoint_position(self, checkpoint_position):
    visit_queue = collections.deque([checkpoint_position])
    restore_ops = []
    while visit_queue:
      current_position = visit_queue.popleft()
      restore_ops.extend(nest.flatten(
          current_position.checkpointable  # pylint: disable=protected-access
          ._single_restoration_from_checkpoint_position(
              checkpoint_position=current_position,
              visit_queue=visit_queue)))
    return restore_ops

  def _single_restoration_from_checkpoint_position(
      self, checkpoint_position, visit_queue):
    self._maybe_initialize_checkpointable()
    checkpoint = checkpoint_position.checkpoint
    if checkpoint.restore_uid > self._update_uid:
      restore_ops = checkpoint_position.restore_ops()
      self._update_uid = checkpoint.restore_uid
    else:
      restore_ops = ()
    for child in checkpoint_position.object_proto.children:
      child_position = _CheckpointPosition(
          checkpoint=checkpoint,
          proto_id=child.node_id)
      local_object = self._lookup_dependency(child.local_name)
      if local_object is None:
        self._deferred_dependencies.setdefault(child.local_name, []).append(
            child_position)
      else:
        if child_position.bind_object(checkpointable=local_object):
          visit_queue.append(child_position)
    return restore_ops

  def _gather_saveables_for_checkpoint(self):
    if not hasattr(self, "get_config"):
      return {}
    try:
      self.get_config()
    except NotImplementedError:
      return {}
    weak_self = weakref.ref(self)
    def _state_callback():
      dereferenced_self = weak_self()
      if dereferenced_self:
        return json.dumps(dereferenced_self,
                          default=serialization.get_json_type,
                          sort_keys=True).encode("utf8")
      else:
        return ""
    return {OBJECT_CONFIG_JSON_KEY: functools.partial(
        PythonStringStateSaveable,
        state_callback=_state_callback)}
