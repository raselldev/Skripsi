from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path
import time
import uuid

import numpy as np
import six


from backend.python import context
from backend.python.framework import constant_op
from backend.python.framework import ops
from backend.python.ops import variables
from backend.python.ops import state_ops
from backend.python.ops import control_flow_ops
from backend.python.ops import io_ops
from backend.python.ops import resource_variable_ops
from backend.core import saver_pb2
#from backend.python.training import saveable_object
from backend.python.training import base as checkpointable
from backend.python.training import checkpoint_management
from backend.python.util import compat
from backend.python.framework import errors_impl as errors


get_checkpoint_state = checkpoint_management.get_checkpoint_state
update_checkpoint_state = checkpoint_management.update_checkpoint_state
generate_checkpoint_state_proto = (
    checkpoint_management.generate_checkpoint_state_proto)
latest_checkpoint = checkpoint_management.latest_checkpoint
checkpoint_exists = checkpoint_management.checkpoint_exists
get_checkpoint_mtimes = checkpoint_management.get_checkpoint_mtimes
remove_checkpoint = checkpoint_management.remove_checkpoint


_VARIABLE_OPS = set(["Variable",
                     "VariableV2",
                     "AutoReloadVariable",
                     "VarHandleOp",
                     "ReadVariableOp"])


class SaveableObject(object):
  def __init__(self, op, specs, name):
    self.op = op
    self.specs = specs
    self.name = name
    self._device = None

  @property
  def device(self):
    if self._device is None:
      self._device = self.specs[0].tensor.device
    return self._device

class SaveSpec(object):
  def __init__(self, tensor, slice_spec, name, dtype=None):
    self._tensor = tensor
    self.slice_spec = slice_spec
    self.name = name
    if callable(self._tensor):
      if dtype is None:
        raise AssertionError(
            "When passing a callable `tensor` to a SaveSpec, an explicit "
            "dtype must be provided.")
      self.dtype = dtype
    else:
      self.dtype = tensor.dtype

  @property
  def tensor(self):
    return self._tensor() if callable(self._tensor) else self._tensor

class BaseSaverBuilder(object):
  #SaveSpec = saveable_object.SaveSpec
  #SaveableObject = saveable_object.SaveableObject

  class VariableSaveable(SaveableObject):

    def __init__(self, var, slice_spec, name):
      spec = SaveSpec(var, slice_spec, name, dtype=var.dtype)
      super(BaseSaverBuilder.VariableSaveable, self).__init__(var, [spec], name)

    def restore(self, restored_tensors, restored_shapes):
      restored_tensor = restored_tensors[0]
      if restored_shapes is not None:
        restored_tensor = array_ops.reshape(restored_tensor, restored_shapes[0])
      return state_ops.assign(
          self.op,
          restored_tensor,
          validate_shape=restored_shapes is None and
          self.op.get_shape().is_fully_defined())

  class ResourceVariableSaveable(SaveableObject):

    def __init__(self, var, slice_spec, name):
      self._var_device = var.device
      self._var_shape = var.shape
      if isinstance(var, ops.Tensor):
        self.handle_op = var.op.inputs[0]
        tensor = var
      elif isinstance(var, resource_variable_ops.ResourceVariable):

        def _read_variable_closure(v):
          def f():
            with ops.device(v.device):
              x = v.read_value()
              with ops.device("/device:CPU:0"):
                return array_ops.identity(x)
          return f

        self.handle_op = var.handle
        tensor = _read_variable_closure(var)
      else:
        raise ValueError(
            "Saveable is neither a resource variable nor a read operation."
            " Got: %s" % repr(var))
      spec = SaveSpec(tensor, slice_spec, name,
                                       dtype=var.dtype)
      super(BaseSaverBuilder.ResourceVariableSaveable, self).__init__(
          var, [spec], name)

    def restore(self, restored_tensors, restored_shapes):
      restored_tensor = restored_tensors[0]
      if restored_shapes is not None:
        restored_tensor = array_ops.reshape(restored_tensor, restored_shapes[0])
      with ops.device(self._var_device):
        restored_tensor = array_ops.identity(restored_tensor)
        return resource_variable_ops.shape_safe_assign_variable_handle(
            self.handle_op, self._var_shape, restored_tensor)

  def __init__(self, write_version=saver_pb2.SaverDef.V2):
    self._write_version = write_version

  def save_op(self, filename_tensor, saveables):
    tensor_names = []
    tensors = []
    tensor_slices = []
    for saveable in saveables:
      for spec in saveable.specs:
        tensor_names.append(spec.name)
        tensors.append(spec.tensor)
        tensor_slices.append(spec.slice_spec)
    if self._write_version == saver_pb2.SaverDef.V1:
      return io_ops._save(
          filename=filename_tensor,
          tensor_names=tensor_names,
          tensors=tensors,
          tensor_slices=tensor_slices)
    elif self._write_version == saver_pb2.SaverDef.V2:
      return io_ops.save_v2(filename_tensor, tensor_names, tensor_slices,
                            tensors)
    else:
      raise RuntimeError("Unexpected write_version: " + self._write_version)

  def bulk_restore(self, filename_tensor, saveables, preferred_shard,
                   restore_sequentially):
    del restore_sequentially
    all_tensors = []
    for saveable in saveables:
      with ops.device(_set_cpu0(saveable.device) if saveable.device else None):
        all_tensors.extend(
            self.restore_op(filename_tensor, saveable, preferred_shard))
    return all_tensors

  def restore_op(self, filename_tensor, saveable, preferred_shard):
    tensors = []
    for spec in saveable.specs:
      tensors.append(
          io_ops.restore_v2(
              filename_tensor,
              [spec.name],
              [spec.slice_spec],
              [spec.dtype])[0])

    return tensors

  def sharded_filename(self, filename_tensor, shard, num_shards):
    return gen_io_ops.sharded_filename(filename_tensor, shard, num_shards)

  def _AddSaveOps(self, filename_tensor, saveables):
    save = self.save_op(filename_tensor, saveables)
    return control_flow_ops.with_dependencies([save], filename_tensor)

  def _AddShardedSaveOpsForV2(self, checkpoint_prefix, per_device):
    _SHARDED_SUFFIX = "_temp_%s/part" % uuid.uuid4().hex
    tmp_checkpoint_prefix = string_ops.string_join(
        [checkpoint_prefix, _SHARDED_SUFFIX])

    num_shards = len(per_device)
    sharded_saves = []
    sharded_prefixes = []
    num_shards_tensor = constant_op.constant(num_shards, name="num_shards")
    last_device = None
    for shard, (device, saveables) in enumerate(per_device):
      last_device = device
      with ops.device(_set_cpu0(device)):
        sharded_filename = self.sharded_filename(tmp_checkpoint_prefix, shard,
                                                 num_shards_tensor)
        sharded_prefixes.append(sharded_filename)
        sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))

    with ops.control_dependencies([x.op for x in sharded_saves]):
      with ops.device(_set_cpu0(last_device)):
        merge_step = gen_io_ops.merge_v2_checkpoints(
            sharded_prefixes, checkpoint_prefix, delete_old_dirs=True)
        with ops.control_dependencies([merge_step]):
          return array_ops.identity(checkpoint_prefix)

  def _AddShardedSaveOps(self, filename_tensor, per_device):
    if self._write_version == saver_pb2.SaverDef.V2:
      return self._AddShardedSaveOpsForV2(filename_tensor, per_device)

    num_shards = len(per_device)
    sharded_saves = []
    num_shards_tensor = constant_op.constant(num_shards, name="num_shards")
    for shard, (device, saveables) in enumerate(per_device):
      with ops.device(device):
        sharded_filename = self.sharded_filename(filename_tensor, shard,
                                                 num_shards_tensor)
        sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))
    with ops.control_dependencies([x.op for x in sharded_saves]):
      return gen_io_ops.sharded_filespec(filename_tensor, num_shards_tensor)

  def _AddRestoreOps(self,
                     filename_tensor,
                     saveables,
                     restore_sequentially,
                     reshape,
                     preferred_shard=-1,
                     name="restore_all"):
    all_tensors = self.bulk_restore(filename_tensor, saveables, preferred_shard,
                                    restore_sequentially)

    assign_ops = []
    idx = 0
    for saveable in saveables:
      shapes = None
      if reshape:
        shapes = []
        for spec in saveable.specs:
          v = spec.tensor
          shape = v.get_shape()
          if not shape.is_fully_defined():
            shape = array_ops.shape(v)
          shapes.append(shape)
      saveable_tensors = all_tensors[idx:idx + len(saveable.specs)]
      idx += len(saveable.specs)
      assign_ops.append(saveable.restore(saveable_tensors, shapes))

    return control_flow_ops.group(*assign_ops, name=name)

  def _AddShardedRestoreOps(self, filename_tensor, per_device,
                            restore_sequentially, reshape):
    sharded_restores = []
    for shard, (device, saveables) in enumerate(per_device):
      with ops.device(device):
        sharded_restores.append(
            self._AddRestoreOps(
                filename_tensor,
                saveables,
                restore_sequentially,
                reshape,
                preferred_shard=shard,
                name="restore_shard"))
    return control_flow_ops.group(*sharded_restores, name="restore_all")

  @staticmethod
  def _IsVariable(v):
    return isinstance(v, ops.Tensor) and v.op.type in _VARIABLE_OPS

  def _GroupByDevices(self, saveables):
    per_device = collections.defaultdict(lambda: [])
    for saveable in saveables:
      canonical_device = set(
          pydev.canonical_name(spec.tensor.device) for spec in saveable.specs)
      if len(canonical_device) != 1:
        raise ValueError("All tensors of a saveable object must be "
                         "on the same device: %s" % saveable.name)
      per_device[canonical_device.pop()].append(saveable)
    return sorted(per_device.items(), key=lambda t: t[0])

  @staticmethod
  def OpListToDict(op_list, convert_variable_to_tensor=True):
    if not isinstance(op_list, (list, tuple, set)):
      raise TypeError("Variables to save should be passed in a dict or a "
                      "list: %s" % op_list)

    op_list = sorted(op_list, key=lambda x: x.name)
    names_to_saveables = {}
    
    for var in op_list:
      if isinstance(var, SaveableObject):
        names_to_saveables[var.name] = var
      elif isinstance(var, variables.PartitionedVariable):
        if var.name in names_to_saveables:
          raise ValueError("At least two variables have the same name: %s" %
                           var.name)
        names_to_saveables[var.name] = var
      elif isinstance(var, variables.Variable) and var._save_slice_info:
        name = var._save_slice_info.full_name
        if name in names_to_saveables:
          if not isinstance(names_to_saveables[name], list):
            raise ValueError("Mixing slices and non-slices with the same name: "
                             "%s" % name)
          names_to_saveables[name].append(var)
        else:
          names_to_saveables[name] = [var]
      elif (isinstance(var, checkpointable.CheckpointableBase)
            and not isinstance(var, variables.Variable)):
        checkpointable_saveables = [
            (factory() if callable(factory) else factory)
            for factory in var._gather_saveables_for_checkpoint().values()]
        names_to_saveables.update(
            BaseSaverBuilder.OpListToDict(checkpointable_saveables))
      else:
        if context.executing_eagerly():
          if not isinstance(var, resource_variable_ops.ResourceVariable):
            raise ValueError(
                "Can only save/restore ResourceVariables when eager execution "
                "is enabled, type: %s." % type(var))
          set_var = names_to_saveables.setdefault(var._shared_name, var)
          if set_var is not var:
            raise ValueError(
                ("Two different ResourceVariable objects with the same "
                 "shared_name '%s' were passed to the Saver. This likely means "
                 "that they were created in different Graphs or isolation "
                 "contexts, and may not be checkpointed together.") %
                (var._shared_name,))
        else:
          if convert_variable_to_tensor:
            if isinstance(var, resource_variable_ops.ResourceVariable):
              var = var._graph_element  
            else:
              var = ops.internal_convert_to_tensor(var, as_ref=True)
            if not BaseSaverBuilder._IsVariable(var):
              raise TypeError("Variable to save is not a Variable: %s" % var)
          if var.op.type == "ReadVariableOp":
            name = var.op.inputs[0].op.name
          else:
            name = var.op.name
          if name in names_to_saveables:
            raise ValueError("At least two variables have the same name: %s" %
                             name)
          names_to_saveables[name] = var

      
    return names_to_saveables

  @staticmethod
  def SaveableObjectsForOp(op, name):
    if not isinstance(name, six.string_types):
      raise TypeError(
          "names_to_saveables must be a dict mapping string names to "
          "checkpointable operations. Name is not a string: %s" % name)
    if isinstance(op, SaveableObject):
      yield op
    elif isinstance(op, (list, tuple, variables.PartitionedVariable)):
      if isinstance(op, variables.PartitionedVariable):
        op = list(op)
      slice_name = None
      
      for variable in op:
        if not isinstance(variable, variables.Variable):
          raise ValueError("Slices must all be Variables: %s" % variable)
        if not variable._save_slice_info:
          raise ValueError("Slices must all be slices: %s" % variable)
        if slice_name is None:
          slice_name = variable._save_slice_info.full_name
        elif slice_name != variable._save_slice_info.full_name:
          raise ValueError(
              "Slices must all be from the same tensor: %s != %s" %
              (slice_name, variable._save_slice_info.full_name))
        if variable.op.type in ["Variable", "VariableV2",
                                "AutoReloadVariable"]:
          yield BaseSaverBuilder.VariableSaveable(
              variable, variable._save_slice_info.spec, name)
        else:
          yield BaseSaverBuilder.ResourceVariableSaveable(
              variable, variable._save_slice_info.spec, name)
      
    elif isinstance(op, checkpointable.CheckpointableBase) and not isinstance(
        op, variables.Variable):
      
      for attr, factory in op._gather_saveables_for_checkpoint().items():
        op = (factory(name + "_" + attr) if callable(factory) else factory)
        for op in BaseSaverBuilder.SaveableObjectsForOp(op, op.name):
          yield op
      
    else:
      if context.executing_eagerly():
        if not isinstance(op, resource_variable_ops.ResourceVariable):
          raise ValueError("Can only save/restore ResourceVariable eager "
                           "mode is enabled, type: %s." % type(op))
        yield BaseSaverBuilder.ResourceVariableSaveable(op, "", name)
      else:
        if isinstance(op, resource_variable_ops.ResourceVariable):
          variable = op._graph_element  
        else:
          variable = ops.internal_convert_to_tensor(op, as_ref=True)
        if not BaseSaverBuilder._IsVariable(variable):
          raise TypeError("names_to_saveables must be a dict mapping string "
                          "names to Tensors/Variables. Not a variable: %s" %
                          variable)
        if variable.op.type in ["Variable", "VariableV2",
                                "AutoReloadVariable"]:
          yield BaseSaverBuilder.VariableSaveable(variable, "", name)
        else:
          yield BaseSaverBuilder.ResourceVariableSaveable(
              variable, "", name)

  def _ValidateAndSliceInputs(self, names_to_saveables):
    if not isinstance(names_to_saveables, dict):
      names_to_saveables = BaseSaverBuilder.OpListToDict(names_to_saveables)

    saveables = []
    seen_ops = set()
    for name, op in sorted(names_to_saveables.items(),
                       
                           key=lambda x: x[0]):
      for converted_saveable_object in self.SaveableObjectsForOp(op, name):
        self._AddSaveable(saveables, seen_ops, converted_saveable_object)
    return saveables

  def _AddSaveable(self, saveables, seen_ops, saveable):
    if saveable.op in seen_ops:
      raise ValueError("The same saveable will be restored with two names: %s" %
                       saveable.name)
    saveables.append(saveable)
    seen_ops.add(saveable.op)

  def build(self,
            names_to_saveables,
            reshape=False,
            sharded=False,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=10000.0,
            name=None,
            restore_sequentially=False,
            filename="model"):
    return self._build_internal(
        names_to_saveables=names_to_saveables,
        reshape=reshape,
        sharded=sharded,
        max_to_keep=max_to_keep,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        name=name,
        restore_sequentially=restore_sequentially,
        filename=filename)

  def _build_internal(self,
                      names_to_saveables,
                      reshape=False,
                      sharded=False,
                      max_to_keep=5,
                      keep_checkpoint_every_n_hours=10000.0,
                      name=None,
                      restore_sequentially=False,
                      filename="model",
                      build_save=True,
                      build_restore=True):
    if not context.executing_eagerly() and (not build_save or
                                            not build_restore):
      raise ValueError("save and restore operations need to be built together "
                       " when eager execution is not enabled.")

    saveables = self._ValidateAndSliceInputs(names_to_saveables)
    if max_to_keep is None:
      max_to_keep = 0

    with ops.name_scope(name, "save",
                        [saveable.op for saveable in saveables]) as name:
      filename_tensor = constant_op.constant(filename or "model")


      if sharded:
        per_device = self._GroupByDevices(saveables)
        if build_save:
          save_tensor = self._AddShardedSaveOps(filename_tensor, per_device)
        if build_restore:
          restore_op = self._AddShardedRestoreOps(filename_tensor, per_device,
                                                  restore_sequentially, reshape)
      else:
        if build_save:
          save_tensor = self._AddSaveOps(filename_tensor, saveables)
        if build_restore:
          restore_op = self._AddRestoreOps(filename_tensor, saveables,
                                           restore_sequentially, reshape)

    if context.executing_eagerly():
      save_tensor_name = save_tensor.numpy() if build_save else ""
      return saver_pb2.SaverDef(
          filename_tensor_name=filename_tensor.numpy(),
          save_tensor_name=save_tensor_name,
          restore_op_name="",
          max_to_keep=max_to_keep,
          sharded=sharded,
          keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
          version=self._write_version)
    else:
      graph = ops.get_default_graph()
      check_collection_list = graph.get_all_collection_keys()
      for collection_type in check_collection_list:
        for element in graph.get_collection(collection_type):
          if isinstance(element, variables.PartitionedVariable):
            try:
              graph.get_operation_by_name(element.name)
            except KeyError:
              element.as_tensor()
      return saver_pb2.SaverDef(
          filename_tensor_name=filename_tensor.name,
          save_tensor_name=save_tensor.name,
          restore_op_name=restore_op.name,
          max_to_keep=max_to_keep,
          sharded=sharded,
          keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
          version=self._write_version)

class BulkSaverBuilder(BaseSaverBuilder):
  def bulk_restore(self, filename_tensor, saveables, preferred_shard,
                   restore_sequentially):
    del restore_sequentially
    restore_specs = []
    for saveable in saveables:
      for spec in saveable.specs:
        restore_specs.append((spec.name, spec.slice_spec, spec.dtype))

    names, slices, dtypes = zip(*restore_specs)
    with ops.device("cpu:0"):
      return io_ops.restore_v2(filename_tensor, names, slices, dtypes)

class Saver(object):
  def __init__(self,
               var_list=None,
               reshape=False,
               sharded=False,
               max_to_keep=5,
               keep_checkpoint_every_n_hours=10000.0,
               name=None,
               restore_sequentially=False,
               saver_def=None,
               builder=None,
               defer_build=False,
               allow_empty=False,
               write_version=saver_pb2.SaverDef.V2,
               pad_step_number=False,
               save_relative_paths=False,
               filename=None):
    if defer_build and var_list:
      raise ValueError(
          "If `var_list` is provided then build cannot be deferred. "
          "Either set defer_build=False or var_list=None.")
    if context.executing_eagerly() and var_list is None:
      raise RuntimeError(
          "When eager execution is enabled, `var_list` must specify a list or "
          "dict of variables to save")
    self._var_list = var_list
    self._reshape = reshape
    self._sharded = sharded
    self._max_to_keep = max_to_keep
    self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    self._name = name
    self._restore_sequentially = restore_sequentially
    self.saver_def = saver_def
    self._builder = builder
    self._is_built = False
    self._allow_empty = allow_empty
    self._is_empty = None
    self._write_version = write_version
    self._pad_step_number = pad_step_number
    self._filename = filename
    self._last_checkpoints = []
    self._checkpoints_to_be_deleted = []
    if context.executing_eagerly():
      self._next_checkpoint_time = (
          time.time() + self._keep_checkpoint_every_n_hours * 3600)
    elif not defer_build:
      self.build()
    if self.saver_def:
      self._check_saver_def()
      self._write_version = self.saver_def.version
    self._save_relative_paths = save_relative_paths
    self._object_restore_saver = None

  def build(self):
    if context.executing_eagerly():
      raise RuntimeError("Use save/restore instead of build in eager mode.")
    self._build(self._filename, build_save=True, build_restore=True)


  def _build(self, checkpoint_path, build_save, build_restore):
    if not context.executing_eagerly():
      if self._is_built:
        return
      self._is_built = True

    if not self.saver_def or context.executing_eagerly():
      if self._builder is None:
        self._builder = BulkSaverBuilder(self._write_version)

      if self._var_list is None:
        
        self._var_list = variables._all_saveable_objects()
      if not self._var_list:
        if self._allow_empty:
          self._is_empty = True
          return
        else:
          raise ValueError("No variables to save")
      self._is_empty = False

      self.saver_def = self._builder._build_internal(  
          self._var_list,
          reshape=self._reshape,
          sharded=self._sharded,
          max_to_keep=self._max_to_keep,
          keep_checkpoint_every_n_hours=self._keep_checkpoint_every_n_hours,
          name=self._name,
          restore_sequentially=self._restore_sequentially,
          filename=checkpoint_path,
          build_save=build_save, build_restore=build_restore)
    elif self.saver_def and self._name:
      self.saver_def.filename_tensor_name = ops.prepend_name_scope(
          self.saver_def.filename_tensor_name, self._name)
      self.saver_def.save_tensor_name = ops.prepend_name_scope(
          self.saver_def.save_tensor_name, self._name)
      self.saver_def.restore_op_name = ops.prepend_name_scope(
          self.saver_def.restore_op_name, self._name)

    self._check_saver_def()
    if not context.executing_eagerly():
      self._next_checkpoint_time = (
          time.time() + self.saver_def.keep_checkpoint_every_n_hours * 3600)

  def _check_saver_def(self):
    if not isinstance(self.saver_def, saver_pb2.SaverDef):
      raise ValueError("saver_def must be a saver_pb2.SaverDef: %s" %
                       self.saver_def)
    if not context.executing_eagerly():
      if not self.saver_def.save_tensor_name:
        raise ValueError("saver_def must specify the save_tensor_name: %s" %
                         str(self.saver_def))
      if not self.saver_def.restore_op_name:
        raise ValueError("saver_def must specify the restore_op_name: %s" %
                         str(self.saver_def))

  def _CheckpointFilename(self, p):
    name, _ = p
    return name

  def _RecordLastCheckpoint(self, latest_save_path):
    if not self.saver_def.max_to_keep:
      return
    for p in self._last_checkpoints:
      if latest_save_path == self._CheckpointFilename(p):
        self._last_checkpoints.remove(p)
    self._last_checkpoints.append((latest_save_path, time.time()))

    if len(self._last_checkpoints) > self.saver_def.max_to_keep:
      self._checkpoints_to_be_deleted.append(self._last_checkpoints.pop(0))

  def _MaybeDeleteOldCheckpoints(self, meta_graph_suffix="meta"):
    if self._checkpoints_to_be_deleted:
      p = self._checkpoints_to_be_deleted.pop(0)
      should_keep = p[1] > self._next_checkpoint_time
      if should_keep:
        self._next_checkpoint_time += (
            self.saver_def.keep_checkpoint_every_n_hours * 3600)
        return

      try:
        checkpoint_management.remove_checkpoint(
            self._CheckpointFilename(p), self.saver_def.version,
            meta_graph_suffix)
      except Exception as e:  
        pass

  def as_saver_def(self):
    return self.saver_def

  def to_proto(self, export_scope=None):
    if export_scope is None:
      return self.saver_def

    if not (self.saver_def.filename_tensor_name.startswith(export_scope) and
            self.saver_def.save_tensor_name.startswith(export_scope) and
            self.saver_def.restore_op_name.startswith(export_scope)):
      return None

    saver_def = saver_pb2.SaverDef()
    saver_def.CopyFrom(self.saver_def)
    saver_def.filename_tensor_name = ops.strip_name_scope(
        saver_def.filename_tensor_name, export_scope)
    saver_def.save_tensor_name = ops.strip_name_scope(
        saver_def.save_tensor_name, export_scope)
    saver_def.restore_op_name = ops.strip_name_scope(
        saver_def.restore_op_name, export_scope)
    return saver_def

  @staticmethod
  def from_proto(saver_def, import_scope=None):
    return Saver(saver_def=saver_def, name=import_scope)

  @property
  def last_checkpoints(self):
    return list(self._CheckpointFilename(p) for p in self._last_checkpoints)

  def set_last_checkpoints(self, last_checkpoints):
    assert isinstance(last_checkpoints, list)
    self._last_checkpoints = [(s, np.inf) for s in last_checkpoints]

  def set_last_checkpoints_with_time(self, last_checkpoints_with_time):
    assert isinstance(last_checkpoints_with_time, list)
    self._last_checkpoints = last_checkpoints_with_time

  def recover_last_checkpoints(self, checkpoint_paths):
    mtimes = checkpoint_management.get_checkpoint_mtimes(checkpoint_paths)
    self.set_last_checkpoints_with_time(list(zip(checkpoint_paths, mtimes)))

  def save(self,
           sess,
           save_path,
           global_step=None,
           latest_filename=None,
           meta_graph_suffix="meta",
           write_meta_graph=True,
           write_state=True,
           strip_default_attrs=False):
    

    
    if not self._is_built and not context.executing_eagerly():
      raise RuntimeError(
          "`build()` should be called before save if defer_build==True")
    if latest_filename is None:
      latest_filename = "checkpoint"
    if self._write_version != saver_pb2.SaverDef.V2:
      print("salah version")

    if os.path.split(latest_filename)[0]:
      raise ValueError("'latest_filename' must not contain path components")

    if global_step is not None:
      if not isinstance(global_step, compat.integral_types):
        global_step = training_util.global_step(sess, global_step)
      checkpoint_file = "%s-%d" % (save_path, global_step)
      if self._pad_step_number:
        checkpoint_file = "%s-%s" % (save_path, "{:08d}".format(global_step))
    else:
      checkpoint_file = save_path
      if os.path.basename(
          save_path) == latest_filename and not self._sharded:
        raise ValueError(
            "'latest_filename' collides with 'save_path': '%s' and '%s'" %
            (latest_filename, save_path))

    if (not context.executing_eagerly() and
        not isinstance(sess, session.SessionInterface)):
      raise TypeError("'sess' must be a Session; %s" % sess)

    save_path_parent = os.path.dirname(save_path)
    if not self._is_empty:
      try:
        if context.executing_eagerly():
          self._build_eager(
              checkpoint_file, build_save=True, build_restore=False)
          model_checkpoint_path = self.saver_def.save_tensor_name
        else:
          model_checkpoint_path = sess.run(
              self.saver_def.save_tensor_name,
              {self.saver_def.filename_tensor_name: checkpoint_file})

        model_checkpoint_path = compat.as_str(model_checkpoint_path)
        if write_state:
          self._RecordLastCheckpoint(model_checkpoint_path)
          checkpoint_management.update_checkpoint_state_internal(
              save_dir=save_path_parent,
              model_checkpoint_path=model_checkpoint_path,
              all_model_checkpoint_paths=self.last_checkpoints,
              latest_filename=latest_filename,
              save_relative_paths=self._save_relative_paths)
          self._MaybeDeleteOldCheckpoints(meta_graph_suffix=meta_graph_suffix)
      except (errors.FailedPreconditionError, errors.NotFoundError) as exc:
        if not gfile.IsDirectory(save_path_parent):
          exc = ValueError(
              "Parent directory of {} doesn't exist, can't save.".format(
                  save_path))
        raise exc

    if write_meta_graph:
      meta_graph_filename = checkpoint_management.meta_graph_filename(
          checkpoint_file, meta_graph_suffix=meta_graph_suffix)
      if not context.executing_eagerly():
        with sess.graph.as_default():
          self.export_meta_graph(
              meta_graph_filename, strip_default_attrs=strip_default_attrs)

    if self._is_empty:
      return None
    else:
      return model_checkpoint_path

  def export_meta_graph(self,
                        filename=None,
                        collection_list=None,
                        as_text=False,
                        export_scope=None,
                        clear_devices=False,
                        clear_extraneous_savers=False,
                        strip_default_attrs=False):
    
    return export_meta_graph(
        filename=filename,
        graph_def=ops.get_default_graph().as_graph_def(add_shapes=True),
        saver_def=self.saver_def,
        collection_list=collection_list,
        as_text=as_text,
        export_scope=export_scope,
        clear_devices=clear_devices,
        clear_extraneous_savers=clear_extraneous_savers,
        strip_default_attrs=strip_default_attrs)

  def restore(self, sess, save_path):
    if self._is_empty:
      return
    if save_path is None:
      raise ValueError("Can't load save_path when it is None.")

    if not checkpoint_management.checkpoint_exists(compat.as_text(save_path)):
      raise ValueError("The passed save_path is not a valid checkpoint: "
                       + compat.as_text(save_path))
    try:
      if context.executing_eagerly():
        self._build_eager(save_path, build_save=False, build_restore=True)
      else:
        sess.run(self.saver_def.restore_op_name,
                 {self.saver_def.filename_tensor_name: save_path})
    except errors.NotFoundError as err:
      try:
        names_to_keys = object_graph_key_mapping(save_path)
      except errors.NotFoundError:
        raise _wrap_restore_error_with_msg(
            err, "a Variable name or other graph key that is missing")

      
      self._object_restore_saver = saver_from_object_based_checkpoint(
          checkpoint_path=save_path,
          var_list=self._var_list,
          builder=self._builder,
          names_to_keys=names_to_keys,
          cached_saver=self._object_restore_saver)
      self._object_restore_saver.restore(sess=sess, save_path=save_path)
    except errors.InvalidArgumentError as err:
      raise _wrap_restore_error_with_msg(
          err, "a mismatch between the current graph and the graph")
