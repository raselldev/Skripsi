from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import copy
import random
import threading


from backend.python.framework import device as pydev
from backend import pywrap_backend
from backend.util import tf_contextlib



GRAPH_MODE = 0
EAGER_MODE = 1

# Default execution mode.
default_execution_mode = GRAPH_MODE


_device_parsing_cache = {}

_MAXINT32 = 2**31 - 1


SYNC = 0
ASYNC = 1

def is_in_graph_mode():
  IS_IN_GRAPH_MODE = lambda: True

class _EagerTensorCache(object):
  def __init__(self, max_items=256, max_tensor_size=10000):
    self._data = collections.OrderedDict()
    self._max_items = max_items
    self._max_tensor_size = max_tensor_size

  def put(self, key, value):
    if value._num_elements() > self._max_tensor_size:  # pylint: disable=protected-access
      return

    self._data[key] = value

    if len(self._data) > self._max_items:
      self._data.popitem(last=False)

  def get(self, key):
    return self._data.get(key, None)

  def flush(self):
    self._data = {}


class _EagerContext(threading.local):
  def __init__(self):
    super(_EagerContext, self).__init__()
    self.device_spec = pydev.DeviceSpec.from_string("")
    self.device_name = self.device_spec.to_string()
    self.mode = default_execution_mode
    self.is_eager = default_execution_mode == EAGER_MODE
    self.scope_name = ""
    self.recording_summaries = False
    self.summary_writer_resource = None
    self.scalar_cache = {}
    self.ones_rank_cache = _EagerTensorCache()
    self.zeros_cache = _EagerTensorCache()
    self.execution_mode = None


ContextSwitch = collections.namedtuple(
    "ContextSwitch", ["is_building_function", "enter_context_fn"])


class _ContextSwitchStack(threading.local):
  def __init__(self, eager):
    super(_ContextSwitchStack, self).__init__()
    self.stack = []
    if eager:
      # Initialize the stack with a pointer to enter the eager context; this
      # ensures that the fact that eager execution was enabled is propagated
      # across threads, since (1) `enable_eager_execution` modifies a
      # process-level flag (`default_execution_mode`) and (2) `__init__` is
      # called each time a threading.local object is used in a separate thread.
      self.push(is_building_function=False, enter_context_fn=eager_mode)

  def push(self, is_building_function, enter_context_fn):
    self.stack.append(
        ContextSwitch(is_building_function, enter_context_fn))

  def pop(self):
    self.stack.pop()


class Context(object):
  def __init__(self,
               config=None,
               device_policy=None,
               execution_mode=None,
               server_def=None):
    self._eager_context = _EagerContext()
    self._context_switches = _ContextSwitchStack(self.executing_eagerly())
    self._context_handle = None
    self._context_devices = None
    self._post_execution_callbacks = []
    self._config = config
    self._seed = None
    self._initialize_lock = threading.Lock()
    self._device_policy = device_policy
    if execution_mode not in (None, SYNC, ASYNC):
      raise ValueError(
          "execution_mode should be None/SYNC/ASYNC. Got %s" % execution_mode)
    if execution_mode is None:
      execution_mode = SYNC
    self._execution_mode = execution_mode
    self._server_def = server_def

  # pylint: enable=redefined-outer-name

  def _set_global_seed(self, seed):
    self._seed = seed
    self._rng = random.Random(self._seed)
    # Also clear the kernel cache, to reset any existing seeds
    if self._context_handle is not None:
      pywrap_backend.TFE_ContextClearCaches(self._context_handle)

  def _internal_operation_seed(self):
    return self._rng.randint(0, _MAXINT32)

  def _initialize_devices(self):
    # Store list of devices
    self._context_devices = []
    device_list = pywrap_backend.TFE_ContextListDevices(
        self._context_handle)
    try:
      self._num_gpus = 0
      for i in range(pywrap_backend.TF_DeviceListCount(device_list)):
        dev_name = pywrap_backend.TF_DeviceListName(device_list, i)
        self._context_devices.append(pydev.canonical_name(dev_name))
        dev_type = pywrap_backend.TF_DeviceListType(device_list, i)
        if dev_type == "GPU":
          self._num_gpus += 1

    finally:
      pywrap_backend.TF_DeleteDeviceList(device_list)

  def _initialize_handle_and_devices(self):
    with self._initialize_lock:
      if self._context_handle is not None:
        return
      assert self._context_devices is None
      opts = pywrap_backend.TFE_NewContextOptions()
      try:
        if self._config is not None:
          config_str = self._config.SerializeToString()
          pywrap_backend.TFE_ContextOptionsSetConfig(opts, config_str)
        if self._device_policy is not None:
          pywrap_backend.TFE_ContextOptionsSetDevicePlacementPolicy(
              opts, self._device_policy)
        if self._execution_mode == ASYNC:
          pywrap_backend.TFE_ContextOptionsSetAsync(opts, True)
        self._context_handle = pywrap_backend.TFE_NewContext(opts)
      finally:
        pywrap_backend.TFE_DeleteContextOptions(opts)
      if self._server_def is not None:
        server_def_str = self._server_def.SerializeToString()
        pywrap_backend.TFE_ContextSetServerDef(self._context_handle, 600,
                                                  server_def_str)

      self._initialize_devices()

  def _clear_caches(self):
    self.scalar_cache().clear()
    self.ones_rank_cache().flush()
    self.zeros_cache().flush()

  def set_server_def(self, server_def, keep_alive_secs=600):
    if not server_def:
      raise ValueError("server_def is None.")
    if not self._context_handle:
      self._server_def = server_def
    else:
      server_def_str = server_def.SerializeToString()
      pywrap_backend.TFE_ContextSetServerDef(self._context_handle,
                                                keep_alive_secs, server_def_str)

      # Clear all the caches in case there are remote tensors in them.
      self._clear_caches()

      self._initialize_devices()

  @property
  def _handle(self):
    ctx = self._context_handle
    if ctx is None:
      self._initialize_handle_and_devices()
      return self._context_handle
    else:
      return ctx

  @property
  def _devices(self):
    devices = self._context_devices
    if devices is None:
      self._initialize_handle_and_devices()
      return self._context_devices
    else:
      return devices

  def __str__(self):
    if self._context_handle is None:
      return "Eager TensorFlow Context. Devices currently uninitialized."
    else:
      devices = self._devices
      lines = ["Eager TensorFlow Context with %d devices" % (len(devices))]
      for i, d in enumerate(devices):
        lines.append("   Device %d: %s" % (i, d))
      return "\n".join(lines)

  @tf_contextlib.contextmanager
  def _mode(self, mode):
    ctx = self._eager_context
    old_mode = ctx.mode
    old_is_eager = ctx.is_eager
    ctx.mode = mode
    ctx.is_eager = mode == EAGER_MODE
    if mode == EAGER_MODE:
      # Entering graph mode does not provide us with sufficient information to
      # record a context switch; graph-based context switches are only logged
      # when a graph is registered as the default graph.
      self.context_switches.push(False, eager_mode)
    try:
      yield
    finally:
      ctx.is_eager = old_is_eager
      ctx.mode = old_mode
      if mode == EAGER_MODE:
        self.context_switches.pop()

  def executing_eagerly(self):
    return self._eager_context.is_eager

  def scalar_cache(self):
    return self._eager_context.scalar_cache

  def ones_rank_cache(self):
    return self._eager_context.ones_rank_cache

  def zeros_cache(self):
    return self._eager_context.zeros_cache

  @property
  def scope_name(self):
    return self._eager_context.scope_name

  @scope_name.setter
  def scope_name(self, s):
    self._eager_context.scope_name = s

  @property
  def summary_writer_resource(self):
    return self._eager_context.summary_writer_resource

  @summary_writer_resource.setter
  def summary_writer_resource(self, resource):
    self._eager_context.summary_writer_resource = resource

  @property
  def device_name(self):
    return self._eager_context.device_name

  @property
  def device_spec(self):
    return self._eager_context.device_spec

  @tf_contextlib.contextmanager
  def device(self, name):
    eager_context = self._eager_context
    old_device_name = eager_context.device_name
    old_device_spec = eager_context.device_spec
    cache_key = (old_device_name, name)
    try:
      new_device_name, new_device_spec = _device_parsing_cache[cache_key]
    except TypeError:
      # Error while trying to compute the cache key.
      raise ValueError("Expecting a string device name. Got %s(%s)" %
                       (type(name), name))
    except KeyError:
      # Handle a cache miss.
      if name is not None:
        if not isinstance(name, str):
          raise ValueError("Expecting a string device name. Got %s(%s)" %
                           (type(name), name))
        device_spec = pydev.DeviceSpec.from_string(name)
        if old_device_name:
          new_device_spec = copy.copy(old_device_spec)
        else:
          new_device_spec = pydev.DeviceSpec.from_string(
              "/job:localhost/replica:0/task:0/device:CPU:0")
        new_device_spec.merge_from(device_spec)
      else:
        new_device_spec = pydev.DeviceSpec.from_string("")
      new_device_name = new_device_spec.to_string()
      _device_parsing_cache[cache_key] = (new_device_name, new_device_spec)

    try:
      eager_context.device_name = new_device_name
      eager_context.device_spec = new_device_spec
      yield
    finally:
      eager_context.device_name = old_device_name
      eager_context.device_spec = old_device_spec

  def devices(self):
    return self._devices

  def get_execution_mode(self):
    mode = self._eager_context.execution_mode
    if mode is None:
      mode = self._execution_mode
    return mode

  def set_execution_mode(self, mode):
    if mode not in (None, SYNC, ASYNC):
      raise ValueError(
          "Execution mode should be None/SYNC/ASYNC. Got %s" % mode)
    if mode is None:
      mode = SYNC
    self._eager_context.execution_mode = mode
    pywrap_backend.TFE_ContextSetAsyncForThread(self._handle, mode == ASYNC)

  @tf_contextlib.contextmanager
  def execution_mode(self, mode):
    old_mode = self.get_execution_mode()
    try:
      self.set_execution_mode(mode)
      yield
    finally:
      self.set_execution_mode(old_mode)

  def async_wait(self):
    pywrap_backend.TFE_ContextAsyncWait(self._handle)

  def async_clear_error(self):
    pywrap_backend.TFE_ContextAsyncClearError(self._handle)

  def num_gpus(self):
    self._initialize_handle_and_devices()
    return self._num_gpus

  def add_function(self, fn):
    pywrap_backend.TFE_ContextAddFunction(self._handle, fn)

  def add_function_def(self, fdef):
    fdef_string = fdef.SerializeToString()
    pywrap_backend.TFE_ContextAddFunctionDef(
        self._handle, fdef_string, len(fdef_string))

  def add_post_execution_callback(self, callback):
    self._post_execution_callbacks.append(callback)

  def clear_post_execution_callbacks(self):
    del self._post_execution_callbacks[:]

  @property
  def post_execution_callbacks(self):
    return self._post_execution_callbacks

  def enable_run_metadata(self):
    pywrap_backend.TFE_ContextEnableRunMetadata(self._handle)

  @tf_contextlib.contextmanager
  def device_policy(self, policy):
    handle = self._handle
    old = pywrap_backend.TFE_ContextGetDevicePlacementPolicy(handle)
    pywrap_backend.TFE_ContextSetThreadLocalDevicePlacementPolicy(
        handle, policy)
    try:
      yield
    finally:
      pywrap_backend.TFE_ContextSetThreadLocalDevicePlacementPolicy(
          handle, old)

  def disable_run_metadata(self):
    if not self._context_handle:
      return
    pywrap_backend.TFE_ContextDisableRunMetadata(self._context_handle)

  def export_run_metadata(self):
    if not self._context_handle:
      return None
    with c_api_util.tf_buffer() as buffer_:
      pywrap_backend.TFE_ContextExportRunMetadata(
          self._context_handle, buffer_)
      proto_data = pywrap_backend.TF_GetBuffer(buffer_)
    run_metadata = config_pb2.RunMetadata()
    run_metadata.ParseFromString(compat.as_bytes(proto_data))
    return run_metadata

  @property
  def context_switches(self):
    return self._context_switches

  def start_step(self):
    pywrap_backend.TFE_ContextStartStep(self._handle)

  def end_step(self):
    pywrap_backend.TFE_ContextEndStep(self._handle)

_context = None
_context_lock = threading.Lock()


def _initialize_context():
  global _context
  with _context_lock:
    if _context is None:
      _context = Context()


def context():
  if _context is None:
    _initialize_context()
  return _context


def executing_eagerly():
  return context().executing_eagerly()


def graph_mode():
  return context()._mode(GRAPH_MODE)  # pylint: disable=protected-access

