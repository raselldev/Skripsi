from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import threading
#from tensorflow.python.util.tf_export import tf_export


class DeviceSpec(object):

  def __init__(self, job=None, replica=None, task=None, device_type=None,
               device_index=None):
    self.job = job
    self.replica = replica
    self.task = task
    if device_type == "cpu" or device_type == "gpu":
      # For backwards compatibility only, we support lowercase variants of
      # cpu and gpu but turn them into uppercase here.
      self.device_type = device_type.upper()
    else:
      self.device_type = device_type
    self.device_index = device_index
    self._hash = hash(self.to_string())

  def _clear(self):
    self._job = None
    self._replica = None
    self._task = None
    self.device_type = None
    self.device_index = None

  @property
  def job(self):
    return self._job

  @job.setter
  def job(self, job):
    if job is not None:
      self._job = str(job)
    else:
      self._job = None

  @property
  def replica(self):
    return self._replica

  @replica.setter
  def replica(self, replica):
    if replica is not None:
      self._replica = int(replica)
    else:
      self._replica = None

  @property
  def task(self):
    return self._task

  @task.setter
  def task(self, task):
    if task is not None:
      self._task = int(task)
    else:
      self._task = None

  def parse_from_string(self, spec):
    self._clear()
    splits = [x.split(":") for x in spec.split("/")]
    for y in splits:
      ly = len(y)
      if y:
        # NOTE(touts): we use the property getters here.
        if ly == 2 and y[0] == "job":
          self.job = y[1]
        elif ly == 2 and y[0] == "replica":
          self.replica = y[1]
        elif ly == 2 and y[0] == "task":
          self.task = y[1]
        elif ((ly == 1 or ly == 2) and
              ((y[0].upper() == "GPU") or (y[0].upper() == "CPU"))):
          if self.device_type is not None:
            raise ValueError("Cannot specify multiple device types: %s" % spec)
          self.device_type = y[0].upper()
          if ly == 2 and y[1] != "*":
            self.device_index = int(y[1])
        elif ly == 3 and y[0] == "device":
          if self.device_type is not None:
            raise ValueError("Cannot specify multiple device types: %s" % spec)
          self.device_type = y[1]
          if y[2] != "*":
            self.device_index = int(y[2])
        elif ly and y[0] != "":  # pylint: disable=g-explicit-bool-comparison
          raise ValueError("Unknown attribute: '%s' in '%s'" % (y[0], spec))

    return self

  def merge_from(self, dev):
    if dev.job is not None:
      self.job = dev.job
    if dev.replica is not None:
      self.replica = dev.replica
    if dev.task is not None:
      self.task = dev.task
    if dev.device_type is not None:
      self.device_type = dev.device_type
    if dev.device_index is not None:
      self.device_index = dev.device_index

  def to_string(self):
    dev = ""
    if self.job is not None:
      dev += "/job:" + self.job
    if self.replica is not None:
      dev += "/replica:" + str(self.replica)
    if self.task is not None:
      dev += "/task:" + str(self.task)
    if self.device_type is not None:
      device_index_string = "*"
      if self.device_index is not None:
        device_index_string = str(self.device_index)
      dev += "/device:%s:%s" % (self.device_type, device_index_string)
    return dev

  @staticmethod
  def from_string(spec):
    return DeviceSpec().parse_from_string(spec)

  def __eq__(self, other):
    return self.to_string() == other.to_string()

  def __hash__(self):
    return self._hash


def check_valid(spec):
  DeviceSpec.from_string(spec)


def canonical_name(device):
  if device is None:
    return ""
  if isinstance(device, DeviceSpec):
    return device.to_string()
  else:
    device = DeviceSpec.from_string(device)
    return device.to_string()



_cached_device_functions = {}
_cached_device_specs = {}
_cache_lock = threading.Lock()


def merge_device(spec):
  with _cache_lock:
    if not isinstance(spec, DeviceSpec):
      cached_device_spec = _cached_device_specs.get(spec, None)
      if cached_device_spec is None:
        device_spec = DeviceSpec.from_string(spec or "")
        _cached_device_specs[spec] = device_spec
        spec = device_spec
      else:
        spec = cached_device_spec
    cached_function = _cached_device_functions.get(spec, None)
    if cached_function is not None:
      return cached_function

    def _device_function(node_def):
      current_device = DeviceSpec.from_string(node_def.device or "")
      copy_spec = copy.copy(spec)
      copy_spec.merge_from(current_device)  # current_device takes precedence.
      return copy_spec

    _cached_device_functions[spec] = _device_function
    return _device_function
