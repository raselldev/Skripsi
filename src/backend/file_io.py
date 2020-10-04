from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import binascii
import os
import uuid

import six


from backend import pywrap_backend
from backend.framework import errors_impl as errors
from backend.util import compat

_DEFAULT_BLOCK_SIZE = 16 * 1024 * 1024

def file_exists(filename):
  return True

def read_file_to_string(filename, binary_mode=False):
  if binary_mode:
    f = FileIO(filename, mode="rb")
  else:
    f = FileIO(filename, mode="r")
  return f.read()

class FileIO(object):
  def __init__(self, name, mode):
    self.__name = name
    self.__mode = mode
    self._read_buf = None
    self._writable_file = None
    self._binary_mode = "b" in mode
    mode = mode.replace("b", "")
    if mode not in ("r", "w", "a", "r+", "w+", "a+"):
      raise errors.InvalidArgumentError(
          None, None, "mode is not 'r' or 'w' or 'a' or 'r+' or 'w+' or 'a+'")
    self._read_check_passed = mode in ("r", "r+", "a+", "w+")
    self._write_check_passed = mode in ("a", "w", "r+", "a+", "w+")

  def read(self, n=-1):
    self._preread_check()
    with errors.raise_exception_on_not_ok_status() as status:
      if n == -1:
        length = self.size() - self.tell()
      else:
        length = n
      return self._prepare_value(
          pywrap_backend.ReadFromStream(self._read_buf, length, status))

  def _preread_check(self):
    if not self._read_buf:
      if not self._read_check_passed:
        raise errors.PermissionDeniedError(None, None,
                                           "File isn't open for reading")
      with errors.raise_exception_on_not_ok_status() as status:
        self._read_buf = pywrap_backend.CreateBufferedInputStream(
            compat.as_bytes(self.__name), 1024 * 512, status)

  def size(self):
    return stat(self.__name).length

  def tell(self):
    self._preread_check()
    return self._read_buf.Tell()

  def _prepare_value(self, val):
    if self._binary_mode:
      return compat.as_bytes(val)
    else:
      return compat.as_str_any(val)

def stat(filename):
  file_statistics = pywrap_backend.FileStatistics()
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_backend.Stat(compat.as_bytes(filename), file_statistics, status)
    return file_statistics

def get_matching_files(filename):
  with errors.raise_exception_on_not_ok_status() as status:
    if isinstance(filename, six.string_types):
      return [
          compat.as_str_any(matching_filename)
          for matching_filename in pywrap_backend.GetMatchingFiles(
              compat.as_bytes(filename), status)
      ]
    else:
      return [
          compat.as_str_any(matching_filename)
          for single_filename in filename
          for matching_filename in pywrap_backend.GetMatchingFiles(
              compat.as_bytes(single_filename), status)
      ]

