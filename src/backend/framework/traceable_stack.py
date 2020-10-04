from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

class TraceableStack(object):
  def __init__(self, existing_stack=None):
    self._stack = existing_stack[:] if existing_stack else []

  def peek_objs(self):
    return [t_obj.obj for t_obj in reversed(self._stack)]

  def peek_traceable_objs(self):
    return list(reversed(self._stack))

  def push_obj(self, obj, offset=0):
    traceable_obj = TraceableObject(obj)
    self._stack.append(traceable_obj)
    return traceable_obj.set_filename_and_line_from_caller(offset + 1)

  def pop_obj(self):
    return self._stack.pop().obj

  def __len__(self):
    return len(self._stack)


class TraceableObject(object):
  SUCCESS, HEURISTIC_USED, FAILURE = (0, 1, 2)

  def __init__(self, obj, filename=None, lineno=None):
    self.obj = obj
    self.filename = filename
    self.lineno = lineno

  def set_filename_and_line_from_caller(self, offset=0):
    local_offset = offset + 1

    frame_records = extract_stack()
    if not frame_records:
      return self.FAILURE
    if len(frame_records) >= local_offset:
      negative_offset = -(local_offset + 1)
      self.filename, self.lineno = frame_records[negative_offset][:2]
      return self.SUCCESS
    else:
      self.filename, self.lineno = frame_records[0][:2]
      return self.HEURISTIC_USED

  def copy_metadata(self):
    return self.__class__(None, filename=self.filename, lineno=self.lineno)

def extract_stack(extract_frame_info_fn=None):
  default_fn = lambda f: None
  extract_frame_info_fn = extract_frame_info_fn or default_fn
  try:
    raise ZeroDivisionError
  except ZeroDivisionError:
    f = sys.exc_info()[2].tb_frame.f_back
  ret = []
  while f is not None:
    lineno = f.f_lineno
    co = f.f_code
    filename = co.co_filename
    name = co.co_name
    frame_globals = f.f_globals
    func_start_lineno = co.co_firstlineno
    frame_info = extract_frame_info_fn(f)
    ret.append((filename, lineno, name, frame_globals, func_start_lineno,
                frame_info))
    f = f.f_back
  ret.reverse()
  return ret
