from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path
import re
import time

from google.protobuf import text_format


from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python import file_io
from tensorflow.python.framework import errors_impl as errors
from tensorflow.core import saver_pb2


def get_checkpoint_state(checkpoint_dir, latest_filename=None):
  ckpt = None
  coord_checkpoint_filename = _GetCheckpointFilename(checkpoint_dir,
                                                     latest_filename)
  f = None
  try:
    if file_io.file_exists(coord_checkpoint_filename):
      file_content = file_io.read_file_to_string(
          coord_checkpoint_filename)
      ckpt = CheckpointState()
      text_format.Merge(file_content, ckpt)
      if not ckpt.model_checkpoint_path:
        raise ValueError("Invalid checkpoint state loaded from "
                         + checkpoint_dir)
      if not os.path.isabs(ckpt.model_checkpoint_path):
        ckpt.model_checkpoint_path = os.path.join(checkpoint_dir,
                                                  ckpt.model_checkpoint_path)
      for i in range(len(ckpt.all_model_checkpoint_paths)):
        p = ckpt.all_model_checkpoint_paths[i]
        if not os.path.isabs(p):
          ckpt.all_model_checkpoint_paths[i] = os.path.join(checkpoint_dir, p)
  except errors.OpError as e:
    logging.warning("%s: %s", type(e).__name__, e)
    logging.warning("%s: Checkpoint ignored", coord_checkpoint_filename)
    return None
  except text_format.ParseError as e:
    logging.warning("%s: %s", type(e).__name__, e)
    logging.warning("%s: Checkpoint ignored", coord_checkpoint_filename)
    return None
  finally:
    if f:
      f.close()
  return ckpt

def update_checkpoint_state(save_dir,
                            model_checkpoint_path,
                            all_model_checkpoint_paths=None,
                            latest_filename=None,
                            all_model_checkpoint_timestamps=None,
                            last_preserved_timestamp=None):
  update_checkpoint_state_internal(
      save_dir=save_dir,
      model_checkpoint_path=model_checkpoint_path,
      all_model_checkpoint_paths=all_model_checkpoint_paths,
      latest_filename=latest_filename,
      save_relative_paths=False,
      all_model_checkpoint_timestamps=all_model_checkpoint_timestamps,
      last_preserved_timestamp=last_preserved_timestamp)

def generate_checkpoint_state_proto(save_dir,
                                    model_checkpoint_path,
                                    all_model_checkpoint_paths=None,
                                    all_model_checkpoint_timestamps=None,
                                    last_preserved_timestamp=None):
  if all_model_checkpoint_paths is None:
    all_model_checkpoint_paths = []

  if (not all_model_checkpoint_paths or
      all_model_checkpoint_paths[-1] != model_checkpoint_path):
    logging.info("%s is not in all_model_checkpoint_paths. Manually adding it.",
                 model_checkpoint_path)
    all_model_checkpoint_paths.append(model_checkpoint_path)

  if (all_model_checkpoint_timestamps
      and (len(all_model_checkpoint_timestamps)
           != len(all_model_checkpoint_paths))):
    raise ValueError(
        ("Checkpoint timestamps, if provided, must match checkpoint paths (got "
         "paths %s and timestamps %s)")
        % (all_model_checkpoint_paths, all_model_checkpoint_timestamps))

  if not os.path.isabs(save_dir):
    if not os.path.isabs(model_checkpoint_path):
      model_checkpoint_path = os.path.relpath(model_checkpoint_path, save_dir)
    for i in range(len(all_model_checkpoint_paths)):
      p = all_model_checkpoint_paths[i]
      if not os.path.isabs(p):
        all_model_checkpoint_paths[i] = os.path.relpath(p, save_dir)

  coord_checkpoint_proto = CheckpointState(
      model_checkpoint_path=model_checkpoint_path,
      all_model_checkpoint_paths=all_model_checkpoint_paths,
      all_model_checkpoint_timestamps=all_model_checkpoint_timestamps,
      last_preserved_timestamp=last_preserved_timestamp)

  return coord_checkpoint_proto

def latest_checkpoint(checkpoint_dir, latest_filename=None):
  ckpt = get_checkpoint_state(checkpoint_dir, latest_filename)
  if ckpt and ckpt.model_checkpoint_path:
    v2_path = _prefix_to_checkpoint_path(ckpt.model_checkpoint_path,
                                         saver_pb2.SaverDef.V2)
    v1_path = _prefix_to_checkpoint_path(ckpt.model_checkpoint_path,
                                         saver_pb2.SaverDef.V1)
    if file_io.get_matching_files(v2_path) or file_io.get_matching_files(
        v1_path):
      return ckpt.model_checkpoint_path
    else:
      logging.error("Couldn't match files for checkpoint %s",
                    ckpt.model_checkpoint_path)
  return None

def checkpoint_exists(checkpoint_prefix):
  pathname = _prefix_to_checkpoint_path(checkpoint_prefix,
                                        saver_pb2.SaverDef.V2)
  if file_io.get_matching_files(pathname):
    return True
  elif file_io.get_matching_files(checkpoint_prefix):
    return True
  else:
    return False

def get_checkpoint_mtimes(checkpoint_prefixes):
  mtimes = []

  def match_maybe_append(pathname):
    fnames = file_io.get_matching_files(pathname)
    if fnames:
      mtimes.append(file_io.stat(fnames[0]).mtime_nsec / 1e9)
      return True
    return False

  for checkpoint_prefix in checkpoint_prefixes:
    pathname = _prefix_to_checkpoint_path(checkpoint_prefix,
                                          saver_pb2.SaverDef.V2)
    if match_maybe_append(pathname):
      continue
    match_maybe_append(checkpoint_prefix)

  return mtimes

def remove_checkpoint(checkpoint_prefix,
                      checkpoint_format_version=saver_pb2.SaverDef.V2,
                      meta_graph_suffix="meta"):
  _delete_file_if_exists(
      meta_graph_filename(checkpoint_prefix, meta_graph_suffix))
  if checkpoint_format_version == saver_pb2.SaverDef.V2:
    _delete_file_if_exists(checkpoint_prefix + ".index")
    _delete_file_if_exists(checkpoint_prefix + ".data-?????-of-?????")
  else:
    _delete_file_if_exists(checkpoint_prefix)

def _GetCheckpointFilename(save_dir, latest_filename):
  if latest_filename is None:
    latest_filename = "checkpoint"
  return os.path.join(save_dir, latest_filename)

def _prefix_to_checkpoint_path(prefix, format_version):
  if format_version == saver_pb2.SaverDef.V2:
    return prefix + ".index"

