import collections as _collections
import six as _six
from backend import context
from backend import execute as execute
from backend.python.framework import op_def_library
#from backend.python.framework import op_def_registry
from backend.core import op_def_pb2

def save_v2(prefix, tensor_names, shape_and_slices, tensors, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "SaveV2", prefix=prefix, tensor_names=tensor_names,
        shape_and_slices=shape_and_slices, tensors=tensors, name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "SaveV2", name,
        _ctx._post_execution_callbacks, prefix, tensor_names,
        shape_and_slices, tensors)
      return _result
    except _core._FallbackException:
      return save_v2_eager_fallback(
          prefix, tensor_names, shape_and_slices, tensors, name=name,
          ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def restore_v2(prefix, tensor_names, shape_and_slices, dtypes, name=None):
  _ctx = context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(dtypes, (list, tuple)):
      raise TypeError(
          "Expected list for 'dtypes' argument to "
          "'restore_v2' Op, not %r." % dtypes)
    dtypes = [execute.make_type(_t, "dtypes") for _t in dtypes]
    _, _, _op = _op_def_lib._apply_op_helper(
        "RestoreV2", prefix=prefix, tensor_names=tensor_names,
        shape_and_slices=shape_and_slices, dtypes=dtypes, name=name)
    _result = _op.outputs[:]
    if not _result:
      return _op
    _inputs_flat = _op.inputs
    _attrs = ("dtypes", _op.get_attr("dtypes"))
    execute.record_gradient(
      "RestoreV2", _inputs_flat, _attrs, _result, name)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "RestoreV2",
        name, _ctx._post_execution_callbacks, prefix, tensor_names,
        shape_and_slices, "dtypes", dtypes)
      return _result
    except _core._FallbackException:
      return restore_v2_eager_fallback(
          prefix, tensor_names, shape_and_slices, dtypes=dtypes, name=name,
          ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
#  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib

_op_def_lib = _InitOpDefLibrary(b"\n\346\001\n\027FixedLengthRecordReader\032\024\n\rreader_handle\030\007\200\001\001\"\027\n\014header_bytes\022\003int\032\002\030\000\"\023\n\014record_bytes\022\003int\"\027\n\014footer_bytes\022\003int\032\002\030\000\"\024\n\thop_bytes\022\003int\032\002\030\000\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000B!\010\032\022\035Use FixedLengthRecordReaderV2\210\001\001\n\332\001\n\031FixedLengthRecordReaderV2\032\021\n\rreader_handle\030\024\"\027\n\014header_bytes\022\003int\032\002\030\000\"\023\n\014record_bytes\022\003int\"\027\n\014footer_bytes\022\003int\032\002\030\000\"\024\n\thop_bytes\022\003int\032\002\030\000\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\"\026\n\010encoding\022\006string\032\002\022\000\210\001\001\nw\n\016IdentityReader\032\024\n\rreader_handle\030\007\200\001\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000B\030\010\032\022\024Use IdentityReaderV2\210\001\001\n\\\n\020IdentityReaderV2\032\021\n\rreader_handle\030\024\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\nY\n\nLMDBReader\032\024\n\rreader_handle\030\007\200\001\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n+\n\rMatchingFiles\022\013\n\007pattern\030\007\032\r\n\tfilenames\030\007\ne\n\022MergeV2Checkpoints\022\027\n\023checkpoint_prefixes\030\007\022\026\n\022destination_prefix\030\007\"\033\n\017delete_old_dirs\022\004bool\032\002(\001\210\001\001\n&\n\010ReadFile\022\014\n\010filename\030\007\032\014\n\010contents\030\007\nF\n\030ReaderNumRecordsProduced\022\024\n\rreader_handle\030\007\200\001\001\032\024\n\020records_produced\030\t\nH\n\032ReaderNumRecordsProducedV2\022\021\n\rreader_handle\030\024\032\024\n\020records_produced\030\t\210\001\001\nH\n\033ReaderNumWorkUnitsCompleted\022\024\n\rreader_handle\030\007\200\001\001\032\023\n\017units_completed\030\t\nJ\n\035ReaderNumWorkUnitsCompletedV2\022\021\n\rreader_handle\030\024\032\023\n\017units_completed\030\t\210\001\001\nK\n\nReaderRead\022\024\n\rreader_handle\030\007\200\001\001\022\023\n\014queue_handle\030\007\200\001\001\032\007\n\003key\030\007\032\t\n\005value\030\007\nb\n\016ReaderReadUpTo\022\024\n\rreader_handle\030\007\200\001\001\022\023\n\014queue_handle\030\007\200\001\001\022\017\n\013num_records\030\t\032\010\n\004keys\030\007\032\n\n\006values\030\007\na\n\020ReaderReadUpToV2\022\021\n\rreader_handle\030\024\022\020\n\014queue_handle\030\024\022\017\n\013num_records\030\t\032\010\n\004keys\030\007\032\n\n\006values\030\007\210\001\001\nJ\n\014ReaderReadV2\022\021\n\rreader_handle\030\024\022\020\n\014queue_handle\030\024\032\007\n\003key\030\007\032\t\n\005value\030\007\210\001\001\n#\n\013ReaderReset\022\024\n\rreader_handle\030\007\200\001\001\n%\n\rReaderResetV2\022\021\n\rreader_handle\030\024\210\001\001\n5\n\022ReaderRestoreState\022\024\n\rreader_handle\030\007\200\001\001\022\t\n\005state\030\007\n7\n\024ReaderRestoreStateV2\022\021\n\rreader_handle\030\024\022\t\n\005state\030\007\210\001\001\n7\n\024ReaderSerializeState\022\024\n\rreader_handle\030\007\200\001\001\032\t\n\005state\030\007\n9\n\026ReaderSerializeStateV2\022\021\n\rreader_handle\030\024\032\t\n\005state\030\007\210\001\001\nn\n\007Restore\022\020\n\014file_pattern\030\007\022\017\n\013tensor_name\030\007\032\014\n\006tensor\"\002dt\"\n\n\002dt\022\004type\"#\n\017preferred_shard\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\n\210\001\n\014RestoreSlice\022\020\n\014file_pattern\030\007\022\017\n\013tensor_name\030\007\022\023\n\017shape_and_slice\030\007\032\014\n\006tensor\"\002dt\"\n\n\002dt\022\004type\"#\n\017preferred_shard\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\no\n\tRestoreV2\022\n\n\006prefix\030\007\022\020\n\014tensor_names\030\007\022\024\n\020shape_and_slices\030\007\032\021\n\007tensors2\006dtypes\"\030\n\006dtypes\022\nlist(type)(\0010\001\210\001\001\nI\n\004Save\022\014\n\010filename\030\007\022\020\n\014tensor_names\030\007\022\t\n\004data2\001T\"\023\n\001T\022\nlist(type)(\0010\001\210\001\001\nf\n\nSaveSlices\022\014\n\010filename\030\007\022\020\n\014tensor_names\030\007\022\025\n\021shapes_and_slices\030\007\022\t\n\004data2\001T\"\023\n\001T\022\nlist(type)(\0010\001\210\001\001\nl\n\006SaveV2\022\n\n\006prefix\030\007\022\020\n\014tensor_names\030\007\022\024\n\020shape_and_slices\030\007\022\021\n\007tensors2\006dtypes\"\030\n\006dtypes\022\nlist(type)(\0010\001\210\001\001\nH\n\017ShardedFilename\022\014\n\010basename\030\007\022\t\n\005shard\030\003\022\016\n\nnum_shards\030\003\032\014\n\010filename\030\007\n=\n\017ShardedFilespec\022\014\n\010basename\030\007\022\016\n\nnum_shards\030\003\032\014\n\010filename\030\007\n\227\001\n\016TFRecordReader\032\024\n\rreader_handle\030\007\200\001\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\"\036\n\020compression_type\022\006string\032\002\022\000B\030\010\032\022\024Use TFRecordReaderV2\210\001\001\n|\n\020TFRecordReaderV2\032\021\n\rreader_handle\030\024\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\"\036\n\020compression_type\022\006string\032\002\022\000\210\001\001\n\225\001\n\016TextLineReader\032\024\n\rreader_handle\030\007\200\001\001\"\034\n\021skip_header_lines\022\003int\032\002\030\000\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000B\030\010\032\022\024Use TextLineReaderV2\210\001\001\nz\n\020TextLineReaderV2\032\021\n\rreader_handle\030\024\"\034\n\021skip_header_lines\022\003int\032\002\030\000\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n^\n\017WholeFileReader\032\024\n\rreader_handle\030\007\200\001\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n]\n\021WholeFileReaderV2\032\021\n\rreader_handle\030\024\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\'\n\tWriteFile\022\014\n\010filename\030\007\022\014\n\010contents\030\007")
