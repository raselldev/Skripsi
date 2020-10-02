import collections as _collections
import six as _six


from backend import execute as _execute
from backend import context as _context
from backend.python.framework import op_def_library as _op_def_library
#from backend.python.framework import op_def_registry as _op_def_registry
from backend.core import op_def_pb2 as _op_def_pb2

_ctc_loss_outputs = ["loss", "gradient"]
_CTCLossOutput = _collections.namedtuple(
    "CTCLoss", _ctc_loss_outputs)

def ctc_loss1(inputs, labels_indices, labels_values, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if preprocess_collapse_repeated is None:
      preprocess_collapse_repeated = False
    preprocess_collapse_repeated = _execute.make_bool(preprocess_collapse_repeated, "preprocess_collapse_repeated")
    if ctc_merge_repeated is None:
      ctc_merge_repeated = True
    ctc_merge_repeated = _execute.make_bool(ctc_merge_repeated, "ctc_merge_repeated")
    if ignore_longer_outputs_than_inputs is None:
      ignore_longer_outputs_than_inputs = False
    ignore_longer_outputs_than_inputs = _execute.make_bool(ignore_longer_outputs_than_inputs, "ignore_longer_outputs_than_inputs")
    _, _, _op = _op_def_lib._apply_op_helper(
        "CTCLoss", inputs=inputs, labels_indices=labels_indices,
        labels_values=labels_values, sequence_length=sequence_length,
        preprocess_collapse_repeated=preprocess_collapse_repeated,
        ctc_merge_repeated=ctc_merge_repeated,
        ignore_longer_outputs_than_inputs=ignore_longer_outputs_than_inputs,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("preprocess_collapse_repeated",
              _op.get_attr("preprocess_collapse_repeated"),
              "ctc_merge_repeated", _op.get_attr("ctc_merge_repeated"),
              "ignore_longer_outputs_than_inputs",
              _op.get_attr("ignore_longer_outputs_than_inputs"))
    _execute.record_gradient(
      "CTCLoss", _inputs_flat, _attrs, _result, name)
    _result = _CTCLossOutput._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "CTCLoss",
        name, _ctx._post_execution_callbacks, inputs, labels_indices,
        labels_values, sequence_length, "preprocess_collapse_repeated",
        preprocess_collapse_repeated, "ctc_merge_repeated",
        ctc_merge_repeated, "ignore_longer_outputs_than_inputs",
        ignore_longer_outputs_than_inputs)
      _result = _CTCLossOutput._make(_result)
      return _result
    except _core._FallbackException:
      return ctc_loss_eager_fallback(
          inputs, labels_indices, labels_values, sequence_length,
          preprocess_collapse_repeated=preprocess_collapse_repeated,
          ctc_merge_repeated=ctc_merge_repeated,
          ignore_longer_outputs_than_inputs=ignore_longer_outputs_than_inputs,
          name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

_ctc_greedy_decoder_outputs = ["decoded_indices", "decoded_values",
                              "decoded_shape", "log_probability"]
_CTCGreedyDecoderOutput = _collections.namedtuple(
    "CTCGreedyDecoder", _ctc_greedy_decoder_outputs)

def ctc_greedy_decoder(inputs, sequence_length, merge_repeated=False, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if merge_repeated is None:
      merge_repeated = False
    merge_repeated = _execute.make_bool(merge_repeated, "merge_repeated")
    _, _, _op = _op_def_lib._apply_op_helper(
        "CTCGreedyDecoder", inputs=inputs, sequence_length=sequence_length,
        merge_repeated=merge_repeated, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("merge_repeated", _op.get_attr("merge_repeated"))
    _execute.record_gradient(
      "CTCGreedyDecoder", _inputs_flat, _attrs, _result, name)
    _result = _CTCGreedyDecoderOutput._make(_result)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "CTCGreedyDecoder", name, _ctx._post_execution_callbacks, inputs,
        sequence_length, "merge_repeated", merge_repeated)
      _result = _CTCGreedyDecoderOutput._make(_result)
      return _result
    except _core._FallbackException:
      return ctc_greedy_decoder_eager_fallback(
          inputs, sequence_length, merge_repeated=merge_repeated, name=name,
          ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
#  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib

_op_def_lib = _InitOpDefLibrary(b"\n\362\001\n\024CTCBeamSearchDecoder\022\n\n\006inputs\030\001\022\023\n\017sequence_length\030\003\032\036\n\017decoded_indices\030\t*\ttop_paths\032\035\n\016decoded_values\030\t*\ttop_paths\032\034\n\rdecoded_shape\030\t*\ttop_paths\032\023\n\017log_probability\030\001\"\025\n\nbeam_width\022\003int(\0010\001\"\024\n\ttop_paths\022\003int(\0010\001\"\032\n\016merge_repeated\022\004bool\032\002(\001\n\240\001\n\020CTCGreedyDecoder\022\n\n\006inputs\030\001\022\023\n\017sequence_length\030\003\032\023\n\017decoded_indices\030\t\032\022\n\016decoded_values\030\t\032\021\n\rdecoded_shape\030\t\032\023\n\017log_probability\030\001\"\032\n\016merge_repeated\022\004bool\032\002(\000\n\342\001\n\007CTCLoss\022\n\n\006inputs\030\001\022\022\n\016labels_indices\030\t\022\021\n\rlabels_values\030\003\022\023\n\017sequence_length\030\003\032\010\n\004loss\030\001\032\014\n\010gradient\030\001\"(\n\034preprocess_collapse_repeated\022\004bool\032\002(\000\"\036\n\022ctc_merge_repeated\022\004bool\032\002(\001\"-\n!ignore_longer_outputs_than_inputs\022\004bool\032\002(\000")
