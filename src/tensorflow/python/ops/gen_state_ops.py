import collections as _collections
import six as _six

from tensorflow.python import context as _context
from tensorflow.python import execute as _execute
from tensorflow.core import op_def_pb2 as _op_def_pb2
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.framework import op_def_registry as _op_def_registry

def variable_v2(shape, dtype, container="", shared_name="", name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    shape = _execute.make_shape(shape, "shape")
    dtype = _execute.make_type(dtype, "dtype")
    if container is None:
      container = ""
    container = _execute.make_str(container, "container")
    if shared_name is None:
      shared_name = ""
    shared_name = _execute.make_str(shared_name, "shared_name")
    _, _, _op = _op_def_lib._apply_op_helper(
        "VariableV2", shape=shape, dtype=dtype, container=container,
        shared_name=shared_name, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("shape", _op.get_attr("shape"), "dtype", _op.get_attr("dtype"),
              "container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _execute.record_gradient(
      "VariableV2", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    raise RuntimeError("variable_v2 op does not support eager execution. Arg 'ref' is a ref.")


  raise RuntimeError("variable_v2 op does not support eager execution. Arg 'ref' is a ref.")

def assign(ref, value, validate_shape=True, use_locking=True, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if validate_shape is None:
      validate_shape = True
    validate_shape = _execute.make_bool(validate_shape, "validate_shape")
    if use_locking is None:
      use_locking = True
    use_locking = _execute.make_bool(use_locking, "use_locking")
    _, _, _op = _op_def_lib._apply_op_helper(
        "Assign", ref=ref, value=value, validate_shape=validate_shape,
        use_locking=use_locking, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "validate_shape",
              _op.get_attr("validate_shape"), "use_locking",
              _op.get_attr("use_locking"))
    _execute.record_gradient(
      "Assign", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    raise RuntimeError("assign op does not support eager execution. Arg 'output_ref' is a ref.")


  raise RuntimeError("assign op does not support eager execution. Arg 'output_ref' is a ref.")

def assign_sub(ref, value, use_locking=False, name=None):
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if use_locking is None:
      use_locking = False
    use_locking = _execute.make_bool(use_locking, "use_locking")
    _, _, _op = _op_def_lib._apply_op_helper(
        "AssignSub", ref=ref, value=value, use_locking=use_locking, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "use_locking",
              _op.get_attr("use_locking"))
    _execute.record_gradient(
      "AssignSub", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    raise RuntimeError("assign_sub op does not support eager execution. Arg 'output_ref' is a ref.")


  raise RuntimeError("assign_sub op does not support eager execution. Arg 'output_ref' is a ref.")

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib

_op_def_lib = _InitOpDefLibrary(b"\nx\n\006Assign\022\013\n\003ref\"\001T\200\001\001\022\n\n\005value\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\"\t\n\001T\022\004type\"\032\n\016validate_shape\022\004bool\032\002(\001\"\027\n\013use_locking\022\004bool\032\002(\001\230\001\001\ns\n\tAssignAdd\022\013\n\003ref\"\001T\200\001\001\022\n\n\005value\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\027\n\013use_locking\022\004bool\032\002(\000\ns\n\tAssignSub\022\013\n\003ref\"\001T\200\001\001\022\n\n\005value\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\027\n\013use_locking\022\004bool\032\002(\000\nF\n\tCountUpTo\022\013\n\003ref\"\001T\200\001\001\032\013\n\006output\"\001T\"\014\n\005limit\022\003int\"\021\n\001T\022\004type:\006\n\0042\002\003\t\nR\n\030DestroyTemporaryVariable\022\013\n\003ref\"\001T\200\001\001\032\n\n\005value\"\001T\"\t\n\001T\022\004type\"\022\n\010var_name\022\006string\nN\n\025IsVariableInitialized\022\017\n\003ref\"\005dtype\200\001\001\032\022\n\016is_initialized\030\n\"\r\n\005dtype\022\004type\230\001\001\nR\n\021ResourceCountUpTo\022\014\n\010resource\030\024\032\013\n\006output\"\001T\"\014\n\005limit\022\003int\"\021\n\001T\022\004type:\006\n\0042\002\003\t\210\001\001\n\203\001\n\024ResourceScatterNdAdd\022\007\n\003ref\030\024\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\"\t\n\001T\022\004type\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\001\210\001\001\n\206\001\n\027ResourceScatterNdUpdate\022\007\n\003ref\030\024\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\"\t\n\001T\022\004type\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\001\210\001\001\n\245\001\n\nScatterAdd\022\013\n\003ref\"\001T\200\001\001\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\000\n\245\001\n\nScatterDiv\022\013\n\003ref\"\001T\200\001\001\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\000\n\232\001\n\nScatterMax\022\013\n\003ref\"\001T\200\001\001\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\"\025\n\001T\022\004type:\n\n\0102\006\023\016\001\002\003\t\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\000\n\232\001\n\nScatterMin\022\013\n\003ref\"\001T\200\001\001\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\"\025\n\001T\022\004type:\n\n\0102\006\023\016\001\002\003\t\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\000\n\245\001\n\nScatterMul\022\013\n\003ref\"\001T\200\001\001\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\000\n\247\001\n\014ScatterNdAdd\022\013\n\003ref\"\001T\200\001\001\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\000\n\247\001\n\014ScatterNdSub\022\013\n\003ref\"\001T\200\001\001\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\000\n\223\001\n\017ScatterNdUpdate\022\013\n\003ref\"\001T\200\001\001\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\"\t\n\001T\022\004type\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\001\n\245\001\n\nScatterSub\022\013\n\003ref\"\001T\200\001\001\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\" \n\001T\022\004type:\025\n\0232\021\001\002\003\004\005\006\010\t\013\014\r\016\021\022\023\026\027\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\000\n\221\001\n\rScatterUpdate\022\013\n\003ref\"\001T\200\001\001\022\023\n\007indices\"\010Tindices\022\014\n\007updates\"\001T\032\022\n\noutput_ref\"\001T\200\001\001\"\t\n\001T\022\004type\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\"\027\n\013use_locking\022\004bool\032\002(\001\n^\n\021TemporaryVariable\032\017\n\003ref\"\005dtype\200\001\001\"\016\n\005shape\022\005shape\"\r\n\005dtype\022\004type\"\026\n\010var_name\022\006string\032\002\022\000\210\001\001\nq\n\010Variable\032\017\n\003ref\"\005dtype\200\001\001\"\016\n\005shape\022\005shape\"\r\n\005dtype\022\004type\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\ns\n\nVariableV2\032\017\n\003ref\"\005dtype\200\001\001\"\016\n\005shape\022\005shape\"\r\n\005dtype\022\004type\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001")
