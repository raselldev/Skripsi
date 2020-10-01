from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_pywrap_backend_internal', [dirname(__file__)])
        except:
            pass
        if fp is not None:
            try:
                _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _pywrap_backend_internal = swig_import_helper()

try:
    _object = object
    _newclass = 1
except AttributeError:
    _newclass = 0

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

def TF_bfloat16_type():
    return _pywrap_backend_internal.TF_bfloat16_type()
TF_bfloat16_type = _pywrap_backend_internal.TF_bfloat16_type

def PyExceptionRegistry_Init(code_to_exc_type_map):
    return _pywrap_backend_internal.PyExceptionRegistry_Init(code_to_exc_type_map)
PyExceptionRegistry_Init = _pywrap_backend_internal.PyExceptionRegistry_Init

def TFE_Py_InitEagerTensor(base_class):
    return _pywrap_backend_internal.TFE_Py_InitEagerTensor(base_class)
TFE_Py_InitEagerTensor = _pywrap_backend_internal.TFE_Py_InitEagerTensor

def RegisterType(type_name, type):
    return _pywrap_backend_internal.RegisterType(type_name, type)
RegisterType = _pywrap_backend_internal.RegisterType

def IsMapping(o):
    return _pywrap_backend_internal.IsMapping(o)

def IsAttrs(o):
    return _pywrap_backend_internal.IsAttrs(o)

def IsSequence(o):
    return _pywrap_backend_internal.IsSequence(o)

def Flatten(nested):
    return _pywrap_backend_internal.Flatten(nested)

def SameNamedtuples(o1, o2):
    return _pywrap_backend_internal.SameNamedtuples(o1, o2)

def TFE_Py_UID():
    return _pywrap_backend_internal.TFE_Py_UID()
TFE_Py_UID = _pywrap_backend_internal.TFE_Py_UID

def TF_NewGraph():
    return _pywrap_backend_internal.TF_NewGraph()
TF_NewGraph = _pywrap_backend_internal.TF_NewGraph

def SetRequireShapeInferenceFns(graph, require):
    return _pywrap_backend_internal.SetRequireShapeInferenceFns(graph, require)
SetRequireShapeInferenceFns = _pywrap_backend_internal.SetRequireShapeInferenceFns

def TF_NewOperation(graph, op_type, oper_name):
    return _pywrap_backend_internal.TF_NewOperation(graph, op_type, oper_name)
TF_NewOperation = _pywrap_backend_internal.TF_NewOperation

def TF_SetAttrValueProto(desc, attr_name, proto):
    return _pywrap_backend_internal.TF_SetAttrValueProto(desc, attr_name, proto)
TF_SetAttrValueProto = _pywrap_backend_internal.TF_SetAttrValueProto

def TF_FinishOperation(desc):
    return _pywrap_backend_internal.TF_FinishOperation(desc)
TF_FinishOperation = _pywrap_backend_internal.TF_FinishOperation

def TF_OperationNumOutputs(oper):
    return _pywrap_backend_internal.TF_OperationNumOutputs(oper)
TF_OperationNumOutputs = _pywrap_backend_internal.TF_OperationNumOutputs

def TF_OperationOutputType(oper_out):
    return _pywrap_backend_internal.TF_OperationOutputType(oper_out)
TF_OperationOutputType = _pywrap_backend_internal.TF_OperationOutputType

class TF_Output(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TF_Output, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TF_Output, name)
    __repr__ = _swig_repr
    __swig_setmethods__["oper"] = _pywrap_backend_internal.TF_Output_oper_set
    __swig_getmethods__["oper"] = _pywrap_backend_internal.TF_Output_oper_get
    if _newclass:
        oper = property(_pywrap_backend_internal.TF_Output_oper_get, _pywrap_backend_internal.TF_Output_oper_set)
    __swig_setmethods__["index"] = _pywrap_backend_internal.TF_Output_index_set
    __swig_getmethods__["index"] = _pywrap_backend_internal.TF_Output_index_get
    if _newclass:
        index = property(_pywrap_backend_internal.TF_Output_index_get, _pywrap_backend_internal.TF_Output_index_set)

    def __init__(self):
        this = _pywrap_backend_internal.new_TF_Output()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _pywrap_backend_internal.delete_TF_Output
    __del__ = lambda self: None
TF_Output_swigregister = _pywrap_backend_internal.TF_Output_swigregister
TF_Output_swigregister(TF_Output)

def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def TF_OperationName(oper):
    return _pywrap_backend_internal.TF_OperationName(oper)
TF_OperationName = _pywrap_backend_internal.TF_OperationName

def GetOperationInputs(oper):
    return _pywrap_backend_internal.GetOperationInputs(oper)
GetOperationInputs = _pywrap_backend_internal.GetOperationInputs

def TF_OperationOpType(oper):
    return _pywrap_backend_internal.TF_OperationOpType(oper)
TF_OperationOpType = _pywrap_backend_internal.TF_OperationOpType

def TF_NewBuffer():
    return _pywrap_backend_internal.TF_NewBuffer()
TF_NewBuffer = _pywrap_backend_internal.TF_NewBuffer

def TF_DeleteBuffer(arg1):
    return _pywrap_backend_internal.TF_DeleteBuffer(arg1)
TF_DeleteBuffer = _pywrap_backend_internal.TF_DeleteBuffer

def TF_OperationGetAttrValueProto(oper, attr_name, output_attr_value):
    return _pywrap_backend_internal.TF_OperationGetAttrValueProto(oper, attr_name, output_attr_value)
TF_OperationGetAttrValueProto = _pywrap_backend_internal.TF_OperationGetAttrValueProto

def TF_GetBuffer(buffer):
    return _pywrap_backend_internal.TF_GetBuffer(buffer)
TF_GetBuffer = _pywrap_backend_internal.TF_GetBuffer

def TF_GraphGetOpDef(graph, op_name, output_op_def):
    return _pywrap_backend_internal.TF_GraphGetOpDef(graph, op_name, output_op_def)
TF_GraphGetOpDef = _pywrap_backend_internal.TF_GraphGetOpDef

def TF_AddInput(desc, input):
    return _pywrap_backend_internal.TF_AddInput(desc, input)
TF_AddInput = _pywrap_backend_internal.TF_AddInput

def TF_GraphGetTensorShapeHelper(graph, output):
    return _pywrap_backend_internal.TF_GraphGetTensorShapeHelper(graph, output)
TF_GraphGetTensorShapeHelper = _pywrap_backend_internal.TF_GraphGetTensorShapeHelper

def TF_OperationGetControlInputs_wrapper(oper):
    return _pywrap_backend_internal.TF_OperationGetControlInputs_wrapper(oper)
TF_OperationGetControlInputs_wrapper = _pywrap_backend_internal.TF_OperationGetControlInputs_wrapper

def TF_OperationToNodeDef(oper, output_node_def):
    return _pywrap_backend_internal.TF_OperationToNodeDef(oper, output_node_def)
TF_OperationToNodeDef = _pywrap_backend_internal.TF_OperationToNodeDef

def TF_OperationDevice(oper):
    return _pywrap_backend_internal.TF_OperationDevice(oper)
TF_OperationDevice = _pywrap_backend_internal.TF_OperationDevice

def TF_NewBufferFromString(proto):
    return _pywrap_backend_internal.TF_NewBufferFromString(proto)
TF_NewBufferFromString = _pywrap_backend_internal.TF_NewBufferFromString

def SetAttr(graph, op, attr_name, attr_value_proto):
    return _pywrap_backend_internal.SetAttr(graph, op, attr_name, attr_value_proto)
SetAttr = _pywrap_backend_internal.SetAttr

def TF_TryEvaluateConstant_wrapper(graph, output):
    return _pywrap_backend_internal.TF_TryEvaluateConstant_wrapper(graph, output)
TF_TryEvaluateConstant_wrapper = _pywrap_backend_internal.TF_TryEvaluateConstant_wrapper

def AddControlInput(graph, op, input):
    return _pywrap_backend_internal.AddControlInput(graph, op, input)
AddControlInput = _pywrap_backend_internal.AddControlInput

def UpdateEdge(graph, new_src, dst):
    return _pywrap_backend_internal.UpdateEdge(graph, new_src, dst)
UpdateEdge = _pywrap_backend_internal.UpdateEdge

class TF_Input(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, TF_Input, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, TF_Input, name)
    __repr__ = _swig_repr
    __swig_setmethods__["oper"] = _pywrap_backend_internal.TF_Input_oper_set
    __swig_getmethods__["oper"] = _pywrap_backend_internal.TF_Input_oper_get
    if _newclass:
        oper = property(_pywrap_backend_internal.TF_Input_oper_get, _pywrap_backend_internal.TF_Input_oper_set)
    __swig_setmethods__["index"] = _pywrap_backend_internal.TF_Input_index_set
    __swig_getmethods__["index"] = _pywrap_backend_internal.TF_Input_index_get
    if _newclass:
        index = property(_pywrap_backend_internal.TF_Input_index_get, _pywrap_backend_internal.TF_Input_index_set)

    def __init__(self):
        this = _pywrap_backend_internal.new_TF_Input()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _pywrap_backend_internal.delete_TF_Input
    __del__ = lambda self: None
TF_Input_swigregister = _pywrap_backend_internal.TF_Input_swigregister
TF_Input_swigregister(TF_Input)

def IsNamedtuple(o, strict):
    return _pywrap_backend_internal.IsNamedtuple(o, strict)
IsNamedtuple = _pywrap_backend_internal.IsNamedtuple

def TF_AddInputList(desc, inputs):
    return _pywrap_backend_internal.TF_AddInputList(desc, inputs)
TF_AddInputList = _pywrap_backend_internal.TF_AddInputList

def TF_GraphSetTensorShape_wrapper(graph, output, dims, unknown_shape):
    return _pywrap_backend_internal.TF_GraphSetTensorShape_wrapper(graph, output, dims, unknown_shape)
TF_GraphSetTensorShape_wrapper = _pywrap_backend_internal.TF_GraphSetTensorShape_wrapper

def SetRequestedDevice(graph, op, device):
    return _pywrap_backend_internal.SetRequestedDevice(graph, op, device)
SetRequestedDevice = _pywrap_backend_internal.SetRequestedDevice

def AssertSameStructure(o1, o2, check_types):
    return _pywrap_backend_internal.AssertSameStructure(o1, o2, check_types)
AssertSameStructure = _pywrap_backend_internal.AssertSameStructure

def TF_OperationOutputConsumers_wrapper(oper_out):
    return _pywrap_backend_internal.TF_OperationOutputConsumers_wrapper(oper_out)
TF_OperationOutputConsumers_wrapper = _pywrap_backend_internal.TF_OperationOutputConsumers_wrapper

def TF_NewSessionOptions(target=None, config=None):
  opts = _TF_NewSessionOptions()
  if target is not None:
    _TF_SetTarget(opts, target)
  if config is not None:
    from backend.python.framework import errors
    config_str = config.SerializeToString()
    _TF_SetConfig(opts, config_str)
  return opts

def _TF_NewSessionOptions():
    return _pywrap_backend_internal._TF_NewSessionOptions()
_TF_NewSessionOptions = _pywrap_backend_internal._TF_NewSessionOptions

def _TF_SetTarget(options, target):
    return _pywrap_backend_internal._TF_SetTarget(options, target)
_TF_SetTarget = _pywrap_backend_internal._TF_SetTarget

def TF_DeleteSessionOptions(arg1):
    return _pywrap_backend_internal.TF_DeleteSessionOptions(arg1)
TF_DeleteSessionOptions = _pywrap_backend_internal.TF_DeleteSessionOptions

def TF_NewSessionRef(graph, opts):
    return _pywrap_backend_internal.TF_NewSessionRef(graph, opts)
TF_NewSessionRef = _pywrap_backend_internal.TF_NewSessionRef

def TF_AddControlInput(desc, input):
    return _pywrap_backend_internal.TF_AddControlInput(desc, input)
TF_AddControlInput = _pywrap_backend_internal.TF_AddControlInput

def TF_DeleteStatus(arg1):
    return _pywrap_backend_internal.TF_DeleteStatus(arg1)
TF_DeleteStatus = _pywrap_backend_internal.TF_DeleteStatus

def TF_NewStatus():
    return _pywrap_backend_internal.TF_NewStatus()
TF_NewStatus = _pywrap_backend_internal.TF_NewStatus

def TF_GetCode(s):
    return _pywrap_backend_internal.TF_GetCode(s)
TF_GetCode = _pywrap_backend_internal.TF_GetCode

def FileExists(filename, out_status):
    return _pywrap_backend_internal.FileExists(filename, out_status)
FileExists = _pywrap_backend_internal.FileExists

def CreateBufferedInputStream(filename, buffer_size, out_status):
    return _pywrap_backend_internal.CreateBufferedInputStream(filename, buffer_size, out_status)
CreateBufferedInputStream = _pywrap_backend_internal.CreateBufferedInputStream

class FileStatistics(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FileStatistics, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FileStatistics, name)
    __repr__ = _swig_repr
    __swig_setmethods__["length"] = _pywrap_backend_internal.FileStatistics_length_set
    __swig_getmethods__["length"] = _pywrap_backend_internal.FileStatistics_length_get
    if _newclass:
        length = property(_pywrap_backend_internal.FileStatistics_length_get, _pywrap_backend_internal.FileStatistics_length_set)
    __swig_setmethods__["mtime_nsec"] = _pywrap_backend_internal.FileStatistics_mtime_nsec_set
    __swig_getmethods__["mtime_nsec"] = _pywrap_backend_internal.FileStatistics_mtime_nsec_get
    if _newclass:
        mtime_nsec = property(_pywrap_backend_internal.FileStatistics_mtime_nsec_get, _pywrap_backend_internal.FileStatistics_mtime_nsec_set)
    __swig_setmethods__["is_directory"] = _pywrap_backend_internal.FileStatistics_is_directory_set
    __swig_getmethods__["is_directory"] = _pywrap_backend_internal.FileStatistics_is_directory_get
    if _newclass:
        is_directory = property(_pywrap_backend_internal.FileStatistics_is_directory_get, _pywrap_backend_internal.FileStatistics_is_directory_set)

    def __init__(self, *args):
        this = _pywrap_backend_internal.new_FileStatistics(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _pywrap_backend_internal.delete_FileStatistics
    __del__ = lambda self: None
FileStatistics_swigregister = _pywrap_backend_internal.FileStatistics_swigregister
FileStatistics_swigregister(FileStatistics)

def Stat(filename, stats, out_status):
    return _pywrap_backend_internal.Stat(filename, stats, out_status)
Stat = _pywrap_backend_internal.Stat

class BufferedInputStream(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BufferedInputStream, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BufferedInputStream, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined")
    __repr__ = _swig_repr
    __swig_destroy__ = _pywrap_backend_internal.delete_BufferedInputStream
    __del__ = lambda self: None

    def Tell(self):
        return _pywrap_backend_internal.BufferedInputStream_Tell(self)

    def Seek(self, position):
        return _pywrap_backend_internal.BufferedInputStream_Seek(self, position)

    def ReadLineAsString(self):
        return _pywrap_backend_internal.BufferedInputStream_ReadLineAsString(self)
BufferedInputStream_swigregister = _pywrap_backend_internal.BufferedInputStream_swigregister
BufferedInputStream_swigregister(BufferedInputStream)

def ReadFromStream(stream, bytes, out_status):
    return _pywrap_backend_internal.ReadFromStream(stream, bytes, out_status)
ReadFromStream = _pywrap_backend_internal.ReadFromStream

def GetMatchingFiles(filename, out_status):
    return _pywrap_backend_internal.GetMatchingFiles(filename, out_status)
GetMatchingFiles = _pywrap_backend_internal.GetMatchingFiles

def ExtendSession(session):
    return _pywrap_backend_internal.ExtendSession(session)
ExtendSession = _pywrap_backend_internal.ExtendSession

def TF_SessionRun_wrapper(session, run_options, inputs, outputs, targets, run_metadata):
    return _pywrap_backend_internal.TF_SessionRun_wrapper(session, run_options, inputs, outputs, targets, run_metadata)
TF_SessionRun_wrapper = _pywrap_backend_internal.TF_SessionRun_wrapper
