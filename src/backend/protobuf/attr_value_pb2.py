# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/framework/attr_value.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from backend.protobuf import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2
from backend.protobuf import tensor_shape_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2
from backend.protobuf import types_pb2 as tensorflow_dot_core_dot_framework_dot_types__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/core/framework/attr_value.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_options=_b('\n\030org.tensorflow.frameworkB\017AttrValueProtosP\001Z=github.com/tensorflow/tensorflow/tensorflow/go/core/framework\370\001\001'),
  serialized_pb=_b('\n*tensorflow/core/framework/attr_value.proto\x12\ntensorflow\x1a&tensorflow/core/framework/tensor.proto\x1a,tensorflow/core/framework/tensor_shape.proto\x1a%tensorflow/core/framework/types.proto\"\xa6\x04\n\tAttrValue\x12\x0b\n\x01s\x18\x02 \x01(\x0cH\x00\x12\x0b\n\x01i\x18\x03 \x01(\x03H\x00\x12\x0b\n\x01\x66\x18\x04 \x01(\x02H\x00\x12\x0b\n\x01\x62\x18\x05 \x01(\x08H\x00\x12$\n\x04type\x18\x06 \x01(\x0e\x32\x14.tensorflow.DataTypeH\x00\x12-\n\x05shape\x18\x07 \x01(\x0b\x32\x1c.tensorflow.TensorShapeProtoH\x00\x12)\n\x06tensor\x18\x08 \x01(\x0b\x32\x17.tensorflow.TensorProtoH\x00\x12/\n\x04list\x18\x01 \x01(\x0b\x32\x1f.tensorflow.AttrValue.ListValueH\x00\x12(\n\x04\x66unc\x18\n \x01(\x0b\x32\x18.tensorflow.NameAttrListH\x00\x12\x15\n\x0bplaceholder\x18\t \x01(\tH\x00\x1a\xe9\x01\n\tListValue\x12\t\n\x01s\x18\x02 \x03(\x0c\x12\r\n\x01i\x18\x03 \x03(\x03\x42\x02\x10\x01\x12\r\n\x01\x66\x18\x04 \x03(\x02\x42\x02\x10\x01\x12\r\n\x01\x62\x18\x05 \x03(\x08\x42\x02\x10\x01\x12&\n\x04type\x18\x06 \x03(\x0e\x32\x14.tensorflow.DataTypeB\x02\x10\x01\x12+\n\x05shape\x18\x07 \x03(\x0b\x32\x1c.tensorflow.TensorShapeProto\x12\'\n\x06tensor\x18\x08 \x03(\x0b\x32\x17.tensorflow.TensorProto\x12&\n\x04\x66unc\x18\t \x03(\x0b\x32\x18.tensorflow.NameAttrListB\x07\n\x05value\"\x92\x01\n\x0cNameAttrList\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x30\n\x04\x61ttr\x18\x02 \x03(\x0b\x32\".tensorflow.NameAttrList.AttrEntry\x1a\x42\n\tAttrEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.tensorflow.AttrValue:\x02\x38\x01\x42o\n\x18org.tensorflow.frameworkB\x0f\x41ttrValueProtosP\x01Z=github.com/tensorflow/tensorflow/tensorflow/go/core/framework\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensorflow_dot_core_dot_framework_dot_tensor__pb2.DESCRIPTOR,tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2.DESCRIPTOR,tensorflow_dot_core_dot_framework_dot_types__pb2.DESCRIPTOR,])



_ATTRVALUE_LISTVALUE = _descriptor.Descriptor(
  name='ListValue',
  full_name='tensorflow.AttrValue.ListValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='s', full_name='tensorflow.AttrValue.ListValue.s', index=0,
      number=2, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='i', full_name='tensorflow.AttrValue.ListValue.i', index=1,
      number=3, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='f', full_name='tensorflow.AttrValue.ListValue.f', index=2,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='b', full_name='tensorflow.AttrValue.ListValue.b', index=3,
      number=5, type=8, cpp_type=7, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='tensorflow.AttrValue.ListValue.type', index=4,
      number=6, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='tensorflow.AttrValue.ListValue.shape', index=5,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tensor', full_name='tensorflow.AttrValue.ListValue.tensor', index=6,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='func', full_name='tensorflow.AttrValue.ListValue.func', index=7,
      number=9, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=492,
  serialized_end=725,
)

_ATTRVALUE = _descriptor.Descriptor(
  name='AttrValue',
  full_name='tensorflow.AttrValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='s', full_name='tensorflow.AttrValue.s', index=0,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='i', full_name='tensorflow.AttrValue.i', index=1,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='f', full_name='tensorflow.AttrValue.f', index=2,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='b', full_name='tensorflow.AttrValue.b', index=3,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='tensorflow.AttrValue.type', index=4,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='tensorflow.AttrValue.shape', index=5,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tensor', full_name='tensorflow.AttrValue.tensor', index=6,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='list', full_name='tensorflow.AttrValue.list', index=7,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='func', full_name='tensorflow.AttrValue.func', index=8,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='placeholder', full_name='tensorflow.AttrValue.placeholder', index=9,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_ATTRVALUE_LISTVALUE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='value', full_name='tensorflow.AttrValue.value',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=184,
  serialized_end=734,
)


AttrValue = _reflection.GeneratedProtocolMessageType('AttrValue', (_message.Message,), dict(
  ListValue = _reflection.GeneratedProtocolMessageType('ListValue', (_message.Message,), dict(
    DESCRIPTOR = _ATTRVALUE_LISTVALUE,
    __module__ = 'tensorflow.core.framework.attr_value_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.AttrValue.ListValue)
    ))
  ,
  DESCRIPTOR = _ATTRVALUE,
  #__module__ = 'tensorflow.core.framework.attr_value_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.AttrValue)
  ))

