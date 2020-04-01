# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/contrib/boosted_trees/proto/quantiles.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/contrib/boosted_trees/proto/quantiles.proto',
  package='boosted_trees',
  syntax='proto3',
  serialized_options=_b('\370\001\001'),
  serialized_pb=_b('\n6tensorflow/contrib/boosted_trees/proto/quantiles.proto\x12\rboosted_trees\"4\n\x0eQuantileConfig\x12\x0b\n\x03\x65ps\x18\x01 \x01(\x01\x12\x15\n\rnum_quantiles\x18\x02 \x01(\x03\"R\n\rQuantileEntry\x12\r\n\x05value\x18\x01 \x01(\x02\x12\x0e\n\x06weight\x18\x02 \x01(\x02\x12\x10\n\x08min_rank\x18\x03 \x01(\x02\x12\x10\n\x08max_rank\x18\x04 \x01(\x02\"E\n\x14QuantileSummaryState\x12-\n\x07\x65ntries\x18\x01 \x03(\x0b\x32\x1c.boosted_trees.QuantileEntry\"M\n\x13QuantileStreamState\x12\x36\n\tsummaries\x18\x01 \x03(\x0b\x32#.boosted_trees.QuantileSummaryStateB\x03\xf8\x01\x01\x62\x06proto3')
)




_QUANTILECONFIG = _descriptor.Descriptor(
  name='QuantileConfig',
  full_name='boosted_trees.QuantileConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='eps', full_name='boosted_trees.QuantileConfig.eps', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_quantiles', full_name='boosted_trees.QuantileConfig.num_quantiles', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=73,
  serialized_end=125,
)


_QUANTILEENTRY = _descriptor.Descriptor(
  name='QuantileEntry',
  full_name='boosted_trees.QuantileEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='boosted_trees.QuantileEntry.value', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight', full_name='boosted_trees.QuantileEntry.weight', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_rank', full_name='boosted_trees.QuantileEntry.min_rank', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_rank', full_name='boosted_trees.QuantileEntry.max_rank', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=127,
  serialized_end=209,
)


_QUANTILESUMMARYSTATE = _descriptor.Descriptor(
  name='QuantileSummaryState',
  full_name='boosted_trees.QuantileSummaryState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='entries', full_name='boosted_trees.QuantileSummaryState.entries', index=0,
      number=1, type=11, cpp_type=10, label=3,
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
  serialized_start=211,
  serialized_end=280,
)


_QUANTILESTREAMSTATE = _descriptor.Descriptor(
  name='QuantileStreamState',
  full_name='boosted_trees.QuantileStreamState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='summaries', full_name='boosted_trees.QuantileStreamState.summaries', index=0,
      number=1, type=11, cpp_type=10, label=3,
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
  serialized_start=282,
  serialized_end=359,
)

_QUANTILESUMMARYSTATE.fields_by_name['entries'].message_type = _QUANTILEENTRY
_QUANTILESTREAMSTATE.fields_by_name['summaries'].message_type = _QUANTILESUMMARYSTATE
DESCRIPTOR.message_types_by_name['QuantileConfig'] = _QUANTILECONFIG
DESCRIPTOR.message_types_by_name['QuantileEntry'] = _QUANTILEENTRY
DESCRIPTOR.message_types_by_name['QuantileSummaryState'] = _QUANTILESUMMARYSTATE
DESCRIPTOR.message_types_by_name['QuantileStreamState'] = _QUANTILESTREAMSTATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

QuantileConfig = _reflection.GeneratedProtocolMessageType('QuantileConfig', (_message.Message,), dict(
  DESCRIPTOR = _QUANTILECONFIG,
  __module__ = 'tensorflow.contrib.boosted_trees.proto.quantiles_pb2'
  # @@protoc_insertion_point(class_scope:boosted_trees.QuantileConfig)
  ))
_sym_db.RegisterMessage(QuantileConfig)

QuantileEntry = _reflection.GeneratedProtocolMessageType('QuantileEntry', (_message.Message,), dict(
  DESCRIPTOR = _QUANTILEENTRY,
  __module__ = 'tensorflow.contrib.boosted_trees.proto.quantiles_pb2'
  # @@protoc_insertion_point(class_scope:boosted_trees.QuantileEntry)
  ))
_sym_db.RegisterMessage(QuantileEntry)

QuantileSummaryState = _reflection.GeneratedProtocolMessageType('QuantileSummaryState', (_message.Message,), dict(
  DESCRIPTOR = _QUANTILESUMMARYSTATE,
  __module__ = 'tensorflow.contrib.boosted_trees.proto.quantiles_pb2'
  # @@protoc_insertion_point(class_scope:boosted_trees.QuantileSummaryState)
  ))
_sym_db.RegisterMessage(QuantileSummaryState)

QuantileStreamState = _reflection.GeneratedProtocolMessageType('QuantileStreamState', (_message.Message,), dict(
  DESCRIPTOR = _QUANTILESTREAMSTATE,
  __module__ = 'tensorflow.contrib.boosted_trees.proto.quantiles_pb2'
  # @@protoc_insertion_point(class_scope:boosted_trees.QuantileStreamState)
  ))
_sym_db.RegisterMessage(QuantileStreamState)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
