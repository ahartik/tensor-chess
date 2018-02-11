# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: board.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='board.proto',
  package='chess',
  syntax='proto2',
  serialized_pb=_b('\n\x0b\x62oard.proto\x12\x05\x63hess\"\x8b\x04\n\x05\x42oard\x12\x12\n\x06layers\x18\x01 \x03(\x04\x42\x02\x10\x01\x12+\n\tpromotion\x18\x02 \x01(\x0e\x32\x12.chess.Board.Layer:\x04MY_Q\x12\x17\n\x0fhalf_move_count\x18\x05 \x01(\x05\x12\x18\n\x10repetition_count\x18\x06 \x01(\x05\x12\x19\n\x11no_progress_count\x18\x07 \x01(\x05\x12\x11\n\tmove_from\x18\x08 \x01(\x05\x12\x0f\n\x07move_to\x18\t \x01(\x05\x12\x17\n\x0f\x65ncoded_move_to\x18\n \x01(\x05\x12\x13\n\x0bgame_result\x18\x0b \x01(\x05\"\xa0\x02\n\x05Layer\x12\x08\n\x04MY_P\x10\x00\x12\x08\n\x04MY_R\x10\x01\x12\x08\n\x04MY_B\x10\x02\x12\x08\n\x04MY_N\x10\x03\x12\x08\n\x04MY_Q\x10\x04\x12\x08\n\x04MY_K\x10\x05\x12\t\n\x05OPP_P\x10\x06\x12\t\n\x05OPP_R\x10\x07\x12\t\n\x05OPP_B\x10\x08\x12\t\n\x05OPP_N\x10\t\x12\t\n\x05OPP_Q\x10\n\x12\t\n\x05OPP_K\x10\x0b\x12\x11\n\rMY_LEGAL_FROM\x10\x0c\x12\x0f\n\x0bMY_LEGAL_TO\x10\r\x12\x12\n\x0eOPP_LEGAL_FROM\x10\x0e\x12\x10\n\x0cOPP_LEGAL_TO\x10\x0f\x12\x14\n\x10MY_CASTLE_RIGHTS\x10\x10\x12\x15\n\x11OPP_CASTLE_RIGHTS\x10\x11\x12\x12\n\x0eOPP_EN_PASSANT\x10\x12\x12\x0e\n\nNUM_LAYERS\x10\x13')
)



_BOARD_LAYER = _descriptor.EnumDescriptor(
  name='Layer',
  full_name='chess.Board.Layer',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MY_P', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MY_R', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MY_B', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MY_N', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MY_Q', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MY_K', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPP_P', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPP_R', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPP_B', index=8, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPP_N', index=9, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPP_Q', index=10, number=10,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPP_K', index=11, number=11,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MY_LEGAL_FROM', index=12, number=12,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MY_LEGAL_TO', index=13, number=13,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPP_LEGAL_FROM', index=14, number=14,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPP_LEGAL_TO', index=15, number=15,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MY_CASTLE_RIGHTS', index=16, number=16,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPP_CASTLE_RIGHTS', index=17, number=17,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPP_EN_PASSANT', index=18, number=18,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NUM_LAYERS', index=19, number=19,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=258,
  serialized_end=546,
)
_sym_db.RegisterEnumDescriptor(_BOARD_LAYER)


_BOARD = _descriptor.Descriptor(
  name='Board',
  full_name='chess.Board',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='layers', full_name='chess.Board.layers', index=0,
      number=1, type=4, cpp_type=4, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001')), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='promotion', full_name='chess.Board.promotion', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=4,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='half_move_count', full_name='chess.Board.half_move_count', index=2,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='repetition_count', full_name='chess.Board.repetition_count', index=3,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='no_progress_count', full_name='chess.Board.no_progress_count', index=4,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='move_from', full_name='chess.Board.move_from', index=5,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='move_to', full_name='chess.Board.move_to', index=6,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='encoded_move_to', full_name='chess.Board.encoded_move_to', index=7,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='game_result', full_name='chess.Board.game_result', index=8,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _BOARD_LAYER,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=23,
  serialized_end=546,
)

_BOARD.fields_by_name['promotion'].enum_type = _BOARD_LAYER
_BOARD_LAYER.containing_type = _BOARD
DESCRIPTOR.message_types_by_name['Board'] = _BOARD
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Board = _reflection.GeneratedProtocolMessageType('Board', (_message.Message,), dict(
  DESCRIPTOR = _BOARD,
  __module__ = 'board_pb2'
  # @@protoc_insertion_point(class_scope:chess.Board)
  ))
_sym_db.RegisterMessage(Board)


_BOARD.fields_by_name['layers'].has_options = True
_BOARD.fields_by_name['layers']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
