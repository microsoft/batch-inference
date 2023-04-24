from typing import ClassVar as _ClassVar
from typing import Optional as _Optional

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message

DESCRIPTOR: _descriptor.FileDescriptor

class Request(_message.Message):
    __slots__ = ["ndarrays"]
    NDARRAYS_FIELD_NUMBER: _ClassVar[int]
    ndarrays: bytes
    def __init__(self, ndarrays: _Optional[bytes] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ["ndarrays"]
    NDARRAYS_FIELD_NUMBER: _ClassVar[int]
    ndarrays: bytes
    def __init__(self, ndarrays: _Optional[bytes] = ...) -> None: ...
