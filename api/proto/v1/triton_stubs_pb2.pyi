from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ServerLiveRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ServerLiveResponse(_message.Message):
    __slots__ = ("live",)
    LIVE_FIELD_NUMBER: _ClassVar[int]
    live: bool
    def __init__(self, live: _Optional[bool] = ...) -> None: ...

class ServerReadyRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ServerReadyResponse(_message.Message):
    __slots__ = ("ready",)
    READY_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    def __init__(self, ready: _Optional[bool] = ...) -> None: ...

class ModelReadyRequest(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class ModelReadyResponse(_message.Message):
    __slots__ = ("ready",)
    READY_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    def __init__(self, ready: _Optional[bool] = ...) -> None: ...

class ModelMetadataRequest(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class ModelMetadataResponse(_message.Message):
    __slots__ = ("name", "versions", "platform", "inputs", "outputs")
    class TensorMetadata(_message.Message):
        __slots__ = ("name", "datatype", "shape")
        NAME_FIELD_NUMBER: _ClassVar[int]
        DATATYPE_FIELD_NUMBER: _ClassVar[int]
        SHAPE_FIELD_NUMBER: _ClassVar[int]
        name: str
        datatype: str
        shape: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, name: _Optional[str] = ..., datatype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSIONS_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    versions: _containers.RepeatedScalarFieldContainer[str]
    platform: str
    inputs: _containers.RepeatedCompositeFieldContainer[ModelMetadataResponse.TensorMetadata]
    outputs: _containers.RepeatedCompositeFieldContainer[ModelMetadataResponse.TensorMetadata]
    def __init__(self, name: _Optional[str] = ..., versions: _Optional[_Iterable[str]] = ..., platform: _Optional[str] = ..., inputs: _Optional[_Iterable[_Union[ModelMetadataResponse.TensorMetadata, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[ModelMetadataResponse.TensorMetadata, _Mapping]]] = ...) -> None: ...

class InferParameter(_message.Message):
    __slots__ = ("bool_param", "int64_param", "string_param", "double_param")
    BOOL_PARAM_FIELD_NUMBER: _ClassVar[int]
    INT64_PARAM_FIELD_NUMBER: _ClassVar[int]
    STRING_PARAM_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_PARAM_FIELD_NUMBER: _ClassVar[int]
    bool_param: bool
    int64_param: int
    string_param: str
    double_param: float
    def __init__(self, bool_param: _Optional[bool] = ..., int64_param: _Optional[int] = ..., string_param: _Optional[str] = ..., double_param: _Optional[float] = ...) -> None: ...

class InferTensorContents(_message.Message):
    __slots__ = ("bool_contents", "int_contents", "int64_contents", "uint_contents", "uint64_contents", "fp32_contents", "fp64_contents", "bytes_contents")
    BOOL_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    INT_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    INT64_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    UINT_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    UINT64_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    FP32_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    FP64_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    BYTES_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    bool_contents: _containers.RepeatedScalarFieldContainer[bool]
    int_contents: _containers.RepeatedScalarFieldContainer[int]
    int64_contents: _containers.RepeatedScalarFieldContainer[int]
    uint_contents: _containers.RepeatedScalarFieldContainer[int]
    uint64_contents: _containers.RepeatedScalarFieldContainer[int]
    fp32_contents: _containers.RepeatedScalarFieldContainer[float]
    fp64_contents: _containers.RepeatedScalarFieldContainer[float]
    bytes_contents: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, bool_contents: _Optional[_Iterable[bool]] = ..., int_contents: _Optional[_Iterable[int]] = ..., int64_contents: _Optional[_Iterable[int]] = ..., uint_contents: _Optional[_Iterable[int]] = ..., uint64_contents: _Optional[_Iterable[int]] = ..., fp32_contents: _Optional[_Iterable[float]] = ..., fp64_contents: _Optional[_Iterable[float]] = ..., bytes_contents: _Optional[_Iterable[bytes]] = ...) -> None: ...

class InferInputTensor(_message.Message):
    __slots__ = ("name", "datatype", "shape", "parameters", "contents")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: InferParameter
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[InferParameter, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATATYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    CONTENTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    datatype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    parameters: _containers.MessageMap[str, InferParameter]
    contents: InferTensorContents
    def __init__(self, name: _Optional[str] = ..., datatype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., parameters: _Optional[_Mapping[str, InferParameter]] = ..., contents: _Optional[_Union[InferTensorContents, _Mapping]] = ...) -> None: ...

class InferRequestedOutputTensor(_message.Message):
    __slots__ = ("name", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: InferParameter
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[InferParameter, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    name: str
    parameters: _containers.MessageMap[str, InferParameter]
    def __init__(self, name: _Optional[str] = ..., parameters: _Optional[_Mapping[str, InferParameter]] = ...) -> None: ...

class ModelInferRequest(_message.Message):
    __slots__ = ("model_name", "model_version", "id", "parameters", "inputs", "outputs", "raw_input_contents")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: InferParameter
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[InferParameter, _Mapping]] = ...) -> None: ...
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    RAW_INPUT_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    model_version: str
    id: str
    parameters: _containers.MessageMap[str, InferParameter]
    inputs: _containers.RepeatedCompositeFieldContainer[InferInputTensor]
    outputs: _containers.RepeatedCompositeFieldContainer[InferRequestedOutputTensor]
    raw_input_contents: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, model_name: _Optional[str] = ..., model_version: _Optional[str] = ..., id: _Optional[str] = ..., parameters: _Optional[_Mapping[str, InferParameter]] = ..., inputs: _Optional[_Iterable[_Union[InferInputTensor, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[InferRequestedOutputTensor, _Mapping]]] = ..., raw_input_contents: _Optional[_Iterable[bytes]] = ...) -> None: ...

class InferOutputTensor(_message.Message):
    __slots__ = ("name", "datatype", "shape", "parameters", "contents")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: InferParameter
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[InferParameter, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATATYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    CONTENTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    datatype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    parameters: _containers.MessageMap[str, InferParameter]
    contents: InferTensorContents
    def __init__(self, name: _Optional[str] = ..., datatype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., parameters: _Optional[_Mapping[str, InferParameter]] = ..., contents: _Optional[_Union[InferTensorContents, _Mapping]] = ...) -> None: ...

class ModelInferResponse(_message.Message):
    __slots__ = ("model_name", "model_version", "id", "parameters", "outputs", "raw_output_contents")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: InferParameter
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[InferParameter, _Mapping]] = ...) -> None: ...
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    RAW_OUTPUT_CONTENTS_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    model_version: str
    id: str
    parameters: _containers.MessageMap[str, InferParameter]
    outputs: _containers.RepeatedCompositeFieldContainer[InferOutputTensor]
    raw_output_contents: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, model_name: _Optional[str] = ..., model_version: _Optional[str] = ..., id: _Optional[str] = ..., parameters: _Optional[_Mapping[str, InferParameter]] = ..., outputs: _Optional[_Iterable[_Union[InferOutputTensor, _Mapping]]] = ..., raw_output_contents: _Optional[_Iterable[bytes]] = ...) -> None: ...

class SystemSharedMemoryRegisterRequest(_message.Message):
    __slots__ = ("name", "key", "offset", "byte_size")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    BYTE_SIZE_FIELD_NUMBER: _ClassVar[int]
    name: str
    key: str
    offset: int
    byte_size: int
    def __init__(self, name: _Optional[str] = ..., key: _Optional[str] = ..., offset: _Optional[int] = ..., byte_size: _Optional[int] = ...) -> None: ...

class SystemSharedMemoryRegisterResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SystemSharedMemoryUnregisterRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class SystemSharedMemoryUnregisterResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
