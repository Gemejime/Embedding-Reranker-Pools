"""KServe v2 binary codec and tensor protocol serialization for Triton Inference Server.

Implements:
1. pack_string_tensor: Triton BYTES tensor wire format (4-byte little-endian uint32 length prefix).
2. unpack_string_tensor: Inverse decoding of Triton BYTES wire format.
3. unpack_fp16_tensor: Fast zero-copy unpacking of FP16 byte buffers into float arrays.
4. pack_fp16_tensor: Packaging float arrays into FP16 byte buffers.
5. build_model_infer_request: Deterministic construction of ModelInferRequest protobufs.
"""

import struct
from collections.abc import Sequence
from typing import Any

import numpy as np

from api.proto.v1.triton_stubs_pb2 import (
    InferInputTensor,
    InferParameter,
    InferRequestedOutputTensor,
    ModelInferRequest,
)


def pack_string_tensor(strings: Sequence[str]) -> bytes:
    """Encode a sequence of UTF-8 strings into the Triton KServe v2 BYTES wire format.

    Each string element is packed as a 4-byte unsigned little-endian integer (<I)
    specifying the byte length, immediately followed by the raw UTF-8 bytes.
    """
    buffer_parts: list[bytes] = []
    for s in strings:
        encoded = s.encode("utf-8")
        buffer_parts.append(struct.pack("<I", len(encoded)))
        buffer_parts.append(encoded)
    return b"".join(buffer_parts)


def unpack_string_tensor(raw_bytes: bytes) -> list[str]:
    """Decode raw bytes from Triton KServe v2 BYTES wire format into UTF-8 strings."""
    strings: list[str] = []
    offset = 0
    total_len = len(raw_bytes)
    while offset < total_len:
        if offset + 4 > total_len:
            break
        (length,) = struct.unpack_from("<I", raw_bytes, offset)
        offset += 4
        if offset + length > total_len:
            break
        s_bytes = raw_bytes[offset : offset + length]
        strings.append(s_bytes.decode("utf-8"))
        offset += length
    return strings


def unpack_fp16_tensor(raw_bytes: bytes, shape: tuple[int, ...]) -> list[list[float]]:
    """Unpack raw FP16 byte buffer into nested list of float32 values matching the given shape.

    Guarantees consistent 2D list[list[float]] return for matrix or vector shapes.
    """
    arr = np.frombuffer(raw_bytes, dtype=np.float16)
    if shape:
        arr = arr.reshape(shape)
    if arr.ndim == 1:
        # Single vector: wrap as single row [[f1, f2, ...]]
        row: list[float] = arr.astype(np.float32).tolist()
        return [row]
    elif arr.ndim == 2:
        res: list[list[float]] = arr.astype(np.float32).tolist()
        return res
    else:
        # Multi-dimensional flatten to 2D
        flattened = arr.reshape(arr.shape[0], -1)
        res_multi: list[list[float]] = flattened.astype(np.float32).tolist()
        return res_multi


def pack_fp16_tensor(data: Sequence[Sequence[float]] | Sequence[float] | np.ndarray) -> bytes:
    """Pack floating point data into raw FP16 byte buffer."""
    if isinstance(data, np.ndarray):
        arr = data.astype(np.float16)
    else:
        arr = np.array(data, dtype=np.float16)
    return bytes(arr.tobytes())


def make_infer_parameter(val: Any) -> InferParameter:
    """Convert a Python primitive into a KServe v2 InferParameter."""
    param = InferParameter()
    if isinstance(val, bool):
        param.bool_param = val
    elif isinstance(val, int):
        param.int64_param = val
    elif isinstance(val, float):
        param.double_param = val
    elif isinstance(val, str):
        param.string_param = val
    else:
        param.string_param = str(val)
    return param


def build_infer_input_tensor(
    name: str,
    datatype: str,
    shape: Sequence[int],
    parameters: dict[str, Any] | None = None,
) -> InferInputTensor:
    """Build a single InferInputTensor protobuf message."""
    tensor = InferInputTensor(name=name, datatype=datatype, shape=list(shape))
    if parameters:
        for k, v in parameters.items():
            tensor.parameters[k].CopyFrom(make_infer_parameter(v))
    return tensor


def build_infer_requested_output_tensor(
    name: str,
    parameters: dict[str, Any] | None = None,
) -> InferRequestedOutputTensor:
    """Build a single InferRequestedOutputTensor protobuf message."""
    output = InferRequestedOutputTensor(name=name)
    if parameters:
        for k, v in parameters.items():
            output.parameters[k].CopyFrom(make_infer_parameter(v))
    return output


def build_single_text_infer_request(
    model_name: str,
    texts: Sequence[str],
    input_name: str = "raw_text",
    output_name: str = "embeddings",
    request_id: str | None = None,
    model_version: str = "",
) -> ModelInferRequest:
    """Build a ModelInferRequest for single-text embedding models (e.g. online_embedder_ensemble)."""
    batch_size = len(texts)
    packed_bytes = pack_string_tensor(texts)

    # Triton dynamic batching expects [batch_size, 1] for TYPE_STRING dims [1]
    input_tensor = build_infer_input_tensor(
        name=input_name,
        datatype="BYTES",
        shape=[batch_size, 1],
    )
    output_tensor = build_infer_requested_output_tensor(name=output_name)

    req = ModelInferRequest(
        model_name=model_name,
        model_version=model_version,
        id=request_id or "",
        inputs=[input_tensor],
        outputs=[output_tensor],
        raw_input_contents=[packed_bytes],
    )
    return req


def build_text_pair_infer_request(
    model_name: str,
    queries: Sequence[str],
    passages: Sequence[str],
    query_input_name: str = "query",
    passage_input_name: str = "passage",
    output_name: str = "logits",
    request_id: str | None = None,
    model_version: str = "",
) -> ModelInferRequest:
    """Build a ModelInferRequest for query-passage reranker models (e.g. online_reranker_ensemble)."""
    if len(queries) != len(passages):
        raise ValueError(
            f"Queries length ({len(queries)}) must match passages length ({len(passages)})"
        )
    batch_size = len(queries)
    packed_queries = pack_string_tensor(queries)
    packed_passages = pack_string_tensor(passages)

    query_tensor = build_infer_input_tensor(
        name=query_input_name,
        datatype="BYTES",
        shape=[batch_size, 1],
    )
    passage_tensor = build_infer_input_tensor(
        name=passage_input_name,
        datatype="BYTES",
        shape=[batch_size, 1],
    )
    output_tensor = build_infer_requested_output_tensor(name=output_name)

    req = ModelInferRequest(
        model_name=model_name,
        model_version=model_version,
        id=request_id or "",
        inputs=[query_tensor, passage_tensor],
        outputs=[output_tensor],
        raw_input_contents=[packed_queries, packed_passages],
    )
    return req


def build_model_infer_request(
    model_name: str,
    inputs: Sequence[InferInputTensor],
    outputs: Sequence[InferRequestedOutputTensor],
    raw_input_contents: Sequence[bytes],
    request_id: str | None = None,
    model_version: str = "",
    parameters: dict[str, Any] | None = None,
) -> ModelInferRequest:
    """Build a generic ModelInferRequest protobuf message."""
    req = ModelInferRequest(
        model_name=model_name,
        model_version=model_version,
        id=request_id or "",
        inputs=list(inputs),
        outputs=list(outputs),
        raw_input_contents=list(raw_input_contents),
    )
    if parameters:
        for k, v in parameters.items():
            req.parameters[k].CopyFrom(make_infer_parameter(v))
    return req
