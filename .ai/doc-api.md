# Microservice Documented API ("Doc API") - RAG Inference Serving

> **MANDATORY INVARIANT**: This document specifies both the internal system contracts, Triton inference protocols, and the **HTTP Doc API & OpenAPI Endpoint** exposed by the microservice.
> **API ACCESSIBILITY REQUIREMENT**: This documentation specification MUST be accessible directly via HTTP endpoints hosted on the microservice's internal operational server (port `8080`):
>
> - `GET /docs` — Interactive OpenAPI 3.1 Documentation via **Swagger UI**.
> - `GET /redoc` — Interactive OpenAPI 3.1 Documentation via **ReDoc**.
> - `GET /openapi.json` & `GET /api/docs` — Machine-readable **OpenAPI 3.1 JSON Specification**.
>
> **MANDATORY MAINTENANCE RULE**: This file and the corresponding OpenAPI specification schemas MUST be updated on **every code change or addition**, regardless of how small the modification is.

---

## 1. Documentation & Probe API Endpoints (OpenAPI 3.1)

Every worker pod (`online-embedding`, `online-reranker`, and `batch-ingestion`) runs an internal FastAPI ASGI server (default port `8080`) providing operational observability, health verification, and OpenAPI documentation:

| Endpoint | Method | Response Format | Description |
| :--- | :--- | :--- | :--- |
| `/docs` | `GET` | `text/html` | Interactive Swagger UI powered by OpenAPI 3.1 |
| `/redoc` | `GET` | `text/html` | Interactive ReDoc UI documentation |
| `/api/docs` | `GET` | `application/json` | Raw OpenAPI 3.1 JSON specification schema |
| `/openapi.json` | `GET` | `application/json` | Standard machine-readable OpenAPI 3.1 JSON specification |
| `/healthz` | `GET` | `application/json` | Liveness probe returning `{"status": "healthy"}` |
| `/ready` | `GET` | `application/json` | Readiness probe verifying GPU caches & Triton engine availability |
| `/metrics` | `GET` | `text/plain` | Prometheus metrics scrape endpoint |

### OpenAPI Root Info Schema

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "RAG Inference Serving Operational & Documentation API",
    "description": "Internal HTTP Documentation, Health, and Operational API for rag-inference-serving (Triton / TensorRT runtime)",
    "version": "1.0.0"
  },
  "paths": {
    "/healthz": {
      "get": {
        "summary": "Liveness Probe",
        "description": "Returns HTTP 200 if the operational probe server and Triton process monitor are running.",
        "responses": {
          "200": {
            "description": "Worker event loop is healthy",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "status": { "type": "string", "example": "healthy" },
                    "timestamp_utc": { "type": "string", "format": "date-time" }
                  },
                  "required": ["status", "timestamp_utc"]
                }
              }
            }
          }
        }
      }
    },
    "/ready": {
      "get": {
        "summary": "Readiness Probe",
        "description": "Verifies that TensorRT model engines are loaded in VRAM and queue latencies are within bounds.",
        "responses": {
          "200": {
            "description": "Worker model engines loaded and ready to accept inference requests",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "status": { "type": "string", "example": "ready" },
                    "pool_type": { "type": "string", "enum": ["online_embedding", "online_reranker", "batch_ingestion"] },
                    "vram_used_bytes": { "type": "integer" },
                    "vram_capacity_bytes": { "type": "integer" },
                    "models_ready": { "type": "array", "items": { "type": "string" } }
                  },
                  "required": ["status", "pool_type", "models_ready"]
                }
              }
            }
          },
          "503": {
            "description": "Models initializing or VRAM allocation threshold breached (>85%)"
          }
        }
      }
    },
    "/api/docs": {
      "get": {
        "summary": "OpenAPI Documentation Schema",
        "description": "Direct JSON output of the OpenAPI 3.1 specification for machine synchronization.",
        "responses": {
          "200": { "description": "OpenAPI 3.1 JSON specification" }
        }
      }
    },
    "/metrics": {
      "get": {
        "summary": "Prometheus Metrics Scrape",
        "security": [
          { "BearerAuth": [] },
          { "ApiKeyAuth": [] },
          {}
        ],
        "responses": {
          "200": { "description": "Prometheus text exposition format" },
          "401": { "description": "Unauthorized - invalid or missing operational API token" }
        }
      }
    }
  },
  "components": {
    "securitySchemes": {
      "BearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "description": "Optional bearer token authentication for operational metrics scraping."
      },
      "ApiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
        "description": "Optional API key header authentication for operational metrics scraping."
      }
    }
  }
}
```

---

## 2. Triton KServe v2 Inference Protocol Specifications

Triton exposes standard KServe v2 Data Plane endpoints for high-throughput tensor inference across gRPC (`8001`) and HTTP (`8000`).

### gRPC Service Definition (`triton_stubs.proto`)

```protobuf
syntax = "proto3";

package inference;

service GRPCInferenceService {
  rpc ServerLive(ServerLiveRequest) returns (ServerLiveResponse);
  rpc ServerReady(ServerReadyRequest) returns (ServerReadyResponse);
  rpc ModelReady(ModelReadyRequest) returns (ModelReadyResponse);
  rpc ModelMetadata(ModelMetadataRequest) returns (ModelMetadataResponse);
  rpc ModelInfer(ModelInferRequest) returns (ModelInferResponse);
  rpc SystemSharedMemoryRegister(SystemSharedMemoryRegisterRequest) returns (SystemSharedMemoryRegisterResponse);
  rpc SystemSharedMemoryStatus(SystemSharedMemoryStatusRequest) returns (SystemSharedMemoryStatusResponse);
  rpc SystemSharedMemoryUnregister(SystemSharedMemoryUnregisterRequest) returns (SystemSharedMemoryUnregisterResponse);
}

message SystemSharedMemoryRegisterRequest {
  string name = 1;
  string key = 2;
  uint64 offset = 3;
  uint64 byte_size = 4;
}

message SystemSharedMemoryRegisterResponse {}

message SystemSharedMemoryStatusRequest {
  string name = 1;
}

message SystemSharedMemoryStatusResponse {
  message RegionStatus {
    string name = 1;
    string key = 2;
    uint64 offset = 3;
    uint64 byte_size = 4;
  }
  map<string, RegionStatus> regions = 1;
}

message SystemSharedMemoryUnregisterRequest {
  string name = 1;
}

message SystemSharedMemoryUnregisterResponse {}

message ModelMetadataResponse {
  message TensorMetadata {
    string name = 1;
    string datatype = 2;
    repeated int64 shape = 3;
  }
  string name = 1;
  repeated string versions = 2;
  string platform = 3;
  repeated TensorMetadata inputs = 4;
  repeated TensorMetadata outputs = 5;
}

message ModelInferRequest {
  string model_name = 1;
  string model_version = 2;
  string id = 3;
  map<string, InferParameter> parameters = 4;
  repeated InferInputTensor inputs = 5;
  repeated InferRequestedOutputTensor outputs = 6;
  repeated bytes raw_input_contents = 7;
}

message InferInputTensor {
  string name = 1;
  string datatype = 2;          // e.g., "FP32", "FP16", "INT32", "BYTES"
  repeated int64 shape = 3;     // [batch_size, sequence_length]
  map<string, InferParameter> parameters = 4;
  InferTensorContents contents = 5;
}

message ModelInferResponse {
  string model_name = 1;
  string model_version = 2;
  string id = 3;
  map<string, InferParameter> parameters = 4;
  repeated InferOutputTensor outputs = 5;
  repeated bytes raw_output_contents = 6;
}

message InferOutputTensor {
  string name = 1;
  string datatype = 2;          // "FP16" or "FP32"
  repeated int64 shape = 3;     // [batch_size, 768] or [batch_size, 1]
  map<string, InferParameter> parameters = 4;
  InferTensorContents contents = 5;
}
```

---

## 3. Dynamic Batching Model Configurations (`config.pbtxt`) & Ensemble Pipelines

To eliminate external Python GIL bottlenecks and provide end-to-end zero-copy execution, each repository pairs its C++ HuggingFace tokenizer with its TensorRT engine via a top-level Triton Ensemble Model (`ensemble_scheduler`).

### A. Online Dense Embedding Repository (`online_embedding_repository`)

#### 1. Ensemble Pipeline: `online_embedder_ensemble`
- **Platform**: `ensemble`
- **Max Batch Size**: `32`
- **Input**: `raw_text` (`TYPE_STRING`, shape `[ 1 ]`)
- **Output**: `embeddings` (`TYPE_FP16`, shape `[ 768 ]`)
- **Internal Pipeline**: Steps `raw_text` $\rightarrow$ `text_tokenizer` (`input_ids`, `attention_mask`) $\rightarrow$ `dense_embedder_trt` (`embeddings`).

```protobuf
name: "online_embedder_ensemble"
platform: "ensemble"
max_batch_size: 32

input [
  {
    name: "raw_text"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "embeddings"
    data_type: TYPE_FP16
    dims: [ 768 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "text_tokenizer"
      model_version: -1
      input_map { key: "raw_text", value: "raw_text" }
      output_map { key: "input_ids", value: "token_ids" }
      output_map { key: "attention_mask", value: "attn_mask" }
    },
    {
      model_name: "dense_embedder_trt"
      model_version: -1
      input_map { key: "input_ids", value: "token_ids" }
      input_map { key: "attention_mask", value: "attn_mask" }
      output_map { key: "embeddings", value: "embeddings" }
    }
  ]
}
```

#### 2. Engine Model: `dense_embedder_trt`
- **Engine**: TensorRT FP16 (`model.plan`)
- **Max Batch Size**: `32`
- **Dynamic Batching**: `max_queue_delay_microseconds: 2000` (2 ms)
- **Target Workload**: Single-sentence query vectors
- **SLO Target**: Standalone latency P99 $\le 10\text{ ms}$ at 1,500 queries/sec per GPU.

```protobuf
name: "dense_embedder_trt"
platform: "tensorrt_plan"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]

output [
  {
    name: "embeddings"
    data_type: TYPE_FP16
    dims: [ 768 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 2000
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

### B. Online Cross-Encoder Reranker Repository (`online_reranker_repository`)

#### 1. Ensemble Pipeline: `online_reranker_ensemble`
- **Platform**: `ensemble`
- **Max Batch Size**: `32`
- **Inputs**: `query` (`TYPE_STRING`, shape `[ 1 ]`), `passage` (`TYPE_STRING`, shape `[ 1 ]`)
- **Output**: `logits` (`TYPE_FP16`, shape `[ 1 ]`)
- **Internal Pipeline**: Steps `query` + `passage` $\rightarrow$ `text_tokenizer` (`input_ids`, `attention_mask`) $\rightarrow$ `cross_encoder_trt` (`logits`).

```protobuf
name: "online_reranker_ensemble"
platform: "ensemble"
max_batch_size: 32

input [
  {
    name: "query"
    data_type: TYPE_STRING
    dims: [ 1 ]
  },
  {
    name: "passage"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP16
    dims: [ 1 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "text_tokenizer"
      model_version: -1
      input_map { key: "query", value: "query" }
      input_map { key: "passage", value: "passage" }
      output_map { key: "input_ids", value: "token_ids" }
      output_map { key: "attention_mask", value: "attn_mask" }
    },
    {
      model_name: "cross_encoder_trt"
      model_version: -1
      input_map { key: "input_ids", value: "token_ids" }
      input_map { key: "attention_mask", value: "attn_mask" }
      output_map { key: "logits", value: "logits" }
    }
  ]
}
```

#### 2. Engine Model: `cross_encoder_trt`
- **Engine**: TensorRT FP16 (`model.plan`)
- **Max Batch Size**: `32`
- **Dynamic Batching**: `max_queue_delay_microseconds: 4000` (4 ms)
- **Target Workload**: Top-10 to top-30 candidate pairs
- **SLO Target**: P99 $\le 30\text{ ms}$ under sustained load.
- **Queue Saturation Fallback**: Saturated queues enforce `default_queue_policy` with `timeout_action: REJECT`, returning `UNAVAILABLE` (gRPC status 14, with "Request timeout expired") or HTTP 503 natively (or `RESOURCE_EXHAUSTED` if translated by specific ingress gateways) to trigger immediate orchestrator bypass to un-reranked RRF scores.

```protobuf
name: "cross_encoder_trt"
platform: "tensorrt_plan"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP16
    dims: [ 1 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 4000
  default_queue_policy {
    max_queue_size: 64
    timeout_action: REJECT
    default_timeout_microseconds: 10000
  }
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

### C. Batch Ingestion Repository (`batch_ingestion_repository`)

#### 1. Ensemble Pipeline: `batch_embedder_ensemble`
- **Platform**: `ensemble`
- **Max Batch Size**: `256`
- **Input**: `raw_text` (`TYPE_STRING`, shape `[ 1 ]`)
- **Output**: `embeddings` (`TYPE_FP16`, shape `[ 768 ]`)

```protobuf
name: "batch_embedder_ensemble"
platform: "ensemble"
max_batch_size: 256

input [
  {
    name: "raw_text"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "embeddings"
    data_type: TYPE_FP16
    dims: [ 768 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "text_tokenizer"
      model_version: -1
      input_map { key: "raw_text", value: "raw_text" }
      output_map { key: "input_ids", value: "token_ids" }
      output_map { key: "attention_mask", value: "attn_mask" }
    },
    {
      model_name: "batch_embedder_trt"
      model_version: -1
      input_map { key: "input_ids", value: "token_ids" }
      input_map { key: "attention_mask", value: "attn_mask" }
      output_map { key: "embeddings", value: "embeddings" }
    }
  ]
}
```

#### 2. Engine Model: `batch_embedder_trt`
- **Engine**: TensorRT FP16 (`model.plan`)
- **Max Batch Size**: `256`
- **Dynamic Batching**: `max_queue_delay_microseconds: 50000` (50 ms)
- **Target Workload**: Bulk document chunk arrays (256–512 tokens/chunk)
- **Throughput Target**: 2,500 to 3,500 chunks/sec per L4/A10G instance.

```protobuf
name: "batch_embedder_trt"
platform: "tensorrt_plan"
max_batch_size: 256

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]

output [
  {
    name: "embeddings"
    data_type: TYPE_FP16
    dims: [ 768 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 64, 128, 256 ]
  max_queue_delay_microseconds: 50000
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

---

## 4. POSIX Shared Memory IPC Protocol (`/dev/shm`) & Pod Boundaries

For high-density batch ingestion, `rag-chunk-lifecycle-service` and ingestion workers utilize local POSIX shared memory buffers to bypass network serialization **when co-located within the same Kubernetes Pod**:

1. **Kubernetes Boundary Invariant**:
   - POSIX shared memory (`/dev/shm`) is mounted as a Kubernetes `emptyDir` RAM volume sized to `8Gi`.
   - **Strict Pod Isolation**: Shared memory regions are shared strictly between containers within the **same Kubernetes Pod**. Cross-pod or cross-node `/dev/shm` access is strictly forbidden to prevent `SIGBUS` memory faults.
   - Remote ingestion workers running on separate nodes use **batched gRPC over HTTP/2** directly.
2. **Buffer Allocation**:
   ```c
   int shm_fd = shm_open("/triton_batch_shm_region", O_CREAT | O_RDWR, 0660);
   ftruncate(shm_fd, region_size_bytes);
   void* ptr = mmap(NULL, region_size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
   ```
3. **Registration with Triton**:
   The caller registers the shared memory region with Triton via gRPC:
   - Call `SystemSharedMemoryRegisterRequest`: specifies `name`, `key`, `offset`, `byte_size`.
4. **Execution**:
   - `ModelInferRequest` references the registered region name instead of packing raw bytes in `raw_input_contents`.
   - Triton reads input tensors directly from RAM and writes output vectors directly back to the output `/dev/shm` region.
5. **Cleanup Lifecycle**:
   - Explicit `SystemSharedMemoryUnregister` followed by `munmap` and `shm_unlink` to prevent orphaned memory segments.

---

## 5. Error Codes & Upstream Fallback Policies

| Status Code | Reason | Upstream Orchestrator Action |
| :--- | :--- | :--- |
| `OK` (0) | Inference succeeded | Consume embeddings or cross-encoder logits |
| `UNAVAILABLE` (14) / `RESOURCE_EXHAUSTED` (8) | Dynamic batch queue saturated beyond timeout ("Request timeout expired" on UNAVAILABLE natively; RESOURCE_EXHAUSTED if translated) | **Immediate fallback**: Bypass cross-encoder reranker and return un-reranked Reciprocal Rank Fusion (RRF) scores directly |
| `DEADLINE_EXCEEDED` (4) | Request exceeded client timeout budget | Abort request; log warning; fall back to un-reranked RRF scores |
| `INVALID_ARGUMENT` (3) | Token length exceeds maximum context window | Deterministically truncate at C++ kernel boundary |
| `UNAVAILABLE` (14) | Triton pod restarting, spot eviction, or queue timeout | Route request to healthy AZ replica or fall back to RRF |

---

## 6. Model Serialization & Engine Compilation CLI Specifications

### A. ONNX Model Export (`scripts/export_onnx.py`)
Exports Hugging Face models to dynamic-axis ONNX computation graphs:
- **Dense Embedder (`--model-type dense`)**:
  - Inlines Mean Pooling and L2 Normalization in the graph.
  - Inputs: `input_ids`, `attention_mask` (dynamic `[batch_size, sequence_length]`).
  - Output: `embeddings` (dynamic `[batch_size, 768]`).
- **Cross-Encoder Reranker (`--model-type reranker`)**:
  - Inputs: `input_ids`, `attention_mask` (dynamic `[batch_size, sequence_length]`).
  - Output: `logits` (dynamic `[batch_size, 1]`).
- **Flags**: `--model-name-or-path`, `--output-path`, `--opset-version 17`, `--dry-run`.

### B. In-Engine Tokenizer Serialization (`scripts/export_tokenizer.py`)
Prepares tokenizer configuration and vocabulary for Triton's `text_tokenizer` backend:
- Enforces deterministic kernel truncation limits:
  - Dense Embedding: max `512` tokens.
  - Cross-Encoder Reranker: max `2048` tokens.
- Generates `model.onnx`, `vocab.txt`, and `tokenizer_config.json`.
- **Flags**: `--model-type`, `--tokenizer-name-or-path`, `--output-dir`, `--max-length`, `--dry-run`.

### C. TensorRT Compilation (`scripts/build_trt_engine.sh` / `scripts/build_trt_engine.py`)
Generates FP16 `.plan` engines using NVIDIA `trtexec` adhering to calibrated shape profiles:
- **`online_embed`**: min `[1, 1]`, opt `[16, 128]`, max `[32, 512]`
- **`online_rerank`**: min `[1, 1]`, opt `[16, 256]`, max `[32, 2048]`
- **`batch_ingestion`**: min `[1, 1]`, opt `[128, 256]`, max `[256, 512]`

---

## 7. Container Supervisor & Declarative Helm Deployments

### A. Dual Process Supervisor (`deployments/docker/entrypoint.sh`)
- Supervises `tritonserver` (ports 8000, 8001, 8002) and `uvicorn probe_server:app` (port 8080).
- Traps `SIGTERM` and `SIGINT` to cleanly signal child processes and drain connections before process exit.
- Hardened non-root user (UID 1000, `triton`).

### B. Parameterized Helm Overlays (`deployments/helm/`)
- `values-online-embed.yaml`: 4–8 replicas, on-demand node group affinity, queue delay $\le 2\text{ ms}$, max batch 32, KEDA autoscaling trigger on `triton_queue_duration_microseconds` $\ge 3000\mu\text{s}$.
- `values-online-rerank.yaml`: 4–12 replicas, on-demand node group affinity, queue delay $\le 4\text{ ms}$, max batch 32, KEDA autoscaling trigger on `triton_queue_duration_microseconds` $\ge 5000\mu\text{s}$.
- `values-batch-ingestion.yaml`: 12–16 replicas, preemptible spot node group with spot tolerations, queue delay $\le 50\text{ ms}$, max batch 256, KEDA autoscaling trigger on Kafka chunk backlog depth.
- **Storage**: 8 GiB RAM `emptyDir` mounted at `/dev/shm`, read-only NVMe CSI persistent volume mounts for model engines (`readOnly: true`).

---

## 8. Client Integration Library & Native Circuit Breaking (Phase 3)

### A. Architectural Overview & Polyglot Contracts
The serving infrastructure provides low-latency gRPC access without intermediate proxy sidecars (ADR-005):
- **Go Microservice Integration (`rag-search-orchestrator`)**:
  - Protobuf definitions in `api/proto/v1/triton_stubs.proto` compile with `option go_package = "api/proto/v1/inference";`.
  - Go callers establish HTTP/2 keepalive connection pools directly to Triton gRPC endpoints (port `8001`).
- **Python Client Library (`internal/client/`)**:
  - Provides reference implementation, used by Python workers (`rag-document-ai-worker`, `rag-eval-governance-worker`), batch ingestion pipelines, and integration tests.
  - Implements zero-copy KServe v2 binary tensor codecs, discrete pool connection pooling, fault-tolerant circuit breaking, and un-reranked RRF fallback.

### B. KServe v2 Binary Tensor Codec (`internal/client/kserve_codec.py`)
Direct binary encoding avoiding JSON serialization overhead:
- **`pack_string_tensor(strings: Sequence[str]) -> bytes`**:
  - Encodes strings into Triton's raw BYTES wire format: a 4-byte little-endian unsigned integer (`<I`) denoting byte length, immediately followed by the UTF-8 encoded string bytes.
- **`unpack_string_tensor(raw_bytes: bytes) -> list[str]`**:
  - Inverse decoding parsing `<I` length-prefixed UTF-8 chunks.
- **`unpack_fp16_tensor(raw_bytes: bytes, shape: tuple[int, ...]) -> list[list[float]]`**:
  - Fast zero-copy unpacking of FP16 byte buffers into float32 nested lists using NumPy.
- **`build_single_text_infer_request` & `build_text_pair_infer_request`**:
  - Helper functions building deterministic `ModelInferRequest` protobuf payloads for embedding and reranker ensembles.

### C. Discrete gRPC Connection Pooling (`internal/client/grpc_pool.py`)
`TritonGRPCChannelPool` manages dedicated, thread-safe HTTP/2 channels for each isolated pool:
- **Keepalive Parameters**:
  - `grpc.keepalive_time_ms = 10000` (10s pings)
  - `grpc.keepalive_timeout_ms = 5000` (5s ack timeout)
  - `grpc.keepalive_permit_without_calls = 0` (idle pinging disabled)
  - `grpc.http2.max_pings_without_data = 0`
  - `grpc.max_receive_message_length = 67108864` (64 MiB)
  - `grpc.max_send_message_length = 67108864` (64 MiB)
- **Pool Targets**:
  - `PoolType.ONLINE_EMBEDDING` $\rightarrow$ `RAG_SERVING_ONLINE_EMBED_GRPC_URL` (default `localhost:8001`)
  - `PoolType.ONLINE_RERANKER` $\rightarrow$ `RAG_SERVING_ONLINE_RERANK_GRPC_URL` (default `localhost:8001`)
  - `PoolType.BATCH_INGESTION` $\rightarrow$ `RAG_SERVING_BATCH_INGEST_GRPC_URL` (default `localhost:8001`)
- **Lifecycle**: Lazy instantiation with lock protection, active liveness/readiness health probes (`is_channel_ready`, `is_pool_ready`), and clean connection draining via `.close()`.

### D. Resilient Circuit Breaker & Un-Reranked RRF Fallback (`internal/client/circuit_breaker.py`)
The `RerankerCircuitBreaker` protects online query latency budgets against Triton queue saturation:
- **State Machine**:
  ```mermaid
  stateDiagram-v2
      [*] --> CLOSED
      CLOSED --> OPEN : 3 consecutive failures (RESOURCE_EXHAUSTED / DEADLINE_EXCEEDED / UNAVAILABLE)
      OPEN --> HALF_OPEN : Recovery timeout (5.0s cooldown elapsed)
      HALF_OPEN --> CLOSED : Probe request succeeds
      HALF_OPEN --> OPEN : Probe request fails
      OPEN --> OPEN : Instant un-reranked RRF fallback (bypasses gRPC)
  ```
- **Tripping Triggers**:
  - `UNAVAILABLE` / `RESOURCE_EXHAUSTED` (Triton queue rejection returning gRPC status 14 "Request timeout expired" natively according to reranker queue policy: 64-request queue limit or 10 ms queue timeout with REJECT action; also trips on `RESOURCE_EXHAUSTED` where translated)
  - `DEADLINE_EXCEEDED` (Client-side 35 ms latency budget breach)
  - `UNAVAILABLE` (Pod restarts or spot node evictions)
- **Deterministic Un-Reranked RRF Fallback**:
  - Preserves initial candidate ranking ($rank(d) \in [1..N]$) and computes:
    $$RRF(d) = \frac{1.0}{k + rank(d)} \quad (k = 60.0)$$
  - Returns `RerankResult` objects marked with `fallback_applied=True` and `fallback_reason`.

### E. High-Performance Search Orchestrator Client (`internal/client/search_client.py`)
- **`embed_query(query: str, request_id: str | None = None) -> list[float]`**:
  - Client deadline: **15 ms** (`timeout=0.015`).
  - Calls `online_embedder_ensemble`, unpacking 768-dimensional FP16 vectors.
- **`rerank(query: str, candidates: list[CandidateDocument], request_id: str | None = None) -> list[RerankResult]`**:
  - Client deadline: **35 ms** (`timeout=0.035`).
  - Automatically chunks candidate sets exceeding `max_batch_size = 32`.
  - Empty candidates list returns `[]` immediately without network roundtrips.
  - Automatically catches server rejections or timeout breaches and returns deterministic RRF fallbacks.
- **Secured Structured Logging**:
  - Emits correlation metrics (`request_id`, `model_name`, `pool_type`, `batch_size`, `compute_latency_ms`).
  - Redacts query text, candidate passages, and raw embedding arrays.

### F. High-Density Batch Ingestion Client (`internal/client/batch_client.py`)
- **`embed_chunks(chunks: Sequence[str], target_host: str = "localhost", use_shm: bool = False) -> list[list[float]]`**:
  - Dynamic micro-batching: splits arbitrary chunk sequences into batches of $N \le 256$ chunks.
  - Dual transport routing:
    - **Pod-Local (`target_host="localhost"`, `use_shm=True`)**: Passes tensors via 8 GiB `/dev/shm` emptyDir using POSIX shared memory and Triton `SystemSharedMemoryRegister`.
    - **Remote Node (`target_host != "localhost"`)**: Validates pod boundaries (`validate_pod_boundary`), rejecting cross-pod SHM with `PodBoundaryViolationError` and streaming batches over HTTP/2 gRPC.

---

## 9. Cluster Load Sweeps, Chaos Resilience & GitOps Delivery (Phase 4)

### A. Quantitative Benchmark & Load Sweeping Suite (`test/benchmark/`)
The benchmark harness validates all calibrated SLIs/SLOs across isolated GPU tiers:

1. **Single-Node Ingestion Throughput (`test/benchmark/test_single_node_embedding_sweep.py`)**:
   - Concurrency and batch size sweeps ($N \in [1, 2, 4, 8, 16, 32, 64, 128, 256]$) with synthetic 256–512 token chunk payloads.
   - Verifies sustained **2,500 to 3,500 chunks/sec** per NVIDIA L4/A10G instance at batch size 256 with 50 ms queue delay ceiling.
   - Evaluates throughput and serialization efficiency delta between pod-local zero-copy `/dev/shm` IPC and remote HTTP/2 gRPC.
   - Verifies active memory remains bounded below 85% ($< 20.4\text{ GiB}$) with zero memory leaks across sustained batch iterations.

2. **Multi-Node Cluster Scaling (`test/benchmark/test_multi_node_ingestion_scaling.py`)**:
   - Simulates horizontal cluster scaling across **12 to 16 preemptible spot GPU nodes**.
   - Validates linear scale-out achieving **$\ge 40{,}000\text{ chunks/sec}$** aggregate cluster throughput without queue stalls, cross-worker contention, or host memory degradation.
   - Verifies round-robin connection pool distribution across multi-instance Triton endpoints in `TritonGRPCChannelPool`.

3. **Online Latency & Reranker Boundary Profiling (`test/benchmark/test_online_latency_profiler.py`)**:
   - Executes mandatory 100-request warm-up sweep prior to latency capture to eliminate JIT compilation and CUDA allocator warmup artifacts.
   - **Online Embedding Pool (Pool A)**: Evaluates single-sentence queries at **1,500 QPS** (batch size 1, queue delay $\le 2\text{ ms}$), asserting standalone latency **P99 $\le 10\text{ ms}$**.
   - **Online Reranker Pool (Pool B)**: Evaluates candidate pairs (10 to 30 candidates pruned to Top-15, queue delay $\le 4\text{ ms}$), asserting latency **P99 $\le 30\text{ ms}$**.
   - **Cumulative Search Budget**: Asserts dense embedding plus cross-encoder reranking completes within the **$\le 40\text{ ms}$ P99** cumulative search budget.

### B. Chaos & Resilience Testing Protocols (`test/chaos/`)

1. **Queue Saturation & Native Backpressure (`test/chaos/test_queue_saturation_backpressure.py`)**:
   - Saturates dynamic batching queues beyond capacity, confirming Triton returns `UNAVAILABLE` (gRPC status 14, "Request timeout expired") or `RESOURCE_EXHAUSTED` (gRPC status 8).
   - Verifies client circuit breaker trips to `OPEN`, immediately returning deterministic un-reranked RRF scores ($RRF(d) = 1.0 / (60.0 + rank(d))$) without cascading failures.
   - Validates complete 3-state transition: `CLOSED` $\rightarrow$ `OPEN` $\rightarrow$ `HALF_OPEN` $\rightarrow$ `CLOSED`.
   - Confirms secured logging under failure conditions: zero query or passage payloads emitted to error logs.

2. **Synthetic CUDA Stalls & GPU Latency Spike Injection (`test/chaos/test_cuda_stall_injection.py`)**:
   - Simulates GPU kernel stalls exceeding deadlines (15 ms embedding, 35 ms reranker).
   - Confirms deadline enforcement (`DEADLINE_EXCEEDED`), rapid gRPC channel recycling, zero leaked channel slots, and transparent RRF fallback.

3. **Hardware Isolation Validation (`test/chaos/test_hardware_isolation.py`)**:
   - Concurrent workload simulation: floods batch ingestion pipeline at peak capacity (batch size 256, spot nodes) while concurrently running sustained 1,500 QPS online search queries on dedicated on-demand nodes.
   - Asserts **zero latency degradation** on online retrieval P99 due to physical node group decoupling (ADR-001, ADR-002).
   - Validates Kubernetes scheduling constraints (`nodeSelector`, `nodeAffinity`, `tolerations`) preventing cross-pool co-location.

4. **Spot Node Preemption & Eviction Resilience (`test/chaos/test_spot_eviction.py`)**:
   - Simulates Kubernetes spot instance termination signals (`SIGTERM`, node drain).
   - Confirms batch ingestion client retry logic, connection failover, task re-queueing without chunk loss, and clean `/dev/shm` resource reclamation preventing orphaned memory segments.

### C. Calibrated KEDA Autoscaling Triggers (`deployments/helm/`)
Autoscaling rules evaluate **average queue delay per request** rather than raw counter rates:

$$\text{Avg Queue Delay } (\mu\text{s}) = \frac{\sum(\text{rate}(\text{nv\_inference\_queue\_duration\_us}[\text{1m}]))}{\text{clamp\_min}(\sum(\text{rate}(\text{nv\_inference\_count}[\text{1m}])), 0.001)}$$

- **Online Embedding Pool (`values-online-embed.yaml`)**:
  - Min/Max Replicas: 4–8.
  - Scale Trigger: Average queue delay $\ge 3000\mu\text{s}$ ($3\text{ ms}$).
  - Warning Alert: Delay $\ge 4000\mu\text{s}$ ($4\text{ ms}$) sustained for $> 1\text{ min}$.
- **Online Reranker Pool (`values-online-rerank.yaml`)**:
  - Min/Max Replicas: 4–12.
  - Scale Trigger: Average queue delay $\ge 5000\mu\text{s}$ ($5\text{ ms}$).
  - Warning Alert: Delay $\ge 8000\mu\text{s}$ ($8\text{ ms}$) sustained for $> 1\text{ min}$.
- **Batch Ingestion Pool (`values-batch-ingestion.yaml`)**:
  - Min/Max Replicas: 12–16.
  - Scale Trigger: Kafka consumer group lag threshold $10{,}000$ chunks.
  - Warning Alert: Backlog $> 500{,}000$ chunks at 16-node spot ceiling.

### D. GitOps ArgoCD Multi-Pool Continuous Delivery (`deployments/argocd/`)

1. **Declarative ApplicationSet (`deployments/argocd/applicationset.yaml`)**:
   - Uses list generators targeting `deployments/helm` with dedicated value overlays:
     - `rag-inference-online-embedding` $\rightarrow$ `values-online-embed.yaml` (on-demand nodes)
     - `rag-inference-online-reranker` $\rightarrow$ `values-online-rerank.yaml` (on-demand nodes)
     - `rag-inference-batch-ingestion` $\rightarrow$ `values-batch-ingestion.yaml` (spot nodes)
   - Configures automated sync (`prune: true`, `selfHeal: true`), namespace creation (`CreateNamespace=true`), and health check status tracking for Triton probes (`/healthz`, `/ready`) and KEDA `ScaledObject` status.

2. **Standalone Application CRDs**:
   - `application-online-embed.yaml`, `application-online-rerank.yaml`, `application-batch-ingestion.yaml` for environment-specific deployments where ApplicationSet controller is not installed.

3. **Canary Progressive Rollout & AnalysisTemplate (`deployments/argocd/`)**:
   - `rollout-canary.yaml`: Manages zero-downtime progressive rollouts of TensorRT FP16 engine updates (`model.plan`) with traffic shifting (20% $\rightarrow$ 50% $\rightarrow$ 100%) and automatic rollback upon error rate or latency breaches.
   - `analysistemplate.yaml`: Defines `triton-vram-and-latency-check` verifying Prometheus metrics (`nv_inference_request_success` and `nv_inference_queue_duration_us`) during progressive rollout steps.

---

## 10. Environment Variables & Operational Configuration Reference

The microservice reads configuration parameters via Pydantic `ServingSettings` with the `RAG_SERVING_` prefix, alongside container supervisor and test runner environment variables:

| Variable Name | Type | Default Value | Description & Constraints |
| :--- | :--- | :--- | :--- |
| `RAG_SERVING_POOL_TYPE` | `string` | `online_embedding` | Hardware compute pool identity: `online_embedding`, `online_reranker`, `batch_ingestion`. |
| `RAG_SERVING_SERVER_HOST` | `string` | `0.0.0.0` | Bind host interface for out-of-band operational ASGI server. |
| `RAG_SERVING_SERVER_PORT` | `integer` | `8080` | HTTP port for operational server hosting `/healthz`, `/ready`, `/metrics`, `/docs`, `/redoc`, `/api/docs`, `/openapi.json`. |
| `RAG_SERVING_TRITON_HTTP_URL` | `string` | `http://127.0.0.1:8000` | Internal pod-local Triton HTTP endpoint for KServe v2 health and readiness probes. |
| `RAG_SERVING_TRITON_GRPC_URL` | `string` | `127.0.0.1:8001` | Internal pod-local Triton gRPC endpoint for KServe v2 binary tensor inference. |
| `RAG_SERVING_ONLINE_EMBED_GRPC_URL` | `string` | `dns:///rag-inference-online-embedding.rag-inference-serving.svc.cluster.local:8001` | Upstream channel target for Online Embedding Pool (Pool A: NVIDIA L4/A10G On-Demand) with client-side round-robin load balancing. |
| `RAG_SERVING_ONLINE_RERANK_GRPC_URL` | `string` | `dns:///rag-inference-online-reranker.rag-inference-serving.svc.cluster.local:8001` | Upstream channel target for Online Reranker Pool (Pool B: NVIDIA L4/A10G On-Demand) with client-side round-robin load balancing. |
| `RAG_SERVING_BATCH_INGEST_GRPC_URL` | `string` | `dns:///rag-inference-batch-ingestion.rag-inference-serving.svc.cluster.local:8001` | Upstream channel target for Batch Ingestion Pool (Preemptible Spot Nodes) with client-side round-robin load balancing. |
| `RAG_SERVING_MAX_QUEUE_DELAY_MICROSECONDS` | `integer` | Pool-calibrated | Triton queue delay timeout in $\mu s$ (`2000` for online embedding, `4000` for online reranker, `50000` for batch ingestion). |
| `RAG_SERVING_MAX_BATCH_SIZE` | `integer` | Pool-calibrated | Dynamic batch size ceiling (`32` for online embedding/reranker, `256` for batch ingestion). |
| `RAG_SERVING_VRAM_CAPACITY_BYTES` | `integer` | `25769803776` | Total GPU VRAM capacity in bytes (24 GiB NVIDIA L4/A10G memory layout). |
| `RAG_SERVING_MAX_STATIC_VRAM_BYTES` | `integer` | `21474836480` | Ceiling for aggregate static model allocations in bytes ($\le 20\text{ GiB}$, ~83.33%). |
| `RAG_SERVING_VRAM_MAX_UTILIZATION_RATIO` | `float` | `0.85` | Active runtime VRAM threshold triggering HTTP 503 traffic shedding ($>85\%$ or $>20.4\text{ GiB}$). |
| `RAG_SERVING_MODELS_EXPECTED` | `JSON array` | Pool-calibrated | List of initialized model engine names required for `/ready` probe to report ready status. |
| `RAG_SERVING_SHM_BASE_PATH` | `string` | `/dev/shm` | Mount path for pod-scoped POSIX shared memory volume (`emptyDir` RAM mount). |
| `RAG_SERVING_GRPC_TLS_ENABLED` | `boolean` | `false` | Enable TLS encryption on outbound gRPC client inference channels. |
| `RAG_SERVING_GRPC_CA_CERT_PATH` | `string` | `null` | Filesystem path to root CA certificate verifying Triton server TLS certificates. |
| `RAG_SERVING_GRPC_CLIENT_CERT_PATH` | `string` | `null` | Filesystem path to client public certificate for mutual TLS (mTLS) authentication. |
| `RAG_SERVING_GRPC_CLIENT_KEY_PATH` | `string` | `null` | Filesystem path to client private key for mutual TLS (mTLS) authentication. |
| `RAG_SERVING_GRPC_SERVER_NAME_OVERRIDE` | `string` | `null` | Optional SNI hostname override for TLS certificate validation. |
| `RAG_SERVING_API_KEY` | `string` | `null` | Optional bearer token / API key securing operational endpoints (`/metrics`). |
| `RAG_SERVING_LOG_LEVEL` | `string` | `INFO` | Structured logging verbosity level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `PROMETHEUS_SERVER_ADDRESS` | `string` | `http://prometheus-k8s.monitoring.svc:9090` | Prometheus query URL for KEDA `triton_queue_duration_microseconds` autoscaling trigger. |
| `PROMETHEUS_BEARER_TOKEN` | `string` | `null` | Optional bearer authentication token for secured Prometheus queries. |
| `KAFKA_BOOTSTRAP_SERVERS` | `string` | `kafka.event-streaming.svc:9092` | Kafka broker endpoints for Batch Ingestion backlog autoscaling trigger. |
| `KAFKA_CONSUMER_GROUP` | `string` | `rag-batch-embedder-group` | Kafka consumer group name monitored by KEDA `ScaledObject`. |
| `KAFKA_TOPIC` | `string` | `rag.ingestion.chunks` | Kafka ingestion topic name delivering chunk embeddings. |
| `KAFKA_SASL_USERNAME` | `string` | `null` | Kafka SASL username for consumer group lag monitoring. |
| `KAFKA_SASL_PASSWORD` | `string` | `null` | Kafka SASL password for consumer group lag monitoring. |
| `KAFKA_SASL_MECHANISM` | `string` | `SCRAM-SHA-512` | Kafka SASL mechanism (`SCRAM-SHA-512` or `PLAIN`). |
| `KAFKA_SECURITY_PROTOCOL` | `string` | `SASL_SSL` | Kafka transport protocol (`SASL_SSL` or `PLAINTEXT`). |
| `KAFKA_SSL_CA_LOCATION` | `string` | `/etc/ssl/certs/kafka-ca.crt` | Path to Kafka broker CA certificate for TLS verification. |
| `HF_TOKEN` | `string` | `null` | Hugging Face access token for gated model downloads during export (`scripts/export_onnx.py`). |
| `AWS_ACCESS_KEY_ID` | `string` | `null` | S3 / MinIO access key ID for pulling serialized TensorRT engines (`model.plan`). |
| `AWS_SECRET_ACCESS_KEY` | `string` | `null` | S3 / MinIO secret access key for pulling serialized TensorRT engines. |
| `AWS_DEFAULT_REGION` | `string` | `us-east-1` | AWS region identifier for model registry storage. |
| `AWS_ENDPOINT_URL` | `string` | `null` | Custom S3 / MinIO / Ceph endpoint URL for cloud-native model engine caches. |
| `MODEL_REGISTRY_STORAGE_URI` | `string` | `s3://rag-model-cache/tensorrt-engines/` | Artifact storage URI containing precompiled TensorRT model engines. |
| `MODEL_REPOSITORY` | `string` | `/models` | Directory path containing compiled Triton model repositories. |
| `TRITON_ARGS` | `string` | `--model-repository=/models --strict-model-config=true --log-verbose=0` | Process supervisor startup flags for `tritonserver`. |
| `PYTHONUNBUFFERED` | `integer` | `1` | Forces unbuffered stdout/stderr output for real-time JSON log shipping. |
| `PYTHONDONTWRITEBYTECODE` | `integer` | `1` | Prevents writing `.pyc` files inside containers. |
| `LIVE_TRITON` | `integer` | `0` | Controls live vs. dry-run Triton execution during automated benchmark sweeps (`0` = mock, `1` = live). |
| `TRITON_ONLINE_GRPC_TARGET` | `string` | `localhost:8001` | Target endpoint for online latency profiler benchmark suite. |
| `TRITON_BATCH_GRPC_TARGET` | `string` | `localhost:8001` | Target endpoint for single-node batch sweep benchmark suite. |
| `TRITON_CLUSTER_ENDPOINTS` | `string` | `localhost:8001` | Comma-separated cluster endpoints for multi-node ingestion scaling benchmarks. |
