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
        "responses": {
          "200": { "description": "Prometheus text exposition format" }
        }
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
  rpc RegisterSharedMemory(RegisterSharedMemoryRequest) returns (RegisterSharedMemoryResponse);
  rpc UnregisterSharedMemory(UnregisterSharedMemoryRequest) returns (UnregisterSharedMemoryResponse);
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
}
```

---

## 3. Dynamic Batching Model Configurations (`config.pbtxt`)

### A. Online Dense Embedding Repository (`online_embedding_repository`)

**Model**: `dense_embedder_trt`
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

**Model**: `cross_encoder_trt`
- **Engine**: TensorRT FP16 (`model.plan`)
- **Max Batch Size**: `32`
- **Dynamic Batching**: `max_queue_delay_microseconds: 4000` (4 ms)
- **Target Workload**: Top-10 to top-30 candidate pairs
- **SLO Target**: P99 $\le 30\text{ ms}$ under sustained load.
- **Queue Saturation Fallback**: If queue delay exceeds limit, return `RESOURCE_EXHAUSTED` to trigger immediate orchestrator bypass to un-reranked RRF scores.

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

**Model**: `batch_embedder_trt`
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

## 4. POSIX Shared Memory IPC Protocol (`/dev/shm`)

For high-density batch ingestion, `rag-chunk-lifecycle-service` and ingestion workers utilize local POSIX shared memory buffers to bypass network serialization:

1. **Mount Specification**: Kubernetes `emptyDir` RAM volume mounted at `/dev/shm`, sized to `8Gi`.
2. **Buffer Allocation**:
   ```c
   int shm_fd = shm_open("/triton_batch_shm_region", O_CREAT | O_RDWR, 0660);
   ftruncate(shm_fd, region_size_bytes);
   void* ptr = mmap(NULL, region_size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
   ```
3. **Registration with Triton**:
   The caller registers the shared memory region with Triton via gRPC:
   - Call `RegisterSharedMemoryRequest`: specifies `name`, `key`, `offset`, `byte_size`.
4. **Execution**:
   - `ModelInferRequest` references the registered region name instead of packing raw bytes in `raw_input_contents`.
   - Triton reads input tensors directly from RAM and writes output vectors directly back to the output `/dev/shm` region.
5. **Cleanup Lifecycle**:
   - Explicit `UnregisterSharedMemory` followed by `munmap` and `shm_unlink` to prevent orphaned memory segments.

---

## 5. Error Codes & Upstream Fallback Policies

| Status Code | Reason | Upstream Orchestrator Action |
| :--- | :--- | :--- |
| `OK` (0) | Inference succeeded | Consume embeddings or cross-encoder logits |
| `RESOURCE_EXHAUSTED` (8) | Dynamic batch queue saturated beyond timeout | **Immediate fallback**: Bypass cross-encoder reranker and return un-reranked Reciprocal Rank Fusion (RRF) scores directly |
| `DEADLINE_EXCEEDED` (4) | Request exceeded client timeout budget | Abort request; log warning; fall back to un-reranked RRF scores |
| `INVALID_ARGUMENT` (3) | Token length exceeds maximum context window | Deterministically truncate at C++ kernel boundary |
| `UNAVAILABLE` (14) | Triton pod restarting or spot eviction | Route request to healthy AZ replica or retry with backoff |
