# Architecture Context - RAG Inference Serving (Embedding & Reranker Pools)

## System Stack & Specifications

| Layer | Technology | Role & Operational Characteristics |
| :--- | :--- | :--- |
| **Inference Engine** | NVIDIA Triton Inference Server 24.x | High-throughput C++ inference execution and dynamic batching |
| **Model Acceleration** | TensorRT Execution Provider (FP16) | GPU kernel optimization, FP16 `.plan` engine execution |
| **Tokenizer Runtime** | Native in-engine C++ HuggingFace Tokenizer | Zero-GIL tokenization pipeline compiled inside Triton (`text_tokenizer`) |
| **GPU Hardware** | NVIDIA L4 / A10G (24 GB VRAM) | Dedicated on-demand (online) and preemptible spot (batch) GPU nodes |
| **Model Storage** | Read-Only NVMe mounts via K8s CSI | Immutable persistent model weights across pod lifecycles |
| **Operational Server**| FastAPI + Uvicorn (ASGI) | OpenAPI 3.1 endpoints (`/docs`, `/api/docs`), `/healthz`, `/ready`, `/metrics` |
| **Data Transport** | gRPC over HTTP/2 + POSIX Shared Memory | Low-latency RPC for search; zero-copy `/dev/shm` IPC for batch ingestion |
| **Autoscaling** | KEDA (Kubernetes Event-driven Autoscaling) | Online: `triton_queue_duration_microseconds`; Batch: Kafka task backlog |
| **Fallback Strategy** | Deterministic RRF Bypass | Immediate fallback to Reciprocal Rank Fusion on reranker saturation |

---

## Production-Grade 5-Layer Backend Architecture

Aligned with *Production Grade Backend Concepts*:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 1: API & Routing                                                 │
│ Direct gRPC (HTTP/2) | Operational FastAPI (OpenAPI 3.1 /docs, /ready) │
│ KServe v2 Protocol   | Input Shape Validation & Kernel Truncation       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Layer 2: Data Persistence & Memory Models                              │
│ Read-Only NVMe Model Storage (model.plan) | POSIX /dev/shm (8 GiB RAM) │
│ Zero Permanent Database State             | Contiguous Tensor Buffers  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Layer 3: Architecture & Scale                                          │
│ Hardware-Isolated Pools: Pool A (Embed), Pool B (Rerank), Batch Pool   │
│ KEDA Metric Autoscaling (Queue Duration & Kafka Lag)                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Layer 4: Security & Identity                                           │
│ Non-Root Execution | Secured Logging (Redacted Queries & Chunks)       │
│ Secrets Decoupling | VRAM Allocation Ceilings (<=83% / 20 GiB)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Layer 5: Observability & Quality                                       │
│ Prometheus Triton Metrics | Native Queue Backpressure & RRF Fallback   │
│ Health/Readiness Probes   | Chaos & Spot Eviction Resilience           │
└────────────────────────────────────────────────────────────────────────┘
```

### Layer 1: API & Routing
- **Entry Points**:
  - Triton Primary Ingress: gRPC port `8001` (KServe v2 Protobuf) and HTTP port `8000`.
  - Operational Probe & Doc Server: Dedicated FastAPI ASGI app on port `8080`.
- **Endpoints & Contracts**:
  - `POST /v2/models/{model_name}/infer`: Binary tensor inference over HTTP/gRPC.
  - `GET /v2/models/{model_name}/ready`: Model readiness check.
  - `GET /v2/health/ready` & `GET /v2/health/live`: Native Triton lifecycle probes.
  - Operational endpoints: `/docs` (Swagger UI), `/redoc` (ReDoc), `/api/docs` & `/openapi.json` (OpenAPI 3.1 JSON), `/healthz`, `/ready`, `/metrics`.
- **Input Validation & Sanitization**:
  - Explicit tensor dimension validation against dynamic shape profiles `[min, opt, max]`.
  - Deterministic padding and truncation handled at the C++ kernel boundary ($> 512$ tokens for embeddings, $> 2048$ tokens for cross-encoders) without Python GIL stalls or memory leaks.
- **Backpressure & Throttling**:
  - Triton native queue duration limits (`max_queue_delay_microseconds`).
  - Saturation triggers immediate `RESOURCE_EXHAUSTED` / HTTP 429 status codes, avoiding queue cascading.
- **Idempotency**:
  - Model inference operations are purely deterministic mathematical vector transformations; identical input payloads produce identical tensor outputs.

### Layer 2: Data Persistence & Memory Models
- **Zero Persistent DB State**: Stateless compute service holding no database tables or mutable transactional records.
- **Model Weights Storage**: Read-only NVMe mounts via Kubernetes CSI drivers for persistent model availability across pod restarts (`/models/online_embedding_repository`, etc.).
- **POSIX Shared Memory (`/dev/shm`)**:
  - Configured at 8 GiB via Kubernetes `emptyDir` RAM volume mounts.
  - High-density chunk arrays mapped directly into shared memory regions, allowing zero-copy pointer registration into Triton without serialization penalties.
  - Unmap and release lifecycle routines prevent orphaned memory segments.

### Layer 3: Architecture & Scale
- **Decoupled Cluster Topologies**:
  - **Online Embedding Pool (Pool A)**: Dedicated NVIDIA L4/A10G, on-demand nodes. Dynamic batching: $\text{max\_batch\_size}=32$, $\text{max\_queue\_delay}=2000\mu\text{s}$ ($2\text{ ms}$).
  - **Online Reranker Pool (Pool B)**: Dedicated NVIDIA L4/A10G, on-demand nodes. Dynamic batching: $\text{max\_batch\_size}=32$, $\text{max\_queue\_delay}=4000\mu\text{s}$ ($4\text{ ms}$).
  - **Batch Ingestion Pool**: Preemptible spot node groups. Dynamic batching: $\text{max\_batch\_size}=256$, $\text{max\_queue\_delay}=50000\mu\text{s}$ ($50\text{ ms}$).
- **Workload Separation**: Isolating cross-encoder matrix multiplications eliminates Streaming Multiprocessor (SM) starvation for embedding queries.
- **Autoscaling Drivers via KEDA**:
  - Online pools: Scales on `triton_queue_duration_microseconds` ($\ge 3\text{ ms}$ for embeddings, $\ge 5\text{ ms}$ for rerankers).
  - Batch ingestion: Scales on Kafka task backlog ($> 500{,}000$ chunks) up to 16 spot GPU nodes.

### Layer 4: Security & Identity
- **Secured Logging Invariant**:
  - Strictly redact user query text, raw document chunks, customer secrets, bearer tokens, and presigned URLs from all logs.
  - Emit machine-parseable JSON logs containing only metadata IDs (`request_id`, `model_name`, `pool_type`, `batch_size`, `latency_ms`).
- **Memory Boundary Safety**:
  - Pods never mount host `/dev/shm` across cluster nodes; shared memory is strictly pod-local `emptyDir` RAM volumes to prevent `SIGBUS` errors.
  - Aggregate engine allocations per GPU capped at $\le 20\text{ GiB}$ ($83\%$ of $24\text{ GB}$) to eliminate CUDA OOM panics.
- **Least-Privilege Execution**: Containers run as non-root (UID 1000) with dropped Linux capabilities and read-only filesystem root.

### Layer 5: Observability, Quality & Probes
- **Prometheus Scrape Endpoints**:
  - Triton native `/v2/metrics` (port `8002`) and operational `/metrics` (port `8080`).
  - Metrics tracked: `triton_inference_request_duration_us`, `triton_inference_queue_duration_us`, `triton_gpu_memory_used_bytes`, `triton_gpu_utilization`.
- **Alerting Thresholds**:
  - Online Embedding Queue Lag: Warning when queue duration $> 4\text{ ms}$ for $> 1\text{ min}$.
  - Online Reranker Queue Lag: Warning when queue duration $> 8\text{ ms}$ for $> 1\text{ min}$.
  - GPU VRAM Saturation: Critical when active memory exceeds $85\%$ ($> 20.4\text{ GiB}$) on any node.
- **Graceful Error Handling & Fallback**:
  - If cross-encoder queue is saturated, upstream `rag-search-orchestrator` falls back instantly to un-reranked RRF scores without breaking user search experience.

---

## Folder Structure

```
rag-inference-serving/
├── config/
│   └── triton/
│       ├── online_embedding_repository/
│       │   ├── text_tokenizer/
│       │   │   ├── 1/
│       │   │   │   └── model.onnx            # C++ tokenization runtime
│       │   │   └── config.pbtxt
│       │   └── dense_embedder_trt/
│       │       ├── 1/
│       │       │   └── model.plan            # FP16 TensorRT engine
│       │       └── config.pbtxt              # Dynamic batching: max 32, delay 2ms
│       ├── online_reranker_repository/
│       │   ├── text_tokenizer/
│       │   │   ├── 1/
│       │   │   │   └── model.onnx
│       │   │   └── config.pbtxt
│       │   └── cross_encoder_trt/
│       │       ├── 1/
│       │       │   └── model.plan            # Isolated FP16 engine
│       │       └── config.pbtxt              # Dynamic batching: max 32, delay 4ms
│       └── batch_ingestion_repository/
│           ├── text_tokenizer/
│           │   ├── 1/
│           │   │   └── model.onnx
│           │   └── config.pbtxt
│           └── batch_embedder_trt/
│               ├── 1/
│               │   └── model.plan            # Throughput-optimized engine
│               └── config.pbtxt              # Dynamic batching: max 256, delay 50ms
├── api/
│   ├── proto/v1/
│   │   └── triton_stubs.proto                # Upstream interface models
│   └── openapi/
│       └── openapi.json                      # Machine-readable OpenAPI 3.1 specification
├── internal/
│   ├── config/settings.py                    # Server configuration & threshold constants
│   ├── server/probe_server.py                # FastAPI operational & OpenAPI documentation server
│   ├── logging/secured_logger.py             # Redacted structured JSON logging engine
│   └── shm/posix_shm.py                      # POSIX shared memory buffer wrapper
├── deployments/
│   ├── docker/
│   │   └── Dockerfile.triton                 # Base image with TensorRT execution runtimes
│   └── helm/
│       ├── Chart.yaml
│       ├── values-online-embed.yaml          # On-demand nodes, queue-based autoscaling
│       ├── values-online-rerank.yaml         # Dedicated reranker deployment
│       └── values-batch-ingestion.yaml       # Spot nodes, lag-based autoscaling
├── scripts/
│   ├── export_onnx.py                        # HuggingFace transformer to ONNX export
│   └── build_trt_engine.sh                   # Trtexec compilation script for FP16 profiles
├── test/
│   ├── benchmark/                            # Single-node & multi-node load sweeps
│   ├── integration/                          # Direct gRPC client validation suites
│   └── chaos/                                # Queue saturation and spot eviction validation
├── Makefile
└── README.md
```

---

## Architectural Decision Records (ADRs)

- **ADR-001: Pool Decoupling (Online vs. Batch)**: Isolates online retrieval from batch ingestion workloads onto separate GPU node groups to guarantee query latency SLOs.
- **ADR-002: Dedicated Cross-Encoder Node Pools**: Isolates compute-intensive cross-encoder rerankers from dense query embedders to avoid SM contention and thread stalls.
- **ADR-003: Standard FP16 Model Compilation**: Mandates FP16 precision across all inference targets in initial releases, deferring INT8 quantization to avoid precision loss.
- **ADR-004: In-Engine C++ Tokenization**: Runs tokenization directly within the Triton engine pipeline using C++ runtimes, removing Python interpreter bottlenecks from the critical path.
- **ADR-005: Elimination of Proxy Data Sidecars**: Connects callers directly to Triton via gRPC and shared memory, avoiding custom proxy processes for high-volume data movement.
