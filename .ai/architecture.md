# Architecture Context - RAG Inference Serving (Embedding & Reranker Pools)

## System Stack & Specifications

| Layer | Technology | Role & Operational Characteristics |
| :--- | :--- | :--- |
| **Inference Engine** | NVIDIA Triton Inference Server 24.x | High-throughput C++ inference execution, dynamic batching, and ensemble pipelines |
| **Model Acceleration** | TensorRT Execution Provider (FP16) | GPU kernel optimization, FP16 `.plan` engine execution |
| **Tokenizer Runtime** | Native in-engine C++ HuggingFace Tokenizer | Zero-GIL tokenization pipeline compiled inside Triton (`text_tokenizer`) |
| **Pipeline Composition**| Triton Ensemble Scheduling (`ensemble_scheduler`) | Chaining `text_tokenizer` $\rightarrow$ TensorRT engine into unified zero-GIL endpoints |
| **GPU Hardware** | NVIDIA L4 / A10G (24 GB VRAM) | Dedicated on-demand (online) and preemptible spot (batch) GPU nodes |
| **Model Storage** | Read-Only NVMe mounts via K8s CSI | Immutable persistent model weights across pod lifecycles |
| **Operational Server**| FastAPI + Uvicorn (ASGI) | OpenAPI 3.1 endpoints (`/docs`, `/api/docs`), `/healthz`, `/ready`, `/metrics` |
| **Data Transport** | gRPC over HTTP/2 + POSIX Shared Memory | Direct gRPC for search/remote ingestion; zero-copy pod-local `/dev/shm` IPC |
| **Sparse Retrieval** | Upstream BM25 (Phase 1) / SPLADE (Phase 2) | Phase 1 sparse search handled in orchestrator; learned sparse models in Phase 2 |
| **Autoscaling** | KEDA (Kubernetes Event-driven Autoscaling) | Online: `triton_queue_duration_microseconds`; Batch: Kafka task backlog |
| **Fallback Strategy** | Deterministic RRF Bypass | Immediate fallback to Reciprocal Rank Fusion on reranker saturation |

---

## Production-Grade 5-Layer Backend Architecture

Aligned with *Production Grade Backend Concepts*:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 1: API & Routing                                                 │
│ Direct gRPC (HTTP/2) | Operational FastAPI (OpenAPI 3.1 /docs, /ready) │
│ KServe v2 Protocol   | Ensemble Pipelines & Dynamic Shape Bound Gates  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ Layer 2: Data Persistence & Memory Models                              │
│ Read-Only NVMe Model Storage (model.plan) | Pod-Local /dev/shm (8 GiB) │
│ Zero Permanent Database State             | Remote Batched gRPC Stream │
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
│ Secrets Decoupling | Static VRAM Cap (<=20 GiB) & Runtime Alert (>85%) │
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
  - Primary Ensemble Endpoints:
    - `POST /v2/models/online_embedder_ensemble/infer`: End-to-end string query embedding (`raw_text` $\rightarrow$ `embeddings`).
    - `POST /v2/models/online_reranker_ensemble/infer`: End-to-end candidate pair reranking (`query` + `passage` $\rightarrow$ `logits`).
    - `POST /v2/models/batch_embedder_ensemble/infer`: End-to-end batch ingestion embedding (`raw_text` $\rightarrow$ `embeddings`).
  - Underlying Component Models:
    - `POST /v2/models/{model_name}/infer`: Direct component model inference (e.g. `dense_embedder_trt`, `cross_encoder_trt`).
    - `GET /v2/models/{model_name}/ready`: Model readiness check.
  - Native Triton lifecycle probes: `GET /v2/health/ready` & `GET /v2/health/live`.
  - Operational endpoints (port `8080`): `/docs` (Swagger UI), `/redoc` (ReDoc), `/api/docs` & `/openapi.json` (OpenAPI 3.1 JSON), `/healthz`, `/ready`, `/metrics`.
- **Dynamic Shape Profiles & Kernel Bounds**:
  - Validated dynamic shape profiles `[min, opt, max]`:
    - Batch dimension: $1 \le \text{batch} \le 32$ (online) and $1 \le \text{batch} \le 256$ (ingestion).
    - Sequence length dimension: $1 \le \text{seq\_len} \le 512$ without unaligned memory access faults.
  - Deterministic truncation & padding handled at the in-engine C++ kernel boundary:
    - Queries & document chunks exceeding $512$ tokens for embeddings are deterministically truncated.
    - Candidate pairs exceeding $2048$ tokens for cross-encoders are deterministically truncated.
    - Eliminates Python GIL stalls, unhandled exceptions, and memory leaks.
- **Backpressure & Throttling**:
  - Dynamic batching wait times are bounded by `max_queue_delay_microseconds` to limit micro-batch accumulation delay prior to execution without autonomously rejecting requests.
  - Saturated queues enforce an explicit overload policy (`default_queue_policy` with `timeout_action: REJECT` and `max_queue_size`), returning `RESOURCE_EXHAUSTED` (gRPC status code 8 or HTTP 503 natively, translated to HTTP 429 when routed through API ingress gateways) under queue saturation to prevent cascading failure and trigger upstream RRF fallback.
- **Idempotency**:
  - Model inference operations are purely deterministic mathematical transformations; identical input payloads yield identical tensor representations.

### Layer 2: Data Persistence & Memory Models
- **Zero Persistent DB State**: Stateless compute microservice holding no database tables or mutable transactional records.
- **Model Weights Storage**: Read-only NVMe mounts via Kubernetes CSI drivers for persistent model availability across pod restarts (`/models/online_embedding_repository`, etc.).
- **Transport & Memory Boundary Safety**:
  - **Pod-Scoped POSIX Shared Memory (`/dev/shm`)**:
    - Mounted as an 8 GiB Kubernetes `emptyDir` RAM volume.
    - Strictly scoped to local container IPC within the **same Kubernetes Pod** (co-located ingestion worker + Triton instance). Cross-pod or cross-node `/dev/shm` sharing is strictly prohibited to eliminate `SIGBUS` faults.
    - Clean `munmap` and `shm_unlink` lifecycle routines prevent orphaned memory segments.
  - **Remote Batch Ingestion Transport**:
    - Remote ingestion workers (`rag-chunk-lifecycle-service` running on CPU-optimized nodes) transmit batched chunks over **batched gRPC over HTTP/2** directly to Triton.

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
  - Emit machine-parseable JSON logs containing only metadata IDs (`request_id`, `model_name`, `pool_type`, `batch_size`, `queue_latency_ms`, `compute_latency_ms`, `status`).
- **Memory Safety & VRAM Boundaries**:
  - **Static Engine Allocation Ceiling**: Total serialized TensorRT engines and static scratch allocations capped at $\le 20\text{ GiB}$ ($83.3\%$ of $24\text{ GB}$ on L4/A10G).
  - **Dynamic Headroom**: The remaining $\approx 4\text{ GiB}$ ($16.7\%$) is strictly reserved for CUDA runtime contexts and dynamic batching activation tensors.
- **Least-Privilege Execution**: Containers run as non-root (UID 1000) with dropped Linux capabilities and read-only filesystem root.

### Layer 5: Observability, Quality & Probes
- **Prometheus Scrape Endpoints**:
  - Triton native `/v2/metrics` (port `8002`) and operational `/metrics` (port `8080`).
  - Metrics tracked: `triton_inference_request_duration_us`, `triton_inference_queue_duration_us`, `triton_gpu_memory_used_bytes`, `triton_gpu_utilization`.
- **Alerting Thresholds**:
  - Online Embedding Queue Lag: Warning when queue duration $> 4\text{ ms}$ for $> 1\text{ min}$.
  - Online Reranker Queue Lag: Warning when queue duration $> 8\text{ ms}$ for $> 1\text{ min}$ (triggers pool autoscaling).
  - GPU VRAM Saturation: Critical when active runtime memory exceeds $85\%$ ($> 20.4\text{ GiB}$) on any node.
  - Ingestion Scale Threshold Alert: Warning when batch task backlog exceeds $500{,}000$ chunks while cluster pool is capped at maximum capacity (16 spot nodes).
- **Graceful Error Handling & Fallback**:
  - If cross-encoder queue is saturated, upstream `rag-search-orchestrator` catches `RESOURCE_EXHAUSTED` and falls back instantly to un-reranked RRF scores without breaking user search experience.

---

## Folder Structure

```
rag-inference-serving/
├── config/
│   └── triton/
│       ├── online_embedding_repository/
│       │   ├── text_tokenizer/
│       │   │   ├── 1/
│       │   │   │   └── model.onnx            # Configured ONNX Runtime tokenizer backend
│       │   │   └── config.pbtxt
│       │   ├── dense_embedder_trt/
│       │   │   ├── 1/
│       │   │   │   └── model.plan            # FP16 TensorRT engine
│       │   │   └── config.pbtxt              # Dynamic batching: max 32, delay 2ms
│       │   └── online_embedder_ensemble/
│       │       └── config.pbtxt              # Chains ONNX Runtime text_tokenizer -> dense_embedder_trt
│       ├── online_reranker_repository/
│       │   ├── text_tokenizer/
│       │   │   ├── 1/
│       │   │   │   └── model.onnx            # Candidate pair ONNX Runtime tokenizer backend
│       │   │   └── config.pbtxt
│       │   ├── cross_encoder_trt/
│       │   │   ├── 1/
│       │   │   │   └── model.plan            # Isolated FP16 engine
│       │   │   └── config.pbtxt              # Dynamic batching: max 32, delay 4ms, queue rejection
│       │   └── online_reranker_ensemble/
│       │       └── config.pbtxt              # Chains ONNX Runtime text_tokenizer -> cross_encoder_trt
│       └── batch_ingestion_repository/
│           ├── text_tokenizer/
│           │   ├── 1/
│           │   │   └── model.onnx            # Bulk chunk ONNX Runtime tokenizer backend
│           │   └── config.pbtxt
│           ├── batch_embedder_trt/
│           │   ├── 1/
│           │   │   └── model.plan            # Throughput-optimized engine
│           │   └── config.pbtxt              # Dynamic batching: max 256, delay 50ms
│           └── batch_embedder_ensemble/
│               └── config.pbtxt              # Chains ONNX Runtime text_tokenizer -> batch_embedder_trt
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
│   ├── chaos/                                # Queue saturation and spot eviction validation
│   └── unit/                                 # Tokenizer conformance, dynamic shape, & config tests
├── Makefile
└── README.md
```

---

## Architectural Decision Records (ADRs)

- **ADR-001: Pool Decoupling (Online vs. Batch)**: Isolates online retrieval from batch ingestion workloads onto separate GPU node groups to guarantee query latency SLOs.
- **ADR-002: Dedicated Cross-Encoder Node Pools**: Isolates compute-intensive cross-encoder rerankers from dense query embedders to avoid SM contention and thread stalls.
- **ADR-003: Standard FP16 Model Compilation**: Mandates FP16 precision across all inference targets in initial releases, deferring INT8 quantization to avoid precision loss.
- **ADR-004: In-Engine C++ Tokenization & Ensembles**: Chains tokenizer models to TensorRT engines via Triton ensemble pipelines (configured with ONNX Runtime backend as scaffolding, with native in-engine C++ HuggingFace tokenization runtime pending), removing Python interpreter bottlenecks from the critical path.
- **ADR-005: Elimination of Proxy Data Sidecars**: Connects callers directly to Triton via gRPC and pod-scoped shared memory, avoiding custom proxy processes for high-volume data movement.

