# Implementation Plan: Phase 1 & Phase 2 Model Serialization, Operational Server, POSIX SHM & Helm Deployments

## 1. Context & Objectives
Based on `tasks/RAG Inference Serving Implementation Plan.pdf`, `tasks/Production Grade Backend Concepts.pdf`, `tasks/Go and Python Microservices Architecture.pdf`, `tasks/Disaggregated Hybrid Model Architecture.pdf`, `GEMINI.md`, and `.ai/` specifications, this phase completes the remaining deliverables of **Phase 1: TensorRT FP16 Serialization & C++ Tokenization** and advances **Phase 2: Decoupled Pool Configuration & Dynamic Batching**.

The target microservice is `rag-inference-serving`, providing horizontally scalable, hardware-accelerated model inference across three physically isolated compute pools:
- **Online Embedding Pool (Pool A)**: Dedicated NVIDIA L4/A10G on-demand instances; $\le 2\text{ ms}$ queue delay, max batch size 32, standalone latency P99 $\le 10\text{ ms}$ at 1,500 QPS.
- **Online Reranker Pool (Pool B)**: Dedicated NVIDIA L4/A10G on-demand instances; $\le 4\text{ ms}$ queue delay, max batch size 32, latency P99 $\le 30\text{ ms}$ for candidate pairs.
- **Batch Ingestion Pool**: Preemptible spot node groups; $\le 50\text{ ms}$ queue delay, max batch size 256, throughput 2,500–3,500 chunks/sec per node, aggregate target $\ge 40,000\text{ chunks/sec}$ across 12–16 spot nodes.

---

## 2. Key Components to Implement

### 1. Operational ASGI Server & Configuration (`internal/config/settings.py`, `internal/server/probe_server.py`)
- **Out-of-Band Operational Server**: Production FastAPI application running on port `8080` alongside Triton.
  - **Strict Data-Plane Isolation**: The ASGI server is purely operational (probes, telemetry, documentation) and **does NOT** proxy model inference calls. Upstream callers (`rag-search-orchestrator`, `rag-chunk-lifecycle-service`) connect directly to Triton's native KServe v2 gRPC (`:8001`) or HTTP (`:8000`) endpoints.
  - **Operational Endpoints**: `GET /healthz` (liveness), `GET /ready` (readiness), `GET /metrics` (Prometheus), `GET /docs`, `GET /redoc`, `GET /api/docs`, `GET /openapi.json`.
  - **Active Readiness & Headroom Guard**:
    - Queries Triton's internal health endpoint (`http://localhost:8000/v2/health/ready`).
    - Monitors GPU VRAM memory usage; if active VRAM exceeds 85% ($>20.4\text{ GiB}$) or static engine allocation exceeds the $20\text{ GiB}$ static ceiling ($83.3\%$), `/ready` returns HTTP 503 (`Service Unavailable`) to trigger automated traffic shedding.
  - **Structured Secured Logging**: Integrated with `internal/logging/secured_logger.py` with RFC 7807/9457 error formatting and zero customer query/chunk text emission.

### 2. Model & Tokenizer Serialization Scripts (`scripts/export_onnx.py`, `scripts/export_tokenizer.py`, `scripts/build_trt_engine.sh`)
- **`scripts/export_onnx.py`**:
  - Exports HuggingFace transformer models to ONNX with dynamic batch and sequence axes.
  - **Dense Embedder (`bge-base-en-v1.5`)**: Bakes Mean Pooling and L2 Normalization directly into the ONNX computation graph so that output tensors strictly conform to `dims: [ 768 ]` (with batch dimension, i.e. `[batch, 768]`) required by `dense_embedder_trt`.
  - **Cross-Encoder Reranker (`bge-reranker-large`)**: Accepts sequence pairs (`query`, `passage`) with proper attention masks and segment token type IDs, outputting scalar classification logits `[batch, 1]` conforming to `cross_encoder_trt`.
- **`scripts/export_tokenizer.py`**:
  - Compiles HuggingFace tokenizers into ONNX/C++ runtime artifacts for Triton's `text_tokenizer` backend.
  - Enforces deterministic kernel truncation and padding: max 512 tokens for dense embeddings, max 2048 tokens for cross-encoders.
  - Populates model repositories (`text_tokenizer/1/model.onnx` and associated vocabulary files).
- **`scripts/build_trt_engine.sh`**:
  - Automated `trtexec` compilation script generating FP16 `.plan` engines adhering to calibrated shape profiles:
    - **Online Embedder**: min `[1, 1]`, opt `[16, 128]`, max `[32, 512]`.
    - **Online Reranker**: min `[1, 1]`, opt `[16, 256]`, max `[32, 2048]`.
    - **Batch Ingestion**: min `[1, 1]`, opt `[128, 256]`, max `[256, 512]`.
  - Includes `--dry-run` and mock compilation fallback flags for automated testing in GPU-less CI environments.

### 3. Reproducible Container Recipe (`deployments/docker/Dockerfile.triton`)
- **Base Image**: NVIDIA Triton Inference Server (`nvcr.io/nvidia/tritonserver:24.04-py3`) with CUDA 12.4 and TensorRT runtimes.
- **Security Hardening**: Non-root user (UID 1000, `triton`), read-only root filesystem compliance, temporary file isolation.
- **Dual Process Supervision**: Lightweight process supervision managing both `tritonserver` (ports 8000/8001/8002) and `uvicorn probe_server` (port 8080), with clean POSIX `SIGTERM`/`SIGINT` signal forwarding for graceful draining.

### 4. POSIX Shared Memory Buffer Management (`internal/shm/posix_shm.py`)
- **Zero-Copy IPC for Ingestion**: Memory-mapped `/dev/shm` buffers on 8 GiB Kubernetes `emptyDir` RAM volumes for co-located ingestion workers.
- **Strict Pod Boundary Safety**:
  - Checks target host and memory segment visibility (refusing non-localhost / cross-pod communication).
  - Throws `PodBoundaryViolationError` if cross-pod or cross-node IPC is attempted, forcing remote callers to use batched gRPC over HTTP/2.
- **Triton KServe v2 IPC Protocol**:
  - Generates Triton System Shared Memory HTTP registration payloads (`POST /v2/systemsharedmemory/region/{REGION_NAME}/register`), distinguished from the gRPC `SystemSharedMemoryRegister` method.
  - Provides deterministic context managers for safe `shm_open`, `mmap`, `munmap`, and `shm_unlink` to prevent kernel memory leaks.

### 5. Declarative Helm Deployments (`deployments/helm/`)
- **Parameterized Helm Chart Structure**:
  - `Chart.yaml`, `values.yaml`, `templates/deployment.yaml`, `templates/service.yaml`, `templates/keda-scaledobject.yaml`, `templates/configmap.yaml`.
- **Pool-Specific Calibrated Value Overlays**:
  - `values-online-embed.yaml`: 4–8 replicas, on-demand node group affinity, queue delay $\le 2\text{ ms}$, max batch 32, KEDA autoscaling trigger on `triton_queue_duration_microseconds` $\ge 3\text{ ms}$.
  - `values-online-rerank.yaml`: 4–12 replicas, on-demand node group affinity, queue delay $\le 4\text{ ms}$, max batch 32, KEDA autoscaling trigger on `triton_queue_duration_microseconds` $\ge 5\text{ ms}$.
  - `values-batch-ingestion.yaml`: 12–16 replicas, preemptible spot node group with spot tolerations, queue delay $\le 50\text{ ms}$, max batch 256, KEDA autoscaling trigger on ingestion task queue backlog depth.
- **Storage & Volume Mounts**: 8 GiB RAM `emptyDir` mounted at `/dev/shm`, read-only NVMe CSI persistent volume mounts for model engines (`readOnly: true`).

### 6. Comprehensive Test Suite (`test/unit/`)
- **`test/unit/test_probe_server.py`**: Validates FastAPI operational endpoints, 503 VRAM threshold shedding, and schema contracts.
- **`test/unit/test_settings.py`**: Validates configuration settings, queue delay ceilings, and VRAM limits across pool types.
- **`test/unit/test_posix_shm.py`**: Validates shared memory allocation, pod boundary enforcement, registration payloads, and leak-free reclamation.
- **`test/unit/test_export_scripts.py`**: Validates ONNX export argument parsing, dynamic axes configuration, and `trtexec` command generation.
- **`test/unit/test_tokenizer_conformance.py`**: Validates tokenizer parity against Hugging Face reference implementations across Unicode and edge-case whitespace.

### 7. Documentation & Specification Synchronization
- Update `api/openapi/openapi.json` to reflect all probe server endpoints.
- Update `.ai/doc-api.md` with complete endpoint schemas, Triton IPC contracts, and query parameters.
- Update `.ai/progress-tracker.md` to record completed milestones and phase transitions.

---

## 3. Engineering Invariants & Security
- **Secured Logging**: All log events pass through `SecuredLogger`; zero customer queries, text chunks, or raw float tensor arrays logged.
- **Backpressure & Fast Fallback**: Queue saturation emits `RESOURCE_EXHAUSTED` to trigger immediate upstream RRF fallback.
- **VRAM Ceilings**: Static engine allocations capped at $\le 20\text{ GiB}$ ($83.3\%$), active alert at $>20.4\text{ GiB}$ ($85\%$).
- **Strict Linting & Typing**: All code must pass `python -m ruff check --fix .`, `python -m ruff format .`, `mypy --strict`, and `pytest`.
