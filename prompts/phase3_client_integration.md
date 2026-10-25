# Implementation Plan: Phase 3 Direct Client Integration & Native Circuit Breaking

## 1. Context & Architectural References
Based on `tasks/RAG Inference Serving Implementation Plan.pdf`, `tasks/Production Grade Backend Concepts.pdf`, `tasks/Go and Python Microservices Architecture.pdf`, `GEMINI.md`, `.ai/architecture.md`, `.ai/doc-api.md`, and `.ai/ai-workflow-rules.md`, this phase implements **Phase 3: Direct Client Integration & Native Circuit Breaking (Weeks 5–6)**.

The `rag-inference-serving` architecture disaggregates serving across three hardware-isolated GPU compute pools:
1. **Online Embedding Pool (Pool A)**: Dedicated NVIDIA L4/A10G instances; $\le 2\text{ ms}$ queue delay, max batch size 32. Standalone latency P99 $\le 10\text{ ms}$ at 1,500 QPS.
2. **Online Reranker Pool (Pool B)**: Dedicated NVIDIA L4/A10G instances; $\le 4\text{ ms}$ queue delay, max batch size 32. Latency P99 $\le 30\text{ ms}$ for 10–30 candidate pairs.
3. **Batch Ingestion Pool**: Preemptible spot node groups; $\le 50\text{ ms}$ queue delay, max batch size 256. Sustains 2,500–3,500 chunks/sec per node, aggregate $\ge 40,000\text{ chunks/sec}$ across 12–16 spot nodes.

### Polyglot Ecosystem & Responsibility Invariants:
- `rag-search-orchestrator` is a Go microservice; this repository provides:
  1. Standard proto3 definitions (`api/proto/v1/triton_stubs.proto`) with Go package options (`option go_package = "api/proto/v1/inference";`).
  2. A fully featured, production-grade Python client library (`internal/client/`) used by Python workers (`rag-document-ai-worker`, `rag-eval-governance-worker`), batch ingestion pipelines, end-to-end integration tests, and serving as the reference implementation.
  3. Complete contract documentation for the Go client in `.ai/doc-api.md`.
- **RRF & Retrieval Ownership**: Reciprocal Rank Fusion (fusing dense + sparse search) and ACL filtering remain strictly in `rag-search-orchestrator`. In `rag-inference-serving`, reranker fallback preserves pre-computed initial ranks and emits deterministic un-reranked RRF scores:
  $$RRF(d) = \frac{1}{k + rank(d)} \quad (k = 60)$$
- **Direct Transport Invariant (ADR-005)**: No intermediate proxy sidecars. Callers connect directly via gRPC over HTTP/2 with keepalives or pod-scoped `/dev/shm` IPC.

---

## 2. Deliverables & Components for Phase 3

### Component 1: Protobuf & KServe v2 Binary Codec (`api/proto/v1/`, `internal/client/kserve_codec.py`)
- Enhance `api/proto/v1/triton_stubs.proto`:
  - Add `option go_package = "api/proto/v1/inference";` for Go orchestrator stub generation.
  - Compile using `grpc_tools.protoc` into:
    - `api/proto/v1/triton_stubs_pb2.py`
    - `api/proto/v1/triton_stubs_pb2_grpc.py`
    - `api/proto/v1/triton_stubs_pb2.pyi`
  - Ensure import compatibility for Python packages.
- Create `internal/client/kserve_codec.py`:
  - `pack_string_tensor(strings: list[str]) -> bytes`: Encodes strings using Triton KServe v2 wire format (4-byte little-endian uint32 length prefix per UTF-8 string).
  - `unpack_fp16_tensor(raw_bytes: bytes, shape: tuple[int, ...]) -> list[list[float]]`: Fast zero-copy unpacking of FP16 byte buffers using NumPy or `struct.unpack`.
  - `build_model_infer_request(...)`: Helper creating deterministic `ModelInferRequest` messages for single-text, text-pair, and raw tensor payloads.

### Component 2: High-Performance gRPC Connection Pooling (`internal/client/grpc_pool.py`)
- `TritonGRPCChannelPool`:
  - Manages discrete, long-lived HTTP/2 channels for decoupled pools:
    - `embedding_target: str` (default from settings or env `RAG_SERVING_ONLINE_EMBED_GRPC_URL`)
    - `reranker_target: str` (default from settings or env `RAG_SERVING_ONLINE_RERANK_GRPC_URL`)
    - `batch_target: str` (default from settings or env `RAG_SERVING_BATCH_INGEST_GRPC_URL`)
  - Configures production HTTP/2 keepalive settings:
    - `grpc.keepalive_time_ms = 10000` (10s pings)
    - `grpc.keepalive_timeout_ms = 5000` (5s ack timeout)
    - `grpc.keepalive_permit_without_calls = 1`
    - `grpc.http2.max_pings_without_data = 0`
    - `grpc.max_receive_message_length = 64 * 1024 * 1024` (64 MiB)
    - `grpc.max_send_message_length = 64 * 1024 * 1024`
  - Thread-safe lazy channel instantiation and clean disposal (`close()` / context manager) draining active channels without socket leaks.
  - Active liveness and readiness probes (`is_channel_ready(pool_type: PoolType) -> bool`).

### Component 3: Resilient Circuit Breaker & RRF Fallback (`internal/client/circuit_breaker.py`)
- `RerankerCircuitBreaker`:
  - State Machine: `CLOSED`, `OPEN`, `HALF_OPEN`.
  - Configurable parameters:
    - `failure_threshold: int = 3` (consecutive failures) or `failure_rate_threshold: float = 0.5`
    - `recovery_timeout_seconds: float = 5.0`
    - `half_open_max_probes: int = 1`
  - Tripping Triggers:
    - `grpc.StatusCode.RESOURCE_EXHAUSTED` (Triton queue rejection when queue saturated)
    - `grpc.StatusCode.DEADLINE_EXCEEDED` (Reranker latency budget of 35ms breached)
    - `grpc.StatusCode.UNAVAILABLE` (Pod restart or eviction)
  - Un-Reranked RRF Fallback:
    - Operates on candidate document initial rankings ($rank \in [1..N]$):
      $$RRF(d) = \frac{1}{k + rank(d)} \quad (k = 60)$$
    - Emits structured `RerankResult` objects marked with `fallback_applied=True` and `fallback_reason`.

### Component 4: High-Performance Search Orchestrator Client (`internal/client/search_client.py`)
- Data models (Pydantic / dataclasses):
  - `CandidateDocument(id: str, passage: str, initial_rank: int, metadata: dict[str, Any] = ...)`
  - `RerankResult(id: str, score: float, rank: int, fallback_applied: bool, fallback_reason: str | None = None)`
- `SearchOrchestratorClient`:
  - `embed_query(query: str, request_id: str | None = None) -> list[float]`:
    - Enforces client deadline: **15 ms** (allows $\le 10\text{ ms}$ compute + network headroom).
    - Unpacks 768-dimensional FP16 vectors from `online_embedder_ensemble`.
  - `rerank(query: str, candidates: list[CandidateDocument], request_id: str | None = None) -> list[RerankResult]`:
    - Validates inputs: if `candidates` is empty, returns `[]` immediately without network call.
    - Chunks candidates exceeding `max_batch_size = 32`.
    - Enforces client deadline: **35 ms** (allows $\le 30\text{ ms}$ compute + network headroom).
    - If circuit is `OPEN` or on queue rejection / timeout, catches error and returns instant un-reranked RRF fallback.
  - Structured Secured Logging: Integrates `get_secured_logger`. Redacts all queries, passages, and float tensors. Logs correlation metrics (`request_id`, `model_name`, `pool_type`, `batch_size`, `latency_ms`).

### Component 5: Batch Ingestion Client (`internal/client/batch_client.py`)
- `BatchIngestionClient`:
  - Dynamic Micro-Batching: Splits large chunk lists into chunks of $N \le 256$.
  - Dual-Transport Support:
    - Pod-Local (`target_host="localhost"`): Leverages `internal.shm.posix_shm.SharedMemoryRegion` and Triton `SystemSharedMemoryRegister` for zero-copy memory transfers.
    - Remote Node (`target_host != "localhost"`): Automatically routes via batched gRPC over HTTP/2 directly to `batch_embedder_ensemble`. Enforces `validate_pod_boundary` to prevent cross-pod SHM crashes.
  - Clean resource reclamation ensuring zero orphaned SHM files in `/dev/shm`.

### Component 6: Comprehensive Unit & Integration Test Suite
- `test/unit/test_kserve_codec.py`: Tests string 4-byte LE packing, FP16 buffer unpacking, and round-trip fidelity.
- `test/unit/test_grpc_pool.py`: Tests channel creation, keepalive options, thread-safety, pool targets, and disposal.
- `test/unit/test_circuit_breaker.py`: Tests state transitions (`CLOSED` -> `OPEN` -> `HALF_OPEN` -> `CLOSED`), failure counting, cooldown timer, and deterministic RRF scoring.
- `test/unit/test_search_client.py`: Tests query embedding, reranking, empty candidates, chunking, `RESOURCE_EXHAUSTED` fast fallback, and zero-payload logging invariants.
- `test/unit/test_batch_client.py`: Tests micro-batching ($N \le 256$), remote gRPC transport, and local `/dev/shm` IPC with pod-boundary validation.
- `test/integration/test_orchestrator_integration.py`: End-to-end integration tests with simulated mock Triton server and simulated queue saturation.

### Component 7: Documentation, OpenAPI & Progress Tracker Updates
- Update `.ai/doc-api.md`: Document Python client interfaces, Go client contract guidelines, circuit breaker state diagram, and RRF fallback specification.
- Update `api/openapi/openapi.json`.
- Update `.ai/progress-tracker.md` to reflect Phase 3 completion.

---

## 3. Pre-Commit Verification & Quality Gates
1. `python -m ruff check --fix .` and `python -m ruff format .` (0 errors).
2. `mypy scripts api internal test` (strict typing passes).
3. `pytest test/unit test/integration` (all tests pass).
4. Secured Logging audit: zero customer queries, text chunks, or embedding vectors emitted in logs.
5. Minimum two Git commits following conventional commit prefixes (`feat:`, `tests:`, `docs:`).
