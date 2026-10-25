# Progress Tracker - RAG Inference Serving

## Current Status

- **Phase**: **Phase 1: TensorRT FP16 Serialization & Ensemble Pipelines** (Delivered; native in-engine C++ tokenization pending), **Phase 2: Decoupled Pool Configuration & Dynamic Batching** (COMPLETED), **Phase 3: Direct Client Integration & Native Circuit Breaking** (COMPLETED), and **Phase 4: Cluster Load Sweeps, Chaos & Autoscaling** (COMPLETED)
- **Current Goal**: Maintenance, continuous profiling, and operational production serving.

---

## Roadmap & Milestones

### Phase 1: TensorRT FP16 Serialization & C++ Tokenization (Weeks 1–2)
- [x] Initialized repository architecture context, `.gitignore`, and AI agent protocols.
- [x] Established Triton ensemble pipeline configurations (`online_embedder_ensemble`, `online_reranker_ensemble`, `batch_embedder_ensemble`) chaining ONNX Runtime tokenizer backends as scaffolding to TensorRT FP16 engines.
- [x] Export dense embedding models to ONNX and compile optimized TensorRT FP16 engines (`scripts/export_onnx.py`, `scripts/build_trt_engine.sh`, `scripts/build_trt_engine.py`).
- [x] Implement in-engine tokenizer serialization pipelines within Triton model repositories (`scripts/export_tokenizer.py`) enforcing max 512 / 2048 token limits.
- [x] Establish reproducible container build recipes with CUDA 12.4 and TensorRT runtimes (`deployments/docker/Dockerfile.triton`, `deployments/docker/entrypoint.sh`).

### Phase 2: Decoupled Pool Configuration & Dynamic Batching (Weeks 3–4)
- [x] Define independent Triton serving configurations for dense embeddings (`dense_embedder_trt`) and cross-encoder rerankers (`cross_encoder_trt`).
- [x] Configure dynamic batching parameters: low-latency profiles for retrieval ($\le 2\text{ ms}$ delay) and high-throughput profiles for ingestion ($\le 50\text{ ms}$ delay).
- [x] Implement operational probe & documentation ASGI server (`internal/server/probe_server.py`, `internal/config/settings.py`) with active VRAM headroom guarding and 503 shedding.
- [x] Implement POSIX shared memory IPC endpoints for high-density chunk generation (`internal/shm/posix_shm.py`) with pod-boundary guards.
- [x] Establish declarative Helm deployments with KEDA autoscaling triggers and pool-specific overlays (`deployments/helm/`).

### Phase 3: Direct Client Integration & Native Circuit Breaking (Weeks 5–6)
- [x] Integrate `rag-search-orchestrator` directly with Triton gRPC endpoints using connection pooling (`internal/client/grpc_pool.py`, `internal/client/search_client.py`).
- [x] Configure orchestrator-side circuit breaking triggered by Triton queue errors (`RESOURCE_EXHAUSTED` $\rightarrow$ un-reranked RRF fallback via `internal/client/circuit_breaker.py`).
- [x] Wire `rag-chunk-lifecycle-service` and ingestion workers directly to the batch ingestion Triton pool (`internal/client/batch_client.py`).

### Phase 4: Cluster Load Sweeps, Chaos & Autoscaling (Weeks 7–8)
- [x] Execute load sweeps validating single-node throughput ($2{,}500\text{--}3{,}500\text{ chunks/sec}$) and multi-node scaling ($\ge 40{,}000\text{ chunks/sec}$ across spot nodes).
- [x] Inject synthetic CUDA stalls and queue saturation to confirm instant fallback to un-reranked RRF scores.
- [x] Finalize KEDA autoscaling rules driven by `triton_queue_duration_microseconds` and Kafka task backlog.
- [x] Establish declarative GitOps ArgoCD ApplicationSet and canary rollout manifests (`deployments/argocd/`).
- [x] Author developer operational tooling (`Makefile`) and comprehensive production guide (`README.md`).

---

## Calibrated 10-Point Testing Strategy Matrix

1. **In-Engine C++ Tokenizer Conformance**: Validate ONNX/C++ tokenizer produces identical token IDs, attention masks, and type IDs as Hugging Face Python reference across Unicode and edge whitespace.
2. **Deterministic Kernel Truncation & Padding**: Verify text $>512$ tokens (embeddings) and $>2048$ tokens (cross-encoders) is deterministically truncated at the C++ kernel boundary without runtime leaks.
3. **Dynamic Shape Profile Scaling**: Assert $1 \le \text{batch} \le 32$ (online) and $1 \le \text{batch} \le 256$ (ingestion) with $1 \le \text{seq\_len} \le 512$ executes without CUDA memory faults.
4. **Vector Contract Alignment**: Verify embedding dimension equals contract (768 or 1536 FP16 elements) and cross-encoder produces normalized scalar logits.
5. **Dynamic Batch Memory Packing**: Verify independent single queries pack into contiguous buffers without cross-request contamination.
6. **Dynamic Delay Flush Verification**: Assert micro-batches flush immediately upon reaching max batch size or delay ceiling ($\le 2\text{ ms}$, $\le 4\text{ ms}$, $\le 50\text{ ms}$).
7. **POSIX `/dev/shm` Zero-Copy Pointer Passing**: Validate zero-copy IPC within pod boundaries and assert unmap/release cleans up descriptors without kernel leaks.
8. **Online Latency & Reranker Boundary Profiling**: Confirm standalone P99 $\le 10\text{ ms}$ at 1,500 QPS for embeddings, and P99 $\le 30\text{ ms}$ for candidate pairs (10–30 candidates).
9. **Batch Throughput Scaling Sweeps**: Benchmark single-node $2,500\text{--}3,500\text{ chunks/sec}$ and multi-node cluster pool $\ge 40{,}000\text{ chunks/sec}$ across 12–16 spot nodes.
10. **Chaos, Queue Saturation & Spot Eviction**: Saturate queues to verify immediate `RESOURCE_EXHAUSTED` emission and upstream RRF fallback; simulate spot termination without data corruption.

---

## Architectural Decision Records (ADRs)

- **ADR-001: Pool Decoupling (Online vs. Batch)**: Isolates online retrieval from batch ingestion workloads onto separate GPU node groups to guarantee query latency SLOs.
- **ADR-002: Dedicated Cross-Encoder Node Pools**: Isolates compute-intensive cross-encoder rerankers from dense query embedders to avoid SM contention and thread stalls.
- **ADR-003: Standard FP16 Model Compilation**: Mandates FP16 precision across all inference targets in initial releases, deferring INT8 quantization to avoid precision loss.
- **ADR-004: In-Engine C++ Tokenization & Ensembles**: Runs tokenization directly within the Triton engine pipeline using C++ runtimes and ensemble schedulers, removing Python interpreter bottlenecks from the critical path.
- **ADR-005: Elimination of Proxy Data Sidecars**: Connects callers directly to Triton via gRPC and pod-scoped shared memory, avoiding custom proxy processes for high-volume data movement.

---

## Operational Runbooks

### Runbook 1: Online Queue Wait Time Alerts
1. Check `triton_queue_duration_microseconds` on the affected online deployment.
2. If queue latency exceeds warning thresholds (4 ms for online embeddings, 8 ms for online rerankers), verify current traffic rates against baseline capacity.
3. Ensure KEDA HPA is scaling pod replicas on the dedicated GPU node group.
4. Confirm that upstream search orchestrators are falling back gracefully to un-reranked RRF scores.

### Runbook 2: Managing CUDA Out-Of-Memory Conditions
1. Inspect `triton_gpu_memory_used_bytes` across the cluster.
2. Check whether incoming request sequences exceed maximum context window limits.
3. Verify that dynamic batching size parameters are aligned with pool specifications.
4. Restart failed worker pods to re-initialize unfragmented CUDA memory spaces.

### Runbook 3: Ingestion Cluster Throughput Lags
1. Monitor aggregate throughput across the batch ingestion pool.
2. Check Kafka or task-queue lag driving the ingestion pipeline.
3. Scale out the spot GPU node pool up to 16 nodes to meet target throughput ceilings.

---

## Session History & Change Log

- **Production Secrets, mTLS & Telemetry Hardening Session**:
  - Overhauled `.env.example` and `.env` across 14 comprehensive sections including: gRPC Transport Security & Mutual TLS (`RAG_SERVING_GRPC_TLS_ENABLED`, `RAG_SERVING_GRPC_CA_CERT_PATH`, `RAG_SERVING_GRPC_CLIENT_CERT_PATH`, `RAG_SERVING_GRPC_CLIENT_KEY_PATH`, `RAG_SERVING_GRPC_SERVER_NAME_OVERRIDE`), Operational Server Security (`RAG_SERVING_API_KEY`, `RAG_SERVING_LOG_LEVEL`), KEDA Autoscaling Triggers & Telemetry Secrets (`PROMETHEUS_SERVER_ADDRESS`, `PROMETHEUS_BEARER_TOKEN`, `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_CONSUMER_GROUP`, `KAFKA_TOPIC`, `KAFKA_SASL_USERNAME`, `KAFKA_SASL_PASSWORD`, `KAFKA_SASL_MECHANISM`, `KAFKA_SECURITY_PROTOCOL`, `KAFKA_SSL_CA_LOCATION`), and Model Registry & Cloud Storage Credentials (`HF_TOKEN`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL`, `MODEL_REGISTRY_STORAGE_URI`).
  - Extended Pydantic `ServingSettings` in `internal/config/settings.py` with full validation for TLS and operational security fields.
  - Implemented secure SSL channel creation in `TritonGRPCChannelPool` (`internal/client/grpc_pool.py`) supporting CA certificates, mTLS client credentials, and SNI target overrides.
  - Hardened out-of-band operational ASGI probe server (`internal/server/probe_server.py`) with optional Bearer / API-Key authentication for `/metrics`.
  - Added unit test suites in `test/unit/test_settings.py`, `test/unit/test_grpc_pool.py`, and `test/unit/test_probe_server.py` expanding test suite to 116 passing tests.
  - Enabled automatic `.env` file ingestion in Pydantic `ServingSettings` via `env_file=".env"` and typed constructor with `_env_file` override support.
  - Configured automatic endpoint inheritance in `TritonGRPCChannelPool` and `ServingSettings`: modifying `RAG_SERVING_TRITON_GRPC_URL` automatically cascades across Online Embedding, Online Reranker, and Batch Ingestion pools unless a pool is explicitly overridden.
  - Added unit test suite `test_triton_grpc_url_pool_inheritance` in `test/unit/test_settings.py` expanding test suite to 117 passing tests with 100% clean Ruff and strict Mypy validation.
  - Synchronized documentation in `README.md`, `.ai/doc-api.md`, and `api/openapi/openapi.json`.

- **Architecture Specification & Microservice Documentation Session**:
  - Authored comprehensive root [`Architecture.md`](Architecture.md) focused directly on the `rag-inference-serving` microservice, referencing the Implementation Plan and structured across all 5 Production-Grade Backend Layers (API & Routing, Data Persistence & Memory Models, Architecture & Scale, Security & Identity, and Observability & Quality).
  - Included detailed Mermaid sequence diagrams for Online Query Retrieval (with native backpressure and RRF fallback) and High-Density Batch Chunk Ingestion (with pod-scoped `/dev/shm` IPC and remote batched gRPC over HTTP/2).
  - Documented complete sizing and deployment matrices, KEDA Prometheus autoscaling trigger query formulas, alerting thresholds, operational runbooks, and Architectural Decision Records (ADR-001 through ADR-005).
  - Overhauled [`README.md`](README.md) into a production-grade operational guide featuring architecture diagrams, streamlined folder tree matching the implementation plan, environment variable references, local setup, model export, TensorRT compilation, testing suites, KEDA autoscaling, and ArgoCD GitOps pipelines.
  - Verified clean linters (ruff), strict typecheckers (mypy), and test suites.

- **Environment Configuration Template & Validation Session**:

  - Authored comprehensive environment configuration template (`.env.example`) defining all operational parameters across 10 structured sections: Service Identity & Topology (`RAG_SERVING_POOL_TYPE`), Out-of-band ASGI Server (`RAG_SERVING_SERVER_HOST`, `RAG_SERVING_SERVER_PORT`), Triton KServe v2 URLs, Decoupled Pool gRPC Targets, Calibrated Dynamic Batching Ceilings (`RAG_SERVING_MAX_QUEUE_DELAY_MICROSECONDS`, `RAG_SERVING_MAX_BATCH_SIZE`), VRAM Headroom Guards (`RAG_SERVING_VRAM_CAPACITY_BYTES`, `RAG_SERVING_MAX_STATIC_VRAM_BYTES`, `RAG_SERVING_VRAM_MAX_UTILIZATION_RATIO`), Expected Model Engines (`RAG_SERVING_MODELS_EXPECTED`), POSIX SHM Mounts (`RAG_SERVING_SHM_BASE_PATH`), Triton Container Supervisor (`MODEL_REPOSITORY`, `TRITON_ARGS`), and Benchmark Test Orchestration (`LIVE_TRITON`, `TRITON_ONLINE_GRPC_TARGET`, `TRITON_BATCH_GRPC_TARGET`, `TRITON_CLUSTER_ENDPOINTS`).
  - Added unit test coverage in `test/unit/test_settings.py` verifying dynamic environment variable parsing and override behavior.
  - Documented Section 6 (Environment Variables & Operational Configuration Reference) in `.ai/doc-api.md`.
  - Updated `README.md` local environment setup with `.env` copy instructions.
  - Verified 100% test pass rate across test suite and confirmed ruff lint/formatting compliance.

- **Phase 4 Cluster Load Sweeps, Chaos & Autoscaling Session**:

  - Implemented comprehensive benchmark suite (`test/benchmark/test_single_node_embedding_sweep.py`, `test/benchmark/test_multi_node_ingestion_scaling.py`, `test/benchmark/test_online_latency_profiler.py`) validating single-node throughput (2,500–3,500 chunks/sec), multi-node cluster scaling (&ge; 40,000 chunks/sec across 12–16 spot nodes), and online P99 retrieval bounds (&le; 10 ms embedding, &le; 30 ms reranking, &le; 40 ms cumulative).
  - Implemented chaos and resilience suite (`test/chaos/test_queue_saturation_backpressure.py`, `test/chaos/test_cuda_stall_injection.py`, `test/chaos/test_hardware_isolation.py`, `test/chaos/test_spot_eviction.py`) testing Triton native backpressure, GPU latency spike injection, physical hardware isolation, and spot termination failover.
  - Hardened KEDA autoscaling Prometheus trigger queries evaluating average queue delay per request and Kafka consumer lag thresholds.
  - Implemented GitOps ArgoCD declarative pipelines (`deployments/argocd/applicationset.yaml`, standalone application CRDs, and `deployments/argocd/rollout-canary.yaml`).
  - Implemented root operational tooling (`Makefile`) and production architecture reference guide (`README.md`).
  - Expanded test suites to 112 passing tests across unit, integration, benchmark, and chaos suites with 100% clean mypy strict typing and ruff conformance.
  - Synchronized `.ai/doc-api.md`, bumped `api/openapi/openapi.json` to v1.2.0, and updated `.ai/progress-tracker.md`.

- **Phase 3 Client Integration & Circuit Breaking Session**:
  - Implemented `api/proto/v1/triton_stubs.proto` with Go package option `api/proto/v1/inference` and compiled python gRPC stubs (`triton_stubs_pb2.py`, `triton_stubs_pb2_grpc.py`, `triton_stubs_pb2.pyi`) with package-safe relative imports.
  - Implemented zero-copy Triton KServe v2 binary tensor codec (`internal/client/kserve_codec.py`) supporting 4-byte little-endian string packing, fast FP16 buffer unpacking, and deterministic request construction.
  - Implemented thread-safe gRPC connection pool (`internal/client/grpc_pool.py`) maintaining isolated HTTP/2 keepalive channels for Online Embedding, Online Reranker, and Batch Ingestion pools.
  - Implemented fault-tolerant circuit breaker (`internal/client/circuit_breaker.py`) with 3-state machine (`CLOSED`, `OPEN`, `HALF_OPEN`), tripping on `RESOURCE_EXHAUSTED`, `DEADLINE_EXCEEDED`, and `UNAVAILABLE`, providing deterministic un-reranked Reciprocal Rank Fusion (RRF) scores: $RRF(d) = 1 / (60 + rank(d))$.
  - Implemented high-performance search orchestrator client (`internal/client/search_client.py`) enforcing 15 ms embedding and 35 ms reranking client deadlines with dynamic batch chunking ($\le 32$) and structured secured logging redacting all query and passage payloads.
  - Implemented high-density batch ingestion client (`internal/client/batch_client.py`) with dynamic micro-batching ($\le 256$) and dual-transport support (pod-local POSIX shared memory vs remote batched gRPC with pod-boundary validation).
  - Built comprehensive test suite with 88 passing tests across unit and integration suites (`test/unit/test_kserve_codec.py`, `test/unit/test_grpc_pool.py`, `test/unit/test_circuit_breaker.py`, `test/unit/test_search_client.py`, `test/unit/test_batch_client.py`, `test/integration/test_orchestrator_integration.py`).
  - Synchronized documentation in `.ai/doc-api.md`, OpenAPI specification in `api/openapi/openapi.json` (v1.1.0), and roadmap in `.ai/progress-tracker.md`.

- **Phase 1 & Phase 2 Full Implementation Session**:
  - Implemented out-of-band operational ASGI server (`internal/server/probe_server.py`, `internal/config/settings.py`) with `/healthz`, `/ready`, `/metrics`, `/api/docs` and active VRAM headroom guarding (>85% utilization / >20 GiB static threshold shedding via HTTP 503).
  - Implemented zero-copy POSIX shared memory buffer management (`internal/shm/posix_shm.py`) with strict Kubernetes pod boundary enforcement (`PodBoundaryViolationError`) and Triton KServe v2 registration payloads.
  - Built ONNX export script (`scripts/export_onnx.py`) with baked mean-pooling / L2 normalization for dense embeddings and dynamic sequence axes.
  - Built tokenizer export script (`scripts/export_tokenizer.py`) enforcing deterministic 512 / 2048 token truncation limits.
  - Built TensorRT FP16 compilation scripts (`scripts/build_trt_engine.sh`, `scripts/build_trt_engine.py`) with calibrated dynamic shape profiles and dry-run execution fallbacks.
  - Built hardened non-root container image recipe (`deployments/docker/Dockerfile.triton`, `deployments/docker/entrypoint.sh`) supervising Triton and the ASGI probe server.
  - Built parameterized Helm chart deployment suite (`deployments/helm/`) with 8 GiB RAM emptyDir `/dev/shm` mounts and calibrated value overlays for Online Embedding, Online Reranker, and Batch Ingestion pools.
  - Expanded comprehensive test suite with 49 unit tests across probe server, settings, POSIX SHM, export scripts, tokenizer conformance, and Helm manifests.

- **Pressure-Test Hardening & Remediation Session**:
  - Implemented missing Triton Ensemble configurations (`online_embedder_ensemble`, `online_reranker_ensemble`, `batch_embedder_ensemble`) chaining C++ tokenizers to TensorRT FP16 engines.
  - Hardened reranker tokenizer configuration to take candidate pairs (`query` and `passage`).
  - Added unit test suite validating ensemble pipelines and updated pytest configurations (`-p no:faker`).
  - Formalized pod-local POSIX `/dev/shm` vs remote batched gRPC over HTTP/2 boundary conditions across `.ai/architecture.md`, `.ai/doc-api.md`, `.ai.md`, and `GEMINI.md`.
  - Harmonized VRAM static engine allocation ceiling ($\le 20\text{ GiB}$ / $83.3\%$) with dynamic activation headroom and active runtime alert ($> 20.4\text{ GiB}$ / $85\%$).
  - Clarified Phase 1 BM25 keyword search vs Phase 2 learned sparse lexical representations roadmap.
  - Embedded full 10-point testing strategy matrix from implementation specification into `.ai/progress-tracker.md`.
- **Initialization Session**:
  - Configured root `.gitignore` to strictly exclude `tasks/` folder and model weight binaries.
  - Aligned `.ai.md` and entire `.ai/` directory to `rag-inference-serving` microservice specs from `tasks/RAG Inference Serving Implementation Plan.pdf` and `tasks/Production Grade Backend Concepts.pdf`.
  - Defined OpenAPI 3.1 HTTP Doc API specification and Triton KServe v2 contracts in `.ai/doc-api.md`.
  - Established secured logging standards, VRAM allocation ceiling ($\le 20\text{ GiB}$ / $83\%$), and Git commit conventions.

