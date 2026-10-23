# Progress Tracker - RAG Inference Serving

## Current Status

- **Phase**: **Phase 1: TensorRT FP16 Serialization & C++ Tokenization** (Initialized & Foundation Scaffolding)
- **Current Goal**: Establish repository scaffolding, GitHub Actions CI pipelines, Triton model configurations, and operational OpenAPI documentation server.

---

## Roadmap & Milestones

### Phase 1: TensorRT FP16 Serialization & C++ Tokenization (Weeks 1–2)
- [x] Initialized repository architecture context, `.gitignore`, and AI agent protocols.
- [x] Established Triton ensemble pipeline configurations (`online_embedder_ensemble`, `online_reranker_ensemble`, `batch_embedder_ensemble`) chaining C++ tokenizers to TensorRT FP16 engines.
- [ ] Export dense embedding models to ONNX and compile optimized TensorRT FP16 engines (`scripts/export_onnx.py`, `scripts/build_trt_engine.sh`).
- [ ] Implement native in-engine C++ HuggingFace tokenizer pipelines within Triton model repositories (`text_tokenizer`).
- [ ] Establish reproducible container build recipes with CUDA 12.4 and TensorRT runtimes (`deployments/docker/Dockerfile.triton`).

### Phase 2: Decoupled Pool Configuration & Dynamic Batching (Weeks 3–4)
- [x] Define independent Triton serving configurations for dense embeddings (`dense_embedder_trt`) and cross-encoder rerankers (`cross_encoder_trt`).
- [x] Configure dynamic batching parameters: low-latency profiles for retrieval ($\le 2\text{ ms}$ delay) and high-throughput profiles for ingestion ($\le 50\text{ ms}$ delay).
- [ ] Implement POSIX shared memory IPC endpoints for high-density chunk generation (`internal/shm/posix_shm.py`) with pod-boundary guards.

### Phase 3: Direct Client Integration & Native Circuit Breaking (Weeks 5–6)
- [ ] Integrate `rag-search-orchestrator` directly with Triton gRPC endpoints using connection pooling.
- [ ] Configure orchestrator-side circuit breaking triggered by Triton queue errors (`RESOURCE_EXHAUSTED` $\rightarrow$ un-reranked RRF fallback).
- [ ] Wire `rag-chunk-lifecycle-service` and ingestion workers directly to the batch ingestion Triton pool.

### Phase 4: Cluster Load Sweeps, Chaos & Autoscaling (Weeks 7–8)
- [ ] Execute load sweeps validating single-node throughput ($2{,}500\text{--}3{,}500\text{ chunks/sec}$) and multi-node scaling ($\ge 40{,}000\text{ chunks/sec}$ across spot nodes).
- [ ] Inject synthetic CUDA stalls to confirm instant fallback to un-reranked RRF scores.
- [ ] Finalize KEDA autoscaling rules driven by `triton_queue_duration_microseconds` and Kafka task backlog.

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
2. If queue latency exceeds 5 ms, verify current traffic rates against baseline capacity.
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

