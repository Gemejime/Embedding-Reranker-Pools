# Implementation Plan: Phase 4 Cluster Load Sweeps, Chaos & Autoscaling

## 1. Context & Architectural References
Based on `tasks/RAG Inference Serving Implementation Plan.pdf`, `tasks/Production Grade Backend Concepts.pdf`, `tasks/Production-Grade Multimodal RAG Architectural Specification.pdf`, `tasks/Disaggregated Hybrid Model Architecture.pdf`, `tasks/Go and Python Microservices Architecture.pdf`, `tasks/My Naming Convention.pdf`, `GEMINI.md`, `.ai/project-overview.md`, `.ai/architecture.md`, `.ai/doc-api.md`, and `.ai/ai-workflow-rules.md`, this plan defines **Phase 4: Cluster Load Sweeps, Chaos & Autoscaling (Weeks 7–8)**.

**Architecture Note**: ArgoCD GitOps deployment has been approved for multi-pool continuous delivery.

Phases 1, 2, and 3 have successfully established:
- **Phase 1**: TensorRT FP16 model export, in-engine C++ tokenizer configurations, and Triton ensemble pipelines (`online_embedder_ensemble`, `online_reranker_ensemble`, `batch_embedder_ensemble`).
- **Phase 2**: Decoupled pool serving configurations, dynamic batching parameters, out-of-band ASGI probe/doc server with VRAM headroom guarding, POSIX shared memory IPC, and Helm deployment charts.
- **Phase 3**: Direct client integration (`rag-search-orchestrator` and `rag-chunk-lifecycle-service` clients), HTTP/2 gRPC connection pooling, zero-copy KServe v2 binary tensor codec, 3-state circuit breaker with native un-reranked RRF fallback, and pod-boundary validation.

Phase 4 completes the microservice specification with cluster-scale load profiling, resilience/chaos testing, finalized KEDA autoscaling rules, GitOps ArgoCD declarative pipelines, operational tooling (`Makefile`), and production documentation (`README.md`).

---

## 2. Deliverables & Components for Phase 4

### Component 1: Benchmark & Load Sweeping Suite (`test/benchmark/`)
Implements quantitative validation for all SLIs/SLOs defined in the specification, featuring dual execution modes (calibrated mock timing harness for deterministic CI/local environments vs `--live-triton` for live GPU clusters):

1. **Single-Node Ingestion Vector Throughput (`test/benchmark/test_single_node_embedding_sweep.py`)**:
   - Concurrency and batch size sweeps ($N \in [1, 2, 4, 8, 16, 32, 64, 128, 256]$) with 256–512 token synthetic chunks.
   - Measures throughput (chunks/sec), latency percentiles (P50, P90, P99), dynamic batch packing efficiency, and verifies target sustained **2,500 to 3,500 chunks/sec** per NVIDIA L4/A10G instance at batch size 256 with 50 ms queue delay ceiling.
   - Evaluates the throughput and serialization efficiency delta between pod-local zero-copy `/dev/shm` IPC and remote batched gRPC over HTTP/2.
   - Validates that active memory remains bounded below 85% ($< 20.4\text{ GiB}$) with zero memory leaks across sustained batch iterations.

2. **Multi-Node Cluster Scaling (`test/benchmark/test_multi_node_ingestion_scaling.py`)**:
   - Simulates horizontal cluster scaling across **12 to 16 spot GPU nodes**.
   - Validates linear scale-out achieving **$\ge 40{,}000\text{ chunks/sec}$** aggregate throughput without queue stalls, cross-worker contention, or host memory degradation.
   - Verifies connection pool scalability and load balancing across multi-instance Triton endpoints in `internal/client/grpc_pool.py`.

3. **Online Latency & Reranker Boundary Profiling (`test/benchmark/test_online_latency_profiler.py`)**:
   - Implements a mandatory 100-request warm-up sweep prior to latency capture to eliminate JIT compilation and CUDA allocator warmup artifacts.
   - **Online Embedding Pool (Pool A)**: Subjects endpoint to **1,500 queries/sec** single-sentence QPS (batch size 1, queue delay $\le 2\text{ ms}$), verifying standalone latency **P99 $\le 10\text{ ms}$**.
   - **Online Reranker Pool (Pool B)**: Benchmarks candidate pairs across search depths (10 to 30 candidates per query adaptively pruned to Top-15 based on RRF scores, queue delay $\le 4\text{ ms}$), verifying latency **P99 $\le 30\text{ ms}$**.
   - **End-to-End Serving Budget**: Asserts dense embedding plus candidate reranking remains within the **$\le 40\text{ ms}$ P99** cumulative search budget.

### Component 2: Chaos & Resilience Suite (`test/chaos/`)
Implements failure injection, timeout enforcement, and architectural isolation testing:

1. **Native Queue Saturation & Backpressure (`test/chaos/test_queue_saturation_backpressure.py`)**:
   - Saturates reranker queue beyond `max_queue_size` and delay thresholds.
   - Asserts Triton native queue policy returns `RESOURCE_EXHAUSTED` (gRPC status code 8) or HTTP 429.
   - Verifies orchestrator circuit breaker catches error, trips to `OPEN`, and returns deterministic un-reranked RRF scores ($RRF(d) = 1 / (60 + rank(d))$) without cascading failures.
   - Validates the complete 3-state transition lifecycle under recovery: `CLOSED` $\rightarrow$ `OPEN` (fail-fast fallback) $\rightarrow$ `HALF_OPEN` (single probe trial) $\rightarrow$ `CLOSED` (traffic resumption).
   - Verifies secured logging under failure conditions: zero query or passage payloads emitted to error logs.

2. **Synthetic CUDA Stalls & GPU Latency Spike Injection (`test/chaos/test_cuda_stall_injection.py`)**:
   - Simulates GPU kernel stalls and prolonged execution delays exceeding client deadlines (15 ms embedding, 35 ms reranker).
   - Confirms deadline enforcement (`DEADLINE_EXCEEDED`), rapid gRPC channel recycling, zero leaked channel slots, and transparent RRF score emission.

3. **Hardware Isolation Validation (`test/chaos/test_hardware_isolation.py`)**:
   - Simulates concurrent workloads: floods batch ingestion pipeline at peak capacity (batch size 256, spot nodes) while simultaneously running sustained 1,500 QPS online search queries on dedicated on-demand nodes.
   - Verifies **zero latency degradation** on online retrieval P99 due to physical node group decoupling (ADR-001, ADR-002).
   - Asserts Kubernetes scheduling constraints (`nodeSelector`, `nodeAffinity`, `tolerations`) prevent cross-pool pod co-location.

4. **Spot Node Preemption & Eviction Resilience (`test/chaos/test_spot_eviction.py`)**:
   - Simulates Kubernetes spot instance termination signals (`SIGTERM`, node drain, simulated spot eviction).
   - Validates batch ingestion client retry logic, connection failover, task re-queueing without chunk loss, and clean `/dev/shm` resource reclamation preventing orphaned memory segments or kernel leaks.

### Component 3: KEDA Autoscaling Hardening & GitOps ArgoCD Integration
1. **Calibrate Trigger Rules in Helm Overlays (`deployments/helm/`)**:
   - Update Prometheus trigger queries to evaluate **average queue delay per request** rather than raw counter rates:
     $$\text{Avg Queue Delay } (\mu\text{s}) = \frac{\sum(\text{rate}(\text{triton\_queue\_duration\_microseconds}[\text{1m}]))}{\text{clamp\_min}(\sum(\text{rate}(\text{triton\_inference\_count}[\text{1m}])), 1)}$$
   - `values-online-embed.yaml`: Scale trigger on average queue delay $\ge 3000\mu\text{s}$ ($3\text{ ms}$); warning alert at $4\text{ ms}$ for $>1\text{ min}$.
   - `values-online-rerank.yaml`: Scale trigger on average queue delay $\ge 5000\mu\text{s}$ ($5\text{ ms}$); warning alert at $8\text{ ms}$ for $>1\text{ min}$.
   - `values-batch-ingestion.yaml`: Scale trigger on Kafka task backlog lag threshold $10{,}000$; warning alert when backlog exceeds $500{,}000$ chunks at maximum 16 spot node ceiling.

2. **GitOps ArgoCD Application Manifests (`deployments/argocd/`)**:
   - Establish declarative ArgoCD `ApplicationSet` (`deployments/argocd/applicationset.yaml`) utilizing a generator to continuously deliver the core `deployments/helm/` chart across the 3 decoupled GPU compute pools:
     - `rag-inference-online-embedding` (targeting on-demand nodes via `values-online-embed.yaml`)
     - `rag-inference-online-reranker` (targeting on-demand nodes via `values-online-rerank.yaml`)
     - `rag-inference-batch-ingestion` (targeting preemptible spot nodes via `values-batch-ingestion.yaml`)
   - Establish standalone `Application` CRDs (`deployments/argocd/application-online-embed.yaml`, `deployments/argocd/application-online-rerank.yaml`, `deployments/argocd/application-batch-ingestion.yaml`) for environment-specific gitops deployments.
   - Configure GitOps automated reconciliation (`syncPolicy: automated: {prune: true, selfHeal: true}`), sync options (`CreateNamespace=true`), and health evaluation tracking Triton probe readiness (`/healthz`, `/ready`) and KEDA `ScaledObject` status.
   - Define progressive rollout patterns (Argo Rollouts) for zero-downtime canary upgrades of TensorRT engines (`model.plan`) without GPU memory fragmentation.

3. **Deployment & GitOps Manifest Test Validation (`test/unit/test_helm_configs.py`, `test/unit/test_argocd_configs.py`)**:
   - Expand unit test coverage verifying KEDA triggers, query definitions, thresholds, and min/max replica boundaries across all three pools.
   - Add test validation for ArgoCD manifests: verify valid YAML syntax, correct Helm repository source path (`deployments/helm`), target namespace bindings, overlay value file paths, automated sync policies, and health check annotations.

### Component 4: Root Operational Tooling & Documentation
1. **`Makefile`**:
   - Standardized developer & CI targets: `lint`, `format`, `typecheck`, `test`, `test-unit`, `test-integration`, `test-benchmark`, `test-chaos`, `build-docker`, and `clean`.
2. **`README.md`**:
   - Complete architectural overview, decoupled pool topology, hardware specifications, instructions for compiling TensorRT engines, running Triton and the ASGI probe server, executing benchmarks and chaos tests, KEDA configuration, GitOps ArgoCD deployment instructions, and API reference.

### Component 5: Documentation, OpenAPI & Progress Synchronization
1. **Update `.ai/doc-api.md`**: Add Phase 4 benchmark interfaces, chaos testing protocols, calibrated KEDA metrics, and ArgoCD GitOps deployment architecture.
2. **Update `api/openapi/openapi.json`**: Bump version to v1.2.0 reflecting operational test and monitoring endpoints.
3. **Update `.ai/progress-tracker.md`**: Mark Phase 4 milestones completed and document test metrics, GitOps deployment manifests, and session history.

---

## 3. Quality Gates & Non-Negotiable Invariants
- **Secured Logging**: Zero customer queries, passages, or raw float tensors logged.
- **Strict Typing**: Mypy strict typing passes across `scripts`, `api`, `internal`, and `test`.
- **Linting & Formatting**: Ruff passes with 0 warnings or errors.
- **All Tests Passing**: Unit, integration, benchmark, and chaos test suites green.
- **Git Commit Standards**: Minimum two commits with conventional prefixes (`feat:`, `tests:`, `docs:`, `chore:`).
