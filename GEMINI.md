# GEMINI Agent Guidelines - RAG Inference Serving

This repository implements the **`rag-inference-serving`** microservice (the hardware-accelerated Embedding & Reranker Pools for the enterprise RAG platform).

---

## 1. Primary Directives & Architecture References

Whenever you are interacting with or developing in this repository, you **MUST** reference and adhere to the guidelines established in the following context files:

1. **Primary Agent Persona & Execution Protocol**:
   - [`.ai/agents.md`](.ai/agents.md) — Role, core operational invariants, and step-by-step implementation protocol.
2. **System Entry Point & Blueprint Context**:
   - [`.ai.md`](.ai.md) — Root operational context and reading hierarchy.
   - [`.ai/project-overview.md`](.ai/project-overview.md) — Product definition, 3 decoupled GPU compute pools, and calibrated SLIs/SLOs.
   - [`.ai/architecture.md`](.ai/architecture.md) — 5 production-grade backend layers, Triton server configuration, and memory models.
   - [`.ai/doc-api.md`](.ai/doc-api.md) — Documented APIs (OpenAPI 3.1 HTTP operational server, Triton KServe v2 contracts, and `/dev/shm` IPC).
   - [`.ai/code-standards.md`](.ai/code-standards.md) — Python 3.11+ / C++ standards, Ruff linting/formatting, mypy strict typing, and test conventions.
   - [`.ai/security.md`](.ai/security.md) — Threat model, memory safety, VRAM bounds, and secured logging invariants.
   - [`.ai/ai-workflow-rules.md`](.ai/ai-workflow-rules.md) — Spec-driven workflow, pre-commit checklist, and non-negotiable guards.
   - [`.ai/progress-tracker.md`](.ai/progress-tracker.md) — Implementation roadmap, ADRs, runbooks, and session logs.

---

## 2. Non-Negotiable Engineering Invariants

### A. Continuous Doc API & OpenAPI Updates
The OpenAPI 3.1 documentation and [`.ai/doc-api.md`](.ai/doc-api.md) MUST be updated on **every code change or addition**, no matter how small or minor the modification is. All operational endpoints (`/docs`, `/redoc`, `/api/docs`, `/openapi.json`) must remain in exact synchronization with running code.

### B. Secured Logging
- All logs must be structured JSON lines.
- **Strictly redact** customer query text, raw document chunk text, raw tensor float arrays, auth tokens, bearer credentials, and presigned URLs before emission.
- Include correlation identifiers (`request_id`, `model_name`, `pool_type`, `batch_size`, `latency_ms`).
- Sanitize error stack traces and internal paths before logging.

### C. Hardware-Isolated Compute Pools
Always maintain strict physical separation between the 3 compute pools:
1. **Online Embedding Pool (Pool A)**: Dedicated NVIDIA L4/A10G GPUs on on-demand nodes; $\le 2\text{ ms}$ queue delay, max batch size $32$. Standalone latency P99 $\le 10\text{ ms}$ at 1,500 QPS.
2. **Online Reranker Pool (Pool B)**: Dedicated NVIDIA L4/A10G GPUs on on-demand nodes; $\le 4\text{ ms}$ queue delay, max batch size $32$. Latency P99 $\le 30\text{ ms}$.
3. **Batch Ingestion Pool**: Preemptible spot node groups; $50\text{ ms}$ queue delay, max batch size $256$. Sustains $2{,}500\text{--}3{,}500\text{ chunks/sec}$ per node.

### D. In-Engine C++ Tokenization & Triton Ensemble Pipelines
- Tokenization must execute directly inside Triton via C++ runtimes (`text_tokenizer`) composed via Triton Ensemble Pipelines (`online_embedder_ensemble`, `online_reranker_ensemble`, `batch_embedder_ensemble`) to eliminate Python GIL bottlenecks and provide single-round-trip inference.
- Models must serialize as FP16 TensorRT engines (`model.plan`).

### E. Native Backpressure & Graceful RRF Fallback
Enforce native queue delay timeouts (`max_queue_delay_microseconds`). When queues saturate, return standard error status codes (`RESOURCE_EXHAUSTED`) to allow upstream orchestrators to fall back instantly to un-reranked Reciprocal Rank Fusion (RRF) scores.

### F. POSIX Shared Memory IPC (`/dev/shm`) & Pod Boundaries
Use pod-scoped Kubernetes `emptyDir` RAM volume mounts ($8\text{ GiB}$) for zero-copy tensor passing in high-throughput batch ingestion strictly for co-located workers. Cross-pod or cross-node `/dev/shm` sharing is strictly prohibited; remote workers communicate over batched gRPC over HTTP/2. Always release and unmap shared memory segments cleanly to prevent memory leaks.

### G. Operational VRAM Ceilings & Dynamic Headroom
Cap aggregate static model allocations per GPU at $\le 20\text{ GiB}$ ($83.3\%$ of $24\text{ GB}$) to leave $4\text{ GiB}$ dedicated headroom for dynamic batching activation memory. Active runtime memory exceeding $85\%$ ($> 20.4\text{ GiB}$) fires a Severity: Critical alert.


---

## 3. Git Commit Standards

- **Git Commit Frequency**: Commit code as you build. Every working session or phase must contain **at least two git commits**.
- **Conventional Commit Patterns**: All git commit messages must strictly adhere to the following prefixes:
  - `feat:` for new features 
  - `fix:` for bug fixes
  - `refactor:` for code refactoring
  - `docs:` for documentation updates
  - `tests:` for tests addition and modification
  - `chore:` for maintenance tasks and environment setup

---

## 4. Pre-Commit Verification Workflow

Before committing any change or completing a task:
1. Run `python -m ruff check --fix .` and `python -m ruff format .`.
2. Run `mypy scripts api internal test`.
3. Run test suites (`pytest test/unit test/integration`).
4. Update [`.ai/doc-api.md`](.ai/doc-api.md) with all new/modified endpoints, parameters, models, or schemas.
5. Verify Secured Logging rules: ensure zero raw text queries or chunk payloads are logged.
6. Update [`.ai/progress-tracker.md`](.ai/progress-tracker.md).
