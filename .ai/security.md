# Security Context - RAG Inference Serving

## Threat Model & Trust Boundaries

- **Upstream Query & Chunk Payloads**: Untrusted input boundary. Queries and document chunks sent to Triton may contain malicious adversarial prompts, oversized text spans, or malformed Unicode.
  - Context boundaries must be enforced deterministically at the C++ tokenizer kernel boundary (truncating inputs exceeding 512 tokens for embeddings, or 2048 tokens for cross-encoders) without raising uncaught exceptions or leaking uninitialized memory.
- **Compute Worker Runtime**: Isolated GPU compute environment.
  - Containers execute with non-root Linux permissions (`runAsNonRoot: true`, non-root UID `1000`).
  - Drop all default Linux capabilities (`capabilities: drop: ["ALL"]`).
  - Read-only root filesystem (`readOnlyRootFilesystem: true`) with strictly scoped writable volumes for temporary scratch (`/tmp`) and local pod shared memory (`/dev/shm` 8 GiB `emptyDir` RAM).
- **Network Boundaries & Transport Security**:
  - Direct gRPC channels between `rag-search-orchestrator`, `rag-chunk-lifecycle-service`, and Triton operate within private VPC networks with mutual TLS (mTLS) where applicable.
  - Triton ports (`8000`, `8001`, `8002`) and operational probe ports (`8080`) are non-public and reachable only by authorized internal mesh services.

---

## Secured Logging Invariants

1. **Strict Content Redaction (Zero Raw Payload Emission)**:
   - User query text, document chunk text bodies, and raw float tensor arrays may contain customer Personally Identifiable Information (PII), Protected Health Information (PHI), or confidential enterprise intellectual property.
   - **Under no circumstances may raw text strings, queries, chunk snippets, or embedding arrays be written to stdout, stderr, or external log collectors.**
2. **Secrets & Authorization Masking**:
   - Bearer tokens, JWT authorization headers, API keys, TLS certificates, and cloud credentials must be sanitized and replaced with `[REDACTED]` prior to logging.
3. **Structured Sanity**:
   - All log lines must be structured JSON containing purely operational metadata:
     - `timestamp_utc`: ISO 8601 UTC timestamp.
     - `correlation_id` / `request_id`: Tracing UUID.
     - `model_name`: e.g. `dense_embedder_trt`, `cross_encoder_trt`.
     - `pool_type`: `online_embedding`, `online_reranker`, or `batch_ingestion`.
     - `batch_size`: Integer count of packed requests.
     - `queue_latency_ms`: Time spent waiting in Triton dynamic batching queue.
     - `compute_latency_ms`: Time spent in TensorRT kernel execution.
     - `status`: Result code (`OK`, `RESOURCE_EXHAUSTED`, `ERROR`).
4. **Error Trace Sanitization**:
   - Exception handlers and crash loggers must sanitize filesystem paths, hostnames, and internal GPU memory addresses before emitting error records.

---

## Memory Safety & GPU Protection

1. **VRAM Allocation Ceilings**:
   - Total engine allocations per GPU must be capped at $\le 20\text{ GiB}$ ($83\%$ of $24\text{ GB}$ on NVIDIA L4/A10G).
   - This safety headroom guarantees that dynamic batching peaks and memory fragmentation will not trigger CUDA out-of-memory (`CUDA OOM`) terminations.
2. **Shared Memory Boundary Isolation**:
   - Shared memory `/dev/shm` must be mounted as a pod-scoped Kubernetes `emptyDir` RAM volume.
   - Cross-pod `/dev/shm` sharing is strictly prohibited to eliminate unexpected container crashes or `SIGBUS` memory faults.
   - All shared memory segments must be unregistered and unmapped cleanly during request teardown.

---

## Git Commit Standards

- **Git Commit Frequency**: Commit code as you build. Every working session or phase must contain **at least two git commits**.
- **Conventional Commit Patterns**: All git commit messages must strictly adhere to the following prefixes:
  - `feat:` for new features 
  - `fix:` for bug fixes
  - `refactor:` for code refactoring
  - `docs:` for documentation updates
  - `tests:` for tests addition and modification
  - `chore:` for maintenance tasks and environment setup
