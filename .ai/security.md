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
   - User query text, document chunk text bodies, raw float tensor arrays, and presigned URLs may contain customer Personally Identifiable Information (PII), Protected Health Information (PHI), or confidential enterprise intellectual property.
   - **Under no circumstances may raw text strings, queries, chunk snippets, embedding arrays, or presigned URLs be written to stdout, stderr, or external log collectors.**
   - `SecuredJSONFormatter` actively sanitizes and suppresses payload-bearing content within unstructured message strings (including embedded queries, chunks, float vector arrays, and presigned URLs) rather than relying solely on forbidden key names in `sanitize_value`.
2. **Secrets & Authorization Masking**:
   - Bearer tokens, Basic authorization headers, API keys, password/secret field variants, TLS certificates, and cloud credentials (including SigV4 signatures) must be sanitized and replaced with `[REDACTED]` prior to logging.
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
   - Dedicated traceback sanitization (`sanitize_traceback`) sanitizes filesystem paths (`[PATH]/<filename>`), hostnames (`[REDACTED_HOST]`), and internal GPU memory addresses (`0x[REDACTED_GPU_ADDR]`) before emitting error records.

---

## Memory Safety & GPU Protection

1. **VRAM Allocation Ceilings & Dynamic Headroom**:
   - **Static Engine Allocation Ceiling**: Total serialized TensorRT model engines (`model.plan`) and static scratch allocations per GPU must be capped at $\le 20\text{ GiB}$ ($83.3\%$ of $24\text{ GB}$ on NVIDIA L4/A10G).
   - **Dynamic Execution Headroom**: The remaining $\approx 4\text{ GiB}$ ($16.7\%$) is strictly reserved for CUDA runtime contexts and dynamic batching activation memory.
   - **Active Saturation Alert**: Active memory allocation exceeding $85\%$ ($> 20.4\text{ GiB}$) on any node fires a **Severity: Critical** Prometheus alert (`triton_gpu_memory_used_bytes`).
2. **Shared Memory Boundary Isolation**:
   - Shared memory `/dev/shm` must be mounted as a pod-scoped Kubernetes `emptyDir` RAM volume (8 GiB).
   - Cross-pod `/dev/shm` sharing is strictly prohibited to eliminate container crashes and `SIGBUS` memory faults. Remote worker pods communicate over batched gRPC.
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
