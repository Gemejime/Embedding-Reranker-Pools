# Code Standards - RAG Inference Serving

## General Principles

- **Hardware & Pool Isolation**: Strictly segregate online embedding, online reranker, and batch ingestion execution profiles to prevent SM thrashing.
- **In-Engine Optimization**: Eliminate Python runtime overhead from the hot path by running HuggingFace tokenization directly inside C++ Triton runtimes and executing models via TensorRT FP16 engines.
- **Deterministic Backpressure & Fail-Fast**: Respect queue timeouts (`max_queue_delay_microseconds`). Never hide queue saturation behind artificial retries; emit `RESOURCE_EXHAUSTED` to allow instant upstream fallback to un-reranked RRF scores.

---

## Python & Code Quality Standards

- **Runtime Target**: Python 3.11+.
- **Authoritative Linter & Formatter**: **Ruff**:
  - Line length limit: `100` characters.
  - Formatter: `python -m ruff format .`
  - Linter & Import Sorter: `python -m ruff check --fix .`
  - Imports: Grouped into standard library, third-party (e.g. `tritonclient`, `fastapi`, `pydantic`), and first-party (`internal`, `scripts`, `api`).
- **Static Type Checking**: Strict `mypy` typing across all Python modules:
  - No untyped `def` functions.
  - Avoid `typing.Any`; prefer Pydantic schemas or strongly typed protobuf classes.
  - Strict null-checking (`strict_optional = true`).

---

## Triton Model Configuration Standards (`config.pbtxt`)

1. **Explicit Data Types & Shapes**:
   - Tokenizer inputs: `TYPE_INT32` with dynamic shapes `[-1]`.
   - Embedder outputs: `TYPE_FP16` with explicit dimension vectors (e.g. `[768]` or `[1536]`).
   - Reranker output: `TYPE_FP16` with scalar logit dimension `[1]`.
2. **Dynamic Batching Discipline**:
   - Online pools: `preferred_batch_size: [8, 16, 32]`, `max_queue_delay_microseconds: 2000` (or `4000` for rerankers).
   - Ingestion pool: `preferred_batch_size: [64, 128, 256]`, `max_queue_delay_microseconds: 50000`.
3. **Instance Groups**:
   - Define exact GPU instance allocation (`count: 2, kind: KIND_GPU, gpus: [0]`).

---

## Secured Logging & Observability

- **Structured JSON Logging**: Emit machine-parseable JSON formatted logs across all operational daemons, scripts, and probe servers.
- **Mandatory Correlation IDs**: Every log event must contain:
  - `request_id`
  - `model_name`
  - `pool_type` (`online_embedding`, `online_reranker`, `batch_ingestion`)
  - `batch_size`
  - `queue_latency_ms`
  - `compute_latency_ms`
- **Security Invariant (Secured Logging)**:
  - **Strict Redaction**: Never log raw text queries, document chunk contents, customer secrets, bearer authorization tokens, or presigned URLs.
  - Error logs must sanitize internal system paths and GPU memory addresses before logging.

---

## Testing & Quality Assurance

- Maintain testing suites across:
  - `test/unit/`: In-engine C++ tokenizer conformance against Hugging Face Python reference implementation, dynamic shape validation, POSIX shared memory buffer serialization.
  - `test/integration/`: Direct gRPC client validation with Triton, queue saturation behavior, and fallback triggers.
  - `test/chaos/`: Synthetic CUDA stalls, queue flooding, and spot node eviction simulations.
- Target: $\ge 80\%$ test coverage on custom Python and Go helper modules.

---

## Doc API & OpenAPI Mandatory Maintenance

- **Continuous Synchronization**: Whenever an operational endpoint, model schema, protobuf contract, or configuration is added or changed, `.ai/doc-api.md` **MUST** be updated immediately in the same change step, no matter how small the code modification is.

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
