# AI Workflow Rules - RAG Inference Serving

## Development Approach

Build this microservice incrementally following a spec-driven workflow:
- The specification documents in [`tasks/`](tasks) and context files in [`.ai/`](.ai) define system invariants, data structures, and operational boundaries.
- Adhere strictly to the decoupled worker design (`online_embedding_repository`, `online_reranker_repository`, and `batch_ingestion_repository`) and the 5 backend architectural layers documented in [`.ai/architecture.md`](.ai/architecture.md).
- Implement against these specs without inferring undocumented dependencies or deviating from architectural decision records (ADRs).

---

## Human-in-the-Loop Rules

- **Plan First**: Propose an implementation plan or prompt in `prompts/<task-name>.md` before making code changes, and await human confirmation.
- **Ambiguity Escalation**: If GPU resource limits, dynamic batching thresholds, or tensor dimension mappings are ambiguous, prompt for clarification before proceeding.
- **Destructive Action Guards**: Always request human confirmation before running destructive operations, container modifications, or major refactors.
- **Demonstrate & Verify**: Provide clear verification steps, unit test output, and linting checks after completing a unit.

---

## Non-Negotiable Invariants

1. **Continuous Doc API & OpenAPI Maintenance**: The [`.ai/doc-api.md`](.ai/doc-api.md) file and OpenAPI 3.1 endpoints (`/docs`, `/redoc`, `/api/docs`, `/openapi.json`) MUST be updated on **every code change or addition**, regardless of how small the modification is.
2. **Secured Logging**: All logs emitted by servers, daemons, or scripts must be structured JSON logs. Strictly redact customer queries, text chunks, raw tensor embeddings, PII/PHI, authorization tokens, and credentials before emission.
3. **Decoupled Node Groups & Hardware Isolation**: Ensure online embedding, online reranker, and batch ingestion pools never run on shared GPUs to protect online query latency SLOs.
4. **Standard FP16 TensorRT Serialization**: Standardize purely on FP16 TensorRT engine serialization, avoiding unvalidated INT8 quantization in initial phases.
5. **In-Engine C++ Tokenization**: Keep tokenization logic within native in-engine C++ pipelines in Triton, eliminating Python GIL bottlenecks.
6. **Native Queue Backpressure**: Saturating queues must emit error status codes to trigger immediate upstream fallback to un-reranked RRF scores.
7. **Git Commit Standards**:
   - **Git Commit Frequency**: Commit code as you build. Every working session or phase must contain **at least two git commits**.
   - **Conventional Commit Patterns**: All git commit messages must strictly adhere to the following prefixes:
     - `feat:` for new features 
     - `fix:` for bug fixes
     - `refactor:` for code refactoring
     - `docs:` for documentation updates
     - `tests:` for tests addition and modification
     - `chore:` for maintenance tasks and environment setup
8. **Quality Gates**: Ruff linting, formatting, and import sorting, along with static `mypy` typing, must pass cleanly before any commit.

---

## Pre-Commit Verification Checklist

Before committing code, submitting changes, or marking a task complete:

1. **Lint & Import Sorting**: Run `python -m ruff check --fix .` (zero errors or warnings).
2. **Code Formatting**: Run `python -m ruff format .` (clean formatting verified).
3. **Type Safety**: Verify static types via `mypy scripts api internal test`.
4. **Automated Tests**: Ensure unit and integration tests are passing in `test/unit/` and `test/integration/`.
5. **Secured Logging Audit**: Verify that no raw text queries, document chunks, or authentication tokens are emitted in logging statements.
6. **Doc API & OpenAPI Sync**: Confirm [`.ai/doc-api.md`](.ai/doc-api.md) reflects any new or modified classes, parameters, endpoints, or storage keys.
7. **Progress Tracker**: Update [`.ai/progress-tracker.md`](.ai/progress-tracker.md) with session notes and completed items.
