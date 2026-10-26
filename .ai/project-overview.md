# RAG Inference Serving - Project Overview

## Overview

The `rag-inference-serving` subsystem provides horizontally scalable, hardware-accelerated model inference across the disaggregated enterprise RAG platform. It serves dense embeddings and cross-encoder rerankers for online query execution and offline batch ingestion workflows. In Phase 1, sparse lexical retrieval is delegated to upstream distributed BM25 inverted shards in `rag-search-orchestrator`, while learned sparse lexical representations (e.g., SPLADE or BGE-M3 multi-head sparse representations) are slated for Phase 2 integration.

The architecture pairs **Triton Inference Server** (handling high-throughput dynamic batching and GPU acceleration via TensorRT execution runtimes and native C++ tokenizer ensembles) with an internal operational FastAPI probe server (handling OpenAPI 3.1 documentation, Kubernetes `/healthz` and `/ready` probes, and Prometheus metrics scraping), alongside upstream direct gRPC channels and pod-scoped POSIX shared memory IPC endpoints.

---

## Compute Pool Topologies

To guarantee latency Service Level Objectives (SLOs) and eliminate Streaming Multiprocessor (SM) contention, the infrastructure is partitioned into three physically isolated compute pools:

| Parameter | Online Embedding Pool (Pool A) | Online Reranker Pool (Pool B) | Batch Ingestion Pool |
| :--- | :--- | :--- | :--- |
| **Target Workload** | Single-sentence query vectors | Top-10 to top-30 candidate pairs | Bulk document chunk arrays (256–512 tokens) |
| **GPU Hardware** | 1x NVIDIA L4 / A10G (24 GB VRAM) | 1x NVIDIA L4 / A10G (24 GB VRAM) | 1x NVIDIA L4 / A10G (Preemptible Spot) |
| **Node Strategy** | On-Demand Kubernetes Node Group | On-Demand Kubernetes Node Group | Preemptible Spot Node Group |
| **Dynamic Batching**| Max Batch: `32`, Queue Delay: $\le 2\text{ ms}$ | Max Batch: `32`, Queue Delay: $\le 4\text{ ms}$ | Max Batch: `256`, Queue Delay: $50\text{ ms}$ |
| **Cluster Sizing** | 4–8 Replicas across multi-AZ | 4–12 Replicas across multi-AZ | 12–16 Replicas (autoscaled on lag) |
| **Autoscaling Driver**| KEDA (`triton_queue_duration_microseconds` $\ge 3\text{ ms}$) | KEDA (`triton_queue_duration_microseconds` $\ge 5\text{ ms}$) | KEDA (Kafka / Task Backlog $> 500{,}000$ chunks) |
| **Transport** | Direct gRPC over HTTP/2 | Direct gRPC over HTTP/2 | Batched gRPC over HTTP/2 (Remote) / POSIX Shared Memory (`/dev/shm` 8 GiB emptyDir RAM for co-located pod-scoped workers) |

---

## Strategic Objectives

1. **Decoupled Cluster Topologies**: Physically isolate online retrieval serving from high-throughput offline batch ingestion to eliminate Streaming Multiprocessor (SM) starvation and dynamic queue thrashing.
2. **Hardware-Isolated Serving**: Separate compute-intensive cross-encoder rerankers from dense embedding models across discrete GPU node pools to prevent matrix multiplication stalls from breaching online embedding latency SLOs.
3. **Direct-to-Runtime Transport**: Upstream search orchestrator (`rag-search-orchestrator`) connects directly to Triton over low-latency gRPC channels with HTTP/2 keepalive pooling. Ingestion workers (`rag-chunk-lifecycle-service`) leverage native batched gRPC over network, or pod-scoped POSIX shared memory (`/dev/shm`) mounts (8 GiB `emptyDir` RAM) when co-located in the same Pod, eliminating intermediate sidecar proxies from data-plane paths.
4. **Native Queue Bounds & Fast Fallback**: Enforce Triton native queue timeouts (`max_queue_delay_microseconds`). On queue saturation, Triton immediately emits gRPC status errors (`RESOURCE_EXHAUSTED`), triggering deterministic, instant fallback to un-reranked Reciprocal Rank Fusion (RRF) scores in the upstream orchestrator.
5. **Standardized Precision Targets**: Standardize purely on FP16 TensorRT engine serialization (`model.plan`) across all initial phases, avoiding precision degradation risks and quantization maintenance overhead.
6. **In-Engine C++ Tokenization & Triton Ensemble Pipelines**: Standardize on native in-engine C++ HuggingFace tokenizer pipelines within the Triton execution graph (`text_tokenizer`), composed into unified Triton ensemble models (`online_embedder_ensemble`, `online_reranker_ensemble`, `batch_embedder_ensemble`) that chain raw text directly to TensorRT engines in a single zero-GIL execution step.

---

## Scope & Exclusions

### In-Scope
- **Triton Runtime Optimization**: Multi-model serving configuration using TensorRT Execution Providers across FP16 precision profiles.
- **In-Engine Preprocessing & Ensembles**: Standardized in-engine C++ HuggingFace tokenization pipelines (`text_tokenizer`) composed via Triton `ensemble_scheduler` pipelines (`online_embedder_ensemble`, `online_reranker_ensemble`, `batch_embedder_ensemble`).
- **Queue-Centric Backpressure**: Triton-native bounded queue enforcement emitting standard error status codes on queue saturation.
- **Direct Transport Configuration**: Direct gRPC with HTTP/2 keepalive pooling between upstream orchestrators and Triton, coupled with shared-memory batched clients for co-located ingestion pipelines.
- **Pool Segmentation**: Declarative Helm deployment configurations separating latency-sensitive online serving pools from preemptible batch ingestion worker pools.
- **Doc API & Operational Probe Server**: Dedicated internal ASGI server exposing OpenAPI 3.1 documentation (`/docs`, `/redoc`, `/api/docs`, `/openapi.json`), health probes (`/healthz`, `/ready`), and Prometheus metrics (`/metrics`).

### Exclusions (Out-of-Scope)
- **Custom Intermediate Go Data Proxies**: Custom Go sidecars handling Arrow Flight data-plane proxying are removed to eliminate memory copies and garbage collection pressure (ADR-005).
- **Phase 1 INT8 / AWQ Calibration**: Post-training INT8 quantization is excluded from Phase 1 to prevent score degradation on embedding spaces.
- **Retrieval Ranking & Merging**: Reciprocal Rank Fusion (RRF) and bitmap ACL filtering remain strictly within `rag-search-orchestrator`.
- **Document Chunking & Preparation**: Text tokenization boundaries and parent-child AST compiling are owned upstream by `rag-chunk-lifecycle-service`.
- **Generative Autoregressive Serving**: Hosting foundation generative LLMs (vLLM clusters or external inference endpoints).

---

## Calibrated Success Criteria (SLIs & SLOs)

- **Standalone Online Dense Embedding Latency**: P99 $\le 10\text{ ms}$ at 1,500 queries/sec per GPU instance (batch size = 1, dynamic queue delay $\le 2\text{ ms}$).
- **Dedicated Online Cross-Encoder Reranking Latency**: P99 $\le 30\text{ ms}$ for candidate pairs under sustained load on dedicated L4/A10G pools.
- **End-to-End Search Serving Budget**: Dense embedding plus candidate reranking $\le 40\text{ ms}$ P99 across decoupled pools.
- **Single-Node Ingestion Vector Throughput**: 2,500 to 3,500 chunks/sec per NVIDIA L4/A10G instance (at 256–512 tokens/chunk, batch size = 256).
- **Aggregate Ingestion Target**: $\ge 40{,}000\text{ chunks/sec}$ scaled across an autoscaled cluster pool of 12 to 16 spot GPU nodes.
- **GPU Stability & Headroom**: Zero CUDA OOM container terminations under peak saturation, with steady-state model engine allocations capped at $\le 20\text{ GiB}$ ($83.3\%$) and active runtime VRAM bounded below $85\%$ ($> 20.4\text{ GiB}$ triggers Critical alerts).

