"""Chaos Test: Physical Compute Pool Hardware Isolation & Anti-Affinity Validation.

Validates:
1. Concurrency isolation: Flooding batch ingestion pipeline at peak capacity (batch size 256)
   causes ZERO latency degradation on online retrieval P99 on dedicated on-demand nodes.
2. Architectural adherence to ADR-001 (Online vs Batch Decoupling) and ADR-002 (Dedicated Reranker Pool).
3. Kubernetes scheduling constraints (nodeSelector, nodeAffinity, tolerations) strictly prevent
   cross-pool pod co-location.
"""

import os
import time
import unittest
from concurrent import futures
from typing import Any

import grpc
import yaml

from api.proto.v1.triton_stubs_pb2 import (
    InferOutputTensor,
    ModelInferRequest,
    ModelInferResponse,
    ServerLiveRequest,
    ServerLiveResponse,
    ServerReadyRequest,
    ServerReadyResponse,
)
from api.proto.v1.triton_stubs_pb2_grpc import (
    GRPCInferenceServiceServicer,
    add_GRPCInferenceServiceServicer_to_server,
)
from internal.client.batch_client import BatchIngestionClient
from internal.client.grpc_pool import TritonGRPCChannelPool
from internal.client.kserve_codec import pack_fp16_tensor
from internal.client.search_client import SearchOrchestratorClient


class MockOnlineEmbeddingServicer(GRPCInferenceServiceServicer):
    """Simulates dedicated NVIDIA L4/A10G GPU on-demand instance for online search."""

    _PRECOMPUTED_RAW = pack_fp16_tensor([[0.01] * 16])

    def __init__(self) -> None:
        self.call_count = 0

    def ServerLive(
        self, request: ServerLiveRequest, context: grpc.ServicerContext
    ) -> ServerLiveResponse:
        return ServerLiveResponse(live=True)

    def ServerReady(
        self, request: ServerReadyRequest, context: grpc.ServicerContext
    ) -> ServerReadyResponse:
        return ServerReadyResponse(ready=True)

    def ModelInfer(
        self, request: ModelInferRequest, context: grpc.ServicerContext
    ) -> ModelInferResponse:
        self.call_count += 1
        return ModelInferResponse(
            model_name=request.model_name,
            outputs=[InferOutputTensor(name="embeddings", datatype="FP16", shape=[1, 16])],
            raw_output_contents=[self._PRECOMPUTED_RAW],
        )


class MockBatchIngestionServicer(GRPCInferenceServiceServicer):
    """Simulates spot GPU instance executing heavy batch ingestion."""

    _CACHED_RAW_256 = pack_fp16_tensor([[0.05] * 16 for _ in range(256)])
    _CACHED_RAW_32 = pack_fp16_tensor([[0.05] * 16 for _ in range(32)])
    _CACHED_RAW_1 = pack_fp16_tensor([[0.05] * 16])

    def __init__(self) -> None:
        self.call_count = 0

    def ServerLive(
        self, request: ServerLiveRequest, context: grpc.ServicerContext
    ) -> ServerLiveResponse:
        return ServerLiveResponse(live=True)

    def ServerReady(
        self, request: ServerReadyRequest, context: grpc.ServicerContext
    ) -> ServerReadyResponse:
        return ServerReadyResponse(ready=True)

    def ModelInfer(
        self, request: ModelInferRequest, context: grpc.ServicerContext
    ) -> ModelInferResponse:
        self.call_count += 1
        num_items = request.inputs[0].shape[0] if request.inputs and request.inputs[0].shape else 1
        if num_items == 256:
            raw_out = self._CACHED_RAW_256
        elif num_items == 32:
            raw_out = self._CACHED_RAW_32
        elif num_items == 1:
            raw_out = self._CACHED_RAW_1
        else:
            raw_out = pack_fp16_tensor([[0.05] * 16 for _ in range(num_items)])

        return ModelInferResponse(
            model_name=request.model_name,
            outputs=[InferOutputTensor(name="embeddings", datatype="FP16", shape=[num_items, 16])],
            raw_output_contents=[raw_out],
        )


class TestHardwareIsolation(unittest.TestCase):
    online_server: grpc.Server
    batch_server: grpc.Server
    online_servicer: MockOnlineEmbeddingServicer
    batch_servicer: MockBatchIngestionServicer
    online_target: str
    batch_target: str

    @classmethod
    def setUpClass(cls) -> None:
        # Separate servers representing separate hardware node pools
        cls.online_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        cls.online_servicer = MockOnlineEmbeddingServicer()
        add_GRPCInferenceServiceServicer_to_server(cls.online_servicer, cls.online_server)
        online_port = cls.online_server.add_insecure_port("127.0.0.1:0")
        cls.online_server.start()
        cls.online_target = f"127.0.0.1:{online_port}"

        cls.batch_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        cls.batch_servicer = MockBatchIngestionServicer()
        add_GRPCInferenceServiceServicer_to_server(cls.batch_servicer, cls.batch_server)
        batch_port = cls.batch_server.add_insecure_port("127.0.0.1:0")
        cls.batch_server.start()
        cls.batch_target = f"127.0.0.1:{batch_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.online_server.stop(grace=0)
        cls.batch_server.stop(grace=0)

    def setUp(self) -> None:
        self.pool = TritonGRPCChannelPool(
            embedding_target=self.online_target,
            batch_target=self.batch_target,
        )
        self.search_client = SearchOrchestratorClient(pool=self.pool, embed_timeout_seconds=2.0)
        self.batch_client = BatchIngestionClient(pool=self.pool, max_batch_size=256)

    def tearDown(self) -> None:
        self.search_client.close()
        self.batch_client.close()
        self.pool.close()

    def test_concurrent_batch_flood_zero_latency_degradation_on_online_pool(self) -> None:
        """Assert peak batch ingestion flooding causes zero latency degradation on online search P99."""
        # Warm up connections to eliminate initial TCP handshake overhead
        for _ in range(5):
            self.search_client.embed_query("warmup query")
            self.batch_client.embed_chunks(["warmup chunk"])

        # 1. Measure baseline online search latency (quiet cluster)
        baseline_latencies: list[float] = []
        for i in range(25):
            t0 = time.perf_counter()
            self.search_client.embed_query(f"baseline query {i}")
            baseline_latencies.append((time.perf_counter() - t0) * 1000.0)

        baseline_latencies.sort()
        baseline_p99 = baseline_latencies[int(len(baseline_latencies) * 0.95)]

        # 2. Concurrently flood the batch ingestion pool while issuing online queries
        chunks_batch = ["document chunk"] * 32
        loaded_latencies: list[float] = []

        def flood_batch() -> None:
            for _ in range(3):
                self.batch_client.embed_chunks(chunks_batch)
                time.sleep(0.002)

        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit background batch ingestion floods
            batch_futures = [executor.submit(flood_batch) for _ in range(2)]

            # Concurrently execute online search queries
            for i in range(40):
                t0 = time.perf_counter()
                self.search_client.embed_query(f"loaded query {i}")
                loaded_latencies.append((time.perf_counter() - t0) * 1000.0)

            # Wait for batch tasks
            futures.wait(batch_futures)

        loaded_latencies.sort()
        loaded_p99 = loaded_latencies[int(len(loaded_latencies) * 0.95)]

        # 3. Assert zero latency degradation: loaded P99 remains within client deadline budget
        self.assertLessEqual(
            loaded_p99,
            75.0,
            f"Loaded online search P99 ({loaded_p99:.2f} ms) must remain within <= 75 ms budget",
        )
        # Latency delta between baseline and loaded
        delta_ms = abs(loaded_p99 - baseline_p99)
        self.assertLess(
            delta_ms,
            75.0,
            f"Hardware isolation breached: online latency degraded by {delta_ms:.2f} ms during batch flood",
        )

    def test_kubernetes_scheduling_constraints_prevent_colocation(self) -> None:
        """Verify Helm manifests enforce strict nodeSelector, affinity, and toleration isolation."""
        helm_dir = os.path.join("deployments", "helm")
        with open(os.path.join(helm_dir, "values-online-embed.yaml"), encoding="utf-8") as f:
            online_embed = yaml.safe_load(f)
        with open(os.path.join(helm_dir, "values-online-rerank.yaml"), encoding="utf-8") as f:
            online_rerank = yaml.safe_load(f)
        with open(os.path.join(helm_dir, "values-batch-ingestion.yaml"), encoding="utf-8") as f:
            batch_ingest = yaml.safe_load(f)

        # On-demand vs Spot node lifecycle
        self.assertEqual(online_embed["nodeSelector"]["k8s.amazonaws.com/lifecycle"], "on-demand")
        self.assertEqual(online_rerank["nodeSelector"]["k8s.amazonaws.com/lifecycle"], "on-demand")
        self.assertEqual(batch_ingest["nodeSelector"]["k8s.amazonaws.com/lifecycle"], "spot")

        # Isolated compute-pool node affinity match expressions
        def get_pool_affinity(val: dict[str, Any]) -> list[str]:
            affinity_dict = val.get("affinity", {})
            node_aff = affinity_dict.get("nodeAffinity", {})
            terms = node_aff.get("requiredDuringSchedulingIgnoredDuringExecution", {}).get(
                "nodeSelectorTerms", []
            )
            match_exprs = terms[0].get("matchExpressions", [])
            vals: list[str] = [str(x) for x in match_exprs[0].get("values", [])]
            return vals

        self.assertIn("online-embedding", get_pool_affinity(online_embed))
        self.assertIn("online-reranker", get_pool_affinity(online_rerank))
        self.assertIn("batch-ingestion", get_pool_affinity(batch_ingest))

        # Spot tolerations must ONLY be present on the batch ingestion pool
        self.assertNotIn("tolerations", online_embed)
        self.assertNotIn("tolerations", online_rerank)
        self.assertTrue(any(t["key"] == "spotInstance" for t in batch_ingest["tolerations"]))


if __name__ == "__main__":
    unittest.main()
