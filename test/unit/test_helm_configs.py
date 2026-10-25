"""Unit tests validating Helm charts and pool-specific values overlays."""

import os
import unittest

import yaml  # type: ignore[import-untyped]


class TestHelmConfigs(unittest.TestCase):
    def setUp(self) -> None:
        self.helm_dir = os.path.join("deployments", "helm")
        self.assertTrue(os.path.isdir(self.helm_dir), "deployments/helm must exist")

    def test_chart_yaml(self) -> None:
        chart_path = os.path.join(self.helm_dir, "Chart.yaml")
        with open(chart_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertEqual(data["name"], "rag-inference-serving")
        self.assertEqual(data["version"], "1.0.0")

    def test_default_values(self) -> None:
        values_path = os.path.join(self.helm_dir, "values.yaml")
        with open(values_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertEqual(data["shmVolume"]["sizeLimit"], "8Gi")
        self.assertTrue(data["modelStorage"]["readOnly"])
        self.assertEqual(data["service"]["operationalPort"], 8080)
        self.assertEqual(data["service"]["grpcPort"], 8001)

    def test_online_embed_overlay(self) -> None:
        embed_path = os.path.join(self.helm_dir, "values-online-embed.yaml")
        with open(embed_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertEqual(data["poolType"], "online_embedding")
        self.assertEqual(data["nodeSelector"]["k8s.amazonaws.com/lifecycle"], "on-demand")
        self.assertEqual(data["env"]["RAG_SERVING_MAX_QUEUE_DELAY_MICROSECONDS"], "2000")
        self.assertEqual(data["env"]["RAG_SERVING_MAX_BATCH_SIZE"], "32")
        trigger = data["autoscaling"]["triggers"][0]
        self.assertEqual(trigger["metadata"]["threshold"], "3000")
        self.assertIn("clamp_min", trigger["metadata"]["query"])
        self.assertIn("nv_inference_queue_duration_us", trigger["metadata"]["query"])
        self.assertEqual(trigger["metadata"]["metricName"], "nv_inference_queue_duration_us")
        self.assertEqual(data["alerts"]["warningQueueDelayMicroseconds"], "4000")

    def test_online_rerank_overlay(self) -> None:
        rerank_path = os.path.join(self.helm_dir, "values-online-rerank.yaml")
        with open(rerank_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertEqual(data["poolType"], "online_reranker")
        self.assertEqual(data["nodeSelector"]["k8s.amazonaws.com/lifecycle"], "on-demand")
        self.assertEqual(data["env"]["RAG_SERVING_MAX_QUEUE_DELAY_MICROSECONDS"], "4000")
        self.assertEqual(data["env"]["RAG_SERVING_MAX_BATCH_SIZE"], "32")
        trigger = data["autoscaling"]["triggers"][0]
        self.assertEqual(trigger["metadata"]["threshold"], "5000")
        self.assertIn("clamp_min", trigger["metadata"]["query"])
        self.assertIn("nv_inference_queue_duration_us", trigger["metadata"]["query"])
        self.assertEqual(trigger["metadata"]["metricName"], "nv_inference_queue_duration_us")
        self.assertEqual(data["alerts"]["warningQueueDelayMicroseconds"], "8000")

    def test_batch_ingestion_overlay(self) -> None:
        batch_path = os.path.join(self.helm_dir, "values-batch-ingestion.yaml")
        with open(batch_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertEqual(data["poolType"], "batch_ingestion")
        self.assertEqual(data["nodeSelector"]["k8s.amazonaws.com/lifecycle"], "spot")
        self.assertEqual(data["env"]["RAG_SERVING_MAX_QUEUE_DELAY_MICROSECONDS"], "50000")
        self.assertEqual(data["env"]["RAG_SERVING_MAX_BATCH_SIZE"], "256")
        tolerations = [t["key"] for t in data.get("tolerations", [])]
        self.assertIn("spotInstance", tolerations)
        self.assertEqual(data["alerts"]["warningBacklogChunks"], "500000")


if __name__ == "__main__":
    unittest.main()
