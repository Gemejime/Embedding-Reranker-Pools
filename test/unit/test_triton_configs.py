"""Unit tests validating Triton model repository configurations."""

import glob
import os
import unittest


class TestTritonConfigs(unittest.TestCase):
    def test_all_repositories_exist(self) -> None:
        repos = [
            "config/triton/online_embedding_repository",
            "config/triton/online_reranker_repository",
            "config/triton/batch_ingestion_repository",
        ]
        for repo in repos:
            self.assertTrue(os.path.isdir(repo), f"Expected repository {repo} to exist")

    def test_online_embedding_config(self) -> None:
        path = "config/triton/online_embedding_repository/dense_embedder_trt/config.pbtxt"
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn('platform: "tensorrt_plan"', content)
        self.assertIn("max_batch_size: 32", content)
        self.assertIn("max_queue_delay_microseconds: 2000", content)

    def test_online_reranker_config(self) -> None:
        path = "config/triton/online_reranker_repository/cross_encoder_trt/config.pbtxt"
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn('platform: "tensorrt_plan"', content)
        self.assertIn("max_batch_size: 32", content)
        self.assertIn("max_queue_delay_microseconds: 4000", content)

    def test_batch_ingestion_config(self) -> None:
        path = "config/triton/batch_ingestion_repository/batch_embedder_trt/config.pbtxt"
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn('platform: "tensorrt_plan"', content)
        self.assertIn("max_batch_size: 256", content)
        self.assertIn("max_queue_delay_microseconds: 50000", content)


if __name__ == "__main__":
    unittest.main()
