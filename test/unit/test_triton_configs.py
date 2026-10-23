"""Unit tests validating Triton model repository configurations."""

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
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn('platform: "tensorrt_plan"', content)
        self.assertIn("max_batch_size: 32", content)
        self.assertIn("max_queue_delay_microseconds: 2000", content)

    def test_online_reranker_config(self) -> None:
        path = "config/triton/online_reranker_repository/cross_encoder_trt/config.pbtxt"
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn('platform: "tensorrt_plan"', content)
        self.assertIn("max_batch_size: 32", content)
        self.assertIn("max_queue_delay_microseconds: 4000", content)

    def test_batch_ingestion_config(self) -> None:
        path = "config/triton/batch_ingestion_repository/batch_embedder_trt/config.pbtxt"
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn('platform: "tensorrt_plan"', content)
        self.assertIn("max_batch_size: 256", content)
        self.assertIn("max_queue_delay_microseconds: 50000", content)

    def test_ensemble_pipeline_configs(self) -> None:
        ensembles = [
            (
                "config/triton/online_embedding_repository/online_embedder_ensemble/config.pbtxt",
                "dense_embedder_trt",
                32,
            ),
            (
                "config/triton/online_reranker_repository/online_reranker_ensemble/config.pbtxt",
                "cross_encoder_trt",
                32,
            ),
            (
                "config/triton/batch_ingestion_repository/batch_embedder_ensemble/config.pbtxt",
                "batch_embedder_trt",
                256,
            ),
        ]
        for path, trt_model, max_batch in ensembles:
            self.assertTrue(os.path.isfile(path), f"Expected ensemble config {path} to exist")
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn('platform: "ensemble"', content)
            self.assertIn(f"max_batch_size: {max_batch}", content)
            self.assertIn('model_name: "text_tokenizer"', content)
            self.assertIn(f'model_name: "{trt_model}"', content)


if __name__ == "__main__":
    unittest.main()
