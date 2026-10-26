"""Unit tests validating Triton model repository configurations and ensemble pipelines."""

import os
import re
import unittest
from typing import Any


def parse_ensemble_steps(content: str) -> list[dict[str, Any]]:
    """Parse step blocks from ensemble_scheduling in a config.pbtxt."""
    steps: list[dict[str, Any]] = []
    parts = re.split(r"(?<=\n)\s*\{\s*(?=\n\s*model_name)", content)
    for p in parts[1:]:
        m_name = re.search(r'model_name:\s*"([^"]+)"', p)
        if not m_name:
            continue
        input_maps = dict(
            re.findall(
                r'input_map\s*\{[^}]*key:\s*"([^"]+)"[^}]*value:\s*"([^"]+)"[^}]*\}',
                p,
            )
        )
        output_maps = dict(
            re.findall(
                r'output_map\s*\{[^}]*key:\s*"([^"]+)"[^}]*value:\s*"([^"]+)"[^}]*\}',
                p,
            )
        )
        steps.append(
            {
                "model_name": m_name.group(1),
                "input_map": input_maps,
                "output_map": output_maps,
            }
        )
    return steps


class TestTritonConfigs(unittest.TestCase):
    def test_all_repositories_exist(self) -> None:
        repos = [
            "config/triton/online_embedding_repository",
            "config/triton/online_reranker_repository",
            "config/triton/batch_ingestion_repository",
        ]
        for repo in repos:
            self.assertTrue(os.path.isdir(repo), f"Expected repository {repo} to exist")
            numeric_version_dirs = [d for _, dirs, _ in os.walk(repo) for d in dirs if d.isdigit()]
            self.assertTrue(
                len(numeric_version_dirs) > 0,
                f"Expected repository {repo} to contain at least one numeric version directory",
            )

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
        self.assertIn("default_queue_policy", content)
        self.assertIn("timeout_action: REJECT", content)

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
                {"raw_text": "raw_text"},
                {"embeddings": "embeddings"},
            ),
            (
                "config/triton/online_reranker_repository/online_reranker_ensemble/config.pbtxt",
                "cross_encoder_trt",
                32,
                {"query": "query", "passage": "passage"},
                {"logits": "logits"},
            ),
            (
                "config/triton/batch_ingestion_repository/batch_embedder_ensemble/config.pbtxt",
                "batch_embedder_trt",
                256,
                {"raw_text": "raw_text"},
                {"embeddings": "embeddings"},
            ),
        ]
        for path, trt_model, max_batch, expected_input_map, expected_output_map in ensembles:
            self.assertTrue(os.path.isfile(path), f"Expected ensemble config {path} to exist")
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn('platform: "ensemble"', content)
            self.assertIn(f"max_batch_size: {max_batch}", content)

            steps = parse_ensemble_steps(content)
            self.assertEqual(len(steps), 2, f"Expected 2 steps in ensemble {path}")

            # Verify step 1: text_tokenizer
            step1 = steps[0]
            self.assertEqual(step1["model_name"], "text_tokenizer")
            self.assertEqual(step1["input_map"], expected_input_map)
            self.assertEqual(
                step1["output_map"],
                {"input_ids": "token_ids", "attention_mask": "attn_mask"},
            )

            # Verify step 2: TensorRT engine
            step2 = steps[1]
            self.assertEqual(step2["model_name"], trt_model)
            self.assertEqual(
                step2["input_map"],
                {"input_ids": "token_ids", "attention_mask": "attn_mask"},
            )
            self.assertEqual(step2["output_map"], expected_output_map)


if __name__ == "__main__":
    unittest.main()
