"""Unit tests for model export, tokenizer serialization, and trtexec scripts."""

import os
import tempfile
import unittest

from scripts.build_trt_engine import (
    SHAPE_PROFILES,
    build_trtexec_cmd,
    compile_engine,
    get_shape_profile,
)
from scripts.export_onnx import (
    export_onnx_model,
    get_io_tensor_names,
    get_onnx_dynamic_axes,
)
from scripts.export_tokenizer import (
    export_tokenizer_artifacts,
    get_tokenizer_max_length,
)


class TestExportScripts(unittest.TestCase):
    def test_onnx_dynamic_axes_dense(self) -> None:
        axes = get_onnx_dynamic_axes("dense")
        self.assertIn("input_ids", axes)
        self.assertIn("attention_mask", axes)
        self.assertIn("embeddings", axes)
        self.assertEqual(axes["embeddings"][0], "batch_size")

    def test_onnx_dynamic_axes_reranker(self) -> None:
        axes = get_onnx_dynamic_axes("reranker")
        self.assertIn("input_ids", axes)
        self.assertIn("attention_mask", axes)
        self.assertIn("token_type_ids", axes)
        self.assertIn("logits", axes)
        self.assertEqual(axes["logits"][0], "batch_size")

    def test_io_tensor_names(self) -> None:
        dense_in, dense_out = get_io_tensor_names("dense")
        self.assertEqual(dense_in, ["input_ids", "attention_mask"])
        self.assertEqual(dense_out, ["embeddings"])

        rerank_in, rerank_out = get_io_tensor_names("reranker")
        self.assertEqual(rerank_in, ["input_ids", "attention_mask", "token_type_ids"])
        self.assertEqual(rerank_out, ["logits"])

    def test_onnx_dry_run_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "model.onnx")
            res = export_onnx_model("dense", "BAAI/bge-base-en-v1.5", out_file, dry_run=True)
            self.assertEqual(res, out_file)
            self.assertTrue(os.path.isfile(out_file))

    def test_tokenizer_max_length_ceilings(self) -> None:
        self.assertEqual(get_tokenizer_max_length("dense"), 512)
        self.assertEqual(get_tokenizer_max_length("reranker"), 2048)

    def test_tokenizer_dry_run_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = export_tokenizer_artifacts(
                "dense", "BAAI/bge-base-en-v1.5", tmp_dir, dry_run=True
            )
            self.assertTrue(os.path.isfile(artifacts["model_path"]))
            self.assertTrue(os.path.isfile(artifacts["vocab_path"]))
            self.assertTrue(os.path.isfile(artifacts["config_path"]))

    def test_trtexec_shape_profiles(self) -> None:
        for profile in ["online_embed", "online_rerank", "batch_ingestion"]:
            prof = get_shape_profile(profile)
            self.assertIn("min", prof)
            self.assertIn("opt", prof)
            self.assertIn("max", prof)

        # Check online embed shape limits
        online_embed = SHAPE_PROFILES["online_embed"]
        self.assertIn("maxShapes=input_ids:32x512", "maxShapes=" + online_embed["max"])

        # Check online rerank shape limits
        online_rerank = SHAPE_PROFILES["online_rerank"]
        self.assertIn("maxShapes=input_ids:32x2048", "maxShapes=" + online_rerank["max"])

        # Check batch ingestion shape limits
        batch_ingest = SHAPE_PROFILES["batch_ingestion"]
        self.assertIn("maxShapes=input_ids:256x512", "maxShapes=" + batch_ingest["max"])

    def test_build_trtexec_cmd_structure(self) -> None:
        cmd = build_trtexec_cmd("model.onnx", "model.plan", "online_embed")
        self.assertEqual(cmd[0], "trtexec")
        self.assertIn("--onnx=model.onnx", cmd)
        self.assertIn("--saveEngine=model.plan", cmd)
        self.assertIn("--fp16", cmd)

    def test_compile_engine_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_plan = os.path.join(tmp_dir, "model.plan")
            res = compile_engine("online_embed", "model.onnx", out_plan, dry_run=True)
            self.assertEqual(res, out_plan)
            self.assertTrue(os.path.isfile(out_plan))


if __name__ == "__main__":
    unittest.main()
