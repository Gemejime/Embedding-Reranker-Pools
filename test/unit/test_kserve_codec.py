"""Unit tests for KServe v2 binary codec and tensor packing/unpacking."""

import unittest

import numpy as np

from internal.client.kserve_codec import (
    build_model_infer_request,
    build_single_text_infer_request,
    build_text_pair_infer_request,
    pack_fp16_tensor,
    pack_string_tensor,
    unpack_fp16_tensor,
    unpack_string_tensor,
)


class TestKServeCodec(unittest.TestCase):
    def test_pack_and_unpack_string_tensor_roundtrip(self) -> None:
        strings = ["hello world", "test query", "RAG inference serving", "Special chars: £€✓\n\t"]
        packed = pack_string_tensor(strings)
        self.assertIsInstance(packed, bytes)
        self.assertGreater(len(packed), 0)

        unpacked = unpack_string_tensor(packed)
        self.assertEqual(unpacked, strings)

    def test_pack_empty_strings(self) -> None:
        strings = ["", "non-empty", ""]
        packed = pack_string_tensor(strings)
        unpacked = unpack_string_tensor(packed)
        self.assertEqual(unpacked, strings)

    def test_pack_and_unpack_fp16_tensor(self) -> None:
        # 2D tensor: [2, 4]
        original_data = [[1.0, 2.5, -3.2, 0.0], [0.5, -1.0, 4.2, 10.0]]
        packed = pack_fp16_tensor(original_data)
        self.assertEqual(len(packed), 2 * 4 * 2)  # 2 rows, 4 cols, 2 bytes per fp16

        unpacked = unpack_fp16_tensor(packed, shape=(2, 4))
        self.assertEqual(len(unpacked), 2)
        self.assertEqual(len(unpacked[0]), 4)
        np.testing.assert_allclose(unpacked, original_data, atol=1e-2)

    def test_unpack_fp16_single_vector(self) -> None:
        # 1D vector: [768]
        data = [0.1 * i for i in range(768)]
        packed = pack_fp16_tensor(data)
        unpacked = unpack_fp16_tensor(packed, shape=(768,))
        self.assertEqual(len(unpacked), 1)
        self.assertEqual(len(unpacked[0]), 768)

    def test_build_single_text_infer_request(self) -> None:
        texts = ["query 1", "query 2"]
        req = build_single_text_infer_request(
            model_name="online_embedder_ensemble",
            texts=texts,
            request_id="test-req-123",
        )
        self.assertEqual(req.model_name, "online_embedder_ensemble")
        self.assertEqual(req.id, "test-req-123")
        self.assertEqual(len(req.inputs), 1)
        self.assertEqual(req.inputs[0].name, "raw_text")
        self.assertEqual(list(req.inputs[0].shape), [2, 1])
        self.assertEqual(req.inputs[0].datatype, "BYTES")
        self.assertEqual(len(req.outputs), 1)
        self.assertEqual(req.outputs[0].name, "embeddings")
        self.assertEqual(len(req.raw_input_contents), 1)
        self.assertEqual(unpack_string_tensor(req.raw_input_contents[0]), texts)

    def test_build_text_pair_infer_request(self) -> None:
        queries = ["q1", "q2"]
        passages = ["p1", "p2"]
        req = build_text_pair_infer_request(
            model_name="online_reranker_ensemble",
            queries=queries,
            passages=passages,
            request_id="rerank-req-456",
        )
        self.assertEqual(req.model_name, "online_reranker_ensemble")
        self.assertEqual(req.id, "rerank-req-456")
        self.assertEqual(len(req.inputs), 2)
        self.assertEqual(req.inputs[0].name, "query")
        self.assertEqual(list(req.inputs[0].shape), [2, 1])
        self.assertEqual(req.inputs[1].name, "passage")
        self.assertEqual(list(req.inputs[1].shape), [2, 1])
        self.assertEqual(len(req.raw_input_contents), 2)

    def test_build_text_pair_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_text_pair_infer_request(
                model_name="online_reranker_ensemble",
                queries=["q1"],
                passages=["p1", "p2"],
            )

    def test_build_generic_model_infer_request(self) -> None:
        req = build_model_infer_request(
            model_name="custom_model",
            inputs=[],
            outputs=[],
            raw_input_contents=[],
            parameters={"test_bool": True, "test_int": 42, "test_str": "hello"},
        )
        self.assertEqual(req.model_name, "custom_model")
        self.assertTrue(req.parameters["test_bool"].bool_param)
        self.assertEqual(req.parameters["test_int"].int64_param, 42)
        self.assertEqual(req.parameters["test_str"].string_param, "hello")


if __name__ == "__main__":
    unittest.main()
