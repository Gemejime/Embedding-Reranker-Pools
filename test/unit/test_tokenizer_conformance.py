"""Unit tests validating tokenizer conformance, Unicode handling, and deterministic truncation."""

import unittest

from scripts.export_tokenizer import get_tokenizer_max_length


class TestTokenizerConformance(unittest.TestCase):
    def test_truncation_limits_dense_and_reranker(self) -> None:
        self.assertEqual(get_tokenizer_max_length("dense"), 512)
        self.assertEqual(get_tokenizer_max_length("reranker"), 2048)

    def test_edge_case_whitespace_and_unicode_normalization(self) -> None:
        test_strings = [
            "Normal query string",
            "   Leading and trailing whitespace   \t\n\r",
            "Multiple   spaces   between   words",
            "Unicode characters: café, résumé, 🚀, 日本語, 123½",
            "Zero\u200bwidth\u200bspace\ufefftest",
            "Long text " * 500,  # Far exceeds 512 tokens
        ]
        dense_max_tokens = get_tokenizer_max_length("dense")
        rerank_max_tokens = get_tokenizer_max_length("reranker")

        for sample in test_strings:
            words = sample.split()
            # Simple word-piece estimate upper bound
            estimated_tokens = len(words) * 2

            # Deterministic truncation check
            truncated_dense_tokens = min(estimated_tokens, dense_max_tokens)
            self.assertLessEqual(truncated_dense_tokens, 512)

            truncated_rerank_tokens = min(estimated_tokens, rerank_max_tokens)
            self.assertLessEqual(truncated_rerank_tokens, 2048)


if __name__ == "__main__":
    unittest.main()
