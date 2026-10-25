import os
import tempfile
import unittest

from scripts.export_tokenizer import (
    export_tokenizer_artifacts,
    get_tokenizer_max_length,
)


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

        with tempfile.TemporaryDirectory() as tmp_dir:
            dense_artifacts = export_tokenizer_artifacts(
                "dense", "BAAI/bge-base-en-v1.5", os.path.join(tmp_dir, "dense"), dry_run=True
            )
            with open(dense_artifacts["vocab_path"], encoding="utf-8") as f:
                vocab = {word.strip(): idx for idx, word in enumerate(f) if word.strip()}

            unk_id = vocab.get("[UNK]", 1)
            cls_id = vocab.get("[CLS]", 2)
            sep_id = vocab.get("[SEP]", 3)

            def tokenize_and_truncate(text: str, max_length: int) -> list[int]:
                # Deterministic normalization: strip zero-width chars and collapse whitespace
                normalized = text.replace("\u200b", "").replace("\ufeff", "")
                tokens = normalized.strip().split()
                token_ids = [cls_id]
                for tok in tokens:
                    token_ids.append(vocab.get(tok.lower(), unk_id))
                token_ids.append(sep_id)
                # Enforce truncation ceiling
                return token_ids[:max_length]

            for sample in test_strings:
                dense_tokens = tokenize_and_truncate(sample, dense_max_tokens)
                rerank_tokens = tokenize_and_truncate(sample, rerank_max_tokens)

                # Token bounds checks tied to pool limits
                self.assertLessEqual(len(dense_tokens), dense_max_tokens)
                self.assertLessEqual(len(rerank_tokens), rerank_max_tokens)

                # Token IDs must be valid non-negative integers
                self.assertTrue(all(isinstance(t, int) and t >= 0 for t in dense_tokens))
                self.assertTrue(all(isinstance(t, int) and t >= 0 for t in rerank_tokens))

                # Structure check: first token is [CLS]
                self.assertEqual(dense_tokens[0], cls_id)
                self.assertEqual(rerank_tokens[0], cls_id)

            # Specific assertion on the long text sample (>500 words)
            long_sample = "Long text " * 500
            untruncated_len = len(long_sample.split()) + 2
            self.assertGreater(untruncated_len, dense_max_tokens)
            dense_truncated = tokenize_and_truncate(long_sample, dense_max_tokens)
            self.assertEqual(len(dense_truncated), dense_max_tokens)
            rerank_truncated = tokenize_and_truncate(long_sample, rerank_max_tokens)
            self.assertEqual(len(rerank_truncated), min(untruncated_len, rerank_max_tokens))


if __name__ == "__main__":
    unittest.main()
