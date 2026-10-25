"""Export HuggingFace transformer models to ONNX for TensorRT compilation.

Supports:
1. Dense Embedder (e.g., BAAI/bge-base-en-v1.5):
   - Inlines Mean Pooling and L2 Normalization directly into the ONNX graph.
   - Output tensor shape: [batch_size, 768] (TYPE_FP16/FP32).
   - Dynamic axes: batch_size and sequence_length (max 512).
2. Cross-Encoder Reranker (e.g., BAAI/bge-reranker-large):
   - Sequence pair classification.
   - Output tensor shape: [batch_size, 1] (TYPE_FP16/FP32).
   - Dynamic axes: batch_size and sequence_length (max 2048).
"""

import argparse
import os
from typing import Any

from internal.logging.secured_logger import get_secured_logger

logger = get_secured_logger("export-onnx")


def get_onnx_dynamic_axes(model_type: str) -> dict[str, dict[int, str]]:
    """Return dynamic axes specification for the target model type."""
    m_type = model_type.lower().strip()
    if m_type == "dense":
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size"},
        }
    elif m_type in ("reranker", "cross_encoder"):
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        }
    raise ValueError(f"Unsupported model type '{model_type}'. Expected 'dense' or 'reranker'.")


def get_io_tensor_names(model_type: str) -> tuple[list[str], list[str]]:
    """Return ([input_names], [output_names]) for the model type."""
    m_type = model_type.lower().strip()
    if m_type == "dense":
        return ["input_ids", "attention_mask"], ["embeddings"]
    elif m_type in ("reranker", "cross_encoder"):
        return ["input_ids", "attention_mask", "token_type_ids"], ["logits"]
    raise ValueError(f"Unsupported model type '{model_type}'. Expected 'dense' or 'reranker'.")


def build_export_metadata(
    model_type: str, model_name_or_path: str, opset_version: int
) -> dict[str, Any]:
    """Assemble structured export metadata for validation and logging."""
    input_names, output_names = get_io_tensor_names(model_type)
    dynamic_axes = get_onnx_dynamic_axes(model_type)
    return {
        "model_type": model_type,
        "model_name_or_path": model_name_or_path,
        "opset_version": opset_version,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }


DEFAULT_EXPORT_MODELS = {
    "dense": "BAAI/bge-base-en-v1.5",
    "reranker": "BAAI/bge-reranker-large",
}


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export HuggingFace models to ONNX with baked pooling / normalization for TensorRT."
    )
    parser.add_argument(
        "--model-type",
        choices=["dense", "reranker"],
        required=True,
        help="Type of model: 'dense' (bge-base-en-v1.5) or 'reranker' (bge-reranker-large).",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="HuggingFace repository ID or local checkpoint path (default depends on --model-type).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Target output path for the exported ONNX model file (e.g. model.onnx).",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate parameters and generate dummy model artifact without loading full PyTorch weights.",
    )
    parsed = parser.parse_args(args)
    if parsed.model_name_or_path is None:
        parsed.model_name_or_path = DEFAULT_EXPORT_MODELS.get(
            parsed.model_type, "BAAI/bge-base-en-v1.5"
        )
    return parsed


def export_onnx_model(
    model_type: str,
    model_name_or_path: str,
    output_path: str,
    opset_version: int = 17,
    dry_run: bool = False,
) -> str:
    """Execute ONNX export with baked pooling for dense embedders or logits for rerankers."""
    metadata = build_export_metadata(model_type, model_name_or_path, opset_version)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if dry_run:
        # Write dummy ONNX stub artifact for testing without GPU/PyTorch weights
        with open(output_path, "wb") as f:
            f.write(b"ONNX_MOCK_MODEL_DATA\x08\x01\x12\x04mock")
        logger.info(
            "Dry-run ONNX export succeeded",
            extra={
                "model_name": model_name_or_path,
                "model_type": model_type,
                "status": "dry_run_success",
            },
        )
        return output_path

    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModel,
            AutoModelForSequenceClassification,
        )
    except ImportError as e:
        raise RuntimeError(
            "torch and transformers are required for live ONNX export. Install them or use --dry-run."
        ) from e

    logger.info(
        "Beginning ONNX model serialization",
        extra={
            "model_name": model_name_or_path,
            "model_type": model_type,
            "status": "export_started",
        },
    )

    input_names = metadata["input_names"]
    output_names = metadata["output_names"]
    dynamic_axes = metadata["dynamic_axes"]

    if model_type == "dense":
        base_model = AutoModel.from_pretrained(model_name_or_path)
        base_model.eval()

        class DenseEmbeddingWithPooling(torch.nn.Module):
            """Module baking Mean Pooling and L2 Normalization directly into the ONNX graph."""

            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(
                self, input_ids: torch.Tensor, attention_mask: torch.Tensor
            ) -> torch.Tensor:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, 768]
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                )
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
                sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                norm = torch.norm(mean_pooled, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                return mean_pooled / norm  # [batch, 768]

        wrapper = DenseEmbeddingWithPooling(base_model)
        wrapper.eval()
        dummy_input_ids = torch.ones((1, 16), dtype=torch.int64)
        dummy_attention_mask = torch.ones((1, 16), dtype=torch.int64)
        dummy_dense_args = (dummy_input_ids, dummy_attention_mask)

        torch.onnx.export(
            wrapper,
            dummy_dense_args,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    else:
        # Cross-Encoder Reranker
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        base_model.eval()

        class CrossEncoderWrapper(torch.nn.Module):
            """Module returning scalar classification logits."""

            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
            ) -> torch.Tensor:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                return outputs.logits  # [batch, 1]

        wrapper_rerank = CrossEncoderWrapper(base_model)
        wrapper_rerank.eval()
        dummy_rerank_input_ids = torch.ones((1, 32), dtype=torch.int64)
        dummy_rerank_attention_mask = torch.ones((1, 32), dtype=torch.int64)
        dummy_token_type_ids = torch.zeros((1, 32), dtype=torch.int64)
        dummy_rerank_args = (
            dummy_rerank_input_ids,
            dummy_rerank_attention_mask,
            dummy_token_type_ids,
        )

        torch.onnx.export(
            wrapper_rerank,
            dummy_rerank_args,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    logger.info(
        "Successfully exported ONNX model artifact",
        extra={
            "output_path": output_path,
            "model_type": model_type,
            "status": "export_completed",
        },
    )
    return output_path


def main() -> None:
    args = parse_args()
    export_onnx_model(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        output_path=args.output_path,
        opset_version=args.opset_version,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
