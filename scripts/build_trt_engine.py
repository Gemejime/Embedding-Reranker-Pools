"""Generate and execute trtexec TensorRT FP16 engine compilation commands.

Enforces calibrated shape profiles across compute pools:
1. Online Embedder:
   - min: [1, 1]
   - opt: [16, 128]
   - max: [32, 512]
2. Online Reranker:
   - min: [1, 1]
   - opt: [16, 256]
   - max: [32, 2048]
3. Batch Ingestion:
   - min: [1, 1]
   - opt: [128, 256]
   - max: [256, 512]

Supports --dry-run and mock compilation for GPU-less CI validation.
"""

import argparse
import os
import shutil
import subprocess

from internal.logging.secured_logger import get_secured_logger

logger = get_secured_logger("build-trt-engine")

SHAPE_PROFILES: dict[str, dict[str, str]] = {
    "online_embed": {
        "min": "input_ids:1x1,attention_mask:1x1",
        "opt": "input_ids:16x128,attention_mask:16x128",
        "max": "input_ids:32x512,attention_mask:32x512",
    },
    "online_rerank": {
        "min": "input_ids:1x1,attention_mask:1x1,token_type_ids:1x1",
        "opt": "input_ids:16x256,attention_mask:16x256,token_type_ids:16x256",
        "max": "input_ids:32x2048,attention_mask:32x2048,token_type_ids:32x2048",
    },
    "batch_ingestion": {
        "min": "input_ids:1x1,attention_mask:1x1",
        "opt": "input_ids:128x256,attention_mask:128x256",
        "max": "input_ids:256x512,attention_mask:256x512",
    },
}


def get_shape_profile(profile_name: str) -> dict[str, str]:
    """Retrieve calibrated min/opt/max shape profile strings."""
    norm_name = profile_name.lower().strip()
    if norm_name not in SHAPE_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile_name}'. Expected one of {list(SHAPE_PROFILES.keys())}."
        )
    return SHAPE_PROFILES[norm_name]


def build_trtexec_cmd(
    onnx_path: str,
    output_engine_path: str,
    profile_name: str,
    workspace_mb: int = 4096,
) -> list[str]:
    """Assemble complete trtexec compilation command line."""
    profile = get_shape_profile(profile_name)
    return [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_engine_path}",
        "--fp16",
        f"--minShapes={profile['min']}",
        f"--optShapes={profile['opt']}",
        f"--maxShapes={profile['max']}",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--verbose",
    ]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for TensorRT engine builder."""
    parser = argparse.ArgumentParser(
        description="Compile ONNX model to TensorRT FP16 engine via trtexec."
    )
    parser.add_argument(
        "--profile",
        choices=["online_embed", "online_rerank", "batch_ingestion"],
        required=True,
        help="Target pool profile defining calibrated dynamic batch shape boundaries.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        required=True,
        help="Input ONNX model filepath.",
    )
    parser.add_argument(
        "--output-engine-path",
        type=str,
        required=True,
        help="Output filepath for compiled TensorRT engine (e.g. model.plan).",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=4096,
        help="TensorRT builder workspace size in megabytes (default: 4096).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print trtexec command and write mock engine without running trtexec.",
    )
    return parser.parse_args(args)


def compile_engine(
    profile_name: str,
    onnx_path: str,
    output_engine_path: str,
    workspace_mb: int = 4096,
    dry_run: bool = False,
) -> str:
    """Execute or simulate TensorRT FP16 compilation."""
    cmd = build_trtexec_cmd(onnx_path, output_engine_path, profile_name, workspace_mb)
    cmd_str = " ".join(cmd)

    os.makedirs(os.path.dirname(os.path.abspath(output_engine_path)), exist_ok=True)

    if dry_run:
        # Write deterministic mock .plan engine artifact for CI validation
        with open(output_engine_path, "wb") as f:
            f.write(b"TENSORRT_FP16_ENGINE_PLAN_BINARY\x00\x00\x01\x00")
        logger.info(
            "Dry-run TensorRT compilation succeeded",
            extra={
                "profile": profile_name,
                "command": cmd_str,
                "output_engine_path": output_engine_path,
                "status": "dry_run_success",
            },
        )
        return output_engine_path

    if not shutil.which("trtexec"):
        raise FileNotFoundError(
            "trtexec executable not found in PATH. Install NVIDIA TensorRT or specify --dry-run for simulation."
        )

    logger.info(
        "Invoking trtexec for TensorRT engine compilation",
        extra={"profile": profile_name, "command": cmd_str, "status": "compiling"},
    )
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    logger.info(
        "TensorRT compilation completed successfully",
        extra={"output_engine_path": output_engine_path, "status": "compiled"},
    )
    return output_engine_path


def main() -> None:
    args = parse_args()
    compile_engine(
        profile_name=args.profile,
        onnx_path=args.onnx_path,
        output_engine_path=args.output_engine_path,
        workspace_mb=args.workspace_mb,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
