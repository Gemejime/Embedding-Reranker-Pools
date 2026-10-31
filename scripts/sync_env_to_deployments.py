"""scripts/sync_env_to_deployments.py

Synchronizes and validates environment configuration between .env.prod and
Kubernetes/Helm deployment manifests (deployments/helm/values.yaml).

Usage:
    python scripts/sync_env_to_deployments.py --sync
    python scripts/sync_env_to_deployments.py --check-only
    python scripts/sync_env_to_deployments.py --env-file .env.prod --check-only
"""

import argparse
import contextlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure UTF-8 output on Windows consoles
reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_stdout):
    with contextlib.suppress(Exception):
        reconfigure_stdout(encoding="utf-8")

reconfigure_stderr = getattr(sys.stderr, "reconfigure", None)
if callable(reconfigure_stderr):
    with contextlib.suppress(Exception):
        reconfigure_stderr(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env.prod"
HELM_VALUES_FILE = REPO_ROOT / "deployments" / "helm" / "values.yaml"

# Keys containing secrets that must never be committed into values.yaml
EXCLUDED_SECRET_KEYWORDS = ("PASSWORD", "SECRET", "TOKEN", "KEY")


def parse_env_file(file_path: Path) -> dict[str, str]:
    """Parse .env formatted file into key-value pairs, stripping comments and excluding secrets."""
    env_vars: dict[str, str] = {}
    if not file_path.exists():
        return env_vars

    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_val = line.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()

        # Handle quoted values or inline comments
        if raw_val.startswith('"'):
            end_quote = raw_val.find('"', 1)
            val = raw_val[1:end_quote] if end_quote != -1 else raw_val.strip('"')
        elif raw_val.startswith("'"):
            end_quote = raw_val.find("'", 1)
            val = raw_val[1:end_quote] if end_quote != -1 else raw_val.strip("'")
        else:
            if "#" in raw_val:
                raw_val = raw_val.split("#", 1)[0]
            val = raw_val.strip()

        # Exclude secrets from being committed into Helm values
        # Retain filesystem paths (such as RAG_SERVING_GRPC_CLIENT_KEY_PATH)
        if key == "RAG_SERVING_GRPC_CLIENT_KEY_PATH" or key.endswith("_KEY_PATH"):
            pass
        elif any(keyword in key.upper() for keyword in EXCLUDED_SECRET_KEYWORDS):
            continue

        env_vars[key] = val
    return env_vars


def normalize_val(val: Any) -> str:
    """Normalize yaml/env values to string representations for drift comparison."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def check_drift(env_vars: dict[str, str], helm_env: dict[str, Any]) -> list[str]:
    """Identify discrepancies between environment variables and Helm values."""
    drift: list[str] = []
    for key, expected_val in env_vars.items():
        if key not in helm_env:
            drift.append(f"Missing key in values.yaml: '{key}' (expected '{expected_val}')")
        else:
            actual_val = normalize_val(helm_env[key])
            if actual_val != normalize_val(expected_val):
                drift.append(
                    f"Mismatch for '{key}': values.yaml='{actual_val}' vs env_file='{expected_val}'"
                )

    for key in helm_env:
        if key != "secretName" and key not in env_vars:
            drift.append(f"Unexpected extra key in values.yaml: '{key}'")

    return drift


def sync_to_deployments(env_vars: dict[str, str], target_file: Path = HELM_VALUES_FILE) -> None:
    """Propagate env_vars into deployments/helm/values.yaml while preserving top-level blocks and secretName."""
    with open(target_file, encoding="utf-8") as f:
        values_data = yaml.safe_load(f) or {}

    current_env = values_data.get("env", {}) or {}
    secret_name = current_env.get("secretName", "")

    # Replace env dictionary while preserving secretName
    new_env: dict[str, str] = dict(env_vars)
    if secret_name:
        new_env["secretName"] = secret_name
    elif "secretName" in current_env:
        new_env["secretName"] = ""
    else:
        new_env["secretName"] = ""

    values_data["env"] = new_env

    with open(target_file, "w", encoding="utf-8") as f:
        yaml.dump(values_data, f, sort_keys=False, default_flow_style=False)

    print(
        json.dumps(
            {
                "status": "success",
                "synced_count": len(env_vars),
                "target_file": str(target_file),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync or check .env.prod against Helm deployment manifests"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to environment file (default: .env.prod)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for drift; exit code 1 if out of sync, 0 if clean",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync environment variables directly into deployments/helm/values.yaml",
    )
    args = parser.parse_args()

    env_file = args.env_file
    if not env_file.exists():
        print(
            json.dumps(
                {
                    "status": "error" if args.check_only else "warning",
                    "message": f"Environment file '{env_file}' does not exist.",
                    "file": str(env_file),
                }
            )
        )
        sys.exit(1 if args.check_only else 0)

    env_vars = parse_env_file(env_file)
    if not env_vars:
        print(
            json.dumps(
                {
                    "status": "error" if args.check_only else "warning",
                    "message": f"No valid environment variables found in '{env_file}'.",
                    "file": str(env_file),
                }
            )
        )
        sys.exit(1 if args.check_only else 0)

    with open(HELM_VALUES_FILE, encoding="utf-8") as f:
        values_data = yaml.safe_load(f) or {}
    helm_env = values_data.get("env", {}) or {}

    drift = check_drift(env_vars, helm_env)

    if args.check_only:
        if drift:
            print(
                json.dumps(
                    {
                        "status": "drift_detected",
                        "env_file": env_file.name,
                        "target_file": str(HELM_VALUES_FILE),
                        "drift_count": len(drift),
                        "issues": drift,
                        "recommendation": (
                            f"Run 'python scripts/sync_env_to_deployments.py "
                            f"--env-file {shlex.quote(str(env_file))} --sync' to resolve."
                        ),
                    }
                )
            )
            sys.exit(1)
        else:
            print(
                json.dumps(
                    {
                        "status": "success",
                        "message": f"deployments/helm/values.yaml is in sync with '{env_file.name}'.",
                        "target_file": str(HELM_VALUES_FILE),
                    }
                )
            )
            sys.exit(0)

    if args.sync:
        sync_to_deployments(env_vars, HELM_VALUES_FILE)
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
