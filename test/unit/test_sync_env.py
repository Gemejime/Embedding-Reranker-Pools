"""Unit tests for scripts/sync_env_to_deployments.py."""

import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.sync_env_to_deployments import (
    DEFAULT_ENV_FILE,
    HELM_VALUES_FILE,
    check_drift,
    normalize_val,
    parse_env_file,
    sync_to_deployments,
)


class TestSyncEnv(unittest.TestCase):
    def test_parse_env_file_excludes_secrets_and_handles_comments(self) -> None:
        sample_env = (
            "# Top level comment\n"
            "ENVIRONMENT=production\n"
            "PORT=8080 # Inline comment\n"
            'QUOTED_VAL="hello # world" # trailing comment\n'
            "SINGLE_QUOTED='single val' # trailing\n"
            "EMPTY_VAL=\n"
            "APP_PASSWORD=supersecret\n"
            "API_SECRET=topsecret\n"
            "AUTH_TOKEN=bearer123\n"
            "PRIVATE_KEY=pk123\n"
            "RAG_SERVING_GRPC_CLIENT_KEY_PATH=/etc/tls/client.key\n"
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            tf.write(sample_env)
            tf_path = Path(tf.name)

        try:
            parsed = parse_env_file(tf_path)
            self.assertEqual(parsed["ENVIRONMENT"], "production")
            self.assertEqual(parsed["PORT"], "8080")
            self.assertEqual(parsed["QUOTED_VAL"], "hello # world")
            self.assertEqual(parsed["SINGLE_QUOTED"], "single val")
            self.assertEqual(parsed["EMPTY_VAL"], "")
            self.assertEqual(parsed["RAG_SERVING_GRPC_CLIENT_KEY_PATH"], "/etc/tls/client.key")

            # Verify secrets are excluded while path setting is retained
            self.assertNotIn("APP_PASSWORD", parsed)
            self.assertNotIn("API_SECRET", parsed)
            self.assertNotIn("AUTH_TOKEN", parsed)
            self.assertNotIn("PRIVATE_KEY", parsed)
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_normalize_val(self) -> None:
        self.assertEqual(normalize_val(None), "")
        self.assertEqual(normalize_val(True), "true")
        self.assertEqual(normalize_val(False), "false")
        self.assertEqual(normalize_val(8080), "8080")
        self.assertEqual(normalize_val("online_embedding"), "online_embedding")

    def test_check_drift_detection(self) -> None:
        env_vars = {"A": "1", "B": "2"}
        helm_env = {"A": "1", "B": "different", "secretName": "my-secret"}

        drift = check_drift(env_vars, helm_env)
        self.assertEqual(len(drift), 1)
        self.assertIn("Mismatch for 'B'", drift[0])

        # Missing key in helm_env
        drift_missing = check_drift({"A": "1", "C": "3"}, helm_env)
        self.assertTrue(any("Missing key in values.yaml: 'C'" in item for item in drift_missing))

        # Unexpected extra key in helm_env (excluding secretName)
        drift_extra = check_drift(
            {"A": "1"}, {"A": "1", "EXTRA_KEY": "val", "secretName": "my-sec"}
        )
        self.assertEqual(len(drift_extra), 1)
        self.assertIn("Unexpected extra key in values.yaml: 'EXTRA_KEY'", drift_extra[0])

    def test_sync_to_deployments_preserves_secret_name_and_structure(self) -> None:
        sample_values = {
            "image": {"repository": "rag-inference-serving", "tag": "1.0.0"},
            "replicaCount": 4,
            "resources": {"limits": {"cpu": "8"}},
            "env": {
                "OLD_KEY": "old",
                "secretName": "existing-k8s-secret",
            },
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            yaml.dump(sample_values, tf)
            tf_path = Path(tf.name)

        try:
            new_env = {"NEW_VAR": "val_123"}
            sync_to_deployments(new_env, target_file=tf_path)

            with open(tf_path, encoding="utf-8") as f:
                updated = yaml.safe_load(f)

            self.assertEqual(updated["replicaCount"], 4)
            self.assertEqual(updated["image"]["repository"], "rag-inference-serving")
            self.assertEqual(updated["resources"]["limits"]["cpu"], "8")
            self.assertEqual(updated["env"]["NEW_VAR"], "val_123")
            self.assertEqual(updated["env"]["secretName"], "existing-k8s-secret")
            self.assertNotIn("OLD_KEY", updated["env"])
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_repository_deployments_zero_drift(self) -> None:
        """Assert repository values.yaml has zero drift against .env.prod."""
        self.assertTrue(DEFAULT_ENV_FILE.exists(), ".env.prod must exist")
        self.assertTrue(HELM_VALUES_FILE.exists(), "deployments/helm/values.yaml must exist")

        env_vars = parse_env_file(DEFAULT_ENV_FILE)
        with open(HELM_VALUES_FILE, encoding="utf-8") as f:
            values_data = yaml.safe_load(f) or {}
        helm_env = values_data.get("env", {}) or {}

        drift = check_drift(env_vars, helm_env)
        self.assertEqual(
            drift,
            [],
            f"Zero drift expected between .env.prod and values.yaml, found: {drift}",
        )

    def test_sync_to_deployments_emits_structured_json(self) -> None:
        import io
        import json
        from contextlib import redirect_stdout

        sample_values = {"env": {"secretName": "my-secret"}}
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            yaml.dump(sample_values, tf)
            tf_path = Path(tf.name)

        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                sync_to_deployments({"TEST_KEY": "123"}, target_file=tf_path)

            output = buf.getvalue().strip()
            record = json.loads(output)
            self.assertEqual(record["status"], "success")
            self.assertEqual(record["synced_count"], 1)
            self.assertEqual(record["target_file"], str(tf_path))
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_missing_source_check_only_exits_with_error(self) -> None:
        import io
        import json
        import sys
        from contextlib import redirect_stdout
        from unittest.mock import patch

        from scripts.sync_env_to_deployments import main

        nonexistent_path = Path("/path/to/definitely/nonexistent/.env.test")
        test_args = [
            "sync_env_to_deployments.py",
            "--env-file",
            str(nonexistent_path),
            "--check-only",
        ]

        buf = io.StringIO()
        with patch.object(sys, "argv", test_args), redirect_stdout(buf):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertEqual(ctx.exception.code, 1)

        output = buf.getvalue().strip()
        record = json.loads(output)
        self.assertEqual(record["status"], "error")
        self.assertIn("does not exist", record["message"])


if __name__ == "__main__":
    unittest.main()
