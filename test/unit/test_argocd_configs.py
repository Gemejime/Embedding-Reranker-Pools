"""Unit tests validating ArgoCD GitOps Application and ApplicationSet manifests."""

import os
import unittest

import yaml  # type: ignore[import-untyped]


class TestArgoCDConfigs(unittest.TestCase):
    def setUp(self) -> None:
        self.argocd_dir = os.path.join("deployments", "argocd")
        self.assertTrue(os.path.isdir(self.argocd_dir), "deployments/argocd must exist")

    def test_applicationset_manifest(self) -> None:
        appset_path = os.path.join(self.argocd_dir, "applicationset.yaml")
        with open(appset_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["apiVersion"], "argoproj.io/v1alpha1")
        self.assertEqual(data["kind"], "ApplicationSet")
        self.assertEqual(data["metadata"]["name"], "rag-inference-serving-pools")

        # Verify generator pools
        generators = data["spec"]["generators"]
        elements = generators[0]["list"]["elements"]
        self.assertEqual(len(elements), 3)

        pools = {elem["pool"]: elem for elem in elements}
        self.assertIn("online-embedding", pools)
        self.assertIn("online-reranker", pools)
        self.assertIn("batch-ingestion", pools)

        self.assertEqual(pools["online-embedding"]["nodeLifecycle"], "on-demand")
        self.assertEqual(pools["online-reranker"]["nodeLifecycle"], "on-demand")
        self.assertEqual(pools["batch-ingestion"]["nodeLifecycle"], "spot")

        # Verify Helm path and sync policy
        template = data["spec"]["template"]["spec"]
        self.assertEqual(template["source"]["path"], "deployments/helm")
        self.assertTrue(template["syncPolicy"]["automated"]["prune"])
        self.assertTrue(template["syncPolicy"]["automated"]["selfHeal"])
        self.assertIn("CreateNamespace=true", template["syncPolicy"]["syncOptions"])

    def test_standalone_application_online_embed(self) -> None:
        path = os.path.join(self.argocd_dir, "application-online-embed.yaml")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["kind"], "Application")
        self.assertEqual(data["metadata"]["name"], "rag-inference-online-embedding")
        self.assertEqual(data["metadata"]["labels"]["lifecycle"], "on-demand")
        self.assertIn("values-online-embed.yaml", data["spec"]["source"]["helm"]["valueFiles"])
        self.assertEqual(data["spec"]["destination"]["namespace"], "rag-inference-serving")
        self.assertTrue(data["spec"]["syncPolicy"]["automated"]["prune"])

    def test_standalone_application_online_rerank(self) -> None:
        path = os.path.join(self.argocd_dir, "application-online-rerank.yaml")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["kind"], "Application")
        self.assertEqual(data["metadata"]["name"], "rag-inference-online-reranker")
        self.assertEqual(data["metadata"]["labels"]["lifecycle"], "on-demand")
        self.assertIn("values-online-rerank.yaml", data["spec"]["source"]["helm"]["valueFiles"])
        self.assertEqual(data["spec"]["destination"]["namespace"], "rag-inference-serving")
        self.assertTrue(data["spec"]["syncPolicy"]["automated"]["prune"])

    def test_standalone_application_batch_ingestion(self) -> None:
        path = os.path.join(self.argocd_dir, "application-batch-ingestion.yaml")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["kind"], "Application")
        self.assertEqual(data["metadata"]["name"], "rag-inference-batch-ingestion")
        self.assertEqual(data["metadata"]["labels"]["lifecycle"], "spot")
        self.assertIn("values-batch-ingestion.yaml", data["spec"]["source"]["helm"]["valueFiles"])
        self.assertEqual(data["spec"]["destination"]["namespace"], "rag-inference-serving")
        self.assertTrue(data["spec"]["syncPolicy"]["automated"]["prune"])

    def test_rollout_canary_manifest(self) -> None:
        path = os.path.join(self.argocd_dir, "rollout-canary.yaml")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["kind"], "Rollout")
        canary = data["spec"]["strategy"]["canary"]
        steps = canary["steps"]
        weights = [s["setWeight"] for s in steps if "setWeight" in s]
        self.assertEqual(weights, [20, 50, 100])

    def test_analysistemplate_manifest(self) -> None:
        path = os.path.join(self.argocd_dir, "analysistemplate.yaml")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["kind"], "AnalysisTemplate")
        self.assertEqual(data["metadata"]["name"], "triton-vram-and-latency-check")
        arg_names = [a["name"] for a in data["spec"]["args"]]
        self.assertIn("service-name", arg_names)
        metric_names = [m["name"] for m in data["spec"]["metrics"]]
        self.assertIn("success-rate", metric_names)


if __name__ == "__main__":
    unittest.main()
