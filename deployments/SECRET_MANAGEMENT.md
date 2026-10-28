# Secret Management & CI/CD Secret Injection

This document describes the secret management architecture and automated CI/CD secret injection pipeline for the `rag-inference-serving` microservice.

---

## 1. Architectural Model: ConfigMap vs. Secret Decoupling

To ensure zero credential leakage and adhere to enterprise compliance, runtime configuration is bifurcated:

1. **Non-Sensitive Configurations (Kubernetes ConfigMap)**:
   - Sourced directly from `.env.prod`.
   - Synchronized to `deployments/helm/values.yaml` via `python scripts/sync_env_to_deployments.py --sync`.
   - Enforced by pre-commit hooks and GitHub Actions CI drift verification (`--check-only`).
   - Pods restart automatically via `checksum/config` sha256 annotation in `deployments/helm/templates/deployment.yaml`.

2. **Sensitive Credentials (Kubernetes Secret)**:
   - Never stored in `.env.prod`, `values.yaml`, or Git history.
   - Automatically excluded by `scripts/sync_env_to_deployments.py` matching `PASSWORD`, `SECRET`, `TOKEN`, `KEY`.
   - Injected dynamically at deployment time via GitHub Actions (`.github/workflows/deploy.yml`) into the Kubernetes cluster.
   - Injected into the container via `envFrom.secretRef` pointing to `rag-inference-secrets`.

---

## 2. GitHub Actions Secrets Inventory

Configure the following secrets in **GitHub Repository Settings &rarr; Secrets and variables &rarr; Actions**:

| Secret Name | Purpose | Target Microservice Env Variable |
| :--- | :--- | :--- |
| `KUBECONFIG` | Base64 or raw kubeconfig granting deployment access to target cluster | *Deployment context only* |
| `PROD_HF_TOKEN` | Hugging Face token for downloading gated/private weights during export | `HF_TOKEN` |
| `PROD_AWS_ACCESS_KEY_ID` | Storage access key for AWS S3, MinIO, or Cloudflare R2 | `AWS_ACCESS_KEY_ID` |
| `PROD_AWS_SECRET_ACCESS_KEY` | Storage secret key for AWS S3, MinIO, or Cloudflare R2 | `AWS_SECRET_ACCESS_KEY` |
| `PROD_RAG_SERVING_API_KEY` | Bearer token securing administrative endpoints (`/metrics`) | `RAG_SERVING_API_KEY` |
| `PROD_KAFKA_SASL_PASSWORD` | Password for Kafka SCRAM-SHA-512 authentication (KEDA triggers) | `KAFKA_SASL_PASSWORD` |

---

## 3. Automated Injection Flow (`.github/workflows/deploy.yml`)

When code merges into `main`, `master`, or `rag-inference-serving`:

```mermaid
flowchart LR
    A["GitHub Push / Dispatch"] --> B["GitHub Actions Runner"]
    B --> C["Fetch GitHub Actions Secrets"]
    C --> D["kubectl create secret generic rag-inference-secrets"]
    D --> E["helm upgrade --install ... --set env.secretName=rag-inference-secrets"]
    E --> F["Pod mounts ConfigMap + Secret via envFrom"]
    F --> G["kubectl rollout status validation"]
```

### Injection Command:
```bash
kubectl create secret generic rag-inference-secrets \
  --namespace rag-inference-serving \
  --from-literal=HF_TOKEN="${{ secrets.PROD_HF_TOKEN }}" \
  --from-literal=AWS_ACCESS_KEY_ID="${{ secrets.PROD_AWS_ACCESS_KEY_ID }}" \
  --from-literal=AWS_SECRET_ACCESS_KEY="${{ secrets.PROD_AWS_SECRET_ACCESS_KEY }}" \
  --from-literal=RAG_SERVING_API_KEY="${{ secrets.PROD_RAG_SERVING_API_KEY }}" \
  --from-literal=KAFKA_SASL_PASSWORD="${{ secrets.PROD_KAFKA_SASL_PASSWORD }}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

---

## 4. Pod-Level Consumption

In `deployments/helm/templates/deployment.yaml`:
```yaml
containers:
  - name: inference-worker
    envFrom:
      - configMapRef:
          name: {{ .Release.Name }}-config
      {{- if .Values.env.secretName }}
      - secretRef:
          name: {{ .Values.env.secretName }}
      {{- end }}
```

When `env.secretName` is set to `rag-inference-secrets`, Kubernetes injects all secret keys directly into the process environment without exposing their values in manifests.
