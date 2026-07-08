# How to Deploy the Live Regime Detector

Step-by-step guide for running Stage 5 locally, in Docker, and on Kubernetes.

---

## Prerequisites

1. **Stages 1–4 completed** — you need a trained model and labeled training data
2. **Network access** to your Prometheus/Cortex endpoint
3. **Python 3.10+** for local testing
4. **Docker** and **kubectl** for cluster deployment

---

## Step 1 — Package artifacts

From the repo root:

```bash
cd StagedBuild/5_Live_Service
pip install -r requirements.txt

# Copy LightGBM model from Stage 4
python3 scripts/package_artifacts.py

# Export frozen encoder thresholds (needs Stage 2 labeled parquet)
python3 scripts/export_encoder_artifacts.py
```

Verify `artifacts/` contains:

```
artifacts/
  model.txt
  feature_cols.json
  encoder_artifacts.json    ← required; created by export script
```

If `export_encoder_artifacts.py` fails with "Input not found", run Stages 1–2
first (see [../HOW_TO_RUN.md](../HOW_TO_RUN.md)).

---

## Step 2 — Configure Prometheus

Edit `config.yaml`:

```yaml
prometheus:
  query_range_url: "http://YOUR-PROMETHEUS:9009/prometheus/api/v1/query_range"
  lookback_days: 8
  step_seconds: 900
```

| Setting | What to change |
|---------|----------------|
| `query_range_url` | Your Prometheus `query_range` endpoint |
| `lookback_days` | Keep at 8 unless you have data gaps (minimum effective: 4) |
| `step_seconds` | Must stay **900** (15-minute bars) |

---

## Step 3 — Test locally (one prediction)

```bash
python3 scripts/run_once.py
```

Expected: JSON printed with `regime`, `confidence`, and probabilities.

Common errors:

| Error | Fix |
|-------|-----|
| `encoder_artifacts.json` not found | Run `export_encoder_artifacts.py` |
| `Need at least 384 bars` | Increase `lookback_days` or check Prometheus data |
| Connection error | Fix `query_range_url`, verify network |
| Missing feature columns | Prometheus queries returning empty — check metric names |

---

## Step 4 — Run the service locally

```bash
python3 src/service.py --config config.yaml
```

- Polls every **900 seconds** (15 minutes)
- API on **http://localhost:8080**
- Metrics on **http://localhost:9109/metrics**

Test endpoints:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/ready
curl http://localhost:8080/prediction
```

---

## Step 5 — Build Docker image

```bash
cd StagedBuild/5_Live_Service
docker build -t btc-regime-detector:latest .
```

Run locally:

```bash
docker run --rm -p 8080:8080 -p 9109:9109 \
  -v "$(pwd)/config.yaml:/app/config.yaml" \
  btc-regime-detector:latest
```

The image bundles `artifacts/` at build time. Rebuild after retraining.

---

## Step 6 — Deploy to Kubernetes

### 6a. Push image to your registry

```bash
docker tag btc-regime-detector:latest YOUR-REGISTRY/btc-regime-detector:v1
docker push YOUR-REGISTRY/btc-regime-detector:v1
```

Update `k8s/deployment.yaml` image field:

```yaml
image: YOUR-REGISTRY/btc-regime-detector:v1
```

### 6b. Edit ConfigMap if needed

Update `k8s/configmap.yaml` with your Prometheus URL if it differs from the default.

### 6c. Apply manifests

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# If you use Prometheus Operator (kube-prometheus-stack):
kubectl apply -f k8s/servicemonitor.yaml
```

### 6d. Verify

```bash
kubectl get pods -l app=btc-regime-detector
kubectl logs -l app=btc-regime-detector -f

# Port-forward API
kubectl port-forward svc/btc-regime-detector 8080:8080
curl http://localhost:8080/prediction

# Port-forward Prometheus exporter
kubectl port-forward svc/btc-regime-detector 9109:9109
curl http://localhost:9109/metrics | grep btc_regime_detector
```

Readiness probe waits for the first successful prediction (~30–60s after start,
depending on Prometheus latency).

---

## Prometheus exporter → Cortex

The service **exports** prediction results as Prometheus metrics on port **9109**.
Prometheus scrapes `/metrics`; Cortex stores the time series via remote-write or
federation. You do **not** need a separate exporter process — it is built into
`src/telemetry.py` and starts with the main service.

### Data flow

```
Prediction loop (every 15m)
    → updates gauges in memory
Prometheus scraper (every 60s)
    → GET :9109/metrics
    → remote_write / federation
Cortex
    → long-term storage
Grafana / Alertmanager / your bots
    → PromQL queries
```

The scrape interval (60s) can be shorter than the prediction interval (15m).
Between predictions, metrics hold the **last successful values** — that is
intentional.

### Enable scraping

**Option A — Prometheus Operator**

Apply `k8s/servicemonitor.yaml`. If your Prometheus instance uses label selectors,
add the matching `release: prometheus` label to the ServiceMonitor metadata.

**Option B — Static Prometheus config**

See `k8s/prometheus-scrape.example.yaml` or rely on pod annotations already set
on the Deployment:

```yaml
prometheus.io/scrape: "true"
prometheus.io/port: "9109"
prometheus.io/path: "/metrics"
```

**Option C — Manual test**

```bash
kubectl port-forward svc/btc-regime-detector 9109:9109
curl -s localhost:9109/metrics | grep btc_regime_detector
```

### Exported metrics

All metrics include `symbol="BTCUSDT"` (configurable via `telemetry.symbol`).

| Metric | Type | Meaning |
|--------|------|---------|
| `btc_regime_detector_regime_value{symbol}` | Gauge | 0=CHOP, 1=UP, 2=DOWN, 3=VOLATILE |
| `btc_regime_detector_confidence{symbol}` | Gauge | Max class probability (0–1) |
| `btc_regime_detector_prob{symbol,regime}` | Gauge | Per-class probability |
| `btc_regime_detector_last_timestamp{symbol}` | Gauge | Unix time of the predicted bar |
| `btc_regime_detector_last_price{symbol}` | Gauge | BTC price at that bar |
| `btc_regime_detector_last_run_success{symbol}` | Gauge | 1=last cycle OK, 0=failed |
| `btc_regime_detector_last_run_unix{symbol}` | Gauge | When the loop last finished |
| `btc_regime_detector_bars_fetched{symbol}` | Gauge | Bars pulled from Prometheus |
| `btc_regime_detector_bars_usable{symbol}` | Gauge | Rows with complete features |
| `btc_regime_detector_prediction_runs_total{symbol,status}` | Counter | success / failure counts |

### Useful PromQL (Grafana / Cortex)

**Current regime name** (map integer to label in Grafana value mappings, or use):

```promql
btc_regime_detector_regime_value{symbol="BTCUSDT"}
```

**High-confidence trending up:**

```promql
btc_regime_detector_regime_value{symbol="BTCUSDT"} == 1
and btc_regime_detector_confidence{symbol="BTCUSDT"} > 0.8
```

**Staleness alert** (no fresh prediction bar in 20 minutes):

```promql
time() - btc_regime_detector_last_timestamp{symbol="BTCUSDT"} > 1200
```

**Prediction loop failing:**

```promql
btc_regime_detector_last_run_success{symbol="BTCUSDT"} == 0
```

**Probability of volatile expansion:**

```promql
btc_regime_detector_prob{symbol="BTCUSDT", regime="VOLATILE_EXPANSION"}
```

### Example Alertmanager rule

```yaml
groups:
  - name: btc-regime-detector
    rules:
      - alert: RegimeDetectorStale
        expr: time() - btc_regime_detector_last_timestamp{symbol="BTCUSDT"} > 1200
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Regime detector has not updated in over 20 minutes"

      - alert: RegimeDetectorFailing
        expr: btc_regime_detector_last_run_success{symbol="BTCUSDT"} == 0
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Regime detector prediction loop is failing"
```

---

## Updating after retraining

When you change anything upstream:

| Changed | Action |
|---------|--------|
| Stage 4 model | `package_artifacts.py` → rebuild Docker image → redeploy |
| Stage 3 encoder settings | `export_encoder_artifacts.py` → rebuild → redeploy |
| Prometheus URL only | Edit ConfigMap → `kubectl apply` → restart pod |
| Poll interval | Edit ConfigMap `poll_interval_seconds` |

---

## Consuming the prediction from another service

### HTTP (simplest)

Poll `GET http://btc-regime-detector:8080/prediction` from your bot every 15 minutes.

### Python example

```python
import requests

resp = requests.get("http://btc-regime-detector:8080/prediction", timeout=10)
data = resp.json()

if data["confidence"] > 0.8:
    regime = data["regime"]
    # switch bot strategy based on regime
```

### Prometheus

Query from Grafana or Alertmanager against Cortex/Prometheus:

```promql
btc_regime_detector_regime_value{symbol="BTCUSDT"}
btc_regime_detector_confidence{symbol="BTCUSDT"}
```

See [HOW_TO_DEPLOY.md](HOW_TO_DEPLOY.md#prometheus-exporter--cortex) for scrape
setup, full metric list, PromQL examples, and alert rules.

---

## Resource sizing

Default requests/limits in `deployment.yaml`:

| Resource | Request | Limit |
|----------|---------|-------|
| Memory | 256Mi | 512Mi |
| CPU | 100m | 500m |

LightGBM inference is lightweight; one replica is sufficient.

---

## Troubleshooting

**Pod stuck Not Ready**
→ Check logs for Prometheus errors. First prediction must succeed before `/ready` returns 200.

**Regime always CHOP with low confidence**
→ Normal when the model is uncertain. Check `prob_*` distribution.

**Predictions differ from backtest**
→ Ensure `encoder_artifacts.json` was exported from the same training data era.
→ Ensure `step_seconds: 900` matches training bars.

**lib/momentum_state.py out of date**
→ Re-copy from Stage 3 when encoder logic changes:
  `cp ../3_Momentum_State_Encoder/momentum_state.py lib/`
