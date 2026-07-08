# Stage 5 — Live Regime Detector

This stage runs the trained model **in production**. Every 15 minutes it:

1. Pulls the latest BTC indicator data from Prometheus
2. Computes the same momentum-state features used during training (Stage 3)
3. Runs the saved LightGBM classifier (Stage 4)
4. Publishes the current market regime

**You do not need to understand machine learning to operate this stage.** Think of
it as a weather app for the market: it tells you what *kind* of market you're in,
not whether to buy or sell.

---

## What problem does this solve?

Your trading bots need to know the **market regime** before choosing a strategy:

| Regime | Plain English | Bot behavior |
|--------|---------------|--------------|
| **CHOP** | Sideways, no clear direction | Mean-reversion or sit out |
| **TRENDING_UP** | Sustained upward momentum | Trend-following longs |
| **TRENDING_DOWN** | Sustained downward momentum | Shorts or defensive exits |
| **VOLATILE_EXPANSION** | Fast, chaotic moves | Reduce size, volatility tactics |

Stages 1–4 **trained** the model offline. Stage 5 **runs** that model live.

---

## How much history does it need?

The model uses **15-minute bars**. At each prediction it looks back at recent history
to compute derivatives and momentum states.

| Requirement | Value |
|-------------|-------|
| **Minimum** | **384 bars ≈ 96 hours (4 days)** — needed for the longest math window |
| **Recommended** | **8 days** — buffer for gaps and accurate state-duration counts |
| **Bar interval** | 900 seconds (15 minutes) — must match training |

It pulls four time series from Prometheus:

- `price` (logged only, not fed to the model)
- `weighted_norm_avg_16h_24h_48h`
- `weighted_deriv_24h_48h_7d`
- `norm_combined_avg`

---

## What does the output look like?

Each prediction cycle produces JSON like:

```json
{
  "timestamp": "2026-04-15T00:15:18+00:00",
  "price": 84250.0,
  "regime": "TRENDING_UP",
  "confidence": 0.83,
  "prob_chop": 0.05,
  "prob_trending_up": 0.83,
  "prob_trending_down": 0.08,
  "prob_volatile_expansion": 0.04,
  "bars_fetched": 768,
  "bars_usable": 385
}
```

### How to read this (beginner guide)

- **`regime`** — the model's best guess (the class with highest probability)
- **`confidence`** — how sure the model is (0.0 to 1.0). Higher = more reliable
- **`prob_*`** — full probability distribution (always sums to 1.0)
- **`price`** — BTC price at that timestamp (for your dashboards)

**Recommended filter:** only act on predictions where `confidence > 0.8`. In
backtests, those were ~92% accurate.

**This is a signal, not a command.** Your bots decide what to do with it.

---

## Key concept: frozen encoder thresholds

Stage 3 learns numeric cutoffs ("thresholds") from historical data — things like
"how small does an indicator need to be before we call it noise?"

Those cutoffs were computed during training and saved in
`artifacts/encoder_artifacts.json`. The live service **loads** them; it does
**not** re-learn them on every poll. That keeps live behavior identical to
training.

If you retrain the model or change Stage 3 settings, re-export encoder artifacts:

```bash
python3 scripts/export_encoder_artifacts.py
```

---

## Files in this directory

| File / folder | Purpose |
|---------------|---------|
| `src/service.py` | Main loop + HTTP API |
| `src/predict.py` | Fetch → encode → classify orchestration |
| `src/prometheus_fetch.py` | Pulls data from Prometheus |
| `src/encoder.py` | Applies frozen Stage 3 features |
| `src/classifier.py` | Runs the LightGBM model |
| `src/telemetry.py` | Exposes Prometheus metrics |
| `lib/momentum_state.py` | Vendored copy of Stage 3 encoder |
| `artifacts/` | `model.txt`, `feature_cols.json`, `encoder_artifacts.json` |
| `config.yaml` | Prometheus URL, poll interval, ports |
| `scripts/run_once.py` | Test one prediction locally |
| `scripts/export_encoder_artifacts.py` | One-time threshold export |
| `scripts/package_artifacts.py` | Copy model from Stage 4 |
| `k8s/` | Kubernetes manifests |
| `HOW_TO_DEPLOY.md` | Step-by-step deployment guide |

---

## Endpoints

When the service is running:

| URL | Purpose |
|-----|---------|
| `GET :8080/health` | Liveness — is the process alive? |
| `GET :8080/ready` | Readiness — has it completed at least one prediction? |
| `GET :8080/prediction` | Latest prediction JSON |
| `GET :9109/metrics` | Prometheus exporter — scrape for Cortex storage |

The exporter is built in. Prometheus scrapes it every ~60s; the model runs every
15 minutes and updates the gauges with the latest prediction.

---

## What this does NOT do

- Does not retrain the model
- Does not place trades
- Does not predict price direction
- Does not guarantee accuracy (test accuracy is ~70%)

---

## Further reading

- [HOW_TO_DEPLOY.md](HOW_TO_DEPLOY.md) — local testing, Docker, Kubernetes
- [../HOW_TO_RUN.md](../HOW_TO_RUN.md) — training pipeline (Stages 1–4)
- [../4_Classifier/WHAT_THIS_DOES.md](../4_Classifier/WHAT_THIS_DOES.md) — model details
