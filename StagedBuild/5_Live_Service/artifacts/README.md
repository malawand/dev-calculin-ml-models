# Artifacts directory

This folder holds everything the live service needs at runtime:

| File | Source | Required |
|------|--------|----------|
| `model.txt` | Stage 4 `train.py` | Yes |
| `feature_cols.json` | Stage 4 `train.py` | Yes |
| `encoder_artifacts.json` | `scripts/export_encoder_artifacts.py` | Yes |
| `class_weights.json` | Stage 4 (informational) | No |
| `metrics.json` | Stage 4 (informational) | No |

## One-time setup

```bash
cd StagedBuild/5_Live_Service

# Copy classifier model from Stage 4
python3 scripts/package_artifacts.py

# Export frozen encoder thresholds (requires Stage 2 labeled parquet)
python3 scripts/export_encoder_artifacts.py
```

`encoder_artifacts.json` is **not** optional. Without it the service cannot encode
live data the same way the model was trained.
