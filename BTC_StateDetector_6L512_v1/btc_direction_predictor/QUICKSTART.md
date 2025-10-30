# Quick Start Guide

## ✅ System Validated

**Smoke test results:**
- ✅ Prometheus connection: WORKING
- ✅ Real BTC data fetched: $105,159.99
- ✅ All dependencies: INSTALLED
- ✅ Pipeline modules: READY

---

## 🚀 Run Your First Training

### 1. Activate Environment
```bash
cd btc_direction_predictor
source venv/bin/activate
```

### 2. Quick Test (2 hours of data, 1 horizon)
```bash
# Test with minimal data first
python quick_train.py
```

This will:
- Fetch 2 hours of BTC data
- Create features
- Train a LightGBM model for 1h horizon
- Generate test metrics
- Save model to `artifacts/`

**Expected output:**
```
Dataset: ~24 rows × ~50 columns
Training samples: ~15
Test samples: ~5
Accuracy: ~50-55% (random baseline)
Time: ~30 seconds
```

### 3. Full Training (4 months, all horizons)
```bash
# Full pipeline (takes 5-10 minutes)
python -m src.pipeline.train
```

This will:
- Fetch 4 months of data (~35,000 rows)
- Create 100+ features
- Train models for 15m, 1h, 4h, 24h
- Generate comprehensive metrics
- Save everything to `artifacts/`

**Expected output:**
```
Dataset: 35,000 rows × 150+ columns
Training: 28,000 samples per horizon
Testing: 7,000 samples per horizon
Models: 4 (one per horizon)
Time: 5-10 minutes
```

### 4. Generate Live Signals
```bash
python -m src.pipeline.infer
```

Output: `artifacts/signal.json`
```json
{
  "timestamp": "2025-10-17T10:00:00Z",
  "horizons": {
    "1h": {
      "signal": "LONG",
      "prob_up": 0.63,
      "confidence": "MEDIUM"
    }
  }
}
```

---

## 📊 Understanding Results

### Metrics Explained

**Classification Metrics:**
- `accuracy`: Overall correctness (target: >53%)
- `f1_up`: F1 score for UP class (target: >0.55)
- `mcc`: Matthews correlation (-1 to 1, target: >0.1)
- `roc_auc`: Area under ROC curve (target: >0.55)

**Trading Metrics:**
- `hit_rate`: % of trades that are correct (target: >52%)
- `sharpe_ratio`: Risk-adjusted return (target: >1.0)
- `max_drawdown`: Worst peak-to-trough loss (target: <-20%)
- `cumulative_return`: Total return over test period

### Performance Targets

| Horizon | Accuracy | F1 | MCC | Sharpe | Interpretation |
|---------|----------|----|----|--------|----------------|
| 15m     | 52-54%   | 0.54 | 0.08 | 0.5-1.0 | Slight edge |
| 1h      | 54-57%   | 0.58 | 0.15 | 1.0-2.0 | Good |
| 4h      | 56-60%   | 0.62 | 0.22 | 1.5-2.5 | Strong |
| 24h     | 58-62%   | 0.65 | 0.28 | 2.0-3.0 | Very strong |

**Why these targets?**
- Random baseline: 50% accuracy
- Profitable trading typically needs: >52% accuracy + good risk management
- Strong signal: >55% accuracy with Sharpe >1.5

---

## 🔧 Customization

### Change Date Range
Edit `config.yaml`:
```yaml
data:
  start: "2024-01-01T00:00:00Z"  # Earlier start
  end: "2025-10-17T00:00:00Z"
```

### Train Specific Horizon
```bash
python -m src.pipeline.train --horizon "4h"
```

### Adjust Model Complexity
Edit `config.yaml`:
```yaml
modeling:
  lightgbm:
    max_depth: [5, 7, 9]      # Deeper trees
    n_estimators: [200, 500]  # More trees
```

---

## 📁 Output Files

After training, you'll have:

```
artifacts/
├── dataset.parquet           # Raw + engineered features
├── models/
│   ├── 1h_lightgbm.pkl      # Trained model
│   ├── 1h_scaler.pkl        # Feature scaler
│   └── 1h_features.json     # Feature list
├── reports/
│   ├── 1h_metrics.json      # Performance metrics
│   └── 1h_feature_importance.json
└── signal.json              # Latest signals
```

---

## 🐛 Troubleshooting

### "No data returned"
- Check Prometheus is running: `curl http://10.1.20.100:9095/api/v1/status/buildinfo`
- Verify BTCUSDT data exists
- Try shorter date range

### "Not enough samples"
- Increase date range in `config.yaml`
- Reduce `rolling_windows` sizes
- Check for data gaps

### Low accuracy (~50%)
- Normal for short training periods
- Need more data (weeks/months)
- Try different horizons (4h, 24h often better)

---

## 🎯 Next Steps

1. **Baseline**: Run quick_train.py to verify everything works
2. **Full Training**: Run full pipeline with 4 months
3. **Evaluate**: Check artifacts/reports/ for metrics
4. **Iterate**:
   - Adjust hyperparameters
   - Add more data
   - Try different features
5. **Deploy**: Use inference pipeline for live signals

---

## 💡 Pro Tips

**For Best Results:**
1. Train on 2+ months of data
2. Focus on 4h and 24h horizons first
3. Use walk-forward validation results (more realistic)
4. Monitor both classification AND trading metrics
5. Compare against buy-and-hold baseline

**Performance Optimization:**
- Use `dataset.parquet` cache (avoid re-fetching)
- Reduce `random_search_iters` for faster training
- Train specific horizons only when iterating

**Production Deployment:**
- Run inference on a schedule (cron)
- Monitor signal confidence
- Implement proper risk management
- Track live performance vs backtest

---

## 📚 Additional Resources

- `README.md` - Full documentation
- `PROJECT_STATUS.md` - Implementation details
- `config.yaml` - All configurable parameters
- `smoke_test.py` - Validation script

---

**Questions?** Check the inline documentation in each module!

