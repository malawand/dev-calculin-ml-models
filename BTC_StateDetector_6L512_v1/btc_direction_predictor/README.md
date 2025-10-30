# Bitcoin Direction Predictor

**Senior ML Engineering Project:** Predict BTC price direction using Prometheus/Cortex time series with self-improving ML pipeline.

## 🎯 Goal

Given Prometheus time series for BTCUSDT, predict direction over multiple horizons (15m, 1h, 4h, 24h) and identify which features have the most predictive power.

**Outputs:**
- Ranked features per horizon
- Cross-validated performance metrics
- Walk-forward backtest results
- Live signal API/files

## 📊 Features

**Data Sources:**
- 1 spot price metric
- 17 moving averages (10m to 14d)
- 20 first derivatives (5m to 30d)
- 200+ second derivatives (acceleration signals)

**Models:**
- Logistic Regression (baseline)
- LightGBM (primary)
- LSTM (sequence-based)

**Feature Selection:**
- Correlation filtering
- Permutation importance
- SHAP values
- Forward stepwise selection

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
# Edit config.yaml with your Cortex endpoint

# 3. Train
python -m src.pipeline.train --config config.yaml

# 4. Inference
python -m src.pipeline.infer --config config.yaml
```

## 📁 Project Structure

```
btc_direction_predictor/
├── config.yaml              # Main configuration
├── requirements.txt         # Dependencies
├── src/
│   ├── data/
│   │   ├── prom.py         # Prometheus client
│   │   └── build_dataset.py # Dataset builder
│   ├── features/
│   │   └── engineering.py   # Feature engineering
│   ├── labels/
│   │   └── labels.py        # Label creation
│   ├── model/
│   │   ├── baselines.py     # LogReg/SVM
│   │   ├── trees.py         # LightGBM
│   │   └── lstm.py          # LSTM classifier
│   ├── select/
│   │   └── selector.py      # Feature selection
│   ├── eval/
│   │   ├── metrics.py       # Performance metrics
│   │   └── walkforward.py   # Time series CV
│   └── pipeline/
│       ├── train.py         # Training orchestration
│       └── infer.py         # Inference/signals
├── artifacts/
│   ├── models/             # Saved models
│   ├── reports/            # Performance reports
│   └── signal.json         # Latest signals
└── notebooks/
    └── EDA.ipynb           # Exploratory analysis
```

## ⚙️ Configuration

Key settings in `config.yaml`:

```yaml
cortex:
  base_url: "http://10.1.20.100:9095"
  symbol: "BTCUSDT"

data:
  start: "2024-06-01T00:00:00Z"
  end: "2025-10-16T00:00:00Z"
  step: "5m"

labels:
  horizons: ["15m", "1h", "4h", "24h"]
  threshold_pct: 0.0  # UP if future_return > 0%

modeling:
  n_splits: 5  # TimeSeriesSplit folds
  random_search_iters: 40
```

## 📈 Evaluation Metrics

- **Classification:** Accuracy, Precision, Recall, F1, MCC, ROC-AUC
- **Trading:** Hit Rate, Avg Return, Sharpe, Max Drawdown

## 🔬 Feature Selection Pipeline

1. **Correlation filter:** Drop one of any pair with |ρ| > 0.95
2. **Permutation importance:** Keep top 30 features
3. **SHAP:** Keep top 12 most important
4. **Forward stepwise:** Find optimal subset (4-12 features)

## 📤 Inference Output

```json
{
  "timestamp": "2025-10-16T10:00:00Z",
  "horizon": "1h",
  "prob_up": 0.63,
  "signal": "LONG",
  "confidence": "MEDIUM"
}
```

## 🛡️ Anti-Leakage Guarantees

- Labels only use future data
- Features only use past data
- Scaling fitted on train fold only
- Walk-forward validation
- No look-ahead bias

## 📊 Example Results

```
=== Horizon: 1h ===
Best Model: LightGBM
Selected Features: 8
  1. deriv1h_prime1h (importance: 0.23)
  2. deriv24h_prime4h (importance: 0.18)
  3. avg4h_lag1 (importance: 0.15)
  ...

Cross-Validation:
  MCC: 0.42
  F1 (UP): 0.68
  ROC-AUC: 0.76

Backtest:
  Hit Rate: 58.3%
  Sharpe: 1.85
  Max DD: -12.4%
```

## 🔄 Self-Improvement

The pipeline automatically:
1. Tries multiple feature sets
2. Searches hyperparameters
3. Validates with walk-forward
4. Selects best configuration
5. Reports performance vs baseline

## 🚨 Known Limitations

- Fetching 200+ metrics takes time (~5-10 min)
- LSTM requires more data for training
- Backtest assumes perfect execution (no realistic slippage model)

## 📝 CLI Usage

```bash
# Train all horizons
python -m src.pipeline.train

# Train specific horizon
python -m src.pipeline.train --horizons "1h"

# Custom date range
python -m src.pipeline.train --start "2024-01-01" --end "2024-12-31"

# Inference
python -m src.pipeline.infer
```

## 🧪 Smoke Test

```bash
# Quick test with 2 days of data
python -m src.pipeline.train --start "2025-10-14" --end "2025-10-16" --horizons "1h"
```

## 📚 Dependencies

- Python 3.10+
- pandas, numpy, scikit-learn
- lightgbm, shap
- torch (for LSTM)
- requests, pyyaml

## 🤝 Contributing

This is a production-ready template. Customize:
- Add more models in `src/model/`
- Extend feature engineering in `src/features/`
- Implement custom metrics in `src/eval/`

## 📄 License

MIT

---

**Built with systematic ML engineering principles:**
✅ Reproducible
✅ Anti-leakage
✅ Walk-forward validated  
✅ Production-ready
✅ Self-documenting
