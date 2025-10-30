# 🚀 PRODUCTION DEPLOYMENT GUIDE

## Overview

This guide will help you deploy the **76.46% accuracy** Bitcoin direction prediction model to production and integrate it with your trading bot.

---

## 📊 Model Information

- **Accuracy**: 76.46% (24-hour horizon)
- **Features**: 6 carefully selected features
- **Training Data**: 1.8 years (Oct 2023 - Oct 2025)
- **Model Type**: LightGBM Classifier
- **Prediction Frequency**: Every 15 minutes (or as needed)

### Winning Features
```
1. deriv7d_prime7d     - 7-day derivative acceleration
2. deriv4d_roc         - 4-day rate of change
3. volatility_24       - 24-period volatility
4. avg30m              - 30-minute moving average
5. avg45m              - 45-minute moving average
6. avg1h               - 1-hour moving average
```

---

## 🛠️ PRODUCTION SETUP

### Step 1: Install Dependencies

```bash
cd "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start"

# Activate virtual environment
source venv/bin/activate

# Install required packages (if not already installed)
pip install pandas numpy scikit-learn lightgbm requests pyyaml
```

### Step 2: Verify Model Files

Ensure these files exist:
```
btc_minimal_start/
├── results/
│   └── BEST_MODEL.json          # Model configuration
├── models/
│   └── best_model.pkl           # Trained model weights
├── config.yaml                   # Configuration file
└── predict_live.py               # Live prediction script
```

---

## 📡 LIVE PREDICTION SYSTEM

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION FLOW                          │
└─────────────────────────────────────────────────────────────┘

  Cortex/Prometheus (10.1.20.60:9009)
           ↓
    [Fetch Latest Data]
           ↓
    [Engineer Features]
           ↓
    [Trained Model] → Prediction
           ↓
    ┌──────────────────────────────────┐
    │  Probability UP: 0.8234 (82.34%) │
    │  Direction: UP                    │
    │  Confidence: HIGH                 │
    │  Signal Strength: STRONG BUY      │
    └──────────────────────────────────┘
           ↓
    [Your Trading Bot]
```

---

## 💻 USAGE

### Option A: One-Time Prediction (Testing)

```bash
cd "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start"
source venv/bin/activate
python predict_live.py --once
```

**Output:**
```
============================================================
               📈 LIVE BITCOIN PREDICTION 📈
============================================================
Timestamp:       2025-10-18T15:30:00
Current Price:   $67,234.50
Horizon:         24h
Probability UP:  0.7823
Direction:       UP
Confidence:      HIGH
Signal Strength: STRONG BUY
============================================================
```

### Option B: Continuous Monitoring (Production)

```bash
python predict_live.py --interval 15
```

This will:
- Make a prediction every 15 minutes
- Log all predictions to `predictions.log`
- Output live signals to console
- Run continuously until stopped (Ctrl+C)

### Option C: API Mode (For Bot Integration)

```bash
python predict_live.py --api --port 8080
```

Then your bot can query:
```bash
curl http://localhost:8080/predict
```

Response:
```json
{
  "timestamp": "2025-10-18T15:30:00Z",
  "current_price": 67234.50,
  "horizon": "24h",
  "probability_up": 0.7823,
  "direction": "UP",
  "confidence": "HIGH",
  "signal_strength": "STRONG_BUY",
  "features_used": [
    "deriv7d_prime7d",
    "deriv4d_roc",
    "volatility_24",
    "avg30m",
    "avg45m",
    "avg1h"
  ]
}
```

---

## 🎯 UNDERSTANDING THE OUTPUT

### 1. Probability UP (Confidence Score)

The model outputs a probability between 0.0 and 1.0:

| Probability | Interpretation | Confidence |
|-------------|----------------|------------|
| 0.85 - 1.00 | Very likely UP | **VERY HIGH** |
| 0.70 - 0.84 | Likely UP | **HIGH** |
| 0.55 - 0.69 | Somewhat likely UP | **MEDIUM** |
| 0.45 - 0.54 | Uncertain | **LOW** |
| 0.31 - 0.44 | Somewhat likely DOWN | **MEDIUM** |
| 0.16 - 0.30 | Likely DOWN | **HIGH** |
| 0.00 - 0.15 | Very likely DOWN | **VERY HIGH** |

### 2. Direction

- **UP**: Price expected to go up in next 24 hours
- **DOWN**: Price expected to go down in next 24 hours
- **NEUTRAL**: Too uncertain to call (probability near 0.50)

### 3. Signal Strength

| Probability | Signal Strength | Recommended Action |
|-------------|-----------------|-------------------|
| ≥ 0.80 | **STRONG BUY** | Open long position (high confidence) |
| 0.70 - 0.79 | **BUY** | Open long position (medium-high confidence) |
| 0.60 - 0.69 | **WEAK BUY** | Consider long position (medium confidence) |
| 0.40 - 0.59 | **HOLD** | Do nothing (too uncertain) |
| 0.31 - 0.39 | **WEAK SELL** | Consider short position (medium confidence) |
| 0.21 - 0.30 | **SELL** | Open short position (medium-high confidence) |
| ≤ 0.20 | **STRONG SELL** | Open short position (high confidence) |

---

## 🤖 TRADING BOT INTEGRATION

### Python Integration Example

```python
import requests
import json

def get_bitcoin_prediction():
    """Fetch prediction from local API"""
    response = requests.get('http://localhost:8080/predict')
    return response.json()

def make_trading_decision(prediction):
    """Convert prediction to trading action"""
    prob = prediction['probability_up']
    price = prediction['current_price']
    
    # Conservative thresholds (recommended)
    if prob >= 0.75:
        return {
            'action': 'BUY',
            'confidence': prediction['confidence'],
            'amount': calculate_position_size(prob),  # Your risk management
            'take_profit': price * 1.02,  # 2% gain
            'stop_loss': price * 0.99     # 1% loss
        }
    elif prob <= 0.25:
        return {
            'action': 'SELL',
            'confidence': prediction['confidence'],
            'amount': calculate_position_size(1 - prob),
            'take_profit': price * 0.98,  # 2% gain on short
            'stop_loss': price * 1.01     # 1% loss on short
        }
    else:
        return {'action': 'HOLD', 'reason': 'Low confidence'}

# Use in your trading loop
while True:
    prediction = get_bitcoin_prediction()
    decision = make_trading_decision(prediction)
    
    if decision['action'] in ['BUY', 'SELL']:
        execute_trade(decision)  # Your trading function
    
    time.sleep(900)  # Wait 15 minutes
```

### Shell Script Integration

```bash
#!/bin/bash
# fetch_signal.sh

PREDICTION=$(python predict_live.py --once --json)
PROB=$(echo $PREDICTION | jq -r '.probability_up')
DIRECTION=$(echo $PREDICTION | jq -r '.direction')

if (( $(echo "$PROB >= 0.75" | bc -l) )); then
    echo "STRONG BUY SIGNAL - Probability: $PROB"
    # Call your trading bot
    ./your_bot.sh BUY $PROB
elif (( $(echo "$PROB <= 0.25" | bc -l) )); then
    echo "STRONG SELL SIGNAL - Probability: $PROB"
    ./your_bot.sh SELL $PROB
else
    echo "HOLD - Probability: $PROB"
fi
```

---

## ⚙️ CONFIGURATION

Edit `config.yaml` to customize:

```yaml
cortex:
  base_url: "10.1.20.60"
  port: 9009
  read_api: "/prometheus/api/v1/query_range"
  symbol: "BTCUSDT"

prediction:
  horizon: "24h"
  update_interval_minutes: 15
  
thresholds:
  strong_buy: 0.80      # Probability threshold for STRONG BUY
  buy: 0.70             # Probability threshold for BUY
  weak_buy: 0.60        # Probability threshold for WEAK BUY
  weak_sell: 0.40       # Probability threshold for WEAK SELL
  sell: 0.30            # Probability threshold for SELL
  strong_sell: 0.20     # Probability threshold for STRONG SELL

logging:
  enabled: true
  file: "predictions.log"
  level: "INFO"
```

---

## 📊 MONITORING

### Log Files

Predictions are automatically logged to `predictions.log`:

```
2025-10-18 15:30:00 | Price: $67,234.50 | Prob: 0.7823 | Direction: UP | Signal: STRONG_BUY
2025-10-18 15:45:00 | Price: $67,189.20 | Prob: 0.7654 | Direction: UP | Signal: BUY
2025-10-18 16:00:00 | Price: $67,345.80 | Prob: 0.8012 | Direction: UP | Signal: STRONG_BUY
```

### Performance Tracking

Track your model's real-world performance:

```bash
python analyze_predictions.py --log predictions.log --days 7
```

This will show:
- Actual accuracy over the past 7 days
- Win rate by signal strength
- ROI if following all signals
- Recommendations for threshold adjustments

---

## 🔄 MAINTENANCE

### Daily Tasks

1. **Check logs** for any errors or anomalies
   ```bash
   tail -100 predictions.log
   ```

2. **Monitor prediction distribution**
   - Are predictions balanced or heavily skewed?
   - Sudden changes might indicate market regime shift

### Weekly Tasks

1. **Review accuracy**
   ```bash
   python analyze_predictions.py --log predictions.log --days 7
   ```

2. **Check data quality**
   - Ensure Cortex/Prometheus is still accessible
   - Verify all 6 features are being fetched correctly

### Monthly Tasks

1. **Retrain model** (optional)
   - Only if market conditions have changed significantly
   - Or if real-world accuracy drops below 65%
   
2. **Backtest** new strategies
   - Test different thresholds
   - Optimize position sizing

---

## ⚠️ IMPORTANT WARNINGS

### 1. 76.46% ≠ 76.46% Win Rate in Trading

The model predicts **direction** with 76.46% accuracy, but:
- **Risk management** is crucial
- Use **stop losses** and **take profits**
- **Position sizing** matters more than prediction accuracy
- Market volatility can cause slippage

### 2. Market Regime Changes

The model was trained on Oct 2023 - Oct 2025 data. It may:
- Perform worse during extreme market conditions
- Need retraining after major market shifts
- Struggle with unprecedented events (regulations, hacks, etc.)

### 3. Latency Matters

- Predictions are based on 15-minute bars
- Execution delays reduce edge
- Use low-latency exchange connections

### 4. Never Trade Blindly

- **Always** use stop losses
- **Never** go all-in on a single prediction
- **Monitor** model performance continuously
- **Have** a kill switch to disable automated trading

---

## 🎯 RECOMMENDED TRADING STRATEGY

### Conservative (Recommended for Beginners)

```yaml
Entry Criteria:
  - Probability ≥ 0.80 (STRONG signals only)
  - Confirm with volume/momentum indicators
  - Risk 1-2% of capital per trade

Exit Strategy:
  - Take profit: 2% gain
  - Stop loss: 1% loss
  - Hold time: 24 hours max
```

### Moderate (For Experienced Traders)

```yaml
Entry Criteria:
  - Probability ≥ 0.70 (STRONG + regular signals)
  - Scale position size with probability
  - Risk 2-3% of capital per trade

Exit Strategy:
  - Take profit: 3% gain
  - Stop loss: 1.5% loss
  - Trailing stop: 1% after 2% gain
```

### Aggressive (High Risk)

```yaml
Entry Criteria:
  - Probability ≥ 0.60
  - Larger positions on higher probability
  - Risk 3-5% of capital per trade

Exit Strategy:
  - Take profit: 5% gain
  - Stop loss: 2% loss
  - Pyramid into winning positions
```

---

## 📞 TROUBLESHOOTING

### Issue: "Cannot connect to Cortex"

**Solution:**
```bash
# Test Cortex connection
curl http://10.1.20.60:9009/api/v1/query?query=up

# If fails, check:
1. Is Cortex running?
2. Is firewall blocking port 9009?
3. Update config.yaml with correct IP/port
```

### Issue: "Model predicts same direction repeatedly"

**Possible causes:**
- Market in strong trend (expected behavior)
- Data feed issue (check if metrics updating)
- Model degradation (check recent accuracy)

**Solution:**
```bash
# Check data freshness
python check_data_quality.py

# If stale, restart data fetching
# If accurate but boring, market might just be trending
```

### Issue: "Accuracy dropped significantly"

**Solution:**
1. Check if market regime changed (bull → bear, low → high volatility)
2. Review recent trades in log
3. Consider retraining with more recent data
4. Reduce position sizes until accuracy recovers

---

## 🚀 QUICK START CHECKLIST

- [ ] Installed all dependencies
- [ ] Verified model files exist
- [ ] Tested connection to Cortex
- [ ] Ran `predict_live.py --once` successfully
- [ ] Configured thresholds in `config.yaml`
- [ ] Set up logging and monitoring
- [ ] Integrated with trading bot
- [ ] Tested with SMALL positions first
- [ ] Set up stop losses and take profits
- [ ] Created kill switch for emergencies

---

## 📚 ADDITIONAL RESOURCES

- **Model Details**: `btc_minimal_start/FINAL_2YEAR_RESULTS.md`
- **Training History**: `btc_minimal_start/results/BEST_MODEL.json`
- **Feature Importance**: Check model file for feature weights
- **Support**: Review logs in `logs/` directory

---

## 🎉 YOU'RE READY!

Your 76.46% accuracy model is production-ready. Start with small positions, monitor closely, and scale up as you gain confidence.

**Remember**: Successful trading = Good Model + Risk Management + Discipline

Good luck! 🚀📈



