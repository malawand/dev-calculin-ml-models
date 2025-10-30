# Bitcoin State Detector - Production Model

> A 95.3% accurate neural network for detecting Bitcoin market state (UP/DOWN/SIDEWAYS) and momentum strength in real-time.

---

## 🎯 Quick Start

### Prerequisites
```bash
Python 3.13+
pip install -r requirements.txt
```

### Run a Prediction
```python
python predict.py
```

### Use in Your Code
```python
from predict import predict_state

result = predict_state(price_array, volume_array)
print(f"Direction: {result['direction']}")  # UP, DOWN, or NONE
print(f"Strength: {result['strength']}")    # 0-100
print(f"Probabilities: {result['probabilities']}")  # [DOWN%, NONE%, UP%]
```

---

## 📊 Performance

- **Overall Accuracy:** 95.3%
  - UP: 98.2%
  - DOWN: 95.4%
  - SIDEWAYS: 90.1%
- **Strength Correlation:** 0.951
- **Inference Time:** 0.1 milliseconds
- **Model Size:** 5 MB

---

## 📁 Files

| File | Description | Size |
|------|-------------|------|
| `direction_model.pkl` | Trained classifier (UP/DOWN/NONE) | ~2 MB |
| `strength_model.pkl` | Trained regressor (momentum strength) | ~2 MB |
| `scaler.pkl` | Feature standardization | ~100 KB |
| `feature_names.json` | List of 22 features | ~1 KB |
| `feature_extractor.py` | Feature calculation code | - |
| `predict.py` | Simple prediction interface | - |

---

## 🔬 How It Works

1. **Input:** Last 5 hours of Bitcoin 1-minute price/volume data (300 data points)
2. **Feature Extraction:** Calculate 22 technical features (derivatives, volatility, volume, deviation)
3. **Model:** 6-layer neural network (512→256→128→64→32→16 neurons)
4. **Output:** 
   - Direction: UP (+1), DOWN (-1), or SIDEWAYS (0)
   - Strength: 0-100 scale
   - Probabilities: [DOWN%, NONE%, UP%]

**Training Data:** 2.5 years (Jan 2023 - Jun 2025), 1,276 samples  
**Framework:** scikit-learn (sklearn)  
**Hardware:** Runs on CPU, no GPU needed

---

## 📖 Documentation

- **[TRAINING_DATA.md](TRAINING_DATA.md)** - What data was used and how
- **[FEATURE_DOCUMENTATION.md](FEATURE_DOCUMENTATION.md)** - How each of 22 features is calculated
- **[AI_CONTEXT.md](AI_CONTEXT.md)** - Full context for AI assistants (use this with Claude/GPT)

---

## 🚀 Usage Examples

### Basic Prediction
```python
import pickle
import numpy as np
from feature_extractor import extract_features

# Load models
with open('direction_model.pkl', 'rb') as f:
    direction_model = pickle.load(f)
with open('strength_model.pkl', 'rb') as f:
    strength_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Your data (5 hours = 300 minutes)
prices = [108500.0, 108510.5, ...]   # 300 values
volumes = [1234.5, 1456.8, ...]      # 300 values

# Extract features
import pandas as pd
df = pd.DataFrame({'price': prices, 'volume': volumes})
features = extract_features(df)

# Prepare for model
feature_names = sorted(features.keys())
X = np.array([[features[fn] for fn in feature_names]])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X_scaled = scaler.transform(X)

# Predict
direction = direction_model.predict(X_scaled)[0]  # -1, 0, or 1
strength = strength_model.predict(X_scaled)[0]    # 0-100
proba = direction_model.predict_proba(X_scaled)[0]  # [down%, none%, up%]

# Interpret
direction_label = {-1: 'DOWN', 0: 'NONE', 1: 'UP'}[direction]
confidence = proba[direction + 1] * 100

print(f"Direction: {direction_label} ({confidence:.1f}% confidence)")
print(f"Strength: {strength:.1f}/100")
print(f"DOWN: {proba[0]*100:.1f}%, NONE: {proba[1]*100:.1f}%, UP: {proba[2]*100:.1f}%")

# Determine lean when sideways
if direction == 0:  # NONE
    lean = 'UP' if proba[2] > proba[0] else 'DOWN' if proba[0] > proba[2] else 'NEUTRAL'
    print(f"Leaning: {lean}")
```

### With Real-Time Data
```python
from predict import predict_state
import requests

# Fetch from your data source (e.g., Prometheus, exchange API, etc.)
def fetch_btc_data():
    # Your implementation here
    # Return: price_array (300 values), volume_array (300 values)
    pass

prices, volumes = fetch_btc_data()
result = predict_state(prices, volumes)

print(f"Current State: {result['direction']}")
print(f"Strength: {result['strength']:.1f}/100")
print(f"Confidence: {result['confidence']:.1f}%")

if result['lean']:
    print(f"Leaning: {result['lean']}")
```

---

## 🎓 The 22 Features

**Price Derivatives (12):**
- Rate of change over 5m, 15m, 30m, 1h, 4h (raw + normalized)
- Acceleration (2nd derivative)
- Rate of change (alternative calculation)

**Volatility (4):**
- Standard deviation of returns over 5m, 15m, 30m, 1h

**Deviation (2):**
- Distance from 1h and 4h average price

**Volume (4):**
- Volume trend over 5m, 15m, 30m
- Volume strength relative to 5h average

See [FEATURE_DOCUMENTATION.md](FEATURE_DOCUMENTATION.md) for formulas.

---

## 🔧 System Requirements

**Minimum:**
- CPU: Any modern CPU (1-2 cores)
- RAM: 4 GB
- Python: 3.8+
- OS: macOS, Linux, Windows

**Recommended:**
- CPU: 2+ cores
- RAM: 8 GB
- Python: 3.13
- OS: macOS or Linux

**GPU:** Not needed! Model runs on CPU in 0.1ms.

**Can run on:**
- ✅ Your laptop
- ✅ $5/month cloud server
- ✅ Raspberry Pi 4
- ✅ AWS t2.micro (free tier)

---

## 📈 Model Architecture

```
Input Layer:     22 features
Hidden Layer 1:  512 neurons (ReLU)
Hidden Layer 2:  256 neurons (ReLU)
Hidden Layer 3:  128 neurons (ReLU)
Hidden Layer 4:  64 neurons (ReLU)
Hidden Layer 5:  32 neurons (ReLU)
Hidden Layer 6:  16 neurons (ReLU)
Output Layer:    3 classes (direction) or 1 value (strength)

Total Parameters: ~300,000
Framework: scikit-learn MLPClassifier/MLPRegressor
Activation: ReLU
Solver: adam
```

---

## 🔄 Retraining

**When to retrain:**
- Quarterly (every 3 months)
- After major market events
- If accuracy drops below 90%

**How to retrain:**
1. Fetch latest 2.5 years of data
2. Run training script (takes 4 seconds)
3. Copy new models to this directory

See [TRAINING_DATA.md](TRAINING_DATA.md) for details.

---

## ⚠️ Important Notes

### What This Model Does:
✅ Detects **current** market state with 95.3% accuracy  
✅ Measures **current** momentum strength with 0.951 correlation  
✅ Shows probability distribution for all directions  
✅ Indicates lean when sideways (UP/DOWN/NEUTRAL)

### What This Model Does NOT Do:
❌ Predict **future** prices (use for current state only)  
❌ Give trading advice (it's a tool, not a strategy)  
❌ Guarantee profits (markets are unpredictable)

### Best Practices:
- Use for confirmation, not sole signal
- Combine with other indicators
- Consider fair value analysis
- Apply proper risk management
- Backtest any strategy

---

## 🐛 Troubleshooting

**Issue:** `FileNotFoundError: direction_model.pkl`  
**Solution:** Make sure you're in the correct directory with all `.pkl` files

**Issue:** `ValueError: X has N features, expecting 22`  
**Solution:** Use the same `feature_extractor.py` that was used during training

**Issue:** Predictions seem off  
**Solution:** 
- Ensure data is 1-minute candles
- Check you have exactly 300 data points (5 hours)
- Verify price/volume arrays are correct

**Issue:** Slow predictions  
**Solution:** Model is fast (0.1ms). If slow, bottleneck is data fetching, not model.

---

## 📜 License

[Add your license here]

---

## 🤝 Contributing

This is a production model. Before making changes:
1. Read [AI_CONTEXT.md](AI_CONTEXT.md) for full context
2. Understand the [FEATURE_DOCUMENTATION.md](FEATURE_DOCUMENTATION.md)
3. Test thoroughly on historical data
4. Validate accuracy >= 90%

---

## 📞 Contact

For questions or issues, refer to the documentation:
- **AI Context:** [AI_CONTEXT.md](AI_CONTEXT.md)
- **Features:** [FEATURE_DOCUMENTATION.md](FEATURE_DOCUMENTATION.md)
- **Training:** [TRAINING_DATA.md](TRAINING_DATA.md)

---

## 🏆 Version History

**v1.0 (October 2025)**
- Initial production release
- 95.3% accuracy
- 6-layer architecture
- 22 features
- Real-time inference < 1ms

---

*Built with ❤️ for accurate Bitcoin state detection*

