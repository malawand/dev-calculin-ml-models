# ✅ Transparency Update - predict_live.py

**Date:** 2025-10-19  
**Update:** Added complete visibility into data sources and query details

---

## 🎯 What Changed

The `predict_live.py` script now shows **everything** about where the data comes from and what's being used for predictions.

---

## 📊 New Information Displayed

### 1. **Cortex Endpoint Details**

```
📡 Cortex Query Details:
   Endpoint: http://10.1.20.60:9009/prometheus/api/v1/query
   Symbol: BTCUSDT
   Query time: 2025-10-19 11:20:09 UTC
```

**What you get:**
- Full Cortex server URL
- API endpoint being used
- Symbol being queried
- Exact query execution time

---

### 2. **Sample Query URL**

```
Sample query: http://10.1.20.60:9009/prometheus/api/v1/query?query=crypto_last_price{symbol="BTCUSDT"}
```

**You can:**
- Copy this URL
- Paste in browser to see raw data
- Use curl to verify manually:
  ```bash
  curl "http://10.1.20.60:9009/prometheus/api/v1/query?query=crypto_last_price%7Bsymbol%3D%22BTCUSDT%22%7D" | python3 -m json.tool
  ```

---

### 3. **Data Timestamp & Freshness**

```
Data timestamp: 2025-10-19 11:19:35
Data age: 33.4s ago
```

**What this tells you:**
- When Cortex last recorded this data
- How stale the data is (should be <60 seconds)
- If >60s, there might be a scraper issue

---

### 4. **All Feature Values**

```
📋 Current Feature Values:
   BTC Price:           $108,313.89
   5m derivative:       -0.1995
   10m derivative:      -0.1438
   15m derivative:      -0.0631
   30m derivative:      -0.0848
   5m average:          $108,322.22
   10m average:         $108,345.91
   15m average:         $108,346.71
   Volume:              12,741.50
   Volume 5m deriv:     0.0960
   Volume 5m avg:       12,736.73
```

**What you see:**
- All 11 features being fed to the model
- Current BTC price in real-time
- Derivatives (rate of change indicators)
- Moving averages
- Volume metrics and trends

---

### 5. **Prediction Target Time**

```
📊 PREDICTION RESULT
   Prediction for: 2025-10-19 11:35:09 (15min from now)
   Direction:      UP
   Confidence:     78.5%
```

**What this tells you:**
- Exact future time the prediction targets
- How far ahead (15 minutes)
- Expected direction
- Model confidence

---

## 🔍 Use Cases

### Debugging
- Check if data is fresh (<60s old)
- Verify features are being fetched correctly
- Identify missing metrics

### Verification
- Copy query URL to browser
- Manually verify Cortex responses
- Compare with exchange prices

### Understanding
- See what data feeds the model
- Learn which metrics matter most
- Track price momentum and trends

### Monitoring
- Ensure Cortex is responsive
- Check for data gaps
- Validate timestamps

---

## 📖 Example: Understanding a Prediction

```
📡 Cortex Query Details:
   Endpoint: http://10.1.20.60:9009/prometheus/api/v1/query
   Symbol: BTCUSDT
   Query time: 2025-10-19 11:20:09 UTC
   Sample query: http://10.1.20.60:9009/prometheus/api/v1/query?query=crypto_last_price{symbol="BTCUSDT"}
   Data timestamp: 2025-10-19 11:19:35
   Data age: 33.4s ago
   ✅ Fetched 11 features successfully

📋 Current Feature Values:
   BTC Price:           $108,313.89
   5m derivative:       -0.1995    ← Falling over 5min
   10m derivative:      -0.1438    ← Falling over 10min
   15m derivative:      -0.0631    ← Falling over 15min
   30m derivative:      -0.0848    ← Falling over 30min
   5m average:          $108,322.22 ← Price below 5m avg
   10m average:         $108,345.91 ← Price below 10m avg
   15m average:         $108,346.71 ← Price below 15m avg
   Volume:              12,741.50
   Volume 5m deriv:     0.0960     ← Volume increasing
   Volume 5m avg:       12,736.73

📊 PREDICTION RESULT
   Prediction for: 2025-10-19 11:35:09 (15min from now)
   Direction:      DOWN             ← Model sees downtrend
   Confidence:     82.3%            ← High confidence

🚨 TRADING SIGNAL
   🔥 HIGH CONFIDENCE SIGNAL: SELL
   Action: Execute SELL with 1-2% position
```

**Analysis:**
1. ✅ Data is fresh (33s old)
2. ✅ All derivatives are negative → price falling
3. ✅ Price below all moving averages → downtrend confirmed
4. ✅ Volume increasing → strong movement
5. ✅ Model predicts DOWN with 82% confidence
6. ✅ **TRADE: SELL/SHORT**

---

## 🧪 Manual Verification

### Verify Current Price

```bash
curl -s "http://10.1.20.60:9009/prometheus/api/v1/query?query=crypto_last_price%7Bsymbol%3D%22BTCUSDT%22%7D" | python3 -m json.tool
```

**Expected Response:**
```json
{
    "status": "success",
    "data": {
        "result": [
            {
                "metric": {
                    "symbol": "BTCUSDT"
                },
                "value": [
                    1760890922.901,
                    "108313.89"
                ]
            }
        ]
    }
}
```

### Verify Derivative

```bash
curl -s "http://10.1.20.60:9009/prometheus/api/v1/query?query=job:crypto_last_price:deriv5m%7Bsymbol%3D%22BTCUSDT%22%7D" | python3 -m json.tool
```

---

## 📚 Related Documentation

- **START_HERE.md** - Quick start guide
- **HOW_TO_USE.md** - Complete usage guide
- **PREDICTION_DETAILS.md** - Deep dive on data transparency
- **AGGRESSIVE_VS_CONSERVATIVE.md** - Model comparison

---

## ⚙️ Configuration

All settings are in `predict_live.py`:

```python
# Cortex configuration
CORTEX_URL = "http://10.1.20.60:9009"
CORTEX_API = "/prometheus/api/v1/query"
SYMBOL = "BTCUSDT"

# Features the model needs (in order!)
REQUIRED_FEATURES = [
    'crypto_last_price',
    'job:crypto_last_price:deriv5m',
    'job:crypto_last_price:deriv10m',
    'job:crypto_last_price:deriv15m',
    'job:crypto_last_price:deriv30m',
    'job:crypto_last_price:avg5m',
    'job:crypto_last_price:avg10m',
    'job:crypto_last_price:avg15m',
    'crypto_volume',
    'job:crypto_volume:deriv5m',
    'job:crypto_volume:avg5m',
]
```

---

## ✅ Summary

**Before this update:**
- ❌ No visibility into data sources
- ❌ Unclear where data comes from
- ❌ Hard to debug issues
- ❌ Can't verify data freshness

**After this update:**
- ✅ Full Cortex endpoint shown
- ✅ Sample query URL to verify manually
- ✅ Data timestamp and age displayed
- ✅ All feature values visible
- ✅ Prediction target time shown
- ✅ Complete transparency for debugging

---

## 🎯 Try It Now

```bash
cd "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start"
source ../btc_lstm_ensemble/venv/bin/activate
python predict_live.py
```

**You'll now see:**
1. Exact Cortex queries
2. All feature values
3. Data timestamps
4. Prediction target time
5. Complete transparency!

---

**Your prediction system is now fully transparent! 🔍**




