# 📊 Prediction Script - Data Transparency

The `predict_live.py` script now shows **complete transparency** about data sources and queries.

---

## 🔍 What You'll See

### 1. **Cortex Query Details**

```
📡 Cortex Query Details:
   Endpoint: http://10.1.20.60:9009/prometheus/api/v1/query
   Symbol: BTCUSDT
   Query time: 2025-10-19 11:20:09 UTC

   Sample query: http://10.1.20.60:9009/prometheus/api/v1/query?query=crypto_last_price{symbol="BTCUSDT"}
   Data timestamp: 2025-10-19 11:19:35
   Data age: 33.4s ago
```

**What this tells you:**
- **Endpoint:** The exact Cortex server and API path being queried
- **Symbol:** Which cryptocurrency pair (BTCUSDT)
- **Query time:** When the prediction script ran
- **Sample query:** Full URL you can paste in browser to see raw data
- **Data timestamp:** When Cortex last updated this metric
- **Data age:** How fresh the data is (should be <60s)

---

### 2. **Current Feature Values**

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

**What this tells you:**
- **BTC Price:** Current spot price from Cortex
- **Derivatives:** Rate of change over different timeframes (negative = falling)
- **Averages:** Moving averages over different windows
- **Volume:** Current trading volume and its trends

---

### 3. **Prediction Target**

```
📊 PREDICTION RESULT
   Prediction for: 2025-10-19 11:35:09 (15min from now)
   Direction:      SIDEWAYS
   Confidence:     41.6%
```

**What this tells you:**
- **Prediction for:** Exact time the prediction is targeting (current time + 15 minutes)
- **Direction:** Expected price movement (UP/DOWN/SIDEWAYS)
- **Confidence:** Model's certainty level (only trade if >70%)

---

## 🔗 Example Queries

You can manually verify the data by querying Cortex directly:

### Current BTC Price
```bash
# URL encoded for curl
curl "http://10.1.20.60:9009/prometheus/api/v1/query?query=crypto_last_price%7Bsymbol%3D%22BTCUSDT%22%7D"

# Or in browser (will auto-encode)
http://10.1.20.60:9009/prometheus/api/v1/query?query=crypto_last_price{symbol="BTCUSDT"}
```

### 5-Minute Derivative
```bash
curl "http://10.1.20.60:9009/prometheus/api/v1/query?query=job:crypto_last_price:deriv5m%7Bsymbol%3D%22BTCUSDT%22%7D"
```

### Current Volume
```bash
curl "http://10.1.20.60:9009/prometheus/api/v1/query?query=crypto_volume%7Bsymbol%3D%22BTCUSDT%22%7D"
```

### Pretty-Print JSON Response
```bash
curl -s "http://10.1.20.60:9009/prometheus/api/v1/query?query=crypto_last_price%7Bsymbol%3D%22BTCUSDT%22%7D" | python3 -m json.tool
```

**Example Response:**
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
                    "108254.22"
                ]
            }
        ]
    }
}
```
- First number: Unix timestamp (1760890922.901)
- Second value: BTC price ($108,254.22)

---

## 📈 Understanding the Features

### Price Derivatives (Rate of Change)

**Positive derivative:** Price is rising
```
5m derivative:  +0.5  → Price rising fast over 5 minutes
```

**Negative derivative:** Price is falling
```
5m derivative:  -0.2  → Price falling over 5 minutes
```

**Near zero:** Price is stable
```
5m derivative:  +0.01 → Price mostly flat
```

### Moving Averages

**Price above averages:** Uptrend
```
BTC Price:      $68,500
5m average:     $68,300  → Price 0.3% above 5m average
```

**Price below averages:** Downtrend
```
BTC Price:      $68,500
5m average:     $68,700  → Price 0.3% below 5m average
```

### Volume Indicators

**Volume derivative positive:** Increasing trading activity
```
Volume 5m deriv: +0.5  → More traders entering
```

**Volume derivative negative:** Decreasing activity
```
Volume 5m deriv: -0.3  → Fewer traders, quieting down
```

---

## 🕐 Time Ranges Explained

### What the Script Fetches

The script uses **instant queries** (most recent value):
- **Query type:** `/api/v1/query` (not range query)
- **Time:** Latest available datapoint (typically <60s old)
- **No historical range:** Just the current snapshot

### Why Instant Queries?

For **live predictions**, you only need current values:
- Current price
- Current derivatives (already calculated by Prometheus)
- Current averages (already calculated by Prometheus)

The model was trained on **historical data** (2.5 years), but for inference it only needs **current conditions**.

---

## 🧪 Debugging Data Issues

### If Data Age > 60 seconds

```
Data age: 125.3s ago  ⚠️  WARNING
```

**This means:**
- Prometheus scraper might be down
- Cortex might be delayed
- Predictions will be less accurate

**Fix:** Check Prometheus scraper status

### If Any Feature Shows 0.0

```
⚠️  Warning: No data for job:crypto_volume:deriv5m, using 0
```

**This means:**
- Metric not available in Cortex
- Recording rule might be missing
- Model will still work but with reduced accuracy

**Fix:** Check Prometheus recording rules

### If Query Fails Completely

```
❌ Error fetching crypto_last_price: Connection refused
```

**This means:**
- Cortex server is down
- Wrong endpoint in config
- Network issue

**Fix:** Verify `CORTEX_URL` in `predict_live.py`

---

## 🔧 Customization

### Change Cortex Endpoint

Edit `predict_live.py`:
```python
CORTEX_URL = "http://YOUR_SERVER:9009"
CORTEX_API = "/prometheus/api/v1/query"
```

### Change Symbol

```python
SYMBOL = "ETHUSDT"  # For Ethereum
SYMBOL = "BTCUSDT"  # For Bitcoin
```

### Add More Metrics

Add to `REQUIRED_FEATURES` list:
```python
REQUIRED_FEATURES = [
    'crypto_last_price',
    'job:crypto_last_price:deriv5m',
    # Add your custom metrics here
    'your_custom_metric',
]
```

**Note:** You'll need to retrain the model if you add features!

---

## 📊 Full Data Flow

```
1. Script runs → 2025-10-19 11:20:09
           ↓
2. Query Cortex → http://10.1.20.60:9009/prometheus/api/v1/query
           ↓
3. Get 11 features → Price, derivatives, volume, etc.
           ↓
4. Data timestamp → 2025-10-19 11:19:35 (33s old)
           ↓
5. Scale features → Normalize using training scaler
           ↓
6. Run model → LightGBM classifier
           ↓
7. Get prediction → UP/DOWN/SIDEWAYS + confidence
           ↓
8. Target time → 2025-10-19 11:35:09 (15 min ahead)
           ↓
9. Show signal → BUY/SELL/NO TRADE
```

---

## ✅ Summary

**What You Now See:**
1. ✅ Exact Cortex endpoint being queried
2. ✅ Sample query URL (paste in browser to verify)
3. ✅ Data timestamp (when data was recorded)
4. ✅ Data age (how fresh the data is)
5. ✅ All 11 feature values being used
6. ✅ Prediction target time (15 min from now)
7. ✅ Complete transparency for debugging

**Use Cases:**
- **Verify data freshness:** Check "Data age"
- **Debug predictions:** See exact feature values
- **Manual verification:** Copy query URL to browser
- **Build confidence:** Full visibility into model inputs

---

**Your prediction script is now fully transparent! 🔍**


