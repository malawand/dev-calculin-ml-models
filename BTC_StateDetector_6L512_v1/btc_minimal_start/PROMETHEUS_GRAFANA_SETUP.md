# 📊 Prometheus & Grafana Monitoring Setup

## Overview

This guide shows you how to expose your Bitcoin direction predictions as Prometheus metrics and visualize them in Grafana with timestamps.

**What You'll Get:**
- Real-time prediction metrics scraped by Prometheus
- Beautiful Grafana dashboards
- Historical prediction tracking
- Alerting on strong signals
- All timestamped and persistent

---

## 🏗️ Architecture

```
┌─────────────────┐
│  BTC Predictor  │
│  (predict_live) │
└────────┬────────┘
         │
         ↓
┌──────────────────────┐
│ Prometheus Exporter  │  ← Exposes /metrics endpoint
│ (port 9100)          │
└──────────┬───────────┘
           │
           ↓ scrapes every 1m
┌──────────────────────┐
│    Prometheus        │  ← Stores time-series data
│  (port 9090)         │
└──────────┬───────────┘
           │
           ↓ queries
┌──────────────────────┐
│      Grafana         │  ← Visualizes data
│    (port 3000)       │
└──────────────────────┘
```

---

## 🚀 Quick Start (Docker)

### Option 1: Run Just the Exporter

```bash
cd "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start"

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Test metrics endpoint
curl http://localhost:9100/metrics
```

### Option 2: Run Exporter Locally (No Docker)

```bash
cd "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start"

# Install prometheus_client
pip install prometheus-client

# Run exporter
python prometheus_exporter.py --port 9100 --interval 15
```

**Metrics will be available at:** `http://localhost:9100/metrics`

---

## 📊 Available Metrics

| Metric Name | Type | Description | Values |
|-------------|------|-------------|--------|
| `btc_prediction_probability_up` | Gauge | Probability price goes UP | 0.0 - 1.0 |
| `btc_prediction_probability_down` | Gauge | Probability price goes DOWN | 0.0 - 1.0 |
| `btc_current_price` | Gauge | Current BTC price in USD | e.g., 67234.50 |
| `btc_prediction_confidence` | Gauge | Confidence level | 0=LOW, 1=MEDIUM, 2=HIGH, 3=VERY_HIGH |
| `btc_prediction_signal` | Gauge | Trading signal strength | -3=STRONG_SELL to 3=STRONG_BUY |
| `btc_prediction_direction` | Gauge | Predicted direction | -1=DOWN, 0=NEUTRAL, 1=UP |
| `btc_prediction_timestamp` | Gauge | Unix timestamp of prediction | Unix epoch |
| `btc_prediction_total` | Counter | Total predictions made | Incrementing |
| `btc_prediction_errors_total` | Counter | Total errors | Incrementing |
| `btc_prediction_latency_seconds` | Histogram | Time to make prediction | Seconds |
| `btc_model_info` | Info | Model metadata | Version, accuracy, etc. |

---

## 🔧 Configure Prometheus to Scrape

### 1. Add to Your Existing Prometheus

Edit your `prometheus.yml`:

```yaml
scrape_configs:
  # Add this job
  - job_name: 'btc_predictions'
    static_configs:
      - targets: ['localhost:9100']  # Or container name if using Docker
    scrape_interval: 1m
    metrics_path: '/metrics'
```

### 2. Or Use the Example Config

```bash
# Copy example config
cp prometheus.yml.example /path/to/your/prometheus/prometheus.yml

# Restart Prometheus
sudo systemctl restart prometheus
# OR
docker restart prometheus
```

### 3. Verify Prometheus is Scraping

1. Open Prometheus UI: `http://localhost:9090`
2. Go to **Status → Targets**
3. Look for `btc_predictions` job
4. Should show status: **UP**

---

## 🎨 Set Up Grafana Dashboard

### 1. Import the Dashboard

1. Open Grafana: `http://localhost:3000` (default login: admin/admin)
2. Go to **Dashboards → Import**
3. Click **Upload JSON file**
4. Select `grafana_dashboard.json`
5. Choose your Prometheus data source
6. Click **Import**

### 2. Or Create Manually

**Create New Dashboard:**
1. Add Panel
2. Select Prometheus data source
3. Use these queries:

```promql
# Current Price
btc_current_price

# Probability UP (as percentage)
btc_prediction_probability_up * 100

# Signal Strength (color-coded)
btc_prediction_signal

# Price with Probability overlay
btc_current_price
btc_prediction_probability_up * 1000  # Scale for visibility
```

---

## 🚨 Set Up Alerts

### 1. Copy Alert Rules

```bash
cp btc_prediction_alerts.yml /path/to/prometheus/rules/

# Add to prometheus.yml
rule_files:
  - 'rules/btc_prediction_alerts.yml'

# Reload Prometheus
curl -X POST http://localhost:9090/-/reload
# OR
sudo systemctl reload prometheus
```

### 2. Configure Alertmanager (Optional)

Example `alertmanager.yml`:

```yaml
route:
  receiver: 'trading-alerts'
  group_by: ['alertname', 'signal']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 1h

receivers:
  - name: 'trading-alerts'
    webhook_configs:
      - url: 'http://your-trading-bot:8080/alerts'
    # Or email
    email_configs:
      - to: 'you@example.com'
        from: 'prometheus@example.com'
        smarthost: 'smtp.gmail.com:587'
```

---

## 📂 File Mode (For Node Exporter)

If you're using Prometheus `node_exporter` with textfile collector:

```bash
# Run in file mode
python prometheus_exporter.py --file-mode --output /var/lib/node_exporter/textfile_collector/btc_predictions.prom

# Configure node_exporter
node_exporter --collector.textfile.directory=/var/lib/node_exporter/textfile_collector
```

**Prometheus will automatically scrape the .prom file!**

---

## 🐳 Full Docker Stack (Optional)

Create `docker-compose.full.yml`:

```yaml
version: '3.8'

services:
  btc-predictor:
    build: .
    container_name: btc-prediction-exporter
    ports:
      - "9100:9100"
    networks:
      - monitoring

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./btc_prediction_alerts.yml:/etc/prometheus/rules/btc_prediction_alerts.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana_dashboard.json:/etc/grafana/provisioning/dashboards/btc_predictions.json
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
```

**Run everything:**
```bash
docker-compose -f docker-compose.full.yml up -d
```

**Access:**
- Grafana: `http://localhost:3000` (admin/admin)
- Prometheus: `http://localhost:9090`
- Metrics: `http://localhost:9100/metrics`

---

## 🔍 Example Queries for Grafana

### Basic Queries

```promql
# Current probability of UP
btc_prediction_probability_up

# Current price
btc_current_price

# Signal strength (bar chart)
btc_prediction_signal

# Confidence level
btc_prediction_confidence
```

### Advanced Queries

```promql
# Prediction accuracy (if you track actual outcomes)
sum(btc_prediction_correct_total) / sum(btc_prediction_total)

# Average signal strength over 1 hour
avg_over_time(btc_prediction_signal[1h])

# Price change rate
rate(btc_current_price[5m])

# Predictions per minute
rate(btc_prediction_total[5m]) * 60

# Error rate
rate(btc_prediction_errors_total[5m])
```

### Combining Metrics

```promql
# Price with probability
btc_current_price and btc_prediction_probability_up > 0.8

# High confidence buy signals
(btc_prediction_signal >= 2) and (btc_prediction_confidence >= 2)
```

---

## 📈 Dashboard Panels Included

The provided Grafana dashboard includes:

1. **Current BTC Price** - Large stat panel
2. **Prediction Probability** - Gauge (0-100%)
3. **Signal Strength** - Color-coded stat (STRONG SELL → STRONG BUY)
4. **Confidence Level** - LOW/MEDIUM/HIGH/VERY HIGH
5. **Price & Probability Chart** - 24h time series
6. **Signal History** - Bar chart of signals over time
7. **Prediction Latency** - Response time monitoring
8. **Total Predictions** - Counter
9. **Prediction Errors** - Error tracking
10. **Model Info** - Static info panel
11. **Prediction Rate** - Predictions per minute

---

## 🔔 Example Alerts You'll Get

Based on `btc_prediction_alerts.yml`:

- **STRONG BUY/SELL** signals (for trading)
- **Exporter Down** (system health)
- **High Error Rate** (quality monitoring)
- **High Latency** (performance issues)
- **Volatility + Strong Signal** (interesting market conditions)

---

## 📊 Retention & Storage

### Prometheus Retention

```yaml
# In prometheus.yml command args
--storage.tsdb.retention.time=90d
--storage.tsdb.retention.size=50GB
```

**Default:** 15 days

**For longer retention:**
- Use Thanos
- Use Cortex (you already have this!)
- Use VictoriaMetrics

### Grafana Data Source Settings

Configure your Prometheus data source in Grafana:
- **URL:** `http://prometheus:9090` (or `http://localhost:9090`)
- **Scrape interval:** `1m`
- **Query timeout:** `60s`

---

## 🧪 Testing

### 1. Test Exporter

```bash
# Check if exporter is running
curl http://localhost:9100/metrics | grep btc_prediction

# Should see output like:
# btc_prediction_probability_up 0.7823
# btc_current_price 67234.50
# btc_prediction_signal 2.0
```

### 2. Test Prometheus

```bash
# Query Prometheus API
curl 'http://localhost:9090/api/v1/query?query=btc_prediction_probability_up'

# Should return JSON with current value
```

### 3. Test Grafana

1. Open dashboard
2. Should see live data updating every minute
3. Zoom in/out to see historical data

---

## 🐛 Troubleshooting

### Exporter Not Starting

```bash
# Check logs
docker-compose logs btc-predictor

# Common issues:
# - Missing config.yaml
# - Missing models directory
# - Cannot connect to Cortex
```

### Prometheus Not Scraping

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify exporter is reachable
curl http://localhost:9100/metrics

# Check prometheus.yml syntax
promtool check config prometheus.yml
```

### No Data in Grafana

1. **Check data source:**
   - Configuration → Data Sources → Prometheus
   - Click "Test" - should be green

2. **Check queries:**
   - Use Explore tab
   - Run simple query: `btc_prediction_probability_up`

3. **Check time range:**
   - Make sure time range includes recent data
   - Try "Last 5 minutes"

---

## 📝 Example Prometheus Metrics Output

```prometheus
# HELP btc_prediction_probability_up Probability that BTC price will go UP in next 24h
# TYPE btc_prediction_probability_up gauge
btc_prediction_probability_up 0.7823

# HELP btc_current_price Current BTC price in USD
# TYPE btc_current_price gauge
btc_current_price 67234.50

# HELP btc_prediction_signal Signal strength
# TYPE btc_prediction_signal gauge
btc_prediction_signal 2.0

# HELP btc_prediction_confidence Confidence score
# TYPE btc_prediction_confidence gauge
btc_prediction_confidence 2.0

# HELP btc_prediction_total Total number of predictions made
# TYPE btc_prediction_total counter
btc_prediction_total 1523

# HELP btc_prediction_latency_seconds Time taken to make a prediction
# TYPE btc_prediction_latency_seconds histogram
btc_prediction_latency_seconds_bucket{le="5.0"} 1450
btc_prediction_latency_seconds_bucket{le="10.0"} 1520
btc_prediction_latency_seconds_bucket{le="+Inf"} 1523
btc_prediction_latency_seconds_sum 7615.2
btc_prediction_latency_seconds_count 1523
```

---

## ✅ Final Checklist

- [ ] Exporter running and accessible at `:9100/metrics`
- [ ] Prometheus scraping the exporter (check Targets page)
- [ ] Grafana dashboard imported
- [ ] Data visible in Grafana
- [ ] Alerts configured (optional)
- [ ] Retention period set appropriately

---

## 🎯 Next Steps

1. **Customize Dashboard**: Add panels for your specific needs
2. **Set Up Alerts**: Configure Alertmanager for notifications
3. **Monitor Performance**: Track accuracy over time
4. **Integrate with Bot**: Use alerts to trigger trades
5. **Backup Data**: Set up Prometheus remote write to Cortex

---

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Guide](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)

---

**Your prediction model is now fully monitored with Prometheus & Grafana!** 🎉📊

All predictions are timestamped, stored, and visualized for easy analysis and trading decisions.



