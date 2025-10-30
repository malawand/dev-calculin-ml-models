#!/bin/bash
# Quick start script for Prometheus exporter

echo "🚀 Starting BTC Prediction Prometheus Exporter..."
echo ""

# Check if prometheus_client is installed
if ! python -c "import prometheus_client" 2>/dev/null; then
    echo "📦 Installing prometheus_client..."
    pip install prometheus-client
    echo ""
fi

# Default values
PORT=${PORT:-9100}
INTERVAL=${INTERVAL:-15}

echo "⚙️  Configuration:"
echo "   Port: $PORT"
echo "   Update Interval: $INTERVAL minutes"
echo "   Metrics URL: http://localhost:$PORT/metrics"
echo ""

# Start exporter
echo "🏁 Starting exporter..."
python prometheus_exporter.py --port $PORT --interval $INTERVAL

# Note: Use Ctrl+C to stop



