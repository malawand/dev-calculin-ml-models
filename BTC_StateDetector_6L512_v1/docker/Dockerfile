# Bitcoin State Detector - Production Docker Image
# Multi-stage build for smaller final image
# Target: Intel Xeon (amd64/x86_64)

# Stage 1: Builder
FROM --platform=linux/amd64 python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal production requirements
COPY docker/requirements-prod.txt /app/requirements-prod.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements-prod.txt

# Stage 2: Runtime
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create directories for outputs
RUN mkdir -p /app/metrics /app/logs

# Copy all necessary project files
COPY btc_momentum_detector/ /app/btc_momentum_detector/
COPY btc_fair_value/ /app/btc_fair_value/
COPY btc_minimal_start/ /app/btc_minimal_start/
COPY btc_ultra_deep_detector/ /app/btc_ultra_deep_detector/
COPY btc_state_detector_production/ /app/btc_state_detector_production/
COPY btc_direction_predictor/ /app/btc_direction_predictor/

# Copy main monitoring scripts
COPY monitor_combined_realtime.py /app/
COPY write_prometheus_metrics.py /app/

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/metrics/btc_ml_metrics.prom') else 1)"

# Expose port for Prometheus metrics (if you set up an HTTP server later)
# EXPOSE 8000

# Labels for metadata
LABEL maintainer="BTC State Detector" \
      version="6L512_v1" \
      description="Bitcoin State Detection and Trading System"

# Default command - runs the combined real-time monitor
CMD ["python", "-u", "monitor_combined_realtime.py"]

