# Bitcoin Direction Predictor - Prometheus Exporter
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY prometheus_exporter.py .
COPY predict_live.py .
COPY config.yaml .
COPY models/ ./models/
COPY results/ ./results/

# Create directory for parent project imports
RUN mkdir -p /app/btc_direction_predictor/src/features /app/btc_direction_predictor/src/labels

# Copy necessary modules from parent project
COPY ../btc_direction_predictor/src/features/ /app/btc_direction_predictor/src/features/
COPY ../btc_direction_predictor/src/labels/ /app/btc_direction_predictor/src/labels/

# Expose Prometheus metrics port
EXPOSE 9100

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:9100/metrics || exit 1

# Run exporter
CMD ["python", "prometheus_exporter.py", "--port", "9100", "--interval", "15"]



