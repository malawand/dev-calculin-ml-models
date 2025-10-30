#!/usr/bin/env python3
"""
Prometheus Metrics Exporter for Bitcoin Direction Predictions
Exposes model predictions as Prometheus metrics for Grafana monitoring
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from prometheus_client import start_http_server, Gauge, Counter, Info, Histogram
from prometheus_client import CollectorRegistry, write_to_textfile

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'btc_direction_predictor'))

from predict_live import LivePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create custom registry
registry = CollectorRegistry()

# Define Prometheus metrics
btc_prediction_probability_up = Gauge(
    'btc_prediction_probability_up',
    'Probability that BTC price will go UP in next 24h (0.0-1.0)',
    registry=registry
)

btc_prediction_probability_down = Gauge(
    'btc_prediction_probability_down',
    'Probability that BTC price will go DOWN in next 24h (0.0-1.0)',
    registry=registry
)

btc_current_price = Gauge(
    'btc_current_price',
    'Current BTC price in USD',
    registry=registry
)

btc_prediction_confidence = Gauge(
    'btc_prediction_confidence',
    'Confidence score: 0=LOW, 1=MEDIUM, 2=HIGH, 3=VERY_HIGH',
    registry=registry
)

btc_prediction_signal = Gauge(
    'btc_prediction_signal',
    'Signal strength: -3=STRONG_SELL, -2=SELL, -1=WEAK_SELL, 0=HOLD, 1=WEAK_BUY, 2=BUY, 3=STRONG_BUY',
    registry=registry
)

btc_prediction_direction = Gauge(
    'btc_prediction_direction',
    'Predicted direction: -1=DOWN, 0=NEUTRAL, 1=UP',
    registry=registry
)

btc_prediction_timestamp = Gauge(
    'btc_prediction_timestamp',
    'Unix timestamp of last prediction',
    registry=registry
)

btc_prediction_total = Counter(
    'btc_prediction_total',
    'Total number of predictions made',
    registry=registry
)

btc_prediction_errors_total = Counter(
    'btc_prediction_errors_total',
    'Total number of prediction errors',
    registry=registry
)

btc_prediction_latency = Histogram(
    'btc_prediction_latency_seconds',
    'Time taken to make a prediction',
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=registry
)

btc_model_info = Info(
    'btc_model',
    'Information about the prediction model',
    registry=registry
)

# Set model info (static)
btc_model_info.info({
    'version': '2.0',
    'accuracy': '76.46',
    'features': '6',
    'horizon': '24h',
    'model_type': 'LightGBM'
})


def map_confidence_to_number(confidence_str):
    """Map confidence string to number for Prometheus"""
    mapping = {
        'LOW': 0,
        'MEDIUM': 1,
        'HIGH': 2,
        'VERY_HIGH': 3
    }
    return mapping.get(confidence_str, 0)


def map_signal_to_number(signal_str):
    """Map signal strength to number for Prometheus"""
    mapping = {
        'STRONG_SELL': -3,
        'SELL': -2,
        'WEAK_SELL': -1,
        'HOLD': 0,
        'WEAK_BUY': 1,
        'BUY': 2,
        'STRONG_BUY': 3
    }
    return mapping.get(signal_str, 0)


def map_direction_to_number(direction_str):
    """Map direction to number for Prometheus"""
    mapping = {
        'DOWN': -1,
        'NEUTRAL': 0,
        'UP': 1
    }
    return mapping.get(direction_str, 0)


def update_metrics(predictor):
    """Fetch prediction and update Prometheus metrics"""
    try:
        start_time = time.time()
        
        # Get prediction
        prediction = predictor.predict()
        
        # Calculate latency
        latency = time.time() - start_time
        btc_prediction_latency.observe(latency)
        
        # Update metrics
        btc_prediction_probability_up.set(prediction['probability_up'])
        btc_prediction_probability_down.set(prediction['probability_down'])
        btc_current_price.set(prediction['current_price'])
        btc_prediction_confidence.set(map_confidence_to_number(prediction['confidence']))
        btc_prediction_signal.set(map_signal_to_number(prediction['signal_strength']))
        btc_prediction_direction.set(map_direction_to_number(prediction['direction']))
        btc_prediction_timestamp.set(time.time())
        
        # Increment counter
        btc_prediction_total.inc()
        
        logger.info(f"✅ Metrics updated: {prediction['direction']} "
                   f"(prob={prediction['probability_up']:.2%}, "
                   f"signal={prediction['signal_strength']}, "
                   f"latency={latency:.2f}s)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating metrics: {e}", exc_info=True)
        btc_prediction_errors_total.inc()
        return False


def export_to_file(output_file='btc_predictions.prom'):
    """
    Export metrics to .prom file for node_exporter textfile collector
    """
    try:
        write_to_textfile(output_file, registry)
        logger.info(f"💾 Metrics exported to {output_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Error exporting to file: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prometheus exporter for BTC predictions')
    parser.add_argument('--port', type=int, default=9100, help='HTTP port for metrics endpoint (default: 9100)')
    parser.add_argument('--interval', type=int, default=15, help='Update interval in minutes (default: 15)')
    parser.add_argument('--file-mode', action='store_true', help='Export to .prom file instead of HTTP server')
    parser.add_argument('--output', default='btc_predictions.prom', help='Output file for file mode')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    logger.info("🚀 Initializing BTC Direction Predictor...")
    predictor = LivePredictor(config_path=args.config)
    logger.info("✅ Predictor ready!")
    
    if args.file_mode:
        # File export mode (for node_exporter textfile collector)
        logger.info(f"📁 Running in FILE MODE - exporting to {args.output}")
        logger.info(f"   Update interval: {args.interval} minutes")
        logger.info(f"   Configure node_exporter with: --collector.textfile.directory=<dir>")
        
        try:
            while True:
                # Update metrics
                if update_metrics(predictor):
                    # Export to file
                    export_to_file(args.output)
                
                # Wait for next update
                logger.info(f"⏰ Waiting {args.interval} minutes until next update...")
                time.sleep(args.interval * 60)
                
        except KeyboardInterrupt:
            logger.info("\n👋 Stopped by user")
    
    else:
        # HTTP server mode (standard Prometheus scraping)
        logger.info(f"🌐 Starting HTTP server on port {args.port}")
        logger.info(f"   Metrics endpoint: http://localhost:{args.port}/metrics")
        logger.info(f"   Update interval: {args.interval} minutes")
        logger.info(f"   Add to prometheus.yml:")
        print(f"""
  - job_name: 'btc_predictions'
    static_configs:
      - targets: ['localhost:{args.port}']
    scrape_interval: 1m
        """)
        
        # Start HTTP server
        start_http_server(args.port, registry=registry)
        
        try:
            # Initial update
            update_metrics(predictor)
            
            # Continuous updates
            while True:
                time.sleep(args.interval * 60)
                update_metrics(predictor)
                
        except KeyboardInterrupt:
            logger.info("\n👋 Stopped by user")


if __name__ == "__main__":
    main()



