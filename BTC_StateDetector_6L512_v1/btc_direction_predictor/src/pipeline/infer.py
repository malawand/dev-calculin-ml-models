"""
Inference pipeline for live signal generation
"""
import yaml
import argparse
import logging
from pathlib import Path
import json
import sys
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.prom import PrometheusClient, get_all_metric_names
from src.features.engineering import engineer_features
from src.model.trees import LightGBMClassifier


class SignalGenerator:
    """Generate trading signals from latest data"""
    
    def __init__(self, config: dict):
        """Initialize signal generator"""
        self.config = config
        self.client = PrometheusClient(
            base_url=config['cortex']['base_url'],
            read_api=config['cortex']['read_api']
        )
        self.symbol = config['cortex']['symbol']
        
    def fetch_latest_data(self, lookback_hours: int = 48) -> pd.DataFrame:
        """Fetch latest data for inference"""
        logger.info(f"Fetching latest {lookback_hours}h of data...")
        
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)
        step = self.config['data']['step']
        
        # Get all metrics
        all_metrics = get_all_metric_names()
        metrics_to_fetch = []
        for category, metric_list in all_metrics.items():
            metrics_to_fetch.extend(metric_list)
        
        # Fetch (using simplified approach for speed)
        dataframes = []
        for metric in metrics_to_fetch[:20]:  # Limit for speed during inference
            try:
                query = self.client.build_metric_query(metric, self.symbol)
                df = self.client.query_range(query, start, end, step)
                if not df.empty:
                    df.columns = [metric.replace('job:crypto_last_price:', '').replace('crypto_last_price', 'price')]
                    dataframes.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch {metric}: {e}")
        
        if not dataframes:
            raise ValueError("No data fetched")
        
        # Merge
        result = dataframes[0]
        for df in dataframes[1:]:
            result = result.join(df, how='outer')
        
        result.sort_index(inplace=True)
        result.fillna(method='ffill', inplace=True)
        result.fillna(method='bfill', inplace=True)
        
        logger.info(f"Fetched {len(result)} rows, {len(result.columns)} columns")
        
        return result
    
    def generate_signals(self, horizons: list = None) -> dict:
        """Generate signals for all horizons"""
        logger.info("=" * 80)
        logger.info("GENERATING LIVE SIGNALS")
        logger.info("=" * 80)
        
        if horizons is None:
            horizons = self.config['labels']['horizons']
        
        # Fetch latest data
        df = self.fetch_latest_data()
        
        # Engineer features
        logger.info("Engineering features...")
        df = engineer_features(df, self.config)
        
        if len(df) == 0:
            raise ValueError("No data after feature engineering")
        
        # Get latest row
        latest = df.iloc[[-1]]
        logger.info(f"Latest timestamp: {latest.index[0]}")
        
        signals = {
            'timestamp': latest.index[0].isoformat(),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'horizons': {}
        }
        
        for horizon in horizons:
            logger.info(f"\nProcessing horizon: {horizon}")
            
            # Load model
            model_path = f'artifacts/models/{horizon}_lightgbm.pkl'
            scaler_path = f'artifacts/models/{horizon}_scaler.pkl'
            feature_path = f'artifacts/models/{horizon}_features.json'
            
            if not Path(model_path).exists():
                logger.warning(f"Model not found for {horizon}, skipping")
                continue
            
            # Load
            model = LightGBMClassifier.load(model_path)
            scaler = joblib.load(scaler_path)
            with open(feature_path) as f:
                feature_cols = json.load(f)
            
            # Prepare features
            available_features = [f for f in feature_cols if f in latest.columns]
            if len(available_features) < len(feature_cols) * 0.8:
                logger.warning(f"Only {len(available_features)}/{len(feature_cols)} features available")
            
            X = latest[available_features].fillna(0).values
            X_scaled = scaler.transform(X)
            
            # Predict
            y_pred = model.predict(X_scaled)[0]
            y_proba = model.predict_proba(X_scaled)[0]
            
            prob_up = float(y_proba[1])
            prob_threshold = self.config['backtest']['prob_threshold']
            
            # Generate signal
            if prob_up >= prob_threshold:
                signal = "LONG"
                confidence = "HIGH" if prob_up >= 0.65 else "MEDIUM"
            else:
                signal = "FLAT"
                confidence = "LOW"
            
            signals['horizons'][horizon] = {
                'signal': signal,
                'prob_up': prob_up,
                'prob_down': float(y_proba[0]),
                'confidence': confidence,
                'model_path': model_path
            }
            
            logger.info(f"  Signal: {signal}")
            logger.info(f"  P(UP): {prob_up:.4f}")
            logger.info(f"  Confidence: {confidence}")
        
        return signals


def inference_pipeline(config_path: str = 'config.yaml'):
    """Execute inference pipeline"""
    logger.info("BITCOIN DIRECTION PREDICTOR - INFERENCE")
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Generate signals
    generator = SignalGenerator(config)
    signals = generator.generate_signals()
    
    # Save signals
    output_dir = Path(config['inference']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON output
    json_path = output_dir / config['inference']['signal_file']
    with open(json_path, 'w') as f:
        json.dump(signals, f, indent=2)
    logger.info(f"\n✓ Signals saved to: {json_path}")
    
    # CSV output
    csv_path = output_dir / 'signal.csv'
    rows = []
    for horizon, data in signals['horizons'].items():
        rows.append({
            'timestamp': signals['timestamp'],
            'horizon': horizon,
            'signal': data['signal'],
            'prob_up': data['prob_up'],
            'confidence': data['confidence']
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"✓ Signals saved to: {csv_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SIGNAL SUMMARY")
    logger.info("="*80)
    for horizon, data in signals['horizons'].items():
        logger.info(f"{horizon:.<20} {data['signal']:.<10} P(UP)={data['prob_up']:.4f}")
    
    return signals


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate Bitcoin direction signals')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    try:
        inference_pipeline(args.config)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()



