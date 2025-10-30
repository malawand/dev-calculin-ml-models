"""
Main training pipeline orchestration
"""
import yaml
import argparse
import logging
from pathlib import Path
import json
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.build_dataset import build_dataset_from_config
from src.features.engineering import engineer_features
from src.labels.labels import create_labels, get_X_y
from src.model.trees import LightGBMClassifier
from src.eval.metrics import evaluate_model, MetricsCalculator
from src.eval.walkforward import WalkForwardValidator


def train_pipeline(config_path: str = 'config.yaml', horizon: str = None):
    """
    Execute full training pipeline.
    
    Args:
        config_path: Path to config file
        horizon: Specific horizon to train (None = all)
    """
    logger.info("=" * 80)
    logger.info("BITCOIN DIRECTION PREDICTOR - TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Config loaded from: {config_path}")
    
    # Create artifacts directory
    Path('artifacts/models').mkdir(parents=True, exist_ok=True)
    Path('artifacts/reports').mkdir(parents=True, exist_ok=True)
    
    # Step 1: Build dataset
    logger.info("\n" + "="*80)
    logger.info("STEP 1: BUILDING DATASET")
    logger.info("="*80)
    
    dataset_path = 'artifacts/dataset.parquet'
    if Path(dataset_path).exists():
        logger.info(f"Loading existing dataset from {dataset_path}")
        import pandas as pd
        df = pd.read_parquet(dataset_path)
    else:
        logger.info("Building new dataset from Prometheus...")
        df = build_dataset_from_config(config, dataset_path)
    
    logger.info(f"Dataset shape: {df.shape}")
    
    # Step 2: Feature engineering
    logger.info("\n" + "="*80)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("="*80)
    
    df = engineer_features(df, config)
    logger.info(f"Engineered dataset shape: {df.shape}")
    
    # Step 3: Create labels
    logger.info("\n" + "="*80)
    logger.info("STEP 3: LABEL CREATION")
    logger.info("="*80)
    
    df = create_labels(df, config)
    
    # Step 4: Train models for each horizon
    horizons = [horizon] if horizon else config['labels']['horizons']
    
    results = {}
    
    for h in horizons:
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING HORIZON: {h}")
        logger.info("="*80)
        
        # Extract features and labels
        exclude_patterns = ['label_', 'return_15m', 'return_1h', 'return_4h', 'return_24h']
        feature_cols = [col for col in df.columns if not any(p in col for p in exclude_patterns)]
        
        X, y = get_X_y(df, h, feature_cols)
        
        # Convert to numpy
        X = X.values
        y = y.values
        
        logger.info(f"Features: {X.shape}")
        logger.info(f"Labels: {y.shape}")
        logger.info(f"Class balance: UP={y.sum()}/{len(y)} ({y.mean():.2%})")
        
        # Train/test split
        validator = WalkForwardValidator(n_splits=config['modeling']['n_splits'])
        test_ratio = config['modeling']['test_size_ratio']
        X_train, X_test, y_train, y_test, train_idx, test_idx = validator.get_train_test_split(X, y, test_ratio)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        logger.info("\nTraining LightGBM...")
        model = LightGBMClassifier(config)
        model.train(X_train_scaled, y_train)
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        
        # Get returns for trading metrics
        return_col = f'return_{h}'
        if return_col in df.columns:
            returns_test = df[return_col].iloc[test_idx].values
        else:
            returns_test = None
        
        metrics = evaluate_model(model, X_test_scaled, y_test, returns_test, config)
        
        # Print metrics
        calc = MetricsCalculator()
        calc.print_metrics(metrics, f"Test Metrics - {h}")
        
        # Feature importance
        logger.info("\nTop 10 Features:")
        importance = model.get_feature_importance(feature_cols)
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feat, imp) in enumerate(sorted_importance, 1):
            logger.info(f"  {i}. {feat:.<40} {imp:.4f}")
        
        # Save model
        model_path = f'artifacts/models/{h}_lightgbm.pkl'
        model.save(model_path)
        
        # Save scaler
        scaler_path = f'artifacts/models/{h}_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save feature names
        feature_path = f'artifacts/models/{h}_features.json'
        with open(feature_path, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        logger.info(f"Features saved to {feature_path}")
        
        # Save metrics
        metrics_path = f'artifacts/reports/{h}_metrics.json'
        # Convert numpy types to native Python types for JSON serialization
        metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save feature importance (convert numpy types to Python types)
        importance_path = f'artifacts/reports/{h}_feature_importance.json'
        importance_json = [[feat, int(imp)] for feat, imp in sorted_importance]
        with open(importance_path, 'w') as f:
            json.dump(importance_json, f, indent=2)
        logger.info(f"Feature importance saved to {importance_path}")
        
        results[h] = {
            'metrics': metrics,
            'model_path': model_path,
            'feature_importance': sorted_importance[:10]
        }
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("="*80)
    
    for h, result in results.items():
        logger.info(f"\n{h}:")
        logger.info(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        logger.info(f"  F1 (UP): {result['metrics']['f1_up']:.4f}")
        logger.info(f"  MCC: {result['metrics']['mcc']:.4f}")
        if 'sharpe_ratio' in result['metrics']:
            logger.info(f"  Sharpe: {result['metrics']['sharpe_ratio']:.4f}")
        logger.info(f"  Model: {result['model_path']}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("1. Review metrics in: artifacts/reports/")
    logger.info("2. Run inference: python -m src.pipeline.infer")
    logger.info("3. Check signals: cat artifacts/signal.json")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Bitcoin direction predictor')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--horizon', default=None, help='Specific horizon to train (e.g., 1h)')
    
    args = parser.parse_args()
    
    try:
        train_pipeline(args.config, args.horizon)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

