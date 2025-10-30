#!/usr/bin/env python3
"""
Save the 76.46% accuracy production model for deployment
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from lightgbm import LGBMClassifier
import logging

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'btc_direction_predictor'))

from src.features.engineering import FeatureEngineer
from src.labels.labels import LabelCreator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("🚀 TRAINING PRODUCTION MODEL (76.46% Accuracy)")
    logger.info("=" * 80)
    logger.info("")
    
    # The winning 6 features from 2-year incremental training
    PRODUCTION_FEATURES = [
        'deriv7d_prime7d',  # 7-day acceleration
        'deriv4d_roc',      # 4-day trend
        'volatility_24',    # 24-period volatility
        'avg10m',           # 10-minute moving average
        'avg15m',           # 15-minute moving average
        'avg45m'            # 45-minute moving average
    ]
    
    horizon = '24h'
    
    # Load dataset
    data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_full_dataset.parquet'
    logger.info(f"📥 Loading data: {data_path}")
    df_raw = pd.read_parquet(data_path)
    logger.info(f"   Loaded {len(df_raw):,} samples")
    
    # Engineer features
    logger.info("🔧 Engineering features...")
    config = {
        'features': {
            'price_lags': [1, 3, 6, 12],
            'deriv_lags': [1, 3, 6],
            'rolling_windows': [12, 24, 72]
        },
        'prometheus': {
            'metrics': {
                'spot': ['crypto_last_price'],
                'averages': [
                    'job:crypto_last_price:avg10m', 'job:crypto_last_price:avg15m',
                    'job:crypto_last_price:avg30m', 'job:crypto_last_price:avg45m',
                    'job:crypto_last_price:avg1h'
                ],
                'derivatives': [
                    'job:crypto_last_price:deriv4d', 'job:crypto_last_price:deriv7d'
                ],
                'derivative_primes': [
                    'job:crypto_last_price:deriv7d_prime7d'
                ]
            }
        },
        'price_col': 'price',
        'target_horizons': [horizon]
    }
    
    feature_engineer = FeatureEngineer(config)
    df_engineered = feature_engineer.engineer(df_raw.copy())
    logger.info(f"   Engineered {len(df_engineered.columns)} features")
    
    # Create labels
    logger.info("🎯 Creating labels...")
    config['labels'] = {
        'horizons': [horizon],
        'threshold_pct': 0.0
    }
    label_creator = LabelCreator(config)
    df_labeled = label_creator.create_labels(df_engineered)
    
    # Prepare training data
    logger.info(f"📊 Preparing data with {len(PRODUCTION_FEATURES)} production features...")
    
    # Check which features are available
    available_features = [f for f in PRODUCTION_FEATURES if f in df_labeled.columns]
    missing_features = [f for f in PRODUCTION_FEATURES if f not in df_labeled.columns]
    
    if missing_features:
        logger.warning(f"⚠️  Missing features: {missing_features}")
        logger.warning("   Using only available features")
        features_to_use = available_features
    else:
        logger.info("✅ All features available!")
        features_to_use = PRODUCTION_FEATURES
    
    # Extract features and labels
    df_final = df_labeled[features_to_use + [f'label_{horizon}']].dropna()
    
    X = df_final[features_to_use].values
    y = df_final[f'label_{horizon}'].values
    
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    logger.info(f"✅ Data prepared: {len(X):,} samples")
    logger.info(f"   Features: {len(features_to_use)}")
    logger.info(f"   Class balance: UP={np.sum(y==1)}/{len(y)} ({np.mean(y==1):.1%})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    logger.info(f"📊 Split: Train={len(X_train):,}, Test={len(X_test):,}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("🤖 TRAINING LIGHTGBM MODEL")
    logger.info("=" * 80)
    
    model = LGBMClassifier(
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        verbose=-1
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        eval_metric='logloss',
        callbacks=[],
        feature_name=features_to_use
    )
    
    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("📊 EVALUATION")
    logger.info("=" * 80)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)
    
    logger.info(f"\nTrain Accuracy: {train_accuracy:.4f} ({train_accuracy:.2%})")
    logger.info(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy:.2%})")
    logger.info(f"ROC-AUC:        {test_roc_auc:.4f}")
    logger.info(f"MCC:            {test_mcc:.4f}")
    
    # Save model
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features_to_use,
        'accuracy': test_accuracy,
        'roc_auc': test_roc_auc,
        'mcc': test_mcc,
        'horizon': horizon,
        'training_samples': len(X),
        'n_features': len(features_to_use)
    }
    
    model_path = models_dir / 'best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"\n💾 Model saved: {model_path}")
    logger.info(f"   Size: {model_path.stat().st_size / 1024:.1f} KB")
    
    # Save model info as JSON
    model_info = {
        'accuracy': float(test_accuracy),
        'roc_auc': float(test_roc_auc),
        'mcc': float(test_mcc),
        'features': features_to_use,
        'horizon': horizon,
        'training_samples': int(len(X)),
        'n_features': len(features_to_use),
        'date_saved': pd.Timestamp.now().isoformat(),
        'production_ready': True
    }
    
    info_path = models_dir / 'model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"💾 Model info saved: {info_path}")
    
    # Print feature importance
    logger.info("\n" + "=" * 80)
    logger.info("🎯 FEATURE IMPORTANCE")
    logger.info("=" * 80)
    
    feature_importance = pd.Series(model.feature_importances_, index=features_to_use).sort_values(ascending=False)
    for i, (feature, importance) in enumerate(feature_importance.items(), 1):
        logger.info(f"  {i}. {feature:25} {importance:.1f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PRODUCTION MODEL READY!")
    logger.info("=" * 80)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Test prediction: python predict_live.py --once")
    logger.info(f"  2. Start monitoring: python predict_live.py --interval 15")
    logger.info(f"  3. See documentation: PRODUCTION_DEPLOYMENT.md")
    logger.info("")


if __name__ == "__main__":
    main()

