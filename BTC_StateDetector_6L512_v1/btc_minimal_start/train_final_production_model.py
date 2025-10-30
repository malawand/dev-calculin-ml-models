#!/usr/bin/env python3
"""
FINAL PRODUCTION MODEL TRAINER
Trains the optimal model discovered from comprehensive experiments:
- Horizon: 4h
- Threshold: 2.0%
- Features: 8 price derivatives
- Expected accuracy: ~65%
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix
)
from lightgbm import LGBMClassifier
import json
import logging
from datetime import datetime
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_horizon_to_periods(horizon: str) -> int:
    """Convert horizon string to 15-min periods."""
    mapping = {
        '1h': 4, '2h': 8, '4h': 16, '8h': 32, 
        '12h': 48, '24h': 96, '48h': 192
    }
    return mapping.get(horizon, 96)

def create_labels(df, price_col, horizon, threshold_pct):
    """Create directional labels with threshold."""
    periods = parse_horizon_to_periods(horizon)
    future_price = df[price_col].shift(-periods)
    price_change_pct = ((future_price - df[price_col]) / df[price_col]) * 100
    
    # Binary: UP if change > threshold, DOWN if change < -threshold
    # Remove samples in between (noise)
    labels = pd.Series(np.nan, index=df.index)
    labels[price_change_pct > threshold_pct] = 1  # UP
    labels[price_change_pct < -threshold_pct] = 0  # DOWN
    
    mask = ~labels.isna()
    return labels[mask], mask

def main():
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("🚀 FINAL PRODUCTION MODEL TRAINING")
    logger.info("="*80)
    logger.info("Configuration:")
    logger.info("  Horizon:   4h")
    logger.info("  Threshold: 2.0%")
    logger.info("  Features:  8 derivatives")
    logger.info("")
    
    # Load data
    data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_with_volume.parquet'
    
    logger.info(f"📥 Loading dataset...")
    df = pd.read_parquet(data_path)
    
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    logger.info(f"   Loaded {len(df):,} samples")
    logger.info(f"   Date range: {df.index.min()} → {df.index.max()}")
    
    # Define optimal features (from experiment)
    optimal_features = [
        'job:crypto_last_price:deriv30d',
        'job:crypto_last_price:deriv8h',
        'job:crypto_last_price:deriv12h',
        'job:crypto_last_price:deriv16h',
        'job:crypto_last_price:deriv14d',
        'job:crypto_last_price:deriv4h',
        'job:crypto_last_price:deriv24h',
        'job:crypto_last_price:deriv3d'
    ]
    
    # Verify features exist
    missing = [f for f in optimal_features if f not in df.columns]
    if missing:
        logger.error(f"Missing features: {missing}")
        return
    
    logger.info(f"\n✅ All {len(optimal_features)} features available")
    
    # Create labels
    logger.info(f"\n🎯 Creating labels (4h horizon, 2% threshold)...")
    price_col = 'crypto_last_price' if 'crypto_last_price' in df.columns else 'price'
    labels, mask = create_labels(df, price_col, '4h', 2.0)
    
    df_subset = df[mask].copy()
    y = labels.values.astype(int)
    
    up_pct = np.mean(y == 1)
    logger.info(f"   Samples: {len(df_subset):,}")
    logger.info(f"   UP:   {up_pct:.2%} ({int(np.sum(y==1)):,} samples)")
    logger.info(f"   DOWN: {1-up_pct:.2%} ({int(np.sum(y==0)):,} samples)")
    
    # Prepare features
    X = df_subset[optimal_features].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"\n🔧 Training with TimeSeriesSplit...")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        logger.info(f"\n  Fold {fold_idx}/5:")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = LGBMClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=7,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5
        
        logger.info(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")
        logger.info(f"    Accuracy:  {acc:.4f} ({acc:.2%})")
        logger.info(f"    Precision: {precision:.4f}")
        logger.info(f"    Recall:    {recall:.4f}")
        logger.info(f"    F1:        {f1:.4f}")
        logger.info(f"    MCC:       {mcc:.4f}")
        logger.info(f"    ROC-AUC:   {auc:.4f}")
        
        fold_results.append({
            'fold': fold_idx,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'roc_auc': auc
        })
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
    
    # Overall metrics
    logger.info(f"\n" + "="*80)
    logger.info("📊 OVERALL PERFORMANCE")
    logger.info("="*80)
    
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    overall_recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    overall_mcc = matthews_corrcoef(all_y_true, all_y_pred)
    overall_auc = roc_auc_score(all_y_true, all_y_proba)
    
    logger.info(f"Accuracy:  {overall_acc:.4f} ({overall_acc:.2%})")
    logger.info(f"Precision: {overall_precision:.4f}")
    logger.info(f"Recall:    {overall_recall:.4f}")
    logger.info(f"F1 Score:  {overall_f1:.4f}")
    logger.info(f"MCC:       {overall_mcc:.4f}")
    logger.info(f"ROC-AUC:   {overall_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"              Predicted")
    logger.info(f"              DOWN    UP")
    logger.info(f"Actual DOWN   {cm[0,0]:4d}  {cm[0,1]:4d}")
    logger.info(f"       UP     {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Train final model on ALL data
    logger.info(f"\n🎯 Training final model on ALL data...")
    
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)
    
    model_final = LGBMClassifier(
        random_state=42,
        n_estimators=300,  # More trees for final model
        learning_rate=0.03,  # Lower learning rate
        num_leaves=31,
        max_depth=7,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )
    model_final.fit(X_scaled_final, y)
    
    logger.info("   ✅ Final model trained")
    
    # Save model
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'final_production_model.pkl'
    scaler_path = model_dir / 'final_production_scaler.pkl'
    config_path = model_dir / 'final_production_config.json'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_final, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_final, f)
    
    config = {
        'horizon': '4h',
        'threshold_pct': 2.0,
        'features': optimal_features,
        'num_features': len(optimal_features),
        'training_samples': len(X),
        'up_ratio': float(up_pct),
        'performance': {
            'accuracy': overall_acc,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'mcc': overall_mcc,
            'roc_auc': overall_auc
        },
        'fold_results': fold_results,
        'trained_at': datetime.now().isoformat(),
        'data_range': {
            'start': str(df_subset.index.min()),
            'end': str(df_subset.index.max())
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\n💾 Model saved:")
    logger.info(f"   Model:  {model_path}")
    logger.info(f"   Scaler: {scaler_path}")
    logger.info(f"   Config: {config_path}")
    
    # Feature importance
    logger.info(f"\n📈 Feature Importance:")
    feature_importance = sorted(
        zip(optimal_features, model_final.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (feat, imp) in enumerate(feature_importance, 1):
        logger.info(f"   {i}. {feat:45s}: {imp:8.1f}")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"\n" + "="*80)
    logger.info(f"✅ TRAINING COMPLETE!")
    logger.info(f"="*80)
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"Final Accuracy: {overall_acc:.2%}")
    logger.info(f"Ready for production!")

if __name__ == "__main__":
    main()



