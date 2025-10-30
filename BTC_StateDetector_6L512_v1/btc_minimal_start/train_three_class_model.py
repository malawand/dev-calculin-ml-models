#!/usr/bin/env python3
"""
THREE-CLASS CLASSIFIER: UP, DOWN, SIDEWAYS
This model detects market regime and only provides signals for clear directional moves.
Training on ALL 2.5 years of data.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
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

def create_three_class_labels(df, price_col, horizon, up_threshold, down_threshold):
    """
    Create three-class labels:
    - 2 = UP (price increases > up_threshold)
    - 1 = SIDEWAYS (price change between thresholds)
    - 0 = DOWN (price decreases < down_threshold)
    """
    periods = parse_horizon_to_periods(horizon)
    future_price = df[price_col].shift(-periods)
    price_change_pct = ((future_price - df[price_col]) / df[price_col]) * 100
    
    # Three classes
    labels = pd.Series(1, index=df.index)  # Default SIDEWAYS
    labels[price_change_pct > up_threshold] = 2  # UP
    labels[price_change_pct < -down_threshold] = 0  # DOWN
    
    mask = ~labels.isna() & ~price_change_pct.isna()
    return labels[mask], mask

def test_multiple_configs():
    """Test multiple horizon and threshold configurations."""
    
    logger.info("="*80)
    logger.info("🚀 THREE-CLASS CLASSIFIER - COMPREHENSIVE SEARCH")
    logger.info("="*80)
    logger.info("Training on ALL 2.5 years of data")
    logger.info("Classes: UP / SIDEWAYS / DOWN")
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
    logger.info(f"   Duration: {(df.index.max() - df.index.min()).days} days")
    
    # Define optimal features (from previous experiments)
    optimal_features = [
        'job:crypto_last_price:deriv3d',
        'job:crypto_last_price:deriv14d',
        'job:crypto_last_price:deriv30d',
        'job:crypto_last_price:deriv24h',
        'job:crypto_last_price:deriv16h',
        'job:crypto_last_price:deriv8h',
        'job:crypto_last_price:deriv12h',
        'job:crypto_last_price:deriv4h'
    ]
    
    # Test configurations
    configs = [
        {'horizon': '4h', 'up_threshold': 1.5, 'down_threshold': 1.5},
        {'horizon': '4h', 'up_threshold': 2.0, 'down_threshold': 2.0},
        {'horizon': '4h', 'up_threshold': 1.0, 'down_threshold': 1.0},
        {'horizon': '8h', 'up_threshold': 2.0, 'down_threshold': 2.0},
        {'horizon': '8h', 'up_threshold': 2.5, 'down_threshold': 2.5},
        {'horizon': '12h', 'up_threshold': 2.0, 'down_threshold': 2.0},
        {'horizon': '12h', 'up_threshold': 3.0, 'down_threshold': 3.0},
    ]
    
    all_results = []
    price_col = 'crypto_last_price' if 'crypto_last_price' in df.columns else 'price'
    
    for config_idx, config in enumerate(configs, 1):
        horizon = config['horizon']
        up_thresh = config['up_threshold']
        down_thresh = config['down_threshold']
        
        logger.info("\n" + "="*80)
        logger.info(f"CONFIG {config_idx}/{len(configs)}")
        logger.info(f"Horizon: {horizon} | UP threshold: {up_thresh}% | DOWN threshold: {down_thresh}%")
        logger.info("="*80)
        
        # Create labels
        labels, mask = create_three_class_labels(df, price_col, horizon, up_thresh, down_thresh)
        df_subset = df[mask].copy()
        y = labels.values.astype(int)
        
        # Class distribution
        up_pct = np.mean(y == 2)
        sideways_pct = np.mean(y == 1)
        down_pct = np.mean(y == 0)
        
        logger.info(f"\n📊 Data Statistics:")
        logger.info(f"   Total samples: {len(df_subset):,}")
        logger.info(f"   UP:       {up_pct:.2%} ({int(np.sum(y==2)):,} samples)")
        logger.info(f"   SIDEWAYS: {sideways_pct:.2%} ({int(np.sum(y==1)):,} samples)")
        logger.info(f"   DOWN:     {down_pct:.2%} ({int(np.sum(y==0)):,} samples)")
        
        # Prepare features
        X = df_subset[optimal_features].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        fold_results = []
        all_y_true = []
        all_y_pred = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
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
                verbose=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            
            fold_results.append({'fold': fold_idx, 'accuracy': acc})
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
        
        # Overall metrics
        overall_acc = accuracy_score(all_y_true, all_y_pred)
        
        # Per-class metrics
        precision_macro = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
        
        logger.info(f"\n📈 Results:")
        logger.info(f"   Overall Accuracy:    {overall_acc:.4f} ({overall_acc:.2%})")
        logger.info(f"   Macro Precision:     {precision_macro:.4f}")
        logger.info(f"   Macro Recall:        {recall_macro:.4f}")
        logger.info(f"   Macro F1:            {f1_macro:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        logger.info(f"\n📊 Confusion Matrix:")
        logger.info(f"                Predicted")
        logger.info(f"                DOWN  SIDE   UP")
        logger.info(f"Actual DOWN    {cm[0,0]:5d} {cm[0,1]:5d} {cm[0,2]:5d}")
        logger.info(f"       SIDE    {cm[1,0]:5d} {cm[1,1]:5d} {cm[1,2]:5d}")
        logger.info(f"       UP      {cm[2,0]:5d} {cm[2,1]:5d} {cm[2,2]:5d}")
        
        # Per-class accuracy
        down_acc = cm[0,0] / cm[0,:].sum() if cm[0,:].sum() > 0 else 0
        side_acc = cm[1,1] / cm[1,:].sum() if cm[1,:].sum() > 0 else 0
        up_acc = cm[2,2] / cm[2,:].sum() if cm[2,:].sum() > 0 else 0
        
        logger.info(f"\n🎯 Per-Class Accuracy:")
        logger.info(f"   DOWN:     {down_acc:.2%}")
        logger.info(f"   SIDEWAYS: {side_acc:.2%}")
        logger.info(f"   UP:       {up_acc:.2%}")
        
        # Directional accuracy (ignoring sideways)
        # Only look at samples where we predicted UP or DOWN
        directional_mask = (np.array(all_y_pred) != 1)
        if directional_mask.sum() > 0:
            y_true_directional = np.array(all_y_true)[directional_mask]
            y_pred_directional = np.array(all_y_pred)[directional_mask]
            
            # Convert to binary: UP (2) -> 1, DOWN (0) -> 0
            y_true_binary = (y_true_directional == 2).astype(int)
            y_pred_binary = (y_pred_directional == 2).astype(int)
            
            directional_acc = accuracy_score(y_true_binary, y_pred_binary)
            logger.info(f"\n💡 Directional Accuracy (when trading):")
            logger.info(f"   Accuracy: {directional_acc:.2%}")
            logger.info(f"   Signals:  {directional_mask.sum():,} / {len(all_y_true):,} ({directional_mask.sum()/len(all_y_true):.1%})")
        else:
            directional_acc = 0
            logger.info(f"\n💡 No directional signals generated")
        
        all_results.append({
            'config_idx': config_idx,
            'horizon': horizon,
            'up_threshold': up_thresh,
            'down_threshold': down_thresh,
            'samples': len(df_subset),
            'up_pct': float(up_pct),
            'sideways_pct': float(sideways_pct),
            'down_pct': float(down_pct),
            'overall_accuracy': overall_acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'down_accuracy': down_acc,
            'sideways_accuracy': side_acc,
            'up_accuracy': up_acc,
            'directional_accuracy': directional_acc,
            'directional_signals_pct': float(directional_mask.sum() / len(all_y_true)) if directional_mask.sum() > 0 else 0
        })
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🏆 FINAL RESULTS - ALL CONFIGURATIONS")
    logger.info("="*80)
    
    sorted_results = sorted(all_results, key=lambda x: x['overall_accuracy'], reverse=True)
    
    logger.info("\n📊 Ranked by Overall Accuracy:\n")
    for i, r in enumerate(sorted_results, 1):
        logger.info(f"{i}. {r['horizon']:4} | ±{r['up_threshold']:.1f}% | "
                   f"Acc: {r['overall_accuracy']:.2%} | "
                   f"Dir Acc: {r['directional_accuracy']:.2%} | "
                   f"Samples: {r['samples']:,}")
    
    logger.info("\n📊 Ranked by Directional Accuracy (when trading):\n")
    sorted_by_dir = sorted(all_results, key=lambda x: x['directional_accuracy'], reverse=True)
    for i, r in enumerate(sorted_by_dir, 1):
        logger.info(f"{i}. {r['horizon']:4} | ±{r['up_threshold']:.1f}% | "
                   f"Dir Acc: {r['directional_accuracy']:.2%} | "
                   f"Signals: {r['directional_signals_pct']:.1%} | "
                   f"Overall: {r['overall_accuracy']:.2%}")
    
    # Best configuration
    best = sorted_results[0]
    logger.info(f"\n" + "="*80)
    logger.info(f"🥇 BEST CONFIGURATION")
    logger.info(f"="*80)
    logger.info(f"Horizon:              {best['horizon']}")
    logger.info(f"Thresholds:           ±{best['up_threshold']}%")
    logger.info(f"Overall Accuracy:     {best['overall_accuracy']:.2%}")
    logger.info(f"Directional Accuracy: {best['directional_accuracy']:.2%}")
    logger.info(f"Trading Signals:      {best['directional_signals_pct']:.1%} of time")
    logger.info(f"Samples:              {best['samples']:,} (from {len(df):,} total)")
    logger.info(f"\nClass Distribution:")
    logger.info(f"  UP:       {best['up_pct']:.1%}")
    logger.info(f"  SIDEWAYS: {best['sideways_pct']:.1%}")
    logger.info(f"  DOWN:     {best['down_pct']:.1%}")
    logger.info(f"\nPer-Class Accuracy:")
    logger.info(f"  DOWN:     {best['down_accuracy']:.2%}")
    logger.info(f"  SIDEWAYS: {best['sideways_accuracy']:.2%}")
    logger.info(f"  UP:       {best['up_accuracy']:.2%}")
    
    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'three_class_results.json'
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'best_config': best,
            'all_results': sorted_results
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved: {results_path}")
    
    # Train final model with best config
    logger.info(f"\n" + "="*80)
    logger.info(f"🎯 TRAINING FINAL MODEL WITH BEST CONFIG")
    logger.info(f"="*80)
    
    labels, mask = create_three_class_labels(
        df, price_col, best['horizon'], 
        best['up_threshold'], best['down_threshold']
    )
    df_subset = df[mask].copy()
    y = labels.values.astype(int)
    X = df_subset[optimal_features].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)
    
    model_final = LGBMClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=7,
        verbose=-1
    )
    model_final.fit(X_scaled_final, y)
    
    logger.info("   ✅ Final model trained on ALL data")
    
    # Save model
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'three_class_model.pkl'
    scaler_path = model_dir / 'three_class_scaler.pkl'
    config_path = model_dir / 'three_class_config.json'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_final, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_final, f)
    
    config_final = {
        'horizon': best['horizon'],
        'up_threshold': best['up_threshold'],
        'down_threshold': best['down_threshold'],
        'features': optimal_features,
        'classes': ['DOWN', 'SIDEWAYS', 'UP'],
        'performance': best,
        'trained_at': datetime.now().isoformat(),
        'training_samples': len(X)
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_final, f, indent=2)
    
    logger.info(f"\n💾 Model saved:")
    logger.info(f"   Model:  {model_path}")
    logger.info(f"   Scaler: {scaler_path}")
    logger.info(f"   Config: {config_path}")
    
    logger.info(f"\n" + "="*80)
    logger.info(f"✅ COMPLETE!")
    logger.info(f"="*80)

if __name__ == "__main__":
    test_multiple_configs()



