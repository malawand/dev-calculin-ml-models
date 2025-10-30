#!/usr/bin/env python3
"""
SCALPING MODEL - Conservative Day Trading
Train on 1-minute data with short horizons for multiple daily trades
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from lightgbm import LGBMClassifier
import json
import logging
from datetime import datetime
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_horizon_to_periods(horizon: str) -> int:
    """Convert horizon to 1-min periods."""
    mapping = {
        '15min': 15, '30min': 30, '45min': 45,
        '1h': 60, '2h': 120, '4h': 240
    }
    return mapping.get(horizon, 60)

def create_three_class_labels(df, price_col, horizon, up_threshold, down_threshold):
    """Create UP/DOWN/SIDEWAYS labels."""
    periods = parse_horizon_to_periods(horizon)
    future_price = df[price_col].shift(-periods)
    price_change_pct = ((future_price - df[price_col]) / df[price_col]) * 100
    
    labels = pd.Series(1, index=df.index)  # SIDEWAYS
    labels[price_change_pct > up_threshold] = 2  # UP
    labels[price_change_pct < -down_threshold] = 0  # DOWN
    
    mask = ~labels.isna() & ~price_change_pct.isna()
    return labels[mask], mask

def main():
    logger.info("="*80)
    logger.info("🎯 SCALPING MODEL TRAINER")
    logger.info("="*80)
    logger.info("Conservative day trading on 1-minute data")
    logger.info("")
    
    # Load 1-minute data
    # Check for full dataset first, fallback to 30-day
    full_data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/scalping_1min_full.parquet'
    short_data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/scalping_1min_30days.parquet'
    
    if full_data_path.exists():
        data_path = full_data_path
        logger.info(f"📥 Loading FULL 2.5-year 1-minute dataset...")
    else:
        data_path = short_data_path
        logger.info(f"📥 Loading 30-day 1-minute dataset...")
    
    df_raw = pd.read_parquet(data_path)
    
    if 'timestamp' not in df_raw.columns:
        df_raw = df_raw.reset_index()
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], utc=True)
    df_raw = df_raw.set_index('timestamp')
    df_raw = df_raw.sort_index()
    
    logger.info(f"   Samples: {len(df_raw):,}")
    logger.info(f"   Features: {len(df_raw.columns)}")
    logger.info(f"   Date range: {df_raw.index.min()} → {df_raw.index.max()}")
    logger.info(f"   Duration: {(df_raw.index.max() - df_raw.index.min()).days} days")
    
    # Features for scalping (short-term focus)
    scalping_features = [
        'crypto_last_price',
        'job:crypto_last_price:deriv5m',
        'job:crypto_last_price:deriv10m',
        'job:crypto_last_price:deriv15m',
        'job:crypto_last_price:deriv30m',
        'job:crypto_last_price:avg5m',
        'job:crypto_last_price:avg10m',
        'job:crypto_last_price:avg15m',
        'crypto_volume',
        'job:crypto_volume:deriv5m',
        'job:crypto_volume:avg5m',
    ]
    
    # Verify features
    available = [f for f in scalping_features if f in df_raw.columns]
    logger.info(f"\n✅ Using {len(available)}/{len(scalping_features)} features")
    
    # Test configurations - AGGRESSIVE SCALPING (prioritize UP/DOWN detection)
    configs = [
        # Ultra-aggressive (catch tiny moves)
        {'horizon': '15min', 'up': 0.03, 'down': 0.03, 'name': '15min_±0.03%'},
        {'horizon': '15min', 'up': 0.05, 'down': 0.05, 'name': '15min_±0.05%'},
        {'horizon': '15min', 'up': 0.08, 'down': 0.08, 'name': '15min_±0.08%'},
        # Fast scalps
        {'horizon': '30min', 'up': 0.05, 'down': 0.05, 'name': '30min_±0.05%'},
        {'horizon': '30min', 'up': 0.08, 'down': 0.08, 'name': '30min_±0.08%'},
        {'horizon': '30min', 'up': 0.10, 'down': 0.10, 'name': '30min_±0.10%'},
        # Medium scalps
        {'horizon': '1h', 'up': 0.08, 'down': 0.08, 'name': '1h_±0.08%'},
        {'horizon': '1h', 'up': 0.10, 'down': 0.10, 'name': '1h_±0.10%'},
        {'horizon': '1h', 'up': 0.15, 'down': 0.15, 'name': '1h_±0.15%'},
        {'horizon': '1h', 'up': 0.20, 'down': 0.20, 'name': '1h_±0.20%'},
    ]
    
    price_col = 'crypto_last_price'
    all_results = []
    
    for config_idx, config in enumerate(configs, 1):
        horizon = config['horizon']
        up_thresh = config['up']
        down_thresh = config['down']
        name = config['name']
        
        logger.info("\n" + "="*80)
        logger.info(f"CONFIG {config_idx}/{len(configs)}: {name}")
        logger.info("="*80)
        
        # Create labels
        labels, mask = create_three_class_labels(df_raw, price_col, horizon, up_thresh, down_thresh)
        df_subset = df_raw[mask].copy()
        y = labels.values.astype(int)
        
        # Distribution
        up_pct = np.mean(y == 2)
        sideways_pct = np.mean(y == 1)
        down_pct = np.mean(y == 0)
        
        logger.info(f"\n📊 Data:")
        logger.info(f"   Samples: {len(df_subset):,}")
        logger.info(f"   UP:       {up_pct:.1%} ({int(np.sum(y==2)):,})")
        logger.info(f"   SIDEWAYS: {sideways_pct:.1%} ({int(np.sum(y==1)):,})")
        logger.info(f"   DOWN:     {down_pct:.1%} ({int(np.sum(y==0)):,})")
        
        # Prepare features
        X = df_subset[available].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Time series CV
        tscv = TimeSeriesSplit(n_splits=5)
        
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LGBMClassifier(
                random_state=42,
                n_estimators=200,
                learning_rate=0.03,
                num_leaves=30,
                max_depth=6,
                class_weight='balanced',  # Handle imbalanced classes
                verbose=-1
            )
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)
            
            # Ensure y_proba has 3 columns (DOWN, SIDEWAYS, UP)
            if y_proba.shape[1] < 3:
                # Missing classes - pad with zeros
                y_proba_full = np.zeros((y_proba.shape[0], 3))
                classes_present = model.classes_
                for i, cls in enumerate(classes_present):
                    y_proba_full[:, cls] = y_proba[:, i]
                y_proba = y_proba_full
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)
        
        # Metrics
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.vstack(all_y_proba)  # Stack list of arrays properly
        
        overall_acc = accuracy_score(all_y_true, all_y_pred)
        cm = confusion_matrix(all_y_true, all_y_pred)
        
        down_acc = cm[0,0] / cm[0,:].sum() if cm[0,:].sum() > 0 else 0
        side_acc = cm[1,1] / cm[1,:].sum() if cm[1,:].sum() > 0 else 0
        up_acc = cm[2,2] / cm[2,:].sum() if cm[2,:].sum() > 0 else 0
        
        # Directional accuracy
        directional_mask = (all_y_pred != 1)
        if directional_mask.sum() > 0:
            y_true_dir = all_y_true[directional_mask]
            y_pred_dir = all_y_pred[directional_mask]
            y_true_bin = (y_true_dir == 2).astype(int)
            y_pred_bin = (y_pred_dir == 2).astype(int)
            directional_acc = accuracy_score(y_true_bin, y_pred_bin)
            directional_pct = directional_mask.sum() / len(all_y_true)
        else:
            directional_acc = 0
            directional_pct = 0
        
        # High confidence signals
        max_proba = np.max(all_y_proba, axis=1)
        high_conf_mask = max_proba > 0.8
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(
                all_y_true[high_conf_mask],
                all_y_pred[high_conf_mask]
            )
            high_conf_pct = high_conf_mask.sum() / len(all_y_true)
        else:
            high_conf_acc = 0
            high_conf_pct = 0
        
        logger.info(f"\n📈 Results:")
        logger.info(f"   Overall Accuracy:     {overall_acc:.2%}")
        logger.info(f"   Directional Accuracy: {directional_acc:.2%} ({directional_pct:.1%} signals)")
        logger.info(f"   High Conf (>80%):     {high_conf_acc:.2%} ({high_conf_pct:.1%} signals)")
        logger.info(f"\n   Per-Class:")
        logger.info(f"     DOWN:     {down_acc:.2%}")
        logger.info(f"     SIDEWAYS: {side_acc:.2%}")
        logger.info(f"     UP:       {up_acc:.2%}")
        
        # Balanced score - heavily prioritize UP/DOWN accuracy
        up_down_avg = (down_acc + up_acc) / 2  # Average of UP and DOWN detection
        balanced_score = (overall_acc * 0.2) + (directional_acc * 0.4) + (high_conf_acc * 0.2) + (up_down_avg * 0.2)
        
        all_results.append({
            'name': name,
            'horizon': horizon,
            'threshold': up_thresh,
            'samples': len(df_subset),
            'overall_accuracy': overall_acc,
            'directional_accuracy': directional_acc,
            'directional_signals_pct': float(directional_pct),
            'high_conf_accuracy': high_conf_acc,
            'high_conf_pct': float(high_conf_pct),
            'balanced_score': balanced_score,
            'down_acc': down_acc,
            'sideways_acc': side_acc,
            'up_acc': up_acc,
        })
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("🏆 SCALPING MODEL RESULTS")
    logger.info("="*80)
    
    sorted_results = sorted(all_results, key=lambda x: x['balanced_score'], reverse=True)
    
    logger.info("\n📊 Ranked by Balanced Score:\n")
    for i, r in enumerate(sorted_results, 1):
        logger.info(f"{i}. {r['name']:15} | Score: {r['balanced_score']:.2%} | "
                   f"Dir: {r['directional_accuracy']:.2%} ({r['directional_signals_pct']:.1%}) | "
                   f"HighConf: {r['high_conf_accuracy']:.2%} ({r['high_conf_pct']:.1%})")
    
    # Best config
    best = sorted_results[0]
    logger.info(f"\n" + "="*80)
    logger.info(f"🥇 BEST SCALPING CONFIGURATION")
    logger.info(f"="*80)
    logger.info(f"Config:                {best['name']}")
    logger.info(f"Horizon:               {best['horizon']}")
    logger.info(f"Threshold:             ±{best['threshold']}%")
    logger.info(f"Balanced Score:        {best['balanced_score']:.2%}")
    logger.info(f"Overall Accuracy:      {best['overall_accuracy']:.2%}")
    logger.info(f"Directional Accuracy:  {best['directional_accuracy']:.2%}")
    logger.info(f"Trading Signals:       {best['directional_signals_pct']:.1%} of time")
    logger.info(f"High Conf Accuracy:    {best['high_conf_accuracy']:.2%}")
    logger.info(f"High Conf Signals:     {best['high_conf_pct']:.1%} of time")
    
    # Save
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'scalping_model_results.json'
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'data_samples': len(df_raw),
            'features': available,
            'best_config': best,
            'all_results': sorted_results
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved: {results_path}")
    
    # Train final model
    logger.info(f"\n" + "="*80)
    logger.info(f"🎯 TRAINING FINAL SCALPING MODEL")
    logger.info(f"="*80)
    
    labels, mask = create_three_class_labels(
        df_raw, price_col, best['horizon'], 
        best['threshold'], best['threshold']
    )
    df_subset = df_raw[mask].copy()
    y = labels.values.astype(int)
    X = df_subset[available].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    model_final = LGBMClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=30,
        max_depth=6,
        class_weight='balanced',  # Handle imbalanced classes
        verbose=-1
    )
    model_final.fit(X_scaled, y)
    
    # Save model
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'scalping_model.pkl'
    scaler_path = model_dir / 'scalping_scaler.pkl'
    config_path = model_dir / 'scalping_config.json'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_final, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_final, f)
    
    with open(config_path, 'w') as f:
        json.dump({
            'horizon': best['horizon'],
            'threshold': best['threshold'],
            'features': available,
            'classes': ['DOWN', 'SIDEWAYS', 'UP'],
            'performance': best,
            'trained_at': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"\n💾 Model saved:")
    logger.info(f"   {model_path}")
    logger.info(f"   {scaler_path}")
    logger.info(f"   {config_path}")
    
    logger.info(f"\n✅ SCALPING MODEL READY!")
    logger.info(f"\n🎯 Next: Fetch 2.5 years of data for better training")

if __name__ == "__main__":
    main()

