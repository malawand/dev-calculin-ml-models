#!/usr/bin/env python3
"""
DAILY DIRECTIONAL PREDICTOR
Predict direction for each day (24h horizon)
Query ALL available data, clean it, and make daily predictions
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from lightgbm import LGBMClassifier
import json
import logging
from datetime import datetime, timedelta
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_daily_labels(df, price_col, up_threshold, down_threshold):
    """
    Create three-class labels for 24h (daily) predictions:
    - 2 = UP (price increases > up_threshold in next 24h)
    - 1 = SIDEWAYS (price change between thresholds)
    - 0 = DOWN (price decreases < down_threshold in next 24h)
    """
    # 24h ahead = 96 periods (15min bars)
    periods = 96
    future_price = df[price_col].shift(-periods)
    price_change_pct = ((future_price - df[price_col]) / df[price_col]) * 100
    
    # Three classes
    labels = pd.Series(1, index=df.index)  # Default SIDEWAYS
    labels[price_change_pct > up_threshold] = 2  # UP
    labels[price_change_pct < -down_threshold] = 0  # DOWN
    
    mask = ~labels.isna() & ~price_change_pct.isna()
    return labels[mask], mask, price_change_pct[mask]

def main():
    logger.info("="*80)
    logger.info("🚀 DAILY DIRECTIONAL PREDICTOR")
    logger.info("="*80)
    logger.info("Goal: Predict daily direction (24h ahead)")
    logger.info("Training on ALL available data")
    logger.info("")
    
    # Load ALL data
    data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_with_volume.parquet'
    
    logger.info(f"📥 Loading dataset...")
    df_raw = pd.read_parquet(data_path)
    
    if 'timestamp' in df_raw.columns:
        df_raw = df_raw.set_index('timestamp')
    df_raw.index = pd.to_datetime(df_raw.index, utc=True)
    df_raw = df_raw.sort_index()
    
    logger.info(f"   Raw samples: {len(df_raw):,}")
    logger.info(f"   Date range: {df_raw.index.min()} → {df_raw.index.max()}")
    logger.info(f"   Duration: {(df_raw.index.max() - df_raw.index.min()).days} days")
    logger.info(f"   Columns: {len(df_raw.columns)}")
    
    # Clean data
    logger.info(f"\n🧹 Cleaning data...")
    
    # Remove rows with too many NaNs
    before_clean = len(df_raw)
    nan_threshold = 0.5  # Drop rows with >50% NaN
    df_raw = df_raw.dropna(thresh=int(len(df_raw.columns) * nan_threshold))
    after_clean = len(df_raw)
    logger.info(f"   Removed {before_clean - after_clean:,} rows with excessive NaNs")
    
    # Fill remaining NaNs with forward fill then backward fill
    df_raw = df_raw.fillna(method='ffill').fillna(method='bfill')
    
    # Remove infinite values
    df_raw = df_raw.replace([np.inf, -np.inf], np.nan)
    df_raw = df_raw.fillna(0)
    
    logger.info(f"   Clean samples: {len(df_raw):,}")
    
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
    
    # Verify features exist
    missing_features = [f for f in optimal_features if f not in df_raw.columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        return
    
    logger.info(f"\n✅ All {len(optimal_features)} features available")
    
    # Test multiple thresholds for daily predictions
    thresholds = [
        {'up': 1.0, 'down': 1.0, 'name': '±1.0%'},
        {'up': 1.5, 'down': 1.5, 'name': '±1.5%'},
        {'up': 2.0, 'down': 2.0, 'name': '±2.0%'},
        {'up': 2.5, 'down': 2.5, 'name': '±2.5%'},
        {'up': 3.0, 'down': 3.0, 'name': '±3.0%'},
    ]
    
    price_col = 'crypto_last_price' if 'crypto_last_price' in df_raw.columns else 'price'
    all_results = []
    
    for threshold_config in thresholds:
        up_thresh = threshold_config['up']
        down_thresh = threshold_config['down']
        name = threshold_config['name']
        
        logger.info("\n" + "="*80)
        logger.info(f"TESTING: 24h (Daily) | Threshold: {name}")
        logger.info("="*80)
        
        # Create labels
        labels, mask, price_changes = create_daily_labels(df_raw, price_col, up_thresh, down_thresh)
        df_subset = df_raw[mask].copy()
        y = labels.values.astype(int)
        
        # Class distribution
        up_pct = np.mean(y == 2)
        sideways_pct = np.mean(y == 1)
        down_pct = np.mean(y == 0)
        
        logger.info(f"\n📊 Data Statistics:")
        logger.info(f"   Total samples: {len(df_subset):,}")
        logger.info(f"   UP:       {up_pct:.1%} ({int(np.sum(y==2)):,} samples)")
        logger.info(f"   SIDEWAYS: {sideways_pct:.1%} ({int(np.sum(y==1)):,} samples)")
        logger.info(f"   DOWN:     {down_pct:.1%} ({int(np.sum(y==0)):,} samples)")
        
        # Check balance
        if sideways_pct > 0.85:
            logger.warning(f"   ⚠️  Too much sideways ({sideways_pct:.1%})")
        elif sideways_pct < 0.55:
            logger.warning(f"   ⚠️  Too little sideways ({sideways_pct:.1%})")
        else:
            logger.info(f"   ✅ Good balance!")
        
        # Prepare features
        X = df_subset[optimal_features].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Time series cross-validation
        logger.info(f"\n🔧 Training with 5-fold TimeSeriesSplit...")
        tscv = TimeSeriesSplit(n_splits=5)
        
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        fold_predictions = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Get test timestamps for per-day analysis
            test_timestamps = df_subset.index[test_idx]
            
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
            y_proba = model.predict_proba(X_test_scaled)
            
            # Store predictions with timestamps
            for i, (ts, true_label, pred_label, proba) in enumerate(zip(test_timestamps, y_test, y_pred, y_proba)):
                fold_predictions.append({
                    'timestamp': ts,
                    'fold': fold_idx,
                    'true_label': int(true_label),
                    'predicted_label': int(pred_label),
                    'prob_down': float(proba[0]),
                    'prob_sideways': float(proba[1]),
                    'prob_up': float(proba[2]),
                    'confidence': float(np.max(proba))
                })
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)
        
        # Overall metrics
        overall_acc = accuracy_score(all_y_true, all_y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        
        # Per-class accuracy
        down_acc = cm[0,0] / cm[0,:].sum() if cm[0,:].sum() > 0 else 0
        side_acc = cm[1,1] / cm[1,:].sum() if cm[1,:].sum() > 0 else 0
        up_acc = cm[2,2] / cm[2,:].sum() if cm[2,:].sum() > 0 else 0
        
        # Directional accuracy (ignoring sideways predictions)
        directional_mask = (np.array(all_y_pred) != 1)
        if directional_mask.sum() > 0:
            y_true_directional = np.array(all_y_true)[directional_mask]
            y_pred_directional = np.array(all_y_pred)[directional_mask]
            
            # Convert to binary: UP (2) -> 1, DOWN (0) -> 0
            y_true_binary = (y_true_directional == 2).astype(int)
            y_pred_binary = (y_pred_directional == 2).astype(int)
            
            directional_acc = accuracy_score(y_true_binary, y_pred_binary)
            directional_signals_pct = directional_mask.sum() / len(all_y_true)
        else:
            directional_acc = 0
            directional_signals_pct = 0
        
        logger.info(f"\n📈 Results:")
        logger.info(f"   Overall Accuracy:     {overall_acc:.2%}")
        logger.info(f"   Per-Class Accuracy:")
        logger.info(f"     DOWN:     {down_acc:.2%}")
        logger.info(f"     SIDEWAYS: {side_acc:.2%}")
        logger.info(f"     UP:       {up_acc:.2%}")
        logger.info(f"   Directional Accuracy: {directional_acc:.2%} (when trading)")
        logger.info(f"   Trading Signals:      {directional_signals_pct:.1%} of time")
        
        logger.info(f"\n📊 Confusion Matrix:")
        logger.info(f"                Predicted")
        logger.info(f"                DOWN  SIDE   UP")
        logger.info(f"Actual DOWN    {cm[0,0]:5d} {cm[0,1]:5d} {cm[0,2]:5d}")
        logger.info(f"       SIDE    {cm[1,0]:5d} {cm[1,1]:5d} {cm[1,2]:5d}")
        logger.info(f"       UP      {cm[2,0]:5d} {cm[2,1]:5d} {cm[2,2]:5d}")
        
        # Save detailed predictions to CSV
        predictions_df = pd.DataFrame(fold_predictions)
        predictions_df = predictions_df.sort_values('timestamp')
        
        # Add human-readable labels
        label_map = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
        predictions_df['true_direction'] = predictions_df['true_label'].map(label_map)
        predictions_df['predicted_direction'] = predictions_df['predicted_label'].map(label_map)
        predictions_df['correct'] = (predictions_df['true_label'] == predictions_df['predicted_label'])
        
        # Group by day
        predictions_df['date'] = predictions_df['timestamp'].dt.date
        daily_stats = predictions_df.groupby('date').agg({
            'correct': 'mean',
            'confidence': 'mean',
            'predicted_label': lambda x: label_map[x.mode()[0]] if len(x.mode()) > 0 else 'UNKNOWN',
            'true_label': lambda x: label_map[x.mode()[0]] if len(x.mode()) > 0 else 'UNKNOWN'
        }).rename(columns={
            'correct': 'daily_accuracy',
            'confidence': 'avg_confidence',
            'predicted_label': 'predicted_direction',
            'true_label': 'actual_direction'
        })
        
        # Save CSVs
        csv_dir = Path(__file__).parent / 'results' / 'daily_predictions'
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        predictions_csv = csv_dir / f'predictions_24h_{name.replace("±", "pm")}.csv'
        daily_csv = csv_dir / f'daily_summary_24h_{name.replace("±", "pm")}.csv'
        
        predictions_df.to_csv(predictions_csv, index=False)
        daily_stats.to_csv(daily_csv)
        
        logger.info(f"\n💾 Saved predictions:")
        logger.info(f"   All predictions: {predictions_csv}")
        logger.info(f"   Daily summary:   {daily_csv}")
        
        # Calculate balanced score
        balanced_score = (overall_acc * 0.4) + (directional_acc * 0.6)
        
        all_results.append({
            'name': name,
            'up_threshold': up_thresh,
            'down_threshold': down_thresh,
            'samples': len(df_subset),
            'up_pct': float(up_pct),
            'sideways_pct': float(sideways_pct),
            'down_pct': float(down_pct),
            'overall_accuracy': overall_acc,
            'down_accuracy': down_acc,
            'sideways_accuracy': side_acc,
            'up_accuracy': up_acc,
            'directional_accuracy': directional_acc,
            'directional_signals_pct': float(directional_signals_pct),
            'balanced_score': balanced_score,
            'predictions_csv': str(predictions_csv),
            'daily_csv': str(daily_csv)
        })
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🏆 FINAL RESULTS - DAILY PREDICTIONS")
    logger.info("="*80)
    
    # Rank by balanced score
    sorted_results = sorted(all_results, key=lambda x: x['balanced_score'], reverse=True)
    
    logger.info("\n📊 Ranked by Balanced Score:\n")
    for i, r in enumerate(sorted_results, 1):
        logger.info(f"{i}. {r['name']:8} | Balanced: {r['balanced_score']:.2%} | "
                   f"Overall: {r['overall_accuracy']:.2%} | Dir: {r['directional_accuracy']:.2%} | "
                   f"Signals: {r['directional_signals_pct']:.1%} | "
                   f"Dist: ↓{r['down_pct']:.0%} ↔{r['sideways_pct']:.0%} ↑{r['up_pct']:.0%}")
    
    # Best configuration
    best = sorted_results[0]
    logger.info(f"\n" + "="*80)
    logger.info(f"🥇 BEST CONFIGURATION FOR DAILY TRADING")
    logger.info(f"="*80)
    logger.info(f"Threshold:            {best['name']}")
    logger.info(f"Balanced Score:       {best['balanced_score']:.2%}")
    logger.info(f"Overall Accuracy:     {best['overall_accuracy']:.2%}")
    logger.info(f"Directional Accuracy: {best['directional_accuracy']:.2%}")
    logger.info(f"Trading Signals:      {best['directional_signals_pct']:.1%} of time")
    logger.info(f"Samples:              {best['samples']:,}")
    logger.info(f"\nClass Distribution:")
    logger.info(f"  UP:       {best['up_pct']:.1%}")
    logger.info(f"  SIDEWAYS: {best['sideways_pct']:.1%}")
    logger.info(f"  DOWN:     {best['down_pct']:.1%}")
    logger.info(f"\nPer-Class Accuracy:")
    logger.info(f"  DOWN:     {best['down_accuracy']:.2%}")
    logger.info(f"  SIDEWAYS: {best['sideways_accuracy']:.2%}")
    logger.info(f"  UP:       {best['up_accuracy']:.2%}")
    logger.info(f"\nPredictions saved to:")
    logger.info(f"  {best['predictions_csv']}")
    logger.info(f"  {best['daily_csv']}")
    
    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_path = results_dir / 'daily_predictor_results.json'
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'horizon': '24h',
            'best_config': best,
            'all_results': sorted_results
        }, f, indent=2)
    
    logger.info(f"\n💾 Summary saved: {results_path}")
    
    # Train final model with best config
    logger.info(f"\n" + "="*80)
    logger.info(f"🎯 TRAINING FINAL DAILY MODEL")
    logger.info(f"="*80)
    
    labels, mask, _ = create_daily_labels(
        df_raw, price_col, best['up_threshold'], best['down_threshold']
    )
    df_subset = df_raw[mask].copy()
    y = labels.values.astype(int)
    X = df_subset[optimal_features].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"   Training on {len(X):,} samples...")
    
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
    
    logger.info("   ✅ Final model trained")
    
    # Save model
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'daily_predictor_model.pkl'
    scaler_path = model_dir / 'daily_predictor_scaler.pkl'
    config_path = model_dir / 'daily_predictor_config.json'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_final, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_final, f)
    
    config_final = {
        'horizon': '24h',
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
    logger.info(f"\nYou now have:")
    logger.info(f"  • Trained daily predictor model")
    logger.info(f"  • Per-day predictions in CSV files")
    logger.info(f"  • Daily accuracy summaries")
    logger.info(f"  • Ready for production trading!")

if __name__ == "__main__":
    main()



