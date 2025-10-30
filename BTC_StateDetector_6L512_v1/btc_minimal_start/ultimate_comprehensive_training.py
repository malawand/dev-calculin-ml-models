#!/usr/bin/env python3
"""
ULTIMATE COMPREHENSIVE TRAINING
Uses ALL metrics provided: derivatives, derivative primes, averages, volume
Tests all combinations systematically to find the absolute best model.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from lightgbm import LGBMClassifier
import json
import logging
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'btc_direction_predictor'))

from src.labels.labels import LabelCreator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_horizon_to_periods(horizon: str) -> int:
    """Convert horizon string to 15-min periods."""
    mapping = {
        '1h': 4, '2h': 8, '4h': 16, '8h': 32, 
        '12h': 48, '24h': 96, '48h': 192
    }
    return mapping.get(horizon, 96)

def create_labels_manual(df, price_col, horizon, threshold_pct=0.0):
    """Create directional labels manually."""
    periods = parse_horizon_to_periods(horizon)
    future_price = df[price_col].shift(-periods)
    price_change_pct = ((future_price - df[price_col]) / df[price_col]) * 100
    
    if threshold_pct > 0:
        # Three classes: UP, DOWN, HOLD
        labels = pd.Series(1, index=df.index)  # Default HOLD
        labels[price_change_pct > threshold_pct] = 2  # UP
        labels[price_change_pct < -threshold_pct] = 0  # DOWN
        # Remove HOLD for binary classification
        mask = labels != 1
        return labels[mask], mask
    else:
        # Binary: UP (1) or DOWN (0)
        labels = (price_change_pct > 0).astype(int)
        mask = ~labels.isna()
        return labels[mask], mask

def categorize_features(all_cols):
    """Categorize all available features."""
    categories = {
        'spot_price': [],
        'price_averages': [],
        'price_derivatives': [],
        'price_derivative_primes': [],
        'volume_raw': [],
        'volume_averages': [],
        'volume_derivatives': [],
        'volume_rates': [],
        'engineered_basic': [],  # volatility, returns, etc
        'engineered_advanced': []  # Everything else
    }
    
    for col in all_cols:
        col_lower = col.lower()
        
        # Skip non-feature columns
        if col in ['timestamp', 'price'] or col.startswith('label_'):
            continue
        
        # Categorize
        if col == 'crypto_last_price':
            categories['spot_price'].append(col)
        elif col == 'crypto_volume':
            categories['volume_raw'].append(col)
        
        # Price metrics
        elif 'job:crypto_last_price:avg' in col:
            categories['price_averages'].append(col)
        elif 'job:crypto_last_price:deriv' in col and 'prime' in col:
            categories['price_derivative_primes'].append(col)
        elif 'job:crypto_last_price:deriv' in col:
            categories['price_derivatives'].append(col)
        
        # Volume metrics
        elif 'job:crypto_volume:avg' in col:
            categories['volume_averages'].append(col)
        elif 'job:crypto_volume:deriv' in col:
            categories['volume_derivatives'].append(col)
        elif 'job:crypto_volume:rate' in col:
            categories['volume_rates'].append(col)
        
        # Engineered features
        elif any(x in col_lower for x in ['volatility', 'return', 'roc', 'zscore', 'spread']):
            categories['engineered_basic'].append(col)
        else:
            categories['engineered_advanced'].append(col)
    
    return categories

def incremental_train_fast(df, all_features, y, start_features, max_add=10):
    """Fast incremental training - add features one by one."""
    current_features = [f for f in start_features if f in all_features]
    remaining = [f for f in all_features if f not in current_features]
    
    if len(current_features) == 0:
        return [], 0.5, []
    
    # Baseline
    X = df[current_features].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LGBMClassifier(random_state=42, n_estimators=50, learning_rate=0.1, verbose=-1)
    model.fit(X_train_scaled, y_train)
    
    best_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    best_features = current_features.copy()
    
    # Try adding features
    for iteration in range(max_add):
        if len(remaining) == 0:
            break
        
        # Rank by correlation
        df_remaining = df[remaining]
        correlations = df_remaining.corrwith(pd.Series(y, index=df.index)).abs().sort_values(ascending=False)
        
        # Try top candidate
        if len(correlations) > 0:
            candidate = correlations.index[0]
            test_features = current_features + [candidate]
            
            X_test_feat = df[test_features].values
            X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=0.0, neginf=0.0)
            
            X_tr, X_te, y_tr, y_te = train_test_split(X_test_feat, y, test_size=0.2, shuffle=False)
            
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)
            
            mdl = LGBMClassifier(random_state=42, n_estimators=50, learning_rate=0.1, verbose=-1)
            mdl.fit(X_tr_sc, y_tr)
            
            acc = accuracy_score(y_te, mdl.predict(X_te_sc))
            
            if acc > best_accuracy:
                best_accuracy = acc
                current_features.append(candidate)
                best_features = current_features.copy()
                remaining.remove(candidate)
            else:
                remaining.remove(candidate)
                break  # Stop if no improvement
        else:
            break
    
    return best_features, best_accuracy, []

def main():
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("🚀 ULTIMATE COMPREHENSIVE TRAINING")
    logger.info("Testing ALL metrics: derivatives, primes, averages, volume")
    logger.info("="*80)
    logger.info(f"Start time: {start_time}")
    logger.info("")
    
    # Load combined data with volume
    data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_with_volume.parquet'
    
    logger.info(f"📥 Loading dataset: {data_path}")
    df = pd.read_parquet(data_path)
    
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    logger.info(f"   Loaded {len(df):,} samples with {len(df.columns)} columns")
    logger.info(f"   Date range: {df.index.min()} → {df.index.max()}")
    
    # Categorize ALL available features
    logger.info(f"\n📊 Categorizing features...")
    categories = categorize_features(df.columns.tolist())
    
    logger.info(f"\n   Feature Categories:")
    for cat_name, features in categories.items():
        if features:
            logger.info(f"   {cat_name:30s}: {len(features):4d} features")
    
    # Show sample of each category
    logger.info(f"\n📋 Sample features from each category:")
    for cat_name, features in categories.items():
        if features:
            logger.info(f"\n   {cat_name}:")
            for feat in features[:5]:
                logger.info(f"      • {feat}")
            if len(features) > 5:
                logger.info(f"      ... and {len(features)-5} more")
    
    # Define experiment grid
    horizons = ['4h', '8h', '12h']  # Focus on shorter horizons
    thresholds = [0.0, 1.0, 2.0]
    
    # Define feature set combinations to test
    feature_combinations = [
        {
            'name': '1_derivatives_only',
            'features': categories['price_derivatives']
        },
        {
            'name': '2_derivative_primes_only',
            'features': categories['price_derivative_primes']
        },
        {
            'name': '3_derivatives_plus_primes',
            'features': categories['price_derivatives'] + categories['price_derivative_primes']
        },
        {
            'name': '4_averages_only',
            'features': categories['price_averages']
        },
        {
            'name': '5_volume_only',
            'features': categories['volume_raw'] + categories['volume_averages'] + categories['volume_derivatives']
        },
        {
            'name': '6_derivatives_plus_volume',
            'features': categories['price_derivatives'] + categories['volume_derivatives']
        },
        {
            'name': '7_primes_plus_volume',
            'features': categories['price_derivative_primes'] + categories['volume_averages']
        },
        {
            'name': '8_averages_plus_derivatives',
            'features': categories['price_averages'] + categories['price_derivatives']
        },
        {
            'name': '9_all_derivatives_family',
            'features': categories['price_derivatives'] + categories['price_derivative_primes'] + categories['volume_derivatives']
        },
        {
            'name': '10_all_price_metrics',
            'features': categories['price_averages'] + categories['price_derivatives'] + categories['price_derivative_primes']
        },
        {
            'name': '11_all_volume_metrics',
            'features': categories['volume_raw'] + categories['volume_averages'] + categories['volume_derivatives'] + categories['volume_rates']
        },
        {
            'name': '12_balanced_mix',
            'features': (
                categories['price_averages'][:10] + 
                categories['price_derivatives'][:10] + 
                categories['price_derivative_primes'][:10] +
                categories['volume_averages'][:5] +
                categories['volume_derivatives'][:5]
            )
        },
        {
            'name': '13_everything',
            'features': []  # Will use all available features
        }
    ]
    
    all_features = []
    for cat in categories.values():
        all_features.extend(cat)
    
    # Replace 'everything' with all features
    for combo in feature_combinations:
        if combo['name'] == '13_everything':
            combo['features'] = all_features
    
    logger.info(f"\n📊 Will test:")
    logger.info(f"   Horizons: {len(horizons)}")
    logger.info(f"   Thresholds: {len(thresholds)}")
    logger.info(f"   Feature Combinations: {len(feature_combinations)}")
    logger.info(f"   Total experiments: {len(horizons) * len(thresholds) * len(feature_combinations)}")
    logger.info("")
    
    all_results = []
    experiment_num = 0
    total_experiments = len(horizons) * len(thresholds) * len(feature_combinations)
    
    for horizon in horizons:
        for threshold_pct in thresholds:
            # Create labels
            logger.info("="*80)
            logger.info(f"Creating labels: {horizon} horizon, {threshold_pct}% threshold")
            logger.info("="*80)
            
            price_col = 'crypto_last_price' if 'crypto_last_price' in df.columns else 'price'
            labels, mask = create_labels_manual(df, price_col, horizon, threshold_pct)
            
            df_subset = df[mask].copy()
            y = labels.values
            
            up_pct = np.mean(y == 1) if threshold_pct == 0 else np.mean(y == 2)
            logger.info(f"   Samples: {len(df_subset):,}")
            logger.info(f"   UP: {up_pct:.1%}")
            
            # Skip if too imbalanced
            if up_pct > 0.9 or up_pct < 0.1:
                logger.warning(f"   ⚠️ Skipping - too imbalanced!")
                continue
            
            for combo in feature_combinations:
                experiment_num += 1
                
                logger.info(f"\n{'='*80}")
                logger.info(f"EXPERIMENT {experiment_num}/{total_experiments}")
                logger.info(f"Horizon: {horizon} | Threshold: {threshold_pct}% | Set: {combo['name']}")
                logger.info(f"{'='*80}")
                
                # Get available features
                available_features = [f for f in combo['features'] if f in df_subset.columns]
                
                if len(available_features) < 3:
                    logger.info(f"   ⏭️ Skipping - only {len(available_features)} features available")
                    continue
                
                logger.info(f"   Available features: {len(available_features)}")
                
                # Select starting features (top 5 by correlation)
                df_feat = df_subset[available_features]
                correlations = df_feat.corrwith(pd.Series(y, index=df_subset.index)).abs().sort_values(ascending=False)
                
                start_features = correlations.head(5).index.tolist()
                logger.info(f"   Starting with top 5:")
                for i, feat in enumerate(start_features, 1):
                    logger.info(f"      {i}. {feat} (corr={correlations[feat]:.4f})")
                
                # Train
                best_features, best_accuracy, _ = incremental_train_fast(
                    df_subset, available_features, y, start_features, max_add=10
                )
                
                logger.info(f"\n   ✅ Result:")
                logger.info(f"      Features: {len(best_features)}")
                logger.info(f"      Accuracy: {best_accuracy:.4f} ({best_accuracy:.2%})")
                
                all_results.append({
                    'experiment_num': experiment_num,
                    'horizon': horizon,
                    'threshold_pct': threshold_pct,
                    'feature_set_name': combo['name'],
                    'available_features': len(available_features),
                    'used_features': len(best_features),
                    'accuracy': best_accuracy,
                    'samples': len(df_subset),
                    'up_ratio': float(up_pct),
                    'best_features': best_features
                })
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL RESULTS")
    logger.info("="*80)
    
    if all_results:
        sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
        
        logger.info(f"\n🏆 TOP 20 CONFIGURATIONS:\n")
        for i, result in enumerate(sorted_results[:20], 1):
            logger.info(f"{i:2}. {result['horizon']:4} | {result['threshold_pct']:4.1f}% | "
                       f"{result['feature_set_name']:30s} | "
                       f"{result['used_features']:3d} feats | "
                       f"Acc: {result['accuracy']:.4f} ({result['accuracy']:.2%})")
        
        # Best overall
        best = sorted_results[0]
        logger.info(f"\n" + "="*80)
        logger.info(f"🥇 ABSOLUTE BEST CONFIGURATION")
        logger.info(f"="*80)
        logger.info(f"Horizon:           {best['horizon']}")
        logger.info(f"Threshold:         {best['threshold_pct']}%")
        logger.info(f"Feature Set:       {best['feature_set_name']}")
        logger.info(f"Features Used:     {best['used_features']}")
        logger.info(f"Accuracy:          {best['accuracy']:.4f} ({best['accuracy']:.2%})")
        logger.info(f"Samples:           {best['samples']:,}")
        logger.info(f"\nBest Features:")
        for i, feat in enumerate(best['best_features'], 1):
            logger.info(f"  {i:2}. {feat}")
        
        # Save results
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / 'ultimate_comprehensive_results.json'
        
        with open(results_path, 'w') as f:
            json.dump({
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_experiments': len(all_results),
                'best_config': best,
                'top_20': sorted_results[:20],
                'all_results': sorted_results
            }, f, indent=2)
        
        logger.info(f"\n💾 Results saved: {results_path}")
        
        # Analysis by category
        logger.info(f"\n" + "="*80)
        logger.info(f"📈 ANALYSIS BY CATEGORY")
        logger.info(f"="*80)
        
        # By horizon
        logger.info(f"\n🎯 Best per Horizon:")
        by_horizon = {}
        for r in sorted_results:
            if r['horizon'] not in by_horizon:
                by_horizon[r['horizon']] = r
        for h in horizons:
            if h in by_horizon:
                r = by_horizon[h]
                logger.info(f"   {h:4} → {r['accuracy']:.4f} ({r['accuracy']:.2%}) "
                           f"[{r['feature_set_name']}, thresh={r['threshold_pct']}%]")
        
        # By feature set
        logger.info(f"\n🎯 Best per Feature Set:")
        by_set = {}
        for r in sorted_results:
            if r['feature_set_name'] not in by_set:
                by_set[r['feature_set_name']] = r
        for combo in feature_combinations:
            name = combo['name']
            if name in by_set:
                r = by_set[name]
                logger.info(f"   {name:30s} → {r['accuracy']:.4f} ({r['accuracy']:.2%}) "
                           f"[{r['horizon']}, thresh={r['threshold_pct']}%]")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\n" + "="*80)
    logger.info(f"✅ COMPLETE!")
    logger.info(f"="*80)
    logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    logger.info(f"Experiments: {len(all_results)}")

if __name__ == "__main__":
    main()



