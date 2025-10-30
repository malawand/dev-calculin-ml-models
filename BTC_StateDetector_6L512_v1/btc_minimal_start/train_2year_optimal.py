#!/usr/bin/env python3
"""
Train on 2 years of data with the optimal 8 features discovered from experiments.

Winning features (73.36% accuracy on 1 year):
- deriv7d_prime7d
- deriv4d_roc
- volatility_24
- avg30m
- avg45m
- avg1h
- avg15m
- avg10m
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
import json

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'btc_direction_predictor'))

from src.features.engineering import FeatureEngineer
from src.labels.labels import LabelCreator

# The 8 winning features
OPTIMAL_FEATURES = [
    'deriv7d_prime7d',
    'deriv4d_roc',
    'volatility_24',
    'avg30m',
    'avg45m',
    'avg1h',
    'avg15m',
    'avg10m'
]

def main():
    print("="*80)
    print("🚀 TRAINING ON 2 YEARS OF DATA - OPTIMAL 8 FEATURES")
    print("="*80)
    print()
    
    # Load dataset (already has ~2 years: 2023-12-30 → 2025-10-17)
    data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_full_dataset.parquet'
    
    if not data_path.exists():
        print(f"❌ Dataset not found at: {data_path}")
        sys.exit(1)
    
    print(f"📥 Loading dataset: {data_path}")
    df_raw = pd.read_parquet(data_path)
    
    # Check date range
    if 'timestamp' in df_raw.columns:
        df_raw = df_raw.set_index('timestamp')
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.sort_index()
    
    date_range_days = (df_raw.index.max() - df_raw.index.min()).days
    date_range_years = date_range_days / 365.25
    
    print(f"   Loaded {len(df_raw)} samples with {len(df_raw.columns)} raw features")
    print(f"   Date range: {df_raw.index.min()} → {df_raw.index.max()}")
    print(f"   Duration: {date_range_days} days ({date_range_years:.2f} years)")
    
    # Engineer features
    print(f"🔧 Engineering features...")
    config = {
        'features': {
            'price_lags': [1, 3, 6, 12],
            'deriv_lags': [1, 3, 6],
            'rolling_windows': [12, 24, 72]
        },
        'labels': {
            'horizons': ['24h'],
            'threshold_pct': 0.0
        }
    }
    
    feature_engineer = FeatureEngineer(config)
    df_engineered = feature_engineer.engineer(df_raw.copy())
    print(f"   Engineered shape: {df_engineered.shape}")
    
    # Create labels
    print(f"🎯 Creating labels for 24h horizon...")
    label_creator = LabelCreator(config)
    df_labeled = label_creator.create_labels(df_engineered)
    
    # Check if optimal features exist
    missing_features = [f for f in OPTIMAL_FEATURES if f not in df_labeled.columns]
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        print(f"   Available features will be used instead")
        available_optimal = [f for f in OPTIMAL_FEATURES if f in df_labeled.columns]
        print(f"   Using {len(available_optimal)} of 8 optimal features")
    else:
        available_optimal = OPTIMAL_FEATURES
        print(f"✅ All 8 optimal features found!")
    
    # Prepare data
    X = df_labeled[available_optimal].values
    y = df_labeled['label_24h'].values
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    print(f"✅ Data prepared: {len(X)} samples with {len(available_optimal)} features")
    print(f"   Class balance: UP={y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")
    
    # Train/Test split (80/20, maintaining temporal order)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Split: Train={len(X_train)}, Test={len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print(f"\n{'='*80}")
    print("🤖 TRAINING LIGHTGBM MODEL")
    print(f"{'='*80}\n")
    
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.01,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print(f"\n{'='*80}")
    print("📊 EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    # Training performance
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    train_mcc = matthews_corrcoef(y_train, y_train_pred)
    
    print(f"Training Set:")
    print(f"  Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  ROC-AUC:   {train_auc:.4f}")
    print(f"  MCC:       {train_mcc:.4f}")
    
    # Test performance
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)
    
    print(f"\nTest Set:")
    print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  ROC-AUC:   {test_auc:.4f}")
    print(f"  MCC:       {test_mcc:.4f}")
    
    # Feature importance
    print(f"\n{'='*80}")
    print("🎯 FEATURE IMPORTANCE")
    print(f"{'='*80}\n")
    
    feature_importance = list(zip(available_optimal, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feat, imp) in enumerate(feature_importance, 1):
        print(f"  {i}. {feat:<25} {imp:>8.1f}")
    
    # Comparison to 1-year result
    print(f"\n{'='*80}")
    print("📊 COMPARISON: 1 YEAR vs 2 YEARS")
    print(f"{'='*80}\n")
    
    one_year_acc = 0.7336
    improvement = test_acc - one_year_acc
    
    print(f"  1 Year (Oct 2024 - Oct 2025):   73.36%")
    print(f"  2 Years (Oct 2023 - Oct 2025):  {test_acc*100:.2f}%")
    print(f"  Difference:                     {improvement*100:+.2f}%")
    
    if test_acc > one_year_acc:
        print(f"\n✅ IMPROVED with more data! (+{improvement*100:.2f}%)")
    elif test_acc > 0.70:
        print(f"\n✅ Still excellent performance (>70%)")
    elif abs(improvement) < 0.05:
        print(f"\n✓  Consistent performance (within 5%)")
    else:
        print(f"\n⚠️  Lower accuracy - possible regime shift or overfitting on 1-year data")
    
    # Save results
    results = {
        'dataset': '2_years',
        'date_range': '2023-10-17 to 2025-10-17',
        'samples': len(X),
        'features': available_optimal,
        'n_features': len(available_optimal),
        'train_accuracy': float(train_acc),
        'train_auc': float(train_auc),
        'train_mcc': float(train_mcc),
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'test_mcc': float(test_mcc),
        'comparison_to_1year': {
            '1_year_accuracy': one_year_acc,
            '2_year_accuracy': float(test_acc),
            'difference': float(improvement)
        },
        'feature_importance': {feat: float(imp) for feat, imp in feature_importance}
    }
    
    output_path = Path('results/2year_optimal_8features.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {output_path}")
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*80}\n")
    
    return results

if __name__ == "__main__":
    main()

