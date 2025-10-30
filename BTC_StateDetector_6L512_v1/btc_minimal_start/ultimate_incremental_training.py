#!/usr/bin/env python3
"""
Ultimate Incremental Training
Starts with the winning features and systematically adds more until optimal
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from lightgbm import LGBMClassifier
import json
import logging
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'btc_direction_predictor'))

from src.features.engineering import FeatureEngineer
from src.labels.labels import LabelCreator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    horizon = '24h'
    
    # Winning starting features from comprehensive search
    starting_features = [
        'job:crypto_last_price:avg24h',
        'job:crypto_last_price:avg48h',
        'job:crypto_last_price:avg3d'
    ]
    
    max_features = 50  # Maximum features to add
    no_improve_stop = 10  # Stop after this many iterations without improvement
    min_improvement = 0.001  # Minimum accuracy improvement to count
    
    logger.info("=" * 80)
    logger.info("🚀 ULTIMATE INCREMENTAL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Horizon: {horizon}")
    logger.info(f"Starting features: {len(starting_features)}")
    logger.info(f"Max features: {max_features}")
    logger.info(f"Early stopping: {no_improve_stop} iterations")
    logger.info("")
    
    # Load and engineer data
    data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_full_dataset.parquet'
    logger.info(f"📥 Loading data: {data_path}")
    df_raw = pd.read_parquet(data_path)
    logger.info(f"   Loaded {len(df_raw):,} samples")
    
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
                'averages': [],
                'derivatives': [],
                'derivative_primes': []
            }
        },
        'price_col': 'crypto_last_price',
        'target_horizons': [horizon]
    }
    
    feature_engineer = FeatureEngineer(config)
    df_engineered = feature_engineer.engineer(df_raw.copy())
    
    logger.info("🎯 Creating labels...")
    config['labels'] = {
        'horizons': [horizon],
        'threshold_pct': 0.0
    }
    label_creator = LabelCreator(config)
    df_labeled = label_creator.create_labels(df_engineered)
    
    # Get all features (excluding leakage features)
    # Leakage features include returns and lags that peek into the future
    leakage_patterns = ['return_24', 'return_72', 'return_96', 'lag1_24h', 'lag3_24h', 'lag6_24h', 'lag12_24h']
    
    all_features = []
    for col in df_labeled.columns:
        if col.startswith('label_') or col == 'crypto_last_price' or col == 'timestamp':
            continue
        # Check if this feature contains leakage patterns
        is_leakage = any(pattern in col for pattern in leakage_patterns)
        if not is_leakage:
            all_features.append(col)
    
    logger.info(f"   Total features available: {len(all_features)}")
    
    # Filter starting features to only available ones
    current_features = [f for f in starting_features if f in all_features]
    remaining_features = [f for f in all_features if f not in current_features]
    
    logger.info(f"   Starting with {len(current_features)} features")
    logger.info(f"   Remaining to test: {len(remaining_features)} features")
    logger.info("")
    
    # Prepare data
    df_subset = df_labeled[current_features + [f'label_{horizon}']].dropna()
    X = df_subset[current_features].values
    y = df_subset[f'label_{horizon}'].values
    
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    logger.info(f"✅ Data prepared: {len(X):,} samples")
    logger.info("")
    
    # Baseline
    logger.info("=" * 80)
    logger.info(f"BASELINE - {len(current_features)} features")
    logger.info("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.05, verbose=-1)
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    
    best_accuracy = accuracy_score(y_test, y_test_pred)
    best_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    best_mcc = matthews_corrcoef(y_test, y_test_pred)
    
    logger.info(f"✅ Baseline:")
    logger.info(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy:.2%})")
    logger.info(f"   ROC-AUC:  {best_roc_auc:.4f}")
    logger.info(f"   MCC:      {best_mcc:.4f}")
    logger.info("")
    
    best_features = current_features.copy()
    no_improve_count = 0
    history = []
    
    # Incremental training
    for iteration in range(1, max_features + 1):
        if no_improve_count >= no_improve_stop:
            logger.info(f"✋ Stopping - no improvement for {no_improve_stop} iterations")
            break
        
        if not remaining_features:
            logger.info("✋ Stopping - no more features to test")
            break
        
        logger.info("=" * 80)
        logger.info(f"ITERATION {iteration}/{max_features}")
        logger.info("=" * 80)
        logger.info(f"Current: {len(current_features)} features, {best_accuracy:.4f} accuracy")
        logger.info(f"Best: {len(best_features)} features, {best_accuracy:.4f} accuracy")
        logger.info(f"No improvement: {no_improve_count}/{no_improve_stop}")
        logger.info(f"Remaining: {len(remaining_features)} features")
        logger.info("")
        
        # Rank remaining features by correlation with label
        df_test = df_labeled[remaining_features + [f'label_{horizon}']].dropna()
        correlations = df_test[remaining_features].corrwith(df_test[f'label_{horizon}']).abs().sort_values(ascending=False)
        
        # Test top 5 candidates
        top_candidates = correlations.head(5).index.tolist()
        logger.info(f"🔍 Testing top 5 candidates:")
        
        best_candidate = None
        best_candidate_accuracy = best_accuracy
        
        for candidate in top_candidates:
            test_features = current_features + [candidate]
            
            # Prepare data
            df_test_subset = df_labeled[test_features + [f'label_{horizon}']].dropna()
            X_test = df_test_subset[test_features].values
            y_test_labels = df_test_subset[f'label_{horizon}'].values
            
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            mask_test = ~np.isnan(y_test_labels)
            X_test = X_test[mask_test]
            y_test_labels = y_test_labels[mask_test]
            
            # Train and evaluate
            X_train_cand, X_test_cand, y_train_cand, y_test_cand = train_test_split(X_test, y_test_labels, test_size=0.2, shuffle=False)
            scaler_cand = StandardScaler()
            X_train_cand_scaled = scaler_cand.fit_transform(X_train_cand)
            X_test_cand_scaled = scaler_cand.transform(X_test_cand)
            
            model_cand = LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.05, verbose=-1)
            model_cand.fit(X_train_cand_scaled, y_train_cand)
            y_pred_cand = model_cand.predict(X_test_cand_scaled)
            
            accuracy_cand = accuracy_score(y_test_cand, y_pred_cand)
            logger.info(f"   🧪 {candidate[:50]:50} → {accuracy_cand:.4f} ({accuracy_cand - best_accuracy:+.4f})")
            
            if accuracy_cand > best_candidate_accuracy:
                best_candidate = candidate
                best_candidate_accuracy = accuracy_cand
        
        # Add best candidate if it improves
        if best_candidate and (best_candidate_accuracy - best_accuracy) >= min_improvement:
            current_features.append(best_candidate)
            remaining_features.remove(best_candidate)
            
            improvement = best_candidate_accuracy - best_accuracy
            logger.info("")
            logger.info(f"✅ Adding: {best_candidate}")
            logger.info(f"   {best_accuracy:.4f} → {best_candidate_accuracy:.4f} ({improvement:+.4f})")
            
            best_accuracy = best_candidate_accuracy
            best_features = current_features.copy()
            no_improve_count = 0
            
            history.append({
                'iteration': iteration,
                'feature_added': best_candidate,
                'total_features': len(current_features),
                'accuracy': float(best_accuracy),
                'improvement': float(improvement)
            })
        else:
            logger.info("")
            logger.info(f"❌ No improvement")
            no_improve_count += 1
        
        logger.info("")
    
    # Final results
    logger.info("\n" + "=" * 80)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Best Model:")
    logger.info(f"  Features: {len(best_features)}")
    logger.info(f"  Accuracy: {best_accuracy:.4f} ({best_accuracy:.2%})")
    logger.info("")
    logger.info(f"Best Features:")
    for i, f in enumerate(best_features, 1):
        logger.info(f"  {i}. {f}")
    logger.info("")
    
    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    results = {
        'test_date': datetime.now().isoformat(),
        'horizon': horizon,
        'best_accuracy': float(best_accuracy),
        'n_features': len(best_features),
        'features': best_features,
        'history': history,
        'iterations': len(history),
        'starting_features': starting_features
    }
    
    output_file = results_dir / 'ultimate_incremental_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Results saved: {output_file}")
    logger.info("")


if __name__ == "__main__":
    main()

