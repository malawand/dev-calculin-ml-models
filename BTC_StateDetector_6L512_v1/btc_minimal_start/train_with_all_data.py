#!/usr/bin/env python3
"""
Incremental training with ALL data (price + volume + derivatives).
Keep iterating until we find the best model!
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

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'btc_direction_predictor'))

from src.features.engineering import FeatureEngineer
from src.labels.labels import LabelCreator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def incremental_train(df, all_features, y, start_features, max_iterations=50, no_improve_stop=8):
    """
    Incremental feature addition training.
    """
    current_features = start_features.copy()
    remaining_features = [f for f in all_features if f not in current_features]
    
    best_accuracy = 0.0
    best_features = current_features.copy()
    no_improve_count = 0
    
    history = []
    
    # Prepare initial data
    X = df[current_features].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Baseline
    logger.info("="*80)
    logger.info(f"BASELINE - {len(current_features)} features")
    logger.info("="*80)
    
    model = LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.05, verbose=-1)
    model.fit(X_train_scaled, y_train)
    
    baseline_pred = model.predict(X_test_scaled)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    best_accuracy = baseline_accuracy
    
    logger.info(f"✅ Baseline Accuracy: {baseline_accuracy:.4f}")
    logger.info(f"   ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]):.4f}")
    
    history.append({
        'iteration': 0,
        'features': current_features.copy(),
        'num_features': len(current_features),
        'accuracy': baseline_accuracy
    })
    
    # Iterative improvement
    for iteration in range(1, max_iterations + 1):
        logger.info("\n" + "="*80)
        logger.info(f"ITERATION {iteration}/{max_iterations}")
        logger.info("="*80)
        logger.info(f"Current: {len(current_features)} features, {best_accuracy:.4f} accuracy")
        logger.info(f"Best: {len(best_features)} features, {best_accuracy:.4f} accuracy")
        logger.info(f"No improvement: {no_improve_count}/{no_improve_stop}")
        
        if len(remaining_features) == 0:
            logger.info("✋ No more features to try!")
            break
        
        # Rank remaining features by correlation
        logger.info(f"🔍 Ranking {len(remaining_features)} remaining features...")
        
        df_remaining = df[remaining_features].copy()
        correlations = df_remaining.corrwith(pd.Series(y, index=df.index)).abs().sort_values(ascending=False)
        
        # Try top 3 candidates
        top_candidates = correlations.head(3)
        logger.info(f"   Top 3 candidates:")
        for i, (feat, corr) in enumerate(top_candidates.items(), 1):
            logger.info(f"      {i}. {feat} (corr={corr:.4f})")
        
        best_candidate = None
        best_candidate_accuracy = best_accuracy
        
        for candidate in top_candidates.index:
            # Test this feature
            test_features = current_features + [candidate]
            X_test_feat = df[test_features].values
            X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=0.0, neginf=0.0)
            
            X_tr, X_te, y_tr, y_te = train_test_split(X_test_feat, y, test_size=0.2, shuffle=False)
            
            scaler_temp = StandardScaler()
            X_tr_scaled = scaler_temp.fit_transform(X_tr)
            X_te_scaled = scaler_temp.transform(X_te)
            
            model_temp = LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.05, verbose=-1)
            model_temp.fit(X_tr_scaled, y_tr)
            
            pred = model_temp.predict(X_te_scaled)
            acc = accuracy_score(y_te, pred)
            
            improvement = acc - best_accuracy
            logger.info(f"   🧪 Testing: {candidate}")
            logger.info(f"      Accuracy: {acc:.4f} ({improvement:+.4f})")
            
            if acc > best_candidate_accuracy:
                best_candidate = candidate
                best_candidate_accuracy = acc
        
        # Add best candidate if it improves
        if best_candidate and best_candidate_accuracy > best_accuracy:
            current_features.append(best_candidate)
            remaining_features.remove(best_candidate)
            
            improvement = best_candidate_accuracy - best_accuracy
            logger.info(f"✅ Adding: {best_candidate}")
            logger.info(f"   {best_accuracy:.4f} → {best_candidate_accuracy:.4f} ({improvement:+.4f})")
            
            best_accuracy = best_candidate_accuracy
            best_features = current_features.copy()
            no_improve_count = 0
            
            history.append({
                'iteration': iteration,
                'features': current_features.copy(),
                'num_features': len(current_features),
                'accuracy': best_candidate_accuracy,
                'added_feature': best_candidate
            })
        else:
            logger.info(f"❌ No improvement")
            no_improve_count += 1
        
        # Check stopping condition
        if no_improve_count >= no_improve_stop:
            logger.info(f"\n✋ Stopping - no improvement for {no_improve_stop} iterations")
            break
    
    return best_features, best_accuracy, history

def main():
    horizon = '24h'
    
    logger.info("="*80)
    logger.info("🚀 INCREMENTAL TRAINING WITH ALL DATA (PRICE + VOLUME)")
    logger.info("="*80)
    logger.info("")
    
    # Load combined dataset
    data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_with_volume.parquet'
    
    if not data_path.exists():
        logger.error(f"❌ Dataset not found: {data_path}")
        sys.exit(1)
    
    logger.info(f"📥 Loading combined dataset: {data_path}")
    df_raw = pd.read_parquet(data_path)
    
    if 'timestamp' in df_raw.columns:
        df_raw = df_raw.set_index('timestamp')
    df_raw.index = pd.to_datetime(df_raw.index, utc=True)
    df_raw = df_raw.sort_index()
    
    # Rename for consistency
    if 'crypto_last_price' in df_raw.columns:
        df_raw = df_raw.rename(columns={'crypto_last_price': 'price'})
    
    logger.info(f"   Loaded {len(df_raw):,} samples")
    logger.info(f"   Date range: {df_raw.index.min()} → {df_raw.index.max()}")
    logger.info(f"   Raw features: {len(df_raw.columns)}")
    
    # Count volume features
    volume_cols = [c for c in df_raw.columns if 'volume' in c.lower()]
    logger.info(f"   Volume features: {len(volume_cols)}")
    
    # Engineer features
    logger.info(f"\n🔧 Engineering features...")
    config = {
        'features': {
            'price_lags': [1, 3, 6, 12],
            'deriv_lags': [1, 3, 6],
            'rolling_windows': [12, 24, 72]
        },
        'prometheus': {
            'metrics': {
                'spot': ['price'],
                'averages': [],
                'derivatives': [],
                'derivative_primes': []
            }
        },
        'price_col': 'price',
        'target_horizons': [horizon],
        'labels': {
            'horizons': [horizon],
            'threshold_pct': 0.0
        }
    }
    
    feature_engineer = FeatureEngineer(config)
    df_engineered = feature_engineer.engineer(df_raw.copy())
    logger.info(f"   Engineered shape: {df_engineered.shape}")
    
    # Create labels
    logger.info(f"\n🎯 Creating labels...")
    label_creator = LabelCreator(config)
    df_labeled = label_creator.create_labels(df_engineered)
    
    # Get all features
    leakage_patterns = ['return_24', 'return_72', 'return_96', 'lag1_24h', 'lag3_24h', 'lag6_24h', 'lag12_24h']
    all_features = []
    for col in df_labeled.columns:
        if col.startswith('label_') or col in ['price', 'timestamp']:
            continue
        is_leakage = any(pattern in col for pattern in leakage_patterns)
        if not is_leakage:
            all_features.append(col)
    
    logger.info(f"   Total features: {len(all_features)}")
    
    # Separate by type for starting features
    volume_features = [f for f in all_features if 'volume' in f.lower()]
    price_features = [f for f in all_features if 'volume' not in f.lower()]
    
    logger.info(f"   Price features: {len(price_features)}")
    logger.info(f"   Volume features: {len(volume_features)}")
    
    # Prepare final dataset
    df_final = df_labeled[all_features + [f'label_{horizon}']].dropna()
    
    X_df = df_final[all_features].select_dtypes(include=[np.number])
    numeric_features = X_df.columns.tolist()
    
    y = df_final[f'label_{horizon}'].values.astype(int)
    mask = ~np.isnan(y)
    
    df_clean = df_final[mask]
    y_clean = y[mask]
    
    logger.info(f"\n✅ Data prepared: {len(df_clean):,} samples with {len(numeric_features)} features")
    logger.info(f"   Class balance: UP={np.sum(y_clean==1)}/{len(y_clean)} ({np.mean(y_clean==1):.1%})")
    
    # Choose strong starting features (mix of volume and price)
    # Start with features we know are good
    starting_candidates = []
    
    # Add top volume features
    if 'crypto_volume' in numeric_features:
        starting_candidates.append('crypto_volume')
    if 'job:crypto_volume:avg1h' in numeric_features:
        starting_candidates.append('job:crypto_volume:avg1h')
    if 'job:crypto_volume:deriv1h' in numeric_features:
        starting_candidates.append('job:crypto_volume:deriv1h')
    
    # Add top price features
    for feat in ['avg15m', 'avg10m', 'avg1h', 'deriv30d_roc', 'volatility_24']:
        if feat in numeric_features and feat not in starting_candidates:
            starting_candidates.append(feat)
    
    # Use first 5 available
    start_features = starting_candidates[:5]
    
    logger.info(f"\n📊 Starting features ({len(start_features)}):")
    for i, f in enumerate(start_features, 1):
        feat_type = "📊 VOL" if 'volume' in f.lower() else "📈 PRICE"
        logger.info(f"   {i}. [{feat_type}] {f}")
    
    # Run incremental training
    logger.info(f"\n🤖 Running incremental training...")
    logger.info(f"   Max iterations: 50")
    logger.info(f"   Early stop: 8 iterations without improvement")
    logger.info("")
    
    best_features, best_accuracy, history = incremental_train(
        df_clean[numeric_features],
        numeric_features,
        y_clean,
        start_features,
        max_iterations=50,
        no_improve_stop=8
    )
    
    # Final results
    logger.info("\n" + "="*80)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Best Model:")
    logger.info(f"  Features: {len(best_features)}")
    logger.info(f"  Accuracy: {best_accuracy:.4f} ({best_accuracy:.2%})")
    logger.info(f"\nBest Features:")
    for i, f in enumerate(best_features, 1):
        feat_type = "📊 VOL" if 'volume' in f.lower() else "📈 PRICE"
        logger.info(f"  {i:2}. [{feat_type}] {f}")
    
    # Count volume features in final model
    vol_count = sum(1 for f in best_features if 'volume' in f.lower())
    logger.info(f"\n📊 Volume features in final model: {vol_count}/{len(best_features)} ({vol_count/len(best_features):.1%})")
    
    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'incremental_with_all_data.json'
    
    results = {
        'horizon': horizon,
        'total_samples': len(df_clean),
        'total_features_available': len(numeric_features),
        'price_features_available': len(price_features),
        'volume_features_available': len(volume_features),
        'best_num_features': len(best_features),
        'best_accuracy': float(best_accuracy),
        'volume_features_in_model': vol_count,
        'best_features': best_features,
        'training_history': history
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Results saved: {results_path}")
    logger.info("\n" + "="*80)
    logger.info("✅ ALL DONE!")
    logger.info("="*80)

if __name__ == "__main__":
    main()



