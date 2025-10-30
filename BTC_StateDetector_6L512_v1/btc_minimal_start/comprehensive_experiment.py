#!/usr/bin/env python3
"""
COMPREHENSIVE EXPERIMENT RUNNER
Tests ALL combinations to find the best model configuration.

Strategy:
1. Test different horizons (4h, 8h, 12h, 24h)
2. Test different thresholds (0%, 0.5%, 1%, 2%)
3. Test different starting feature sets (volume-only, price-only, mixed, derivatives, etc.)
4. Run incremental training for each
5. Report best configurations
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
import json
import logging
from datetime import datetime
import itertools

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'btc_direction_predictor'))

from src.features.engineering import FeatureEngineer
from src.labels.labels import LabelCreator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_train_test(X, y, features_to_use):
    """Quick training to evaluate a feature set."""
    try:
        X_subset = X[features_to_use].values
        X_subset = np.nan_to_num(X_subset, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, shuffle=False
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LGBMClassifier(
            random_state=42, 
            n_estimators=100, 
            learning_rate=0.05, 
            verbose=-1
        )
        model.fit(X_train_scaled, y_train)
        
        pred = model.predict(X_test_scaled)
        pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, pred)
        try:
            roc_auc = roc_auc_score(y_test, pred_proba)
        except:
            roc_auc = 0.5
        
        return accuracy, roc_auc
    except Exception as e:
        logger.warning(f"Error in quick_train_test: {e}")
        return 0.5, 0.5

def main():
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("🚀 COMPREHENSIVE EXPERIMENT RUNNER")
    logger.info("="*80)
    logger.info(f"Start time: {start_time}")
    logger.info("")
    
    # Load data
    data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_with_volume.parquet'
    
    logger.info(f"📥 Loading dataset: {data_path}")
    df_raw = pd.read_parquet(data_path)
    
    if 'timestamp' in df_raw.columns:
        df_raw = df_raw.set_index('timestamp')
    df_raw.index = pd.to_datetime(df_raw.index, utc=True)
    df_raw = df_raw.sort_index()
    
    if 'crypto_last_price' in df_raw.columns:
        df_raw = df_raw.rename(columns={'crypto_last_price': 'price'})
    
    logger.info(f"   Loaded {len(df_raw):,} samples")
    
    # Define experiment grid
    horizons = ['4h', '8h', '12h', '24h']
    thresholds = [0.0, 0.5, 1.0, 2.0]  # Percentage
    
    all_results = []
    experiment_num = 0
    total_experiments = len(horizons) * len(thresholds)
    
    logger.info(f"\n📊 Will run {total_experiments} experiments")
    logger.info(f"   Horizons: {horizons}")
    logger.info(f"   Thresholds: {thresholds}%")
    logger.info("")
    
    for horizon in horizons:
        for threshold_pct in thresholds:
            experiment_num += 1
            
            logger.info("="*80)
            logger.info(f"EXPERIMENT {experiment_num}/{total_experiments}")
            logger.info(f"Horizon: {horizon} | Threshold: {threshold_pct}%")
            logger.info("="*80)
            
            # Engineer features
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
                    'threshold_pct': threshold_pct
                }
            }
            
            feature_engineer = FeatureEngineer(config)
            df_engineered = feature_engineer.engineer(df_raw.copy())
            
            # Create labels
            label_creator = LabelCreator(config)
            df_labeled = label_creator.create_labels(df_engineered)
            
            # Get features
            leakage_patterns = ['return_24', 'return_72', 'return_96', 'lag1_24h', 'lag3_24h', 'lag6_24h', 'lag12_24h']
            all_features = []
            for col in df_labeled.columns:
                if col.startswith('label_') or col in ['price', 'timestamp']:
                    continue
                is_leakage = any(pattern in col for pattern in leakage_patterns)
                if not is_leakage:
                    all_features.append(col)
            
            # Categorize features
            volume_features = [f for f in all_features if 'volume' in f.lower()]
            avg_features = [f for f in all_features if 'avg' in f and 'volume' not in f.lower()]
            deriv_features = [f for f in all_features if 'deriv' in f and 'prime' not in f and 'volume' not in f.lower()]
            prime_features = [f for f in all_features if 'prime' in f]
            vol_stat_features = [f for f in all_features if 'volatility' in f]
            other_features = [f for f in all_features if f not in volume_features + avg_features + deriv_features + prime_features + vol_stat_features]
            
            logger.info(f"📊 Feature breakdown:")
            logger.info(f"   Volume: {len(volume_features)}")
            logger.info(f"   Price Averages: {len(avg_features)}")
            logger.info(f"   Derivatives: {len(deriv_features)}")
            logger.info(f"   Derivative Primes: {len(prime_features)}")
            logger.info(f"   Volatility: {len(vol_stat_features)}")
            logger.info(f"   Other: {len(other_features)}")
            
            # Prepare data
            df_final = df_labeled[all_features + [f'label_{horizon}']].dropna()
            X_df = df_final[all_features].select_dtypes(include=[np.number])
            numeric_features = X_df.columns.tolist()
            
            y = df_final[f'label_{horizon}'].values.astype(int)
            mask = ~np.isnan(y)
            
            df_clean = df_final[mask]
            y_clean = y[mask]
            
            # Check class balance
            up_pct = np.mean(y_clean == 1)
            logger.info(f"   Samples: {len(df_clean):,}")
            logger.info(f"   Class balance: UP={up_pct:.1%}, DOWN={(1-up_pct):.1%}")
            
            # Skip if too imbalanced
            if up_pct > 0.9 or up_pct < 0.1:
                logger.warning(f"⚠️  Skipping - too imbalanced!")
                continue
            
            # Define starting feature sets to test
            feature_sets = {
                'volume_only': volume_features[:5] if len(volume_features) >= 5 else volume_features,
                'price_avg_only': avg_features[:5] if len(avg_features) >= 5 else avg_features,
                'derivatives_only': deriv_features[:5] if len(deriv_features) >= 5 else deriv_features,
                'volatility_focused': vol_stat_features[:3] + volume_features[:2] if len(vol_stat_features) >= 3 and len(volume_features) >= 2 else vol_stat_features + volume_features,
                'mixed_balanced': (
                    volume_features[:2] + 
                    avg_features[:2] + 
                    deriv_features[:1]
                ) if all([len(volume_features) >= 2, len(avg_features) >= 2, len(deriv_features) >= 1]) else (volume_features + avg_features)[:5],
            }
            
            # Test each starting set
            set_results = []
            for set_name, start_features in feature_sets.items():
                # Filter to only available features
                start_features = [f for f in start_features if f in numeric_features]
                
                if len(start_features) < 2:
                    logger.info(f"   ⏭️  Skipping {set_name} - not enough features")
                    continue
                
                logger.info(f"\n   🧪 Testing: {set_name} ({len(start_features)} features)")
                
                # Quick test
                acc, auc = quick_train_test(df_clean[numeric_features], y_clean, start_features)
                
                logger.info(f"      Accuracy: {acc:.4f} ({acc:.2%})")
                logger.info(f"      ROC-AUC:  {auc:.4f}")
                
                set_results.append({
                    'set_name': set_name,
                    'num_start_features': len(start_features),
                    'start_features': start_features,
                    'accuracy': acc,
                    'roc_auc': auc
                })
            
            # Find best starting set
            if set_results:
                best_set = max(set_results, key=lambda x: x['accuracy'])
                
                logger.info(f"\n   ✅ Best starting set: {best_set['set_name']}")
                logger.info(f"      Accuracy: {best_set['accuracy']:.4f} ({best_set['accuracy']:.2%})")
                
                all_results.append({
                    'experiment_num': experiment_num,
                    'horizon': horizon,
                    'threshold_pct': threshold_pct,
                    'samples': len(df_clean),
                    'up_ratio': float(up_pct),
                    'total_features': len(numeric_features),
                    'best_set_name': best_set['set_name'],
                    'best_accuracy': best_set['accuracy'],
                    'best_roc_auc': best_set['roc_auc'],
                    'best_start_features': best_set['start_features'],
                    'all_sets': set_results
                })
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("📊 EXPERIMENT SUMMARY")
    logger.info("="*80)
    
    if all_results:
        # Sort by accuracy
        sorted_results = sorted(all_results, key=lambda x: x['best_accuracy'], reverse=True)
        
        logger.info(f"\n🏆 TOP 10 CONFIGURATIONS:\n")
        for i, result in enumerate(sorted_results[:10], 1):
            logger.info(f"{i:2}. Horizon: {result['horizon']:4} | Threshold: {result['threshold_pct']:4.1f}% | "
                       f"Set: {result['best_set_name']:20} | "
                       f"Accuracy: {result['best_accuracy']:.4f} ({result['best_accuracy']:.2%})")
        
        # Best overall
        best = sorted_results[0]
        logger.info(f"\n" + "="*80)
        logger.info(f"🥇 BEST CONFIGURATION")
        logger.info(f"="*80)
        logger.info(f"Horizon:        {best['horizon']}")
        logger.info(f"Threshold:      {best['threshold_pct']}%")
        logger.info(f"Starting Set:   {best['best_set_name']}")
        logger.info(f"Accuracy:       {best['best_accuracy']:.4f} ({best['best_accuracy']:.2%})")
        logger.info(f"ROC-AUC:        {best['best_roc_auc']:.4f}")
        logger.info(f"Samples:        {best['samples']:,}")
        logger.info(f"Features Used:  {len(best['best_start_features'])}")
        logger.info(f"\nTop Features:")
        for i, feat in enumerate(best['best_start_features'][:10], 1):
            logger.info(f"  {i}. {feat}")
        
        # Save results
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / 'comprehensive_experiments.json'
        
        with open(results_path, 'w') as f:
            json.dump({
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_experiments': len(all_results),
                'best_config': best,
                'all_results': sorted_results
            }, f, indent=2)
        
        logger.info(f"\n💾 Results saved: {results_path}")
        
        # Insights
        logger.info(f"\n" + "="*80)
        logger.info(f"📈 KEY INSIGHTS")
        logger.info(f"="*80)
        
        # Best horizon
        by_horizon = {}
        for r in sorted_results:
            h = r['horizon']
            if h not in by_horizon or r['best_accuracy'] > by_horizon[h]['best_accuracy']:
                by_horizon[h] = r
        
        logger.info(f"\n🎯 Best accuracy per horizon:")
        for h in horizons:
            if h in by_horizon:
                logger.info(f"   {h:4} → {by_horizon[h]['best_accuracy']:.4f} ({by_horizon[h]['best_accuracy']:.2%}) "
                           f"[threshold={by_horizon[h]['threshold_pct']}%, set={by_horizon[h]['best_set_name']}]")
        
        # Best threshold
        by_threshold = {}
        for r in sorted_results:
            t = r['threshold_pct']
            if t not in by_threshold or r['best_accuracy'] > by_threshold[t]['best_accuracy']:
                by_threshold[t] = r
        
        logger.info(f"\n🎯 Best accuracy per threshold:")
        for t in thresholds:
            if t in by_threshold:
                logger.info(f"   {t:4.1f}% → {by_threshold[t]['best_accuracy']:.4f} ({by_threshold[t]['best_accuracy']:.2%}) "
                           f"[horizon={by_threshold[t]['horizon']}, set={by_threshold[t]['best_set_name']}]")
        
        # Best feature set
        by_set = {}
        for r in sorted_results:
            s = r['best_set_name']
            if s not in by_set or r['best_accuracy'] > by_set[s]['best_accuracy']:
                by_set[s] = r
        
        logger.info(f"\n🎯 Best accuracy per feature set:")
        for s in sorted(by_set.keys()):
            logger.info(f"   {s:20} → {by_set[s]['best_accuracy']:.4f} ({by_set[s]['best_accuracy']:.2%}) "
                       f"[horizon={by_set[s]['horizon']}, threshold={by_set[s]['threshold_pct']}%]")
    else:
        logger.warning("❌ No valid results!")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\n" + "="*80)
    logger.info(f"✅ ALL EXPERIMENTS COMPLETE!")
    logger.info(f"="*80)
    logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Experiments run: {len(all_results)}")

if __name__ == "__main__":
    main()



