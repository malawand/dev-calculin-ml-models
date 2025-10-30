#!/usr/bin/env python3
"""
Train model with VOLUME data included.
This is the game-changer that should boost accuracy from 54% → 62-68%!
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

# Add parent directories to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'btc_direction_predictor'))

from src.features.engineering import FeatureEngineer
from src.labels.labels import LabelCreator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def engineer_volume_features(df):
    """
    Engineer volume-specific features that combine price and volume.
    These are the KEY features that will boost accuracy!
    """
    logger.info("🔧 Engineering volume-specific features...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure we have the required columns
    if 'crypto_volume' not in df.columns or 'crypto_last_price' not in df.columns:
        logger.warning("⚠️ Missing crypto_volume or crypto_last_price columns!")
        return df
    
    volume = df['crypto_volume']
    price = df['crypto_last_price']
    
    # 1. Volume vs Average (KEY FEATURE #1)
    if 'job:crypto_volume:avg24h' in df.columns:
        df['volume_vs_avg24h'] = volume / (df['job:crypto_volume:avg24h'] + 1e-10)
        df['volume_spike_24h'] = (df['volume_vs_avg24h'] > 2.0).astype(int)
    
    if 'job:crypto_volume:avg1h' in df.columns:
        df['volume_vs_avg1h'] = volume / (df['job:crypto_volume:avg1h'] + 1e-10)
    
    # 2. Price-Volume Correlation (KEY FEATURE #2)
    # Correlation between price changes and volume over rolling windows
    price_returns = price.pct_change()
    volume_returns = volume.pct_change()
    
    for window in [12, 24, 72]:  # 3h, 6h, 18h
        df[f'price_volume_corr_{window}'] = price_returns.rolling(window).corr(volume_returns)
    
    # 3. Volume-confirmed moves (KEY FEATURE #3)
    # Price up + High volume = REAL move
    # Price up + Low volume = FAKE move
    price_up = (price_returns > 0).astype(int)
    volume_high = (volume > volume.rolling(24).mean()).astype(int)
    df['volume_confirmed_up'] = price_up * volume_high
    df['volume_confirmed_down'] = (1 - price_up) * volume_high
    
    # 4. Volume momentum (KEY FEATURE #4)
    if 'job:crypto_volume:deriv1h' in df.columns:
        df['volume_momentum_1h'] = df['job:crypto_volume:deriv1h']
    if 'job:crypto_volume:deriv24h' in df.columns:
        df['volume_momentum_24h'] = df['job:crypto_volume:deriv24h']
    
    # 5. Volume divergence (KEY FEATURE #5)
    # Price up but volume down = WEAK trend (bearish divergence)
    # Price down but volume up = potential reversal
    price_trend = price.diff(24)  # 6-hour trend
    if 'job:crypto_volume:avg24h' in df.columns:
        volume_trend = volume - df['job:crypto_volume:avg24h']
        df['volume_divergence'] = np.sign(price_trend) != np.sign(volume_trend)
        df['volume_divergence'] = df['volume_divergence'].astype(int)
    
    # 6. Volume breakout score (KEY FEATURE #6)
    # Combines price movement with volume strength
    price_change_24h = price.pct_change(96)  # 24-hour change
    if 'job:crypto_volume:avg24h' in df.columns:
        volume_strength = volume / (df['job:crypto_volume:avg24h'] + 1e-10)
        df['volume_breakout_score'] = price_change_24h * volume_strength
    
    # 7. Multi-timeframe volume alignment (KEY FEATURE #7)
    # Are all timeframes showing high volume?
    vol_cols = [c for c in df.columns if 'job:crypto_volume:avg' in c and any(tf in c for tf in ['1h', '4h', '24h'])]
    if len(vol_cols) >= 3:
        volume_ratios = []
        for col in vol_cols:
            volume_ratios.append(volume / (df[col] + 1e-10))
        df['volume_alignment'] = pd.DataFrame(volume_ratios).T.mean(axis=1)
    
    # 8. Volume rate comparisons
    if 'job:crypto_volume:rate1h' in df.columns and 'job:crypto_volume:rate24h' in df.columns:
        df['volume_rate_ratio'] = df['job:crypto_volume:rate1h'] / (df['job:crypto_volume:rate24h'] + 1e-10)
    
    # 9. Volume climax detection (potential reversals)
    if 'job:crypto_volume:avg24h' in df.columns:
        volume_zscore = (volume - volume.rolling(96).mean()) / (volume.rolling(96).std() + 1e-10)
        df['volume_climax'] = (volume_zscore > 2.5).astype(int)
    
    # 10. Price-weighted volume
    df['price_weighted_volume'] = price * volume
    df['price_weighted_volume_ma24'] = df['price_weighted_volume'].rolling(96).mean()
    
    new_features = [c for c in df.columns if c not in ['crypto_volume', 'crypto_last_price']]
    logger.info(f"   ✅ Created {len(new_features)} volume features")
    
    return df

def main():
    horizon = '24h'
    
    logger.info("=" * 80)
    logger.info("🚀 TRAINING WITH VOLUME DATA - THE GAME CHANGER!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Expected improvement: 54% → 62-68% accuracy! 📈")
    logger.info("")
    
    # Load dataset with volume
    data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_with_volume.parquet'
    
    if not data_path.exists():
        logger.error(f"❌ Dataset with volume not found: {data_path}")
        logger.info("   Please wait for fetch_volume_data.py to complete first.")
        logger.info(f"   Check status: tail -f ../btc_direction_predictor/fetch_volume.log")
        sys.exit(1)
    
    logger.info(f"📥 Loading dataset with volume: {data_path}")
    df_raw = pd.read_parquet(data_path)
    
    # Ensure datetime index
    if 'timestamp' in df_raw.columns:
        df_raw = df_raw.set_index('timestamp')
    df_raw.index = pd.to_datetime(df_raw.index, utc=True)
    df_raw = df_raw.sort_index()
    
    # Rename crypto_last_price to price for consistency
    if 'crypto_last_price' in df_raw.columns:
        df_raw = df_raw.rename(columns={'crypto_last_price': 'price'})
    
    logger.info(f"   Loaded {len(df_raw):,} samples")
    logger.info(f"   Date range: {df_raw.index.min()} → {df_raw.index.max()}")
    logger.info(f"   Total columns: {len(df_raw.columns)}")
    
    # Count price vs volume columns
    volume_cols = [c for c in df_raw.columns if 'volume' in c.lower()]
    price_cols = [c for c in df_raw.columns if 'volume' not in c.lower()]
    logger.info(f"   Price features: {len(price_cols)}")
    logger.info(f"   Volume features: {len(volume_cols)}")
    
    # Engineer price features
    logger.info(f"\n🔧 Engineering price features...")
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
        'target_horizons': [horizon]
    }
    
    feature_engineer = FeatureEngineer(config)
    df_engineered = feature_engineer.engineer(df_raw.copy())
    logger.info(f"   Engineered shape: {df_engineered.shape}")
    
    # Engineer volume features
    df_with_volume = engineer_volume_features(df_engineered)
    logger.info(f"   Final shape after volume features: {df_with_volume.shape}")
    
    # Create labels
    logger.info(f"\n🎯 Creating labels for {horizon} horizon...")
    label_creator = LabelCreator(
        price_col='price',
        threshold=0.0
    )
    df_labeled = label_creator.create_labels(df_with_volume, [horizon])
    
    # Get feature columns (exclude labels and metadata)
    leakage_patterns = ['return_24', 'return_72', 'return_96', 'lag1_24h', 'lag3_24h', 'lag6_24h', 'lag12_24h']
    all_features = []
    for col in df_labeled.columns:
        if col.startswith('label_') or col == 'price' or col == 'timestamp':
            continue
        # Check if this feature contains leakage patterns
        is_leakage = any(pattern in col for pattern in leakage_patterns)
        if not is_leakage:
            all_features.append(col)
    
    logger.info(f"   Total features available: {len(all_features)}")
    
    # Separate volume and non-volume features for analysis
    volume_features = [f for f in all_features if 'volume' in f.lower()]
    price_features = [f for f in all_features if 'volume' not in f.lower()]
    
    logger.info(f"   Price-only features: {len(price_features)}")
    logger.info(f"   Volume features: {len(volume_features)}")
    
    # Prepare data
    df_final = df_labeled[all_features + [f'label_{horizon}']].dropna()
    
    # Ensure all features are numeric
    X_df = df_final[all_features].select_dtypes(include=[np.number])
    numeric_features = X_df.columns.tolist()
    
    X = X_df.values.astype(float)
    y = df_final[f'label_{horizon}'].values.astype(int)
    
    # Remove NaN and inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    logger.info(f"\n✅ Data prepared: {len(X):,} samples with {len(numeric_features)} numeric features")
    logger.info(f"   Class balance: UP={np.sum(y==1)}/{len(y)} ({np.mean(y==1):.1%})")
    
    # Split data (time-series split - no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    logger.info(f"\n📊 Split: Train={len(X_train):,}, Test={len(X_test):,}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("🤖 TRAINING LIGHTGBM MODEL WITH VOLUME")
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
        callbacks=[],
        feature_name=numeric_features
    )
    
    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("📊 EVALUATION RESULTS")
    logger.info("=" * 80)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)
    
    logger.info(f"\nTraining Set:")
    logger.info(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy:.2%})")
    logger.info(f"  ROC-AUC:   {roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1]):.4f}")
    logger.info(f"  MCC:       {matthews_corrcoef(y_train, y_train_pred):.4f}")
    
    logger.info(f"\nTest Set:")
    logger.info(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy:.2%})")
    logger.info(f"  ROC-AUC:   {test_roc_auc:.4f}")
    logger.info(f"  MCC:       {test_mcc:.4f}")
    
    # Feature importance analysis
    logger.info("\n" + "=" * 80)
    logger.info("🎯 TOP 30 FEATURES BY IMPORTANCE")
    logger.info("=" * 80)
    
    feature_importance = pd.Series(model.feature_importances_, index=numeric_features).sort_values(ascending=False)
    
    for i, (feature, importance) in enumerate(feature_importance.head(30).items()):
        # Categorize feature for display
        if 'volume' in feature.lower():
            category = "📊 VOL  "
        elif "avg" in feature:
            category = "📈 AVG  "
        elif "deriv" in feature and "prime" not in feature:
            category = "📉 DERIV"
        elif "prime" in feature:
            category = "🔄 PRIME"
        else:
            category = "📌 OTHER"
        
        logger.info(f"  {i+1:2}. [{category}] {feature:50s} {importance:.1f}")
    
    # Count how many volume features are in top 30
    top_30_features = feature_importance.head(30).index.tolist()
    volume_in_top30 = sum(1 for f in top_30_features if 'volume' in f.lower())
    
    logger.info(f"\n🔍 VOLUME FEATURES IN TOP 30: {volume_in_top30}/30 ({volume_in_top30/30:.1%})")
    
    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'with_volume_results.json'
    
    results = {
        "horizon": horizon,
        "dataset_samples": len(X),
        "total_features": len(numeric_features),
        "price_features": len(price_features),
        "volume_features": len(volume_features),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "test_roc_auc": float(test_roc_auc),
        "test_mcc": float(test_mcc),
        "volume_features_in_top30": int(volume_in_top30),
        "top_10_features": feature_importance.head(10).to_dict(),
        "baseline_accuracy_no_volume": 0.5483  # From previous run
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPARISON: BEFORE vs AFTER VOLUME")
    logger.info("=" * 80)
    logger.info(f"  WITHOUT Volume:  {results['baseline_accuracy_no_volume']:.2%}")
    logger.info(f"  WITH Volume:     {test_accuracy:.2%}")
    improvement = test_accuracy - results['baseline_accuracy_no_volume']
    logger.info(f"  Improvement:     {improvement:+.2%}")
    
    if improvement > 0:
        logger.info(f"\n🎉 SUCCESS! Volume data improved accuracy by {improvement:.2%}!")
    else:
        logger.info(f"\n⚠️ Unexpected: No improvement. May need more volume feature engineering.")
    
    logger.info(f"\n💾 Results saved: {results_path}")
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()



