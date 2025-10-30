#!/usr/bin/env python3
"""
Smart Comprehensive Feature Search
Tests ALL available Prometheus metrics systematically
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
import re

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'btc_direction_predictor'))

from src.features.engineering import FeatureEngineer
from src.labels.labels import LabelCreator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_timeframe(col_name):
    """Extract timeframe from column name"""
    # Match patterns like: avg10m, deriv1h, deriv3d_prime4d, etc.
    match = re.search(r'(\d+)(m|h|d)', col_name)
    if match:
        val, unit = int(match.group(1)), match.group(2)
        if unit == 'm':
            return val / (60 * 24)  # Convert to days
        elif unit == 'h':
            return val / 24
        else:
            return val
    return None


def categorize_metrics(columns):
    """Intelligently categorize all available metrics"""
    categories = {
        'avg_ultra_short': [],  # <1h
        'avg_short': [],         # 1h-4h
        'avg_medium': [],        # 4h-24h
        'avg_long': [],          # >24h
        'deriv_ultra_short': [],
        'deriv_short': [],
        'deriv_medium': [],
        'deriv_long': [],
        'deriv_prime_short': [],
        'deriv_prime_medium': [],
        'deriv_prime_long': [],
        'volatility': [],
        'engineered': []
    }
    
    for col in columns:
        if col.startswith('label_') or col == 'crypto_last_price' or col == 'timestamp':
            continue
            
        tf = parse_timeframe(col)
        
        if 'prime' in col:
            if tf and tf < 1/6:  # <4h
                categories['deriv_prime_short'].append(col)
            elif tf and tf < 1:  # 4h-24h
                categories['deriv_prime_medium'].append(col)
            else:
                categories['deriv_prime_long'].append(col)
        elif ':avg' in col:
            if tf and tf < 1/24:  # <1h
                categories['avg_ultra_short'].append(col)
            elif tf and tf < 1/6:  # 1h-4h
                categories['avg_short'].append(col)
            elif tf and tf < 1:  # 4h-24h
                categories['avg_medium'].append(col)
            else:
                categories['avg_long'].append(col)
        elif ':deriv' in col:
            if tf and tf < 1/24:  # <1h
                categories['deriv_ultra_short'].append(col)
            elif tf and tf < 1/6:  # 1h-4h
                categories['deriv_short'].append(col)
            elif tf and tf < 1:  # 4h-24h
                categories['deriv_medium'].append(col)
            else:
                categories['deriv_long'].append(col)
        elif 'volatility' in col or 'return' in col or 'zscore' in col or 'momentum' in col or 'roc' in col:
            if 'volatility' in col:
                categories['volatility'].append(col)
            else:
                categories['engineered'].append(col)
        else:
            categories['engineered'].append(col)
    
    # Remove empty categories
    categories = {k: v for k, v in categories.items() if len(v) > 0}
    
    return categories


def generate_smart_combinations(categories):
    """Generate intelligent starting combinations"""
    combinations = []
    
    # Single category best performers
    for cat_name, cat_features in categories.items():
        if len(cat_features) >= 3:
            # Take first 3 from each category
            combinations.append((f'{cat_name}_top3', cat_features[:3]))
    
    # Multi-scale combinations
    if 'avg_ultra_short' in categories and 'avg_medium' in categories and 'avg_long' in categories:
        combinations.append(('avg_multi_scale', [
            categories['avg_ultra_short'][0],
            categories['avg_medium'][0],
            categories['avg_long'][0]
        ]))
    
    if 'deriv_ultra_short' in categories and 'deriv_medium' in categories and 'deriv_long' in categories:
        combinations.append(('deriv_multi_scale', [
            categories['deriv_ultra_short'][0],
            categories['deriv_medium'][0],
            categories['deriv_long'][0]
        ]))
    
    if 'deriv_prime_short' in categories and 'deriv_prime_medium' in categories and 'deriv_prime_long' in categories:
        combinations.append(('prime_multi_scale', [
            categories['deriv_prime_short'][0],
            categories['deriv_prime_medium'][0],
            categories['deriv_prime_long'][0]
        ]))
    
    # Mixed strategies
    if 'avg_short' in categories and 'deriv_medium' in categories and 'volatility' in categories:
        combinations.append(('mixed_classic', [
            categories['avg_short'][0],
            categories['deriv_medium'][0],
            categories['volatility'][0]
        ]))
    
    if 'deriv_prime_medium' in categories and 'avg_medium' in categories and 'engineered' in categories:
        combinations.append(('mixed_advanced', [
            categories['deriv_prime_medium'][0],
            categories['avg_medium'][0],
            categories['engineered'][0] if categories['engineered'] else categories['volatility'][0]
        ]))
    
    # Volatility-focused
    if 'volatility' in categories and len(categories['volatility']) >= 2:
        if 'deriv_long' in categories:
            combinations.append(('vol_trend', 
                categories['volatility'][:2] + [categories['deriv_long'][0]]))
    
    return combinations


class SmartComprehensiveSearcher:
    """Smart search through all available metrics"""
    
    def __init__(self, horizon='24h'):
        self.horizon = horizon
        self.results = []
        
        logger.info("=" * 80)
        logger.info("🔍 SMART COMPREHENSIVE FEATURE SEARCH")
        logger.info("=" * 80)
        logger.info(f"Horizon: {horizon}")
        logger.info("")
        
        # Load and prepare data
        data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_full_dataset.parquet'
        logger.info(f"📥 Loading raw data: {data_path}")
        df_raw = pd.read_parquet(data_path)
        logger.info(f"   Loaded {len(df_raw):,} samples with {len(df_raw.columns)} columns")
        
        # Categorize available metrics
        logger.info("\n🗂️  Categorizing metrics...")
        self.categories = categorize_metrics(df_raw.columns)
        for cat_name, cat_features in self.categories.items():
            logger.info(f"   {cat_name:25} {len(cat_features):3} features")
        
        # Generate combinations
        logger.info("\n🎲 Generating test combinations...")
        self.combinations = generate_smart_combinations(self.categories)
        logger.info(f"   Generated {len(self.combinations)} combinations to test")
        logger.info("")
        
        # Engineer features
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
                    'averages': list(self.categories.get('avg_ultra_short', [])) + 
                               list(self.categories.get('avg_short', [])) +
                               list(self.categories.get('avg_medium', [])) +
                               list(self.categories.get('avg_long', [])),
                    'derivatives': list(self.categories.get('deriv_ultra_short', [])) +
                                  list(self.categories.get('deriv_short', [])) +
                                  list(self.categories.get('deriv_medium', [])) +
                                  list(self.categories.get('deriv_long', [])),
                    'derivative_primes': list(self.categories.get('deriv_prime_short', [])) +
                                        list(self.categories.get('deriv_prime_medium', [])) +
                                        list(self.categories.get('deriv_prime_long', []))
                }
            },
            'price_col': 'crypto_last_price',
            'target_horizons': [horizon]
        }
        
        feature_engineer = FeatureEngineer(config)
        df_engineered = feature_engineer.engineer(df_raw.copy())
        
        # Create labels
        logger.info("🎯 Creating labels...")
        config['labels'] = {
            'horizons': [horizon],
            'threshold_pct': 0.0
        }
        label_creator = LabelCreator(config)
        df_labeled = label_creator.create_labels(df_engineered)
        
        # Get all feature columns
        self.all_features = [col for col in df_labeled.columns 
                            if not col.startswith('label_') 
                            and col != 'crypto_last_price' 
                            and col != 'timestamp']
        
        self.df = df_labeled
        logger.info(f"   Total features after engineering: {len(self.all_features)}")
        logger.info("")
    
    def test_combination(self, name, starting_features):
        """Test a single starting combination"""
        logger.info("=" * 80)
        logger.info(f"🧪 TESTING: {name}")
        logger.info("=" * 80)
        logger.info(f"Starting features ({len(starting_features)}):")
        for f in starting_features:
            logger.info(f"   - {f}")
        
        # Check which features are available in the engineered dataset
        available_start = [f for f in starting_features if f in self.all_features]
        if len(available_start) < len(starting_features):
            missing = set(starting_features) - set(available_start)
            logger.warning(f"⚠️  Some features not in engineered dataset (this is OK, they may not survive feature engineering)")
            if len(available_start) == 0:
                logger.error("❌ No starting features available, skipping")
                return None
        
        logger.info(f"✅ Using {len(available_start)} available features")
        
        # Prepare data
        try:
            df_subset = self.df[available_start + [f'label_{self.horizon}']].dropna()
            X = df_subset[available_start].values
            y = df_subset[f'label_{self.horizon}'].values
            
            # Handle NaN/inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            
            if len(X) < 100:
                logger.error(f"❌ Not enough samples ({len(X)}), skipping")
                return None
            
            logger.info(f"📊 Data: {len(X):,} samples × {len(available_start)} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            model = LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.05, verbose=-1)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_test_pred = model.predict(X_test_scaled)
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, y_test_proba)
            mcc = matthews_corrcoef(y_test, y_test_pred)
            
            logger.info(f"📊 Results:")
            logger.info(f"   Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
            logger.info(f"   ROC-AUC:   {roc_auc:.4f}")
            logger.info(f"   MCC:       {mcc:.4f}")
            logger.info("")
            
            result = {
                'name': name,
                'starting_features': starting_features,
                'available_features': available_start,
                'n_features': len(available_start),
                'samples': len(X),
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'mcc': float(mcc),
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Error testing {name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_tests(self):
        """Run all combination tests"""
        logger.info("🚀 Starting smart comprehensive search...")
        logger.info(f"   Total combinations to test: {len(self.combinations)}")
        logger.info("")
        
        for i, (name, features) in enumerate(self.combinations, 1):
            logger.info(f"\n[{i}/{len(self.combinations)}] Testing: {name}")
            self.test_combination(name, features)
        
        # Analyze results
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze and report best results"""
        if not self.results:
            logger.error("❌ No results to analyze")
            return
        
        # Sort by accuracy
        sorted_results = sorted(self.results, key=lambda x: x['accuracy'], reverse=True)
        
        logger.info("\n\n" + "=" * 80)
        logger.info("📊 SMART COMPREHENSIVE SEARCH RESULTS")
        logger.info("=" * 80)
        logger.info("")
        
        logger.info("🏆 ALL RESULTS (RANKED BY ACCURACY):")
        logger.info("")
        for i, result in enumerate(sorted_results, 1):
            logger.info(f"{i}. {result['name']:30} | "
                       f"Accuracy: {result['accuracy']:.2%} | "
                       f"ROC-AUC: {result['roc_auc']:.4f} | "
                       f"MCC: {result['mcc']:.4f} | "
                       f"Features: {result['n_features']}")
            logger.info(f"   {result['available_features']}")
            logger.info("")
        
        logger.info("=" * 80)
        
        # Save results
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        output_file = results_dir / 'smart_comprehensive_search_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'horizon': self.horizon,
                'total_tests': len(self.results),
                'categories': {k: len(v) for k, v in self.categories.items()},
                'results': sorted_results
            }, f, indent=2)
        
        logger.info(f"💾 Results saved: {output_file}")
        logger.info("")
        
        # Print winner
        winner = sorted_results[0]
        logger.info("=" * 80)
        logger.info("🎉 WINNING COMBINATION!")
        logger.info("=" * 80)
        logger.info(f"Name:      {winner['name']}")
        logger.info(f"Accuracy:  {winner['accuracy']:.2%}")
        logger.info(f"ROC-AUC:   {winner['roc_auc']:.4f}")
        logger.info(f"MCC:       {winner['mcc']:.4f}")
        logger.info(f"Features:  {winner['available_features']}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("💡 NEXT STEP: Run incremental training with these features")
        logger.info(f"   python incremental_simple.py --starting-features {' '.join(winner['available_features'][:3])}")


def main():
    searcher = SmartComprehensiveSearcher(horizon='24h')
    searcher.run_all_tests()


if __name__ == "__main__":
    main()



