#!/usr/bin/env python3
"""
Comprehensive Feature Search
Systematically tests ALL available metrics to find optimal feature combinations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


# Define all available metric categories
METRIC_CATEGORIES = {
    'derivative_primes_short': [
        'deriv1h_prime1h', 'deriv2h_prime1h', 'deriv4h_prime1h',
        'deriv4h_prime2h', 'deriv8h_prime2h'
    ],
    'derivative_primes_medium': [
        'deriv8h_prime4h', 'deriv12h_prime4h', 'deriv12h_prime8h',
        'deriv16h_prime8h', 'deriv24h_prime8h'
    ],
    'derivative_primes_long': [
        'deriv24h_prime12h', 'deriv24h_prime16h', 'deriv24h_prime24h',
        'deriv48h_prime24h', 'deriv3d_prime24h', 'deriv7d_prime7d'
    ],
    'derivatives_short': [
        'deriv5m', 'deriv10m', 'deriv15m', 'deriv30m', 'deriv45m', 'deriv1h'
    ],
    'derivatives_medium': [
        'deriv2h', 'deriv4h', 'deriv8h', 'deriv12h', 'deriv16h'
    ],
    'derivatives_long': [
        'deriv24h', 'deriv48h', 'deriv3d', 'deriv4d', 'deriv5d', 'deriv6d', 'deriv7d'
    ],
    'averages_very_short': [
        'avg10m', 'avg15m', 'avg30m', 'avg45m'
    ],
    'averages_short': [
        'avg1h', 'avg2h', 'avg4h', 'avg8h'
    ],
    'averages_medium': [
        'avg12h', 'avg24h', 'avg48h'
    ],
    'averages_long': [
        'avg3d', 'avg4d', 'avg5d', 'avg6d', 'avg7d', 'avg14d'
    ],
    'volatility': [
        'volatility_12', 'volatility_24', 'volatility_72'
    ],
    'momentum': [
        'deriv4d_roc', 'deriv7d_roc', 'deriv14d_roc', 'deriv30d_roc'
    ],
    'spreads': [
        'avg10m_spread', 'avg1h_spread', 'avg12h_spread', 'avg24h_spread'
    ]
}


# Define starting combinations to test
STARTING_COMBINATIONS = [
    # Best from previous runs
    ('best_known', ['deriv7d_prime7d', 'deriv4d_roc', 'volatility_24']),
    
    # Derivative prime focused
    ('prime_short', ['deriv4h_prime2h', 'deriv8h_prime2h', 'deriv12h_prime4h']),
    ('prime_medium', ['deriv12h_prime8h', 'deriv16h_prime8h', 'deriv24h_prime12h']),
    ('prime_long', ['deriv24h_prime24h', 'deriv48h_prime24h', 'deriv7d_prime7d']),
    
    # Derivative focused (various timeframes)
    ('deriv_short', ['deriv15m', 'deriv30m', 'deriv1h']),
    ('deriv_medium', ['deriv4h', 'deriv8h', 'deriv12h']),
    ('deriv_long', ['deriv24h', 'deriv3d', 'deriv7d']),
    
    # Moving average focused
    ('avg_very_short', ['avg10m', 'avg15m', 'avg30m']),
    ('avg_short', ['avg1h', 'avg2h', 'avg4h']),
    ('avg_medium', ['avg12h', 'avg24h', 'avg48h']),
    ('avg_long', ['avg3d', 'avg7d', 'avg14d']),
    
    # Volatility + trend
    ('vol_trend', ['volatility_24', 'deriv7d_roc', 'deriv30d_roc']),
    
    # Multi-scale combinations
    ('multi_scale_1', ['deriv1h_prime1h', 'deriv12h_prime8h', 'deriv7d_prime7d']),
    ('multi_scale_2', ['avg15m', 'avg4h', 'avg3d']),
    ('multi_scale_3', ['deriv30m', 'deriv8h', 'deriv3d']),
    
    # Momentum focused
    ('momentum', ['deriv4d_roc', 'deriv7d_roc', 'deriv14d_roc']),
    
    # Mixed strategies
    ('mixed_1', ['deriv7d_prime7d', 'avg1h', 'volatility_24']),
    ('mixed_2', ['deriv12h_prime8h', 'avg4h', 'deriv7d_roc']),
    ('mixed_3', ['deriv24h_prime12h', 'avg12h', 'volatility_72']),
    
    # Ultra short-term
    ('ultra_short', ['deriv5m', 'deriv10m', 'avg10m']),
    
    # Ultra long-term
    ('ultra_long', ['deriv14d', 'deriv30d', 'avg14d']),
]


class ComprehensiveSearcher:
    """Systematically search through all feature combinations"""
    
    def __init__(self, horizon='24h'):
        self.horizon = horizon
        self.results = []
        
        # Load data once
        logger.info("=" * 80)
        logger.info("🔍 COMPREHENSIVE FEATURE SEARCH")
        logger.info("=" * 80)
        logger.info(f"Horizon: {horizon}")
        logger.info(f"Testing {len(STARTING_COMBINATIONS)} different starting combinations")
        logger.info("")
        
        # Load and prepare data
        data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_full_dataset.parquet'
        logger.info(f"📥 Loading data: {data_path}")
        df_raw = pd.read_parquet(data_path)
        logger.info(f"   Loaded {len(df_raw):,} samples")
        
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
                    'averages': [f'job:crypto_last_price:avg{t}' for t in ['10m', '15m', '30m', '45m', '1h', '2h', '4h', '8h', '12h', '24h', '48h', '3d', '4d', '5d', '6d', '7d', '14d']],
                    'derivatives': [f'job:crypto_last_price:deriv{t}' for t in ['5m', '10m', '15m', '30m', '45m', '1h', '2h', '4h', '8h', '12h', '16h', '24h', '48h', '3d', '4d', '5d', '6d', '7d', '14d', '30d']],
                    'derivative_primes': []
                }
            },
            'price_col': 'price',
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
                            and col != 'price' 
                            and col != 'timestamp']
        
        self.df = df_labeled
        logger.info(f"   Total features available: {len(self.all_features)}")
        logger.info("")
    
    def test_combination(self, name, starting_features, max_additional=10):
        """Test a single starting combination"""
        logger.info("=" * 80)
        logger.info(f"🧪 TESTING: {name}")
        logger.info("=" * 80)
        logger.info(f"Starting features: {starting_features}")
        
        # Check which features are available
        available_start = [f for f in starting_features if f in self.all_features]
        if len(available_start) < len(starting_features):
            missing = set(starting_features) - set(available_start)
            logger.warning(f"⚠️  Missing features: {missing}")
            if len(available_start) == 0:
                logger.error("❌ No starting features available, skipping")
                return None
        
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
            
            logger.info(f"✅ Data prepared: {len(X):,} samples with {len(available_start)} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train baseline
            model = LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.05, verbose=-1)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_test_pred = model.predict(X_test_scaled)
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, y_test_proba)
            
            logger.info(f"📊 Results:")
            logger.info(f"   Accuracy: {accuracy:.4f} ({accuracy:.2%})")
            logger.info(f"   ROC-AUC:  {roc_auc:.4f}")
            logger.info("")
            
            result = {
                'name': name,
                'starting_features': starting_features,
                'available_features': available_start,
                'n_features': len(available_start),
                'samples': len(X),
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Error testing {name}: {e}")
            return None
    
    def run_all_tests(self):
        """Run all combination tests"""
        logger.info("🚀 Starting comprehensive search...")
        logger.info(f"   Total combinations to test: {len(STARTING_COMBINATIONS)}")
        logger.info("")
        
        for i, (name, features) in enumerate(STARTING_COMBINATIONS, 1):
            logger.info(f"[{i}/{len(STARTING_COMBINATIONS)}] Testing: {name}")
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
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE SEARCH RESULTS")
        logger.info("=" * 80)
        logger.info("")
        
        logger.info("🏆 TOP 10 COMBINATIONS BY ACCURACY:")
        logger.info("")
        for i, result in enumerate(sorted_results[:10], 1):
            logger.info(f"{i}. {result['name']:20} | "
                       f"Accuracy: {result['accuracy']:.2%} | "
                       f"ROC-AUC: {result['roc_auc']:.4f} | "
                       f"Features: {result['n_features']}")
            logger.info(f"   Starting: {result['available_features']}")
            logger.info("")
        
        # Best by category
        logger.info("=" * 80)
        logger.info("🎯 BEST BY CATEGORY:")
        logger.info("")
        
        categories = {
            'Derivative Primes': ['prime_short', 'prime_medium', 'prime_long'],
            'Derivatives': ['deriv_short', 'deriv_medium', 'deriv_long'],
            'Moving Averages': ['avg_very_short', 'avg_short', 'avg_medium', 'avg_long'],
            'Mixed': ['mixed_1', 'mixed_2', 'mixed_3'],
            'Multi-Scale': ['multi_scale_1', 'multi_scale_2', 'multi_scale_3']
        }
        
        for category, names in categories.items():
            category_results = [r for r in self.results if r['name'] in names]
            if category_results:
                best = max(category_results, key=lambda x: x['accuracy'])
                logger.info(f"{category:20} | Best: {best['name']:20} | {best['accuracy']:.2%}")
        
        logger.info("")
        logger.info("=" * 80)
        
        # Save results
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        output_file = results_dir / 'comprehensive_search_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'horizon': self.horizon,
                'total_tests': len(self.results),
                'results': sorted_results
            }, f, indent=2)
        
        logger.info(f"💾 Results saved: {output_file}")
        logger.info("")
        
        # Print winner
        winner = sorted_results[0]
        logger.info("=" * 80)
        logger.info("🎉 WINNING COMBINATION!")
        logger.info("=" * 80)
        logger.info(f"Name:     {winner['name']}")
        logger.info(f"Accuracy: {winner['accuracy']:.2%}")
        logger.info(f"ROC-AUC:  {winner['roc_auc']:.4f}")
        logger.info(f"Features: {winner['available_features']}")
        logger.info("=" * 80)


def main():
    searcher = ComprehensiveSearcher(horizon='24h')
    searcher.run_all_tests()


if __name__ == "__main__":
    main()



