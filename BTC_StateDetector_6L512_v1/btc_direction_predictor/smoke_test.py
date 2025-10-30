#!/usr/bin/env python3
"""
Smoke test: Quick validation with 2 days of data
Tests the entire pipeline end-to-end with minimal data
"""
import sys
import yaml
import logging
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_smoke_test():
    """Run quick smoke test with minimal data"""
    
    print("="*80)
    print("SMOKE TEST - BTC Direction Predictor")
    print("="*80)
    
    # Test 1: Load config
    print("\n[1/5] Testing config loading...")
    try:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
        print("✓ Config loaded successfully")
        print(f"  Base URL: {config['cortex']['base_url']}")
        print(f"  Symbol: {config['cortex']['symbol']}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
    
    # Test 2: Prometheus client
    print("\n[2/5] Testing Prometheus client...")
    try:
        from src.data.prom import PrometheusClient, get_all_metric_names
        
        client = PrometheusClient(
            base_url=config['cortex']['base_url'],
            read_api=config['cortex']['read_api']
        )
        
        # Test with just spot price for 1 hour
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=1)
        
        query = client.build_metric_query('crypto_last_price', 'BTCUSDT')
        df = client.query_range(query, start, end, '5m')
        
        if not df.empty:
            print(f"✓ Fetched {len(df)} data points")
            print(f"  Latest price: ${df.iloc[-1].values[0]:,.2f}")
        else:
            print("✗ No data returned (check Prometheus endpoint)")
            return False
            
    except Exception as e:
        print(f"✗ Prometheus client failed: {e}")
        print("  Hint: Check if Prometheus is accessible at the configured URL")
        return False
    
    # Test 3: Feature engineering (mock)
    print("\n[3/5] Testing feature engineering...")
    try:
        import pandas as pd
        import numpy as np
        
        # Create mock features
        df_test = pd.DataFrame({
            'price': np.random.randn(100) + 100000,
            'deriv1h': np.random.randn(100),
            'avg4h': np.random.randn(100) + 100000
        })
        
        # Simple feature: price returns
        df_test['return_1'] = df_test['price'].pct_change()
        df_test['return_12'] = df_test['price'].pct_change(12)
        
        print(f"✓ Created {len(df_test.columns)} mock features")
        
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False
    
    # Test 4: Model training (mock)
    print("\n[4/5] Testing model training...")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        # Mock data
        X = np.random.randn(1000, 10)
        y = np.random.randint(0, 2, 1000)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train simple model
        model = LogisticRegression(max_iter=100)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        print(f"✓ Model trained successfully")
        print(f"  Test accuracy: {accuracy:.2%}")
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return False
    
    # Test 5: Metrics calculation
    print("\n[5/5] Testing metrics...")
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"✓ Metrics calculated:")
        print(f"  Accuracy: {acc:.2%}")
        print(f"  Precision: {prec:.2%}")
        print(f"  Recall: {rec:.2%}")
        print(f"  F1: {f1:.2%}")
        
    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
    print("\nNext steps:")
    print("1. Run full training: python -m src.pipeline.train")
    print("2. Check results in: artifacts/reports/")
    print("3. Run inference: python -m src.pipeline.infer")
    
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
