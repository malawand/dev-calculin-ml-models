#!/usr/bin/env python3
"""
Simplified Incremental Training

Uses the same data pipeline as btc_lstm_ensemble but adds features incrementally.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import sys

# Add parent paths
sys.path.append(str(Path(__file__).parent.parent / 'btc_lstm_ensemble'))
sys.path.append(str(Path(__file__).parent.parent / 'btc_direction_predictor'))

from src.features.engineering import FeatureEngineer
from src.labels.labels import LabelCreator


def load_and_prepare_data(horizon='24h'):
    """Load and prepare data same way as btc_lstm_ensemble"""
    
    # Load cached RAW data
    data_path = Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_full_dataset.parquet'
    print(f"📥 Loading data: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"   Loaded {len(df)} samples with {len(df.columns)} raw features")
    
    # Engineer features
    print(f"🔧 Engineering features...")
    config = {
        'features': {
            'price_lags': [1, 3, 6, 12],
            'deriv_lags': [1, 3, 6],
            'rolling_windows': [12, 24, 72]
        }
    }
    
    feature_engineer = FeatureEngineer(config)
    df_engineered = feature_engineer.engineer(df.copy())
    print(f"   Engineered shape: {df_engineered.shape}")
    
    # Create labels
    print(f"🎯 Creating labels...")
    config['labels'] = {
        'horizons': [horizon],
        'threshold_pct': 0.0
    }
    label_creator = LabelCreator(config)
    df_labeled = label_creator.create_labels(df_engineered)
    
    # Get feature columns
    feature_cols = [col for col in df_labeled.columns 
                   if not col.startswith('label_') 
                   and col != 'price' 
                   and col != 'timestamp']
    
    print(f"   Total features: {len(feature_cols)}")
    
    return df_labeled, feature_cols


def create_sequences(X, y, lookback=24):
    """Create LSTM sequences"""
    X_seq, y_seq = [], []
    
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


def train_model(X_train_seq, y_train_seq, X_test_seq, y_test_seq, num_features, epochs=15):
    """Train simple LSTM model"""
    
    # Build model
    hidden_size = min(32, max(16, num_features * 2))
    model = nn.Sequential(
        nn.LSTM(num_features, hidden_size, 2, batch_first=True, dropout=0.3),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid()
    )
    
    # Actually that won't work because LSTM outputs (batch, seq, hidden)
    # Let me use a proper LSTM class
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=32):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            out = self.fc(h_n[-1])
            return self.sigmoid(out)
    
    model = SimpleLSTM(num_features, hidden_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    batch_size = 64
    best_acc = 0.0
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        
        for i in range(0, len(X_train_seq), batch_size):
            batch_X = torch.FloatTensor(X_train_seq[i:i+batch_size])
            batch_y = torch.FloatTensor(y_train_seq[i:i+batch_size]).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(torch.FloatTensor(X_test_seq))
            val_pred = (val_out >= 0.5).float()
            val_acc = (val_pred == torch.FloatTensor(y_test_seq).unsqueeze(1)).float().mean().item()
        
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:
                break
    
    # Final eval
    model.eval()
    with torch.no_grad():
        test_out = model(torch.FloatTensor(X_test_seq))
        test_pred = (test_out >= 0.5).float().numpy()
        test_prob = test_out.numpy()
    
    accuracy = accuracy_score(y_test_seq, test_pred)
    try:
        roc_auc = roc_auc_score(y_test_seq, test_prob)
    except:
        roc_auc = 0.5
    
    return accuracy, roc_auc


def main():
    print("="*80)
    print("🚀 INCREMENTAL FEATURE TRAINING (Simplified)")
    print("="*80)
    
    # Load data
    df, all_features = load_and_prepare_data(horizon='24h')
    
    # Starting features
    starting_features = ['deriv30d_roc', 'volatility_24', 'avg14d_spread']
    
    # Check if features exist
    available_starting = [f for f in starting_features if f in all_features]
    if len(available_starting) < len(starting_features):
        print(f"\n⚠️  Some starting features missing:")
        print(f"   Requested: {starting_features}")
        print(f"   Available: {available_starting}")
        print(f"\n   Using first 3 available features instead...")
        available_starting = all_features[:3]
    
    print(f"\n📊 Starting features: {available_starting}")
    
    # Prepare data
    # Ensure all features are numeric
    X_df = df[all_features].select_dtypes(include=[np.number])
    numeric_features = X_df.columns.tolist()
    
    X = X_df.values.astype(float)
    y = df['label_24h'].values.astype(int)
    
    # Remove NaN and inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    # Update feature list to only numeric ones
    all_features = numeric_features
    
    print(f"✅ Data prepared: {len(X)} samples with {len(all_features)} numeric features")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Initialize
    current_features = available_starting.copy()
    current_accuracy = 0.0
    best_accuracy = 0.0
    best_features = current_features.copy()
    no_improve_count = 0
    history = []
    
    # Train baseline
    print(f"\n{'='*80}")
    print(f"BASELINE - {len(current_features)} features")
    print(f"{'='*80}")
    
    # Get indices for current features
    feature_indices = [all_features.index(f) for f in current_features]
    X_train_current = X_train[:, feature_indices]
    X_test_current = X_test[:, feature_indices]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_current)
    X_test_scaled = scaler.transform(X_test_current)
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback=24)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback=24)
    
    print(f"Training with {len(current_features)} features...")
    current_accuracy, current_roc_auc = train_model(
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, 
        len(current_features), epochs=20
    )
    
    print(f"✅ Baseline Accuracy: {current_accuracy:.4f}")
    print(f"   ROC-AUC: {current_roc_auc:.4f}")
    
    best_accuracy = current_accuracy
    
    history.append({
        'iteration': 0,
        'action': 'init',
        'features': current_features.copy(),
        'num_features': len(current_features),
        'accuracy': current_accuracy,
        'roc_auc': current_roc_auc
    })
    
    # Incremental loop
    max_iterations = 20
    no_improve_stop = 5
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/{max_iterations}")
        print(f"{'='*80}")
        print(f"Current: {len(current_features)} features, {current_accuracy:.4f} accuracy")
        print(f"Best: {len(best_features)} features, {best_accuracy:.4f} accuracy")
        print(f"No improvement: {no_improve_count}/{no_improve_stop}")
        
        if no_improve_count >= no_improve_stop:
            print(f"\n✋ Stopping - no improvement for {no_improve_stop} iterations")
            break
        
        # Rank remaining features by correlation
        remaining = [f for f in all_features if f not in current_features]
        
        if not remaining:
            print(f"\n✋ No more features to try")
            break
        
        print(f"\n🔍 Ranking {len(remaining)} remaining features...")
        
        correlations = []
        for feat in remaining[:50]:  # Only check top 50 to save time
            feat_idx = all_features.index(feat)
            try:
                corr = np.corrcoef(X[:, feat_idx], y)[0, 1]
                if not np.isnan(corr):
                    correlations.append((feat, abs(corr)))
            except:
                pass
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   Top 3 candidates:")
        for i, (feat, corr) in enumerate(correlations[:3], 1):
            print(f"      {i}. {feat} (corr={corr:.4f})")
        
        # Try top 3
        best_new = None
        best_new_acc = current_accuracy
        
        for feat, _ in correlations[:3]:
            test_features = current_features + [feat]
            test_indices = [all_features.index(f) for f in test_features]
            
            X_train_test = X_train[:, test_indices]
            X_test_test = X_test[:, test_indices]
            
            scaler_test = StandardScaler()
            X_train_test_scaled = scaler_test.fit_transform(X_train_test)
            X_test_test_scaled = scaler_test.transform(X_test_test)
            
            X_train_seq_test, y_train_seq_test = create_sequences(X_train_test_scaled, y_train, 24)
            X_test_seq_test, y_test_seq_test = create_sequences(X_test_test_scaled, y_test, 24)
            
            print(f"\n   🧪 Testing: {feat}")
            acc, roc = train_model(
                X_train_seq_test, y_train_seq_test,
                X_test_seq_test, y_test_seq_test,
                len(test_features), epochs=15
            )
            
            improvement = acc - current_accuracy
            print(f"      Accuracy: {acc:.4f} ({improvement:+.4f})")
            
            if acc > best_new_acc:
                best_new = feat
                best_new_acc = acc
        
        # Add best if improvement
        if best_new and (best_new_acc - current_accuracy) >= 0.001:
            print(f"\n✅ Adding: {best_new}")
            print(f"   {current_accuracy:.4f} → {best_new_acc:.4f} (+{best_new_acc-current_accuracy:.4f})")
            
            current_features.append(best_new)
            current_accuracy = best_new_acc
            no_improve_count = 0
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_features = current_features.copy()
            
            history.append({
                'iteration': iteration,
                'action': 'add',
                'feature': best_new,
                'features': current_features.copy(),
                'num_features': len(current_features),
                'accuracy': current_accuracy,
                'improvement': best_new_acc - current_accuracy
            })
        else:
            print(f"\n❌ No improvement")
            no_improve_count += 1
            
            history.append({
                'iteration': iteration,
                'action': 'skip',
                'features': current_features.copy(),
                'num_features': len(current_features),
                'accuracy': current_accuracy
            })
    
    # Final results
    print(f"\n{'='*80}")
    print(f"🎉 TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nBest Model:")
    print(f"  Features: {len(best_features)}")
    print(f"  Accuracy: {best_accuracy:.4f}")
    print(f"\nBest Features:")
    for i, feat in enumerate(best_features, 1):
        print(f"  {i}. {feat}")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/incremental_final.json', 'w') as f:
        json.dump({
            'best_features': best_features,
            'best_accuracy': float(best_accuracy),
            'history': history
        }, f, indent=2)
    
    print(f"\n💾 Results saved: results/incremental_final.json")


if __name__ == "__main__":
    main()

