#!/usr/bin/env python3
"""
Train Minimal Model - Start with Just 3 Features

The simplest possible Bitcoin direction predictor.
Uses only 3 hand-picked features to establish baseline.

Usage:
    python train_minimal.py --horizon 24h
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from btc_lstm_ensemble.lstm_model import LSTMDirectionClassifier
from btc_lstm_ensemble.data_loader import load_cached_data, create_sequences


# ============================================================================
# MINIMAL FEATURE SET - Just 3 Features
# ============================================================================
MINIMAL_FEATURES = [
    'deriv30d_roc',      # Long-term trend (30-day derivative)
    'volatility_24',     # Market volatility
    'avg14d_spread',     # Mean reversion signal
]

print("="*80)
print("🎯 MINIMAL MODEL - 3 Features Only")
print("="*80)
print(f"\nStarting features:")
for i, feat in enumerate(MINIMAL_FEATURES, 1):
    print(f"  {i}. {feat}")
print(f"\n{'='*80}\n")


def train_minimal_model(horizon='24h', epochs=30, batch_size=64):
    """
    Train LSTM with minimal 3 features
    
    Args:
        horizon: Prediction horizon
        epochs: Training epochs
        batch_size: Batch size
        
    Returns:
        accuracy, roc_auc, model
    """
    print(f"🚀 Training minimal model for {horizon} horizon\n")
    
    # 1. Load data
    print("📥 Loading cached data...")
    df, all_features = load_cached_data(horizon)
    print(f"   Total features available: {len(all_features)}")
    print(f"   Using only {len(MINIMAL_FEATURES)} features\n")
    
    # 2. Check if minimal features exist
    missing = [f for f in MINIMAL_FEATURES if f not in all_features]
    if missing:
        print(f"❌ Missing features: {missing}")
        print(f"   Available features: {all_features[:20]}...")
        return None, None, None
    
    # 3. Prepare data with minimal features
    print("🔧 Preparing data...")
    X = df[MINIMAL_FEATURES].values
    y = df['label_' + horizon].values
    
    # Remove NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    print(f"   Valid samples: {len(X)}")
    print(f"   Features shape: {X.shape}")
    print(f"   Class balance: UP={y.sum()}/{len(y)} ({100*y.mean():.1f}%)\n")
    
    if len(X) < 1000:
        print("❌ Not enough data!")
        return None, None, None
    
    # 4. Split data
    print("📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples\n")
    
    # 5. Scale features
    print("⚖️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Create sequences
    print("📦 Creating sequences...")
    lookback = 24
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback)
    
    print(f"   Train sequences: {len(X_train_seq)}")
    print(f"   Test sequences: {len(X_test_seq)}")
    print(f"   Sequence shape: {X_train_seq.shape}\n")
    
    # 7. Build model
    print("🤖 Building LSTM model...")
    model = LSTMDirectionClassifier(
        input_size=len(MINIMAL_FEATURES),
        hidden_sizes=[32, 32],  # Smaller model for fewer features
        dropout=0.3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Input size: {len(MINIMAL_FEATURES)} features")
    print(f"   LSTM layers: [32, 32]")
    print(f"   Dropout: 0.3\n")
    
    # 8. Train
    print("🎓 Training model...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}\n")
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    patience = 5
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training
        for i in range(0, len(X_train_seq), batch_size):
            batch_X = torch.FloatTensor(X_train_seq[i:i+batch_size])
            batch_y = torch.FloatTensor(y_train_seq[i:i+batch_size]).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_test_seq))
            val_predicted = (val_outputs >= 0.5).float()
            val_correct = (val_predicted == torch.FloatTensor(y_test_seq).unsqueeze(1)).sum().item()
            val_acc = 100 * val_correct / len(y_test_seq)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:2d}/{epochs} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            # Save best model
            torch.save(model.state_dict(), 'models/minimal_3feat_best.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n   ⚠️  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    # 9. Final evaluation
    print(f"\n{'='*80}")
    print("📊 FINAL EVALUATION")
    print("="*80)
    
    model.load_state_dict(torch.load('models/minimal_3feat_best.pt'))
    model.eval()
    
    with torch.no_grad():
        test_outputs = model(torch.FloatTensor(X_test_seq))
        test_predicted = (test_outputs >= 0.5).float().numpy()
        test_probs = test_outputs.numpy()
    
    accuracy = accuracy_score(y_test_seq, test_predicted)
    try:
        roc_auc = roc_auc_score(y_test_seq, test_probs)
    except:
        roc_auc = 0.5
    
    print(f"\nMinimal Model (3 features):")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Features:  {len(MINIMAL_FEATURES)}")
    print(f"\n{'='*80}\n")
    
    # 10. Save results
    results = {
        'horizon': horizon,
        'features': MINIMAL_FEATURES,
        'num_features': len(MINIMAL_FEATURES),
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'train_samples': len(X_train_seq),
        'test_samples': len(X_test_seq),
        'epochs_trained': epoch + 1,
        'best_epoch': epoch + 1 - no_improve
    }
    
    Path('results').mkdir(exist_ok=True)
    with open('results/minimal_3feat_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: results/minimal_3feat_results.json")
    print(f"💾 Model saved to: models/minimal_3feat_best.pt")
    
    # Save scaler
    import pickle
    with open('models/minimal_3feat_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to: models/minimal_3feat_scaler.pkl\n")
    
    return accuracy, roc_auc, model


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Minimal 3-Feature Model')
    parser.add_argument('--horizon', type=str, default='24h',
                       choices=['15m', '1h', '4h', '24h'],
                       help='Prediction horizon')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Train
    accuracy, roc_auc, model = train_minimal_model(
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if accuracy is not None:
        print("✅ Training complete!")
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review results: cat results/minimal_3feat_results.json")
        print(f"   2. Run incremental training: python incremental_train.py")
        print(f"   3. Compare to baseline: python compare_to_baseline.py\n")
    else:
        print("❌ Training failed!")


if __name__ == "__main__":
    main()



