#!/usr/bin/env python3
"""
Incremental Feature Training - Add Features One at a Time

Start with minimal features, add one at a time, keep only if improves accuracy.
Continue until no improvement for N iterations or max features reached.

Usage:
    python incremental_train.py --start-features 3 --max-features 20
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
from tqdm import tqdm
import sys
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from btc_lstm_ensemble.lstm_model import LSTMDirectionClassifier
from btc_lstm_ensemble.data_loader import load_cached_data, create_sequences


class IncrementalTrainer:
    """Manages incremental feature addition/removal"""
    
    def __init__(self, horizon='24h', max_features=20, max_iterations=50, 
                 no_improve_stop=5, min_improvement=0.001):
        self.horizon = horizon
        self.max_features = max_features
        self.max_iterations = max_iterations
        self.no_improve_stop = no_improve_stop
        self.min_improvement = min_improvement
        
        # Load all data once
        print(f"📥 Loading all available data...")
        
        # Use already-engineered data from btc_lstm_ensemble
        data_path = Path(__file__).parent.parent / 'btc_lstm_ensemble/data_cache_24h.pkl'
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data cache not found: {data_path}\n"
                "Please run btc_lstm_ensemble training first to generate cache."
            )
        
        import pickle
        with open(data_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.df = cache_data['df']
        self.all_features = cache_data['features']
        
        print(f"   Loaded {len(self.df)} samples with {len(self.all_features)} features")
        print(f"   Label: label_{horizon}")
        
        # Progress tracking
        self.history = []
        self.current_features = []
        self.current_accuracy = 0.0
        self.current_roc_auc = 0.0
        self.best_accuracy = 0.0
        self.best_features = []
        self.no_improve_count = 0
        
    def train_with_features(self, features, epochs=20, batch_size=64, verbose=False):
        """
        Train LSTM with given features
        
        Returns:
            accuracy, roc_auc
        """
        # Prepare data
        X = self.df[features].values
        y = self.df['label_' + self.horizon].values
        
        # Remove NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 1000:
            return 0.0, 0.5
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create sequences
        lookback = 24
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback)
        
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            return 0.0, 0.5
        
        # Build model (smaller for fewer features)
        hidden_size = min(32, max(16, len(features) * 2))
        model = LSTMDirectionClassifier(
            input_size=len(features),
            hidden_sizes=[hidden_size, hidden_size],
            dropout=0.3
        )
        
        # Train
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_val_acc = 0.0
        patience = 3
        no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            
            # Training
            for i in range(0, len(X_train_seq), batch_size):
                batch_X = torch.FloatTensor(X_train_seq[i:i+batch_size])
                batch_y = torch.FloatTensor(y_train_seq[i:i+batch_size]).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(torch.FloatTensor(X_test_seq))
                val_predicted = (val_outputs >= 0.5).float()
                val_correct = (val_predicted == torch.FloatTensor(y_test_seq).unsqueeze(1)).sum().item()
                val_acc = val_correct / len(y_test_seq)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        
        # Final evaluation
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
        
        return accuracy, roc_auc
    
    def rank_remaining_features(self, max_candidates=10):
        """
        Rank remaining features by correlation with target
        
        Returns:
            List of (feature, correlation) tuples
        """
        remaining = [f for f in self.all_features if f not in self.current_features]
        
        if not remaining:
            return []
        
        # Calculate correlation with target
        y = self.df['label_' + self.horizon]
        correlations = []
        
        for feature in remaining:
            try:
                x = self.df[feature]
                # Remove NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 100:
                    continue
                
                corr = np.corrcoef(x[mask], y[mask])[0, 1]
                if not np.isnan(corr):
                    correlations.append((feature, abs(corr)))
            except:
                pass
        
        # Sort by correlation
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        return correlations[:max_candidates]
    
    def try_add_feature(self, feature):
        """
        Try adding a feature, return accuracy improvement
        
        Returns:
            (improved, accuracy, roc_auc)
        """
        test_features = self.current_features + [feature]
        
        print(f"   🧪 Testing: {feature}")
        accuracy, roc_auc = self.train_with_features(test_features, epochs=15)
        
        improvement = accuracy - self.current_accuracy
        improved = improvement >= self.min_improvement
        
        print(f"      Accuracy: {accuracy:.4f} ({improvement:+.4f}) {'✅' if improved else '❌'}")
        
        return improved, accuracy, roc_auc
    
    def run(self, starting_features):
        """
        Run incremental training
        
        Args:
            starting_features: List of initial features to start with
        """
        print("="*80)
        print("🚀 INCREMENTAL FEATURE TRAINING")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Horizon: {self.horizon}")
        print(f"  Starting features: {len(starting_features)}")
        print(f"  Max features: {self.max_features}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Stop after no improvement: {self.no_improve_stop}")
        print(f"  Min improvement threshold: {self.min_improvement:.4f}")
        print(f"\n{'='*80}\n")
        
        # Initialize with starting features
        self.current_features = starting_features.copy()
        
        print(f"📊 BASELINE - Training with {len(self.current_features)} initial features")
        print(f"   Features: {', '.join(self.current_features)}")
        self.current_accuracy, self.current_roc_auc = self.train_with_features(
            self.current_features, epochs=20
        )
        print(f"   Accuracy: {self.current_accuracy:.4f}")
        print(f"   ROC-AUC: {self.current_roc_auc:.4f}\n")
        
        self.best_accuracy = self.current_accuracy
        self.best_features = self.current_features.copy()
        
        # Save initial state
        self.history.append({
            'iteration': 0,
            'action': 'init',
            'features': self.current_features.copy(),
            'num_features': len(self.current_features),
            'accuracy': self.current_accuracy,
            'roc_auc': self.current_roc_auc
        })
        
        # Incremental training loop
        for iteration in range(1, self.max_iterations + 1):
            print(f"{'='*80}")
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*80}")
            print(f"Current: {len(self.current_features)} features, {self.current_accuracy:.4f} accuracy")
            print(f"Best: {len(self.best_features)} features, {self.best_accuracy:.4f} accuracy")
            print(f"No improvement count: {self.no_improve_count}/{self.no_improve_stop}\n")
            
            # Check stopping conditions
            if len(self.current_features) >= self.max_features:
                print(f"✋ Reached max features ({self.max_features})")
                break
            
            if self.no_improve_count >= self.no_improve_stop:
                print(f"✋ No improvement for {self.no_improve_stop} iterations")
                break
            
            # Rank remaining features
            print(f"🔍 Ranking remaining features by correlation...")
            candidates = self.rank_remaining_features(max_candidates=5)
            
            if not candidates:
                print(f"   No more features to try!")
                break
            
            print(f"   Top 5 candidates:")
            for i, (feat, corr) in enumerate(candidates[:5], 1):
                print(f"      {i}. {feat} (corr={corr:.4f})")
            print()
            
            # Try adding best candidates
            best_improvement = 0.0
            best_new_feature = None
            best_new_accuracy = self.current_accuracy
            best_new_roc_auc = self.current_roc_auc
            
            for feature, _ in candidates[:3]:  # Try top 3
                improved, accuracy, roc_auc = self.try_add_feature(feature)
                
                if improved and (accuracy - self.current_accuracy) > best_improvement:
                    best_improvement = accuracy - self.current_accuracy
                    best_new_feature = feature
                    best_new_accuracy = accuracy
                    best_new_roc_auc = roc_auc
            
            # Add best feature if found
            if best_new_feature:
                print(f"\n✅ Adding feature: {best_new_feature}")
                print(f"   Accuracy: {self.current_accuracy:.4f} → {best_new_accuracy:.4f} (+{best_improvement:.4f})")
                
                self.current_features.append(best_new_feature)
                self.current_accuracy = best_new_accuracy
                self.current_roc_auc = best_new_roc_auc
                self.no_improve_count = 0
                
                if self.current_accuracy > self.best_accuracy:
                    self.best_accuracy = self.current_accuracy
                    self.best_features = self.current_features.copy()
                
                self.history.append({
                    'iteration': iteration,
                    'action': 'add',
                    'feature': best_new_feature,
                    'features': self.current_features.copy(),
                    'num_features': len(self.current_features),
                    'accuracy': self.current_accuracy,
                    'roc_auc': self.current_roc_auc,
                    'improvement': best_improvement
                })
            else:
                print(f"\n❌ No improvement found")
                self.no_improve_count += 1
                
                self.history.append({
                    'iteration': iteration,
                    'action': 'skip',
                    'features': self.current_features.copy(),
                    'num_features': len(self.current_features),
                    'accuracy': self.current_accuracy,
                    'roc_auc': self.current_roc_auc
                })
            
            # Save progress
            self.save_progress()
            
            print()
        
        # Final results
        print("\n" + "="*80)
        print("🎉 INCREMENTAL TRAINING COMPLETE!")
        print("="*80)
        print(f"\nFinal Model:")
        print(f"  Features: {len(self.best_features)}")
        print(f"  Accuracy: {self.best_accuracy:.4f}")
        print(f"  Iterations: {iteration}")
        print(f"\nBest Features:")
        for i, feat in enumerate(self.best_features, 1):
            print(f"  {i}. {feat}")
        print()
        
        # Train final model
        print("🎓 Training final model with best features...")
        final_accuracy, final_roc_auc = self.train_with_features(
            self.best_features, epochs=30, verbose=True
        )
        
        print(f"\nFinal Performance:")
        print(f"  Accuracy: {final_accuracy:.4f}")
        print(f"  ROC-AUC: {final_roc_auc:.4f}")
        
        # Save final results
        self.save_final_results(final_accuracy, final_roc_auc)
        
    def save_progress(self):
        """Save progress to JSON"""
        Path('results').mkdir(exist_ok=True)
        
        progress = {
            'horizon': self.horizon,
            'current_features': self.current_features,
            'current_accuracy': float(self.current_accuracy),
            'current_roc_auc': float(self.current_roc_auc),
            'best_features': self.best_features,
            'best_accuracy': float(self.best_accuracy),
            'history': self.history
        }
        
        with open('results/incremental_progress.json', 'w') as f:
            json.dump(progress, f, indent=2)
    
    def save_final_results(self, final_accuracy, final_roc_auc):
        """Save final results"""
        results = {
            'horizon': self.horizon,
            'features': self.best_features,
            'num_features': len(self.best_features),
            'accuracy': float(final_accuracy),
            'roc_auc': float(final_roc_auc),
            'iterations': len(self.history),
            'history': self.history
        }
        
        with open('results/incremental_final.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: results/incremental_final.json")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Incremental Feature Training')
    parser.add_argument('--horizon', type=str, default='24h',
                       choices=['15m', '1h', '4h', '24h'],
                       help='Prediction horizon')
    parser.add_argument('--start-features', type=int, default=3,
                       help='Number of starting features')
    parser.add_argument('--max-features', type=int, default=20,
                       help='Maximum features to add')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='Maximum iterations')
    parser.add_argument('--no-improve-stop', type=int, default=5,
                       help='Stop after N iterations without improvement')
    
    args = parser.parse_args()
    
    # Starting features (minimal set)
    starting_features = [
        'deriv30d_roc',      # Long-term trend
        'volatility_24',     # Volatility
        'avg14d_spread',     # Mean reversion
    ]
    
    print(f"\n🎯 Starting with {len(starting_features)} features:")
    for feat in starting_features:
        print(f"   - {feat}")
    print()
    
    # Create trainer
    trainer = IncrementalTrainer(
        horizon=args.horizon,
        max_features=args.max_features,
        max_iterations=args.max_iterations,
        no_improve_stop=args.no_improve_stop
    )
    
    # Run
    trainer.run(starting_features)


if __name__ == "__main__":
    main()

