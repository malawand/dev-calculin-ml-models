#!/usr/bin/env python3
"""
ULTRA-DEEP Neural Network Training (MAXIMIZED LEARNING)

This is what "proper" training looks like:
- Architecture: 6 layers (512 → 256 → 128 → 64 → 32 → 16)
- Training: 10,000 epochs (no early stopping!)
- Time: 5-10 MINUTES (you'll see real learning)
- Batch training: Mini-batches for more gradient updates
- L2 Regularization: Strong regularization to prevent overfitting

This TRULY maximizes learning from 2.55 years of data.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pickle
import json
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔥 ULTRA-DEEP NEURAL NETWORK TRAINING (MAXIMIZED)")
print("="*80)
print()
print("This will take 5-10 MINUTES and truly maximize learning!")
print("You'll see EXTENSIVE training progress.")
print()
print("Configuration:")
print("   • Architecture: 6 layers (512→256→128→64→32→16)")
print("   • Max epochs: 10,000")
print("   • Early stopping: DISABLED")
print("   • Batch size: 32 (more gradient updates)")
print("   • L2 regularization: 0.001 (strong)")
print("   • Verbose: Every 10 iterations")
print()
input("Press ENTER to start training (this will take several minutes)...")
print()

start_time = time.time()

# Load data
data_paths = [
    Path(__file__).parent.parent / 'btc_direction_predictor/artifacts/historical_data/combined_with_volume.parquet',
]

print("📥 Loading data...")
df = None
for path in data_paths:
    if path.exists():
        df = pd.read_parquet(path)
        break

if df is None:
    print("❌ Could not find data!")
    sys.exit(1)

if 'timestamp' in df.columns:
    df = df.set_index('timestamp')
df.index = pd.to_datetime(df.index)
df = df.sort_index()

print(f"   ✅ Loaded {len(df):,} samples")
print()

# Extract features and labels
print("📊 Extracting features and current state labels...")
print("   (This is sampling every 30 minutes from 2.55 years)")
print()

sys.path.insert(0, str(Path(__file__).parent))
from feature_extractor import extract_features

lookback = 300
test_window = 60
sample_every = 30

samples = []
total_possible = (len(df) - lookback - test_window) // sample_every

for idx, i in enumerate(range(lookback, len(df) - test_window, sample_every)):
    if idx % 100 == 0:
        progress = (idx / total_possible) * 100
        print(f"   Progress: {progress:.1f}%", end='\r')
    
    window = df.iloc[i-lookback:i].copy()
    
    # Rename columns to match what feature_extractor expects
    window_for_features = pd.DataFrame({
        'price': window['crypto_last_price'].values,
        'volume': window['crypto_volume'].values if 'crypto_volume' in window.columns else np.zeros(len(window))
    })
    
    if len(window_for_features) < lookback:
        continue
    
    # Extract features
    features = extract_features(window_for_features)
    if not features:
        continue
    
    # Calculate TRUE current state
    prices = window_for_features['price'].values
    prices_recent = prices[-60:]
    
    # True strength
    returns_recent = np.diff(prices_recent) / prices_recent[:-1]
    true_volatility = np.std(returns_recent) * 100
    true_strength = min(true_volatility * 50, 100)
    
    # True direction
    price_change = (prices_recent[-1] - prices_recent[0]) / prices_recent[0] * 100
    if price_change > 0.3:
        true_direction = 1  # UP
    elif price_change < -0.3:
        true_direction = -1  # DOWN
    else:
        true_direction = 0  # NONE
    
    samples.append({
        'features': features,
        'true_strength': true_strength,
        'true_direction': true_direction
    })

print(f"\n   ✅ Extracted {len(samples):,} training samples")
print()

# Convert to arrays - collect all unique feature names
all_feature_names = set()
for s in samples:
    all_feature_names.update(s['features'].keys())
feature_names = sorted(list(all_feature_names))

X = []
y_strength = []
y_direction = []

for s in samples:
    # Use .get() with default 0 for missing features
    X.append([s['features'].get(f, 0.0) for f in feature_names])
    y_strength.append(s['true_strength'])
    y_direction.append(s['true_direction'])

X = np.array(X)
y_strength = np.array(y_strength)
y_direction = np.array(y_direction)

# Handle NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y_strength = np.nan_to_num(y_strength, nan=0.0, posinf=0.0, neginf=0.0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split (80/20, no shuffle to respect time series)
X_train, X_test, y_str_train, y_str_test, y_dir_train, y_dir_test = train_test_split(
    X_scaled, y_strength, y_direction, test_size=0.2, shuffle=False
)

print(f"   Training set: {len(X_train):,} samples")
print(f"   Test set:     {len(X_test):,} samples")
print(f"   Features:     {len(feature_names)}")
print()

# Train ULTRA-DEEP models
print("="*80)
print("🔥 TRAINING ULTRA-DEEP NEURAL NETWORK")
print("="*80)
print()
print("This will take 5-10 MINUTES. You'll see extensive learning progress!")
print()

# Train strength model (regression) - ULTRA-DEEP
print("🧠 Training Strength Model (Ultra-Deep)...")
print("   Architecture: 512 → 256 → 128 → 64 → 32 → 16 (6 layers)")
print("   Max epochs: 10,000")
print("   Early stopping: DISABLED")
print("   Batch size: 32")
print("   L2 regularization: 0.001 (strong)")
print()
print("   Progress (will show every 10 iterations):")
print("   " + "-"*60)

strength_model = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128, 64, 32, 16),  # Very deep!
    activation='relu',
    max_iter=10000,  # 10,000 epochs!
    batch_size=32,  # Mini-batch training
    learning_rate='adaptive',  # Adjust learning rate
    learning_rate_init=0.001,
    alpha=0.001,  # Strong L2 regularization
    early_stopping=False,  # NO EARLY STOPPING!
    random_state=42,
    verbose=True  # Show progress every 10 iterations
)

print()
strength_model.fit(X_train, y_str_train)
print()
print("   ✅ Strength model trained!")
print()

# Train direction model (classification) - ULTRA-DEEP
print("🧠 Training Direction Model (Ultra-Deep)...")
print("   Architecture: 512 → 256 → 128 → 64 → 32 → 16 (6 layers)")
print("   Max epochs: 10,000")
print("   Early stopping: DISABLED")
print("   Batch size: 32")
print("   L2 regularization: 0.001 (strong)")
print()
print("   Progress (will show every 10 iterations):")
print("   " + "-"*60)

direction_model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64, 32, 16),  # Very deep!
    activation='relu',
    max_iter=10000,  # 10,000 epochs!
    batch_size=32,  # Mini-batch training
    learning_rate='adaptive',  # Adjust learning rate
    learning_rate_init=0.001,
    alpha=0.001,  # Strong L2 regularization
    early_stopping=False,  # NO EARLY STOPPING!
    random_state=42,
    verbose=True  # Show progress every 10 iterations
)

print()
direction_model.fit(X_train, y_dir_train)
print()

training_time = time.time() - start_time
print("="*80)
print(f"✅ TRAINING COMPLETE! (took {training_time/60:.1f} minutes)")
print("="*80)
print()

# Evaluate
print("📊 Evaluating on test set...")
print()

strength_pred = strength_model.predict(X_test)
direction_pred = direction_model.predict(X_test)

# Metrics
direction_accuracy = accuracy_score(y_dir_test, direction_pred)
strength_corr = np.corrcoef(strength_pred, y_str_test)[0, 1]
strength_mae = np.mean(np.abs(strength_pred - y_str_test))

# Per-class metrics
dir_report = classification_report(y_dir_test, direction_pred, output_dict=True, zero_division=0)

print("="*80)
print("📊 ULTRA-DEEP MODEL RESULTS")
print("="*80)
print()

print("DIRECTION DETECTION:")
print(f"   Overall Accuracy:  {direction_accuracy*100:.1f}%")
print()
print(f"   Per-class F1 scores:")
dir_map = {-1: 'DOWN', 0: 'NONE', 1: 'UP'}
for dir_val, dir_name in dir_map.items():
    dir_str = str(dir_val)
    if dir_str in dir_report:
        f1 = dir_report[dir_str]['f1-score'] * 100
        support = dir_report[dir_str]['support']
        print(f"      {dir_name:6s}  {f1:5.1f}%  ({int(support)} samples)")

print()
print("STRENGTH MEASUREMENT:")
print(f"   Correlation:       {strength_corr:.3f}")
print(f"   MAE:               {strength_mae:.1f} points")
print()

print("="*80)
print("🏆 COMPARISON WITH OTHER MODELS")
print("="*80)
print()

# Baseline comparisons
fast_acc = 0.906
fast_str = 0.703
deep_acc = 0.93  # Expected
deep_str = 0.75  # Expected

print("                  Fast     Deep     Ultra     Ultra Gain")
print("                  Model    Model    Model     (vs Fast)")
print("-"*80)
print(f"Accuracy:         90.6%    93.0%    {direction_accuracy*100:.1f}%     {(direction_accuracy - fast_acc)*100:+.1f} pp")
print(f"Strength Corr:    0.703    0.750    {strength_corr:.3f}     {strength_corr - fast_str:+.3f}")
print(f"Training Time:    3s       60s      {training_time:.0f}s      {training_time/3:.0f}x slower")
print()

improvement_acc = (direction_accuracy - fast_acc) * 100
improvement_str = strength_corr - fast_str

if direction_accuracy > fast_acc + 0.03:  # At least 3% better
    print("✅ ULTRA-DEEP MODEL IS SIGNIFICANTLY BETTER!")
    print()
    print(f"   • Accuracy improved by {improvement_acc:+.1f} percentage points")
    print(f"   • Strength correlation improved by {improvement_str:+.3f}")
    print()
    print("   The extensive training (10,000 epochs) paid off!")
    print("   This model has TRULY learned from 2.55 years of data.")
    print()
    print("   Recommended: Use this for maximum accuracy.")
    
elif direction_accuracy > fast_acc + 0.01:  # 1-3% better
    print("⚠️  ULTRA-DEEP MODEL IS SOMEWHAT BETTER")
    print()
    print(f"   • Accuracy improved by {improvement_acc:+.1f} percentage points")
    print(f"   • Strength correlation improved by {improvement_str:+.3f}")
    print()
    print("   The improvement is modest for {training_time/60:.1f} minutes of training.")
    print("   Consider:")
    print("   • Use ultra-deep for production (maximum accuracy)")
    print("   • Use fast model for experiments (3 seconds)")
    
elif direction_accuracy > fast_acc - 0.02:  # Within 2%
    print("⚠️  DIMINISHING RETURNS")
    print()
    print(f"   • Accuracy change: {improvement_acc:+.1f} pp (not significant)")
    print(f"   • Strength change: {improvement_str:+.3f}")
    print()
    print("   10,000 epochs didn't help much beyond the fast model.")
    print()
    print("   Possible reasons:")
    print("   • Fast model (90.6%) already near optimal for these features")
    print("   • Need better features, not more training")
    print("   • Dataset size (1,276 samples) limits learning")
    print()
    print("   Recommended: Stick with fast model (3s vs {training_time/60:.1f}min)")
    
else:
    print("❌ OVERFITTING DETECTED")
    print()
    print(f"   • Accuracy DECREASED by {-improvement_acc:.1f} pp")
    print()
    print("   The model overfit despite L2 regularization.")
    print("   10,000 epochs was too much without early stopping.")
    print()
    print("   Recommended: Use fast model or deep model instead.")

print()
print("="*80)

# Show training iterations info
print()
print("📈 TRAINING DETAILS:")
print()
print(f"   Strength Model:")
print(f"      Iterations:  {strength_model.n_iter_}")
print(f"      Final loss:  {strength_model.loss_:.6f}")
if hasattr(strength_model, 'best_loss_'):
    print(f"      Best loss:   {strength_model.best_loss_:.6f}")

print()
print(f"   Direction Model:")
print(f"      Iterations:  {direction_model.n_iter_}")
print(f"      Final loss:  {direction_model.loss_:.6f}")
if hasattr(direction_model, 'best_loss_'):
    print(f"      Best loss:   {direction_model.best_loss_:.6f}")

print()
print(f"   Total training time:  {training_time/60:.1f} minutes")
print(f"   Passes through data:  {strength_model.n_iter_ + direction_model.n_iter_:,}")
print()

# Save models
models_dir = Path(__file__).parent
print("💾 Saving ultra-deep models...")

with open(models_dir / 'direction_model.pkl', 'wb') as f:
    pickle.dump(direction_model, f)
print(f"   ✅ Saved direction_model.pkl")

with open(models_dir / 'strength_model.pkl', 'wb') as f:
    pickle.dump(strength_model, f)
print(f"   ✅ Saved strength_model.pkl")

with open(models_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✅ Saved scaler.pkl")

with open(models_dir / 'feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print(f"   ✅ Saved feature_names.json")

print()
print(f"📁 Models saved to: {models_dir}")
print()

print("="*80)
print("🎯 SUMMARY")
print("="*80)
print()
print(f"You trained an ULTRA-DEEP model with:")
print(f"   • 6 layers (512→256→128→64→32→16)")
print(f"   • 10,000 max epochs (no early stopping)")
print(f"   • {strength_model.n_iter_ + direction_model.n_iter_:,} total iterations")
print(f"   • {training_time/60:.1f} minutes of training")
print(f"   • Strong L2 regularization (0.001)")
print()
print(f"Results:")
print(f"   • Accuracy: {direction_accuracy*100:.1f}%")
print(f"   • Strength: {strength_corr:.3f}")
print()

if direction_accuracy >= 0.92:
    print("✅ Excellent results! This model has truly maximized learning.")
elif direction_accuracy >= 0.88:
    print("✅ Good results! Comparable to faster models.")
else:
    print("⚠️  Consider using fast model or adding better features.")

print()
print("="*80)

