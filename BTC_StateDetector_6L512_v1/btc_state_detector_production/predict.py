#!/usr/bin/env python3
"""
Simple prediction interface for Bitcoin State Detector

Usage:
    from predict import predict_state
    result = predict_state(price_array, volume_array)
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from feature_extractor import extract_features

# Load models (cached after first import)
_models_loaded = False
_direction_model = None
_strength_model = None
_scaler = None

def load_models():
    """Load trained models from disk (cached)"""
    global _models_loaded, _direction_model, _strength_model, _scaler
    
    if _models_loaded:
        return
    
    models_dir = Path(__file__).parent
    
    with open(models_dir / 'direction_model.pkl', 'rb') as f:
        _direction_model = pickle.load(f)
    
    with open(models_dir / 'strength_model.pkl', 'rb') as f:
        _strength_model = pickle.load(f)
    
    with open(models_dir / 'scaler.pkl', 'rb') as f:
        _scaler = pickle.load(f)
    
    _models_loaded = True

def predict_state(prices, volumes):
    """
    Predict current Bitcoin market state
    
    Args:
        prices: Array of 300 price values (last 5 hours, 1-minute candles)
        volumes: Array of 300 volume values (last 5 hours, 1-minute candles)
    
    Returns:
        dict: {
            'direction': str,          # 'UP', 'DOWN', or 'NONE'
            'direction_code': int,     # 1, -1, or 0
            'strength': float,         # 0-100
            'confidence': float,       # 0-100 (% confidence in direction)
            'probabilities': dict,     # {'DOWN': %, 'NONE': %, 'UP': %}
            'lean': str or None,       # 'UP', 'DOWN', 'NEUTRAL', or None (only when NONE)
            'raw_probabilities': list  # [down%, none%, up%] as floats
        }
    
    Example:
        >>> prices = [108500.0, 108510.5, ...]  # 300 values
        >>> volumes = [1234.5, 1456.8, ...]     # 300 values
        >>> result = predict_state(prices, volumes)
        >>> print(result['direction'])
        'UP'
        >>> print(result['confidence'])
        89.2
        >>> print(result['probabilities'])
        {'DOWN': 5.3, 'NONE': 5.5, 'UP': 89.2}
    """
    # Load models if not already loaded
    load_models()
    
    # Validate input
    prices = np.array(prices)
    volumes = np.array(volumes)
    
    if len(prices) < 300:
        raise ValueError(f"Need 300 price values (5 hours), got {len(prices)}")
    if len(volumes) < 300:
        raise ValueError(f"Need 300 volume values (5 hours), got {len(volumes)}")
    
    # Use last 300 values if more provided
    prices = prices[-300:]
    volumes = volumes[-300:]
    
    # Create DataFrame
    df = pd.DataFrame({
        'price': prices,
        'volume': volumes
    })
    
    # Extract features
    features = extract_features(df)
    
    # Prepare for model
    feature_names = sorted(features.keys())
    X = np.array([[features[fn] for fn in feature_names]])
    
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale
    X_scaled = _scaler.transform(X)
    
    # Predict
    direction_code = _direction_model.predict(X_scaled)[0]  # -1, 0, or 1
    strength = _strength_model.predict(X_scaled)[0]         # 0-100
    proba = _direction_model.predict_proba(X_scaled)[0]     # [down%, none%, up%]
    
    # Interpret
    direction_labels = {-1: 'DOWN', 0: 'NONE', 1: 'UP'}
    direction = direction_labels[direction_code]
    confidence = proba[direction_code + 1] * 100
    
    # Determine lean when NONE
    lean = None
    if direction_code == 0:
        if proba[2] > proba[0]:
            lean = 'UP'
        elif proba[0] > proba[2]:
            lean = 'DOWN'
        else:
            lean = 'NEUTRAL'
    
    return {
        'direction': direction,
        'direction_code': int(direction_code),
        'strength': float(strength),
        'confidence': float(confidence),
        'probabilities': {
            'DOWN': float(proba[0] * 100),
            'NONE': float(proba[1] * 100),
            'UP': float(proba[2] * 100)
        },
        'lean': lean,
        'raw_probabilities': [float(p) for p in proba]
    }

def main():
    """Example usage"""
    print("Bitcoin State Detector - Example")
    print("="*50)
    print()
    
    # Generate example data (normally you'd fetch from your data source)
    print("Generating example data (5 hours of 1-minute candles)...")
    np.random.seed(42)
    
    # Simulate an uptrend
    base_price = 108000.0
    prices = []
    for i in range(300):
        # Uptrend with noise
        trend = i * 2.0  # 2.0 per minute = +600 over 5 hours
        noise = np.random.randn() * 50
        prices.append(base_price + trend + noise)
    
    # Simulate volume
    volumes = np.random.uniform(1000, 2000, 300)
    
    print(f"Price range: ${prices[0]:,.2f} → ${prices[-1]:,.2f}")
    print()
    
    # Predict
    print("Running prediction...")
    result = predict_state(prices, volumes)
    
    print()
    print("="*50)
    print("RESULTS")
    print("="*50)
    print()
    print(f"Direction:   {result['direction']}")
    print(f"Strength:    {result['strength']:.1f}/100")
    print(f"Confidence:  {result['confidence']:.1f}%")
    print()
    print("Probabilities:")
    for direction, prob in result['probabilities'].items():
        print(f"  {direction:5s}: {prob:5.1f}%")
    
    if result['lean']:
        print()
        print(f"Leaning:     {result['lean']}")
    
    print()
    print("="*50)

if __name__ == '__main__':
    main()

