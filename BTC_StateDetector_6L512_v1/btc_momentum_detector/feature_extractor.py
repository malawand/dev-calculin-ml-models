"""
Shared feature extraction for state detection
Extracts 23 features (no 7-day metrics - original version)
"""
import numpy as np
import pandas as pd

def extract_features(df):
    """
    Extract features for state detection
    
    Args:
        df: DataFrame with 'price' and optionally 'volume' columns
        
    Returns:
        dict: Feature names and values
    """
    prices = df['price'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(prices))
    
    features = {}
    
    # Price features (multiple timeframes: 5min to 4h)
    for period in [5, 15, 30, 60, 120, 240]:
        if len(prices) >= period:
            # ROC (Rate of Change)
            roc = (prices[-1] - prices[-period]) / prices[-period]
            features[f'roc_{period}'] = roc
            
            # Volatility (standard deviation of returns)
            returns = np.diff(prices[-period:]) / prices[-period:-1]
            features[f'vol_{period}'] = np.std(returns)
            
            # Trend strength (correlation with time)
            x = np.arange(period)
            y = prices[-period:]
            if len(x) == len(y):
                corr = np.corrcoef(x, y)[0, 1]
                features[f'trend_{period}'] = corr
    
    # Volume features (4 timeframes)
    for period in [15, 30, 60, 120]:
        if len(volumes) >= period and volumes.sum() > 0:
            recent_vol = np.mean(volumes[-period//2:])
            avg_vol = np.mean(volumes[-period:])
            if avg_vol > 0:
                features[f'vol_ratio_{period}'] = recent_vol / avg_vol
    
    # Acceleration (change in momentum)
    if len(prices) >= 60:
        roc_recent = (prices[-1] - prices[-30]) / prices[-30]
        roc_earlier = (prices[-30] - prices[-60]) / prices[-60]
        features['acceleration'] = roc_recent - roc_earlier
    
    return features
