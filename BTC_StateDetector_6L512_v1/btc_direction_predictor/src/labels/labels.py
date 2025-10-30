"""
Label creation for directional prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def parse_horizon(horizon: str) -> int:
    """
    Convert horizon string to number of 5-minute steps.
    
    Args:
        horizon: e.g., "15m", "1h", "4h", "24h"
        
    Returns:
        Number of 5-minute steps
    """
    mapping = {
        '5m': 1,
        '15m': 3,
        '30m': 6,
        '1h': 12,
        '2h': 24,
        '4h': 48,
        '8h': 96,
        '12h': 144,
        '24h': 288
    }
    
    if horizon not in mapping:
        raise ValueError(f"Unsupported horizon: {horizon}. Supported: {list(mapping.keys())}")
    
    return mapping[horizon]


class LabelCreator:
    """Create directional labels for multiple horizons"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        self.horizons = config['labels']['horizons']
        self.threshold = config['labels']['threshold_pct'] / 100.0  # Convert to decimal
        
    def create_labels(self, df: pd.DataFrame, price_col: str = None) -> pd.DataFrame:
        """
        Create labels for all horizons.
        
        Args:
            df: DataFrame with features
            price_col: Name of price column
            
        Returns:
            DataFrame with labels added
        """
        logger.info("=" * 80)
        logger.info("CREATING LABELS")
        logger.info("=" * 80)
        
        df = df.copy()
        
        # Find price column if not specified
        if price_col is None:
            price_col = self._get_price_column(df)
        
        logger.info(f"Using price column: {price_col}")
        logger.info(f"Threshold: {self.threshold * 100:.2f}%")
        
        for horizon in self.horizons:
            logger.info(f"Creating label for horizon: {horizon}")
            df = self._create_horizon_label(df, price_col, horizon)
            
            # Log class distribution
            label_col = f'label_{horizon}'
            class_dist = df[label_col].value_counts(normalize=True)
            logger.info(f"  Class distribution:")
            logger.info(f"    DOWN (0): {class_dist.get(0, 0):.2%}")
            logger.info(f"    UP (1): {class_dist.get(1, 0):.2%}")
        
        return df
    
    def _get_price_column(self, df: pd.DataFrame) -> str:
        """Identify the spot price column"""
        candidates = ['price', 'crypto_last_price']
        for col in candidates:
            if col in df.columns:
                return col
        raise ValueError(f"No price column found. Available: {df.columns.tolist()}")
    
    def _create_horizon_label(self, df: pd.DataFrame, price_col: str, horizon: str) -> pd.DataFrame:
        """Create label for a single horizon"""
        steps = parse_horizon(horizon)
        
        # Calculate future return
        # return = (price[t+H] - price[t]) / price[t]
        future_price = df[price_col].shift(-steps)
        current_price = df[price_col]
        
        df[f'return_{horizon}'] = (future_price - current_price) / current_price
        
        # Create binary label
        # 1 = UP if return > threshold
        # 0 = DOWN/FLAT if return <= threshold
        df[f'label_{horizon}'] = (df[f'return_{horizon}'] > self.threshold).astype(int)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excludes labels and returns)"""
        exclude_patterns = ['label_', 'return_1h', 'return_4h', 'return_15m', 'return_24h']
        
        feature_cols = []
        for col in df.columns:
            if not any(pattern in col for pattern in exclude_patterns):
                feature_cols.append(col)
        
        return feature_cols


def create_labels(df: pd.DataFrame, config: Dict, price_col: str = None) -> pd.DataFrame:
    """Convenience function for label creation"""
    creator = LabelCreator(config)
    return creator.create_labels(df, price_col)


def get_X_y(df: pd.DataFrame, horizon: str, feature_cols: List[str] = None) -> tuple:
    """
    Extract features and labels for a specific horizon.
    
    Args:
        df: DataFrame with features and labels
        horizon: Horizon to extract label for
        feature_cols: List of feature columns (if None, auto-detect)
        
    Returns:
        (X, y) tuple where X is features and y is labels
    """
    label_col = f'label_{horizon}'
    
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col} not found")
    
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        exclude_patterns = ['label_', 'return_']
        feature_cols = [col for col in df.columns if not any(p in col for p in exclude_patterns)]
    
    # Drop rows with NaN in label
    df_clean = df.dropna(subset=[label_col])
    
    X = df_clean[feature_cols]
    y = df_clean[label_col]
    
    logger.info(f"Extracted X: {X.shape}, y: {y.shape}")
    logger.info(f"Class balance: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y



