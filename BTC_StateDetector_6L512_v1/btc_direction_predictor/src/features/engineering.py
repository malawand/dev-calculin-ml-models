"""
Feature engineering with strict anti-leakage guarantees
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from .advanced_derivatives import add_advanced_derivative_features
from .ultra_advanced_features import add_ultra_advanced_features

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features from raw metrics without data leakage"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        self.price_lags = config['features']['price_lags']
        self.deriv_lags = config['features']['deriv_lags']
        self.rolling_windows = config['features']['rolling_windows']
        
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features.
        
        Args:
            df: Raw metrics DataFrame with timestamp index
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        df = df.copy()
        initial_shape = df.shape
        
        # Identify price column
        price_col = self._get_price_column(df)
        logger.info(f"Using price column: {price_col}")
        
        # 1. Price-based features
        df = self._add_price_features(df, price_col)
        
        # 2. Derivative-based features
        df = self._add_derivative_features(df)
        
        # 3. Average-based features
        df = self._add_average_features(df)
        
        # 4. Rolling statistics
        df = self._add_rolling_features(df, price_col)
        
        # 5. Derivative prime features (if available)
        df = self._add_derivative_prime_features(df)
        
        # 6. Advanced derivative analysis (momentum regimes, divergences, etc.)
        df = add_advanced_derivative_features(df)
        
        # 7. Ultra-advanced features (interactions, microstructure, stat arb, etc.)
        df = add_ultra_advanced_features(df)
        
        # Drop rows with NaN (from rolling/lag operations)
        df.dropna(inplace=True)
        
        final_shape = df.shape
        logger.info(f"Initial shape: {initial_shape}")
        logger.info(f"Final shape: {final_shape}")
        logger.info(f"Features created: {final_shape[1] - initial_shape[1]}")
        logger.info(f"Rows dropped (NaN): {initial_shape[0] - final_shape[0]}")
        
        return df
    
    def _get_price_column(self, df: pd.DataFrame) -> str:
        """Identify the spot price column"""
        candidates = ['price', 'crypto_last_price']
        for col in candidates:
            if col in df.columns:
                return col
        raise ValueError(f"No price column found. Available: {df.columns.tolist()}")
    
    def _add_price_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add price-based features"""
        logger.info("Adding price features...")
        
        # Returns at different lags
        for lag in self.price_lags:
            df[f'return_{lag}'] = df[price_col].pct_change(lag)
        
        # Log returns
        df['log_return_1'] = np.log(df[price_col] / df[price_col].shift(1))
        
        return df
    
    def _add_derivative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features from derivative columns"""
        logger.info("Adding derivative features...")
        
        deriv_cols = [col for col in df.columns if 'deriv' in col and 'prime' not in col]
        logger.info(f"Found {len(deriv_cols)} derivative columns")
        
        for col in deriv_cols:
            # Lagged values
            for lag in self.deriv_lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            
            # Rate of change
            df[f'{col}_roc'] = df[col].pct_change()
        
        return df
    
    def _add_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features from average columns"""
        logger.info("Adding average features...")
        
        avg_cols = [col for col in df.columns if 'avg' in col]
        logger.info(f"Found {len(avg_cols)} average columns")
        
        price_col = self._get_price_column(df)
        
        for col in avg_cols:
            # Lagged values
            for lag in [1, 3]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            
            # Spread from price
            df[f'{col}_spread'] = (df[price_col] - df[col]) / df[col]
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add rolling window statistics"""
        logger.info("Adding rolling features...")
        
        for window in self.rolling_windows:
            # Rolling mean and std
            rolling_mean = df[price_col].rolling(window).mean()
            rolling_std = df[price_col].rolling(window).std()
            
            # Z-score
            df[f'zscore_{window}'] = (df[price_col] - rolling_mean) / rolling_std
            
            # Volatility
            df[f'volatility_{window}'] = df[f'return_1'].rolling(window).std()
        
        return df
    
    def _add_derivative_prime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features from second derivatives (acceleration)"""
        logger.info("Adding derivative prime features...")
        
        prime_cols = [col for col in df.columns if 'prime' in col]
        logger.info(f"Found {len(prime_cols)} derivative prime columns")
        
        for col in prime_cols:
            # Lagged values
            df[f'{col}_lag1'] = df[col].shift(1)
            
            # Sign changes (momentum reversals)
            df[f'{col}_sign'] = np.sign(df[col])
            df[f'{col}_sign_change'] = df[f'{col}_sign'].diff().abs()
        
        # Clean up: Replace inf/-inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop rows with NaN
        initial_rows = len(df)
        df.dropna(inplace=True)
        logger.info(f"Rows dropped (NaN after cleanup): {initial_rows - len(df)}")
        
        return df


def engineer_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Convenience function for feature engineering"""
    engineer = FeatureEngineer(config)
    return engineer.engineer(df)

