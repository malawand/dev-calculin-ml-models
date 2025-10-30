"""
Ultra-advanced features for maximum directional prediction accuracy.
Goes beyond basic derivatives to capture complex market dynamics.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from itertools import combinations

logger = logging.getLogger(__name__)


class UltraAdvancedFeatureEngine:
    """
    Extremely sophisticated feature engineering for directional prediction.
    Focuses on feature interactions, market microstructure, and regime dependencies.
    """
    
    def __init__(self):
        pass
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ultra-advanced features.
        
        Args:
            df: DataFrame with existing features
            
        Returns:
            DataFrame with ultra-advanced features added
        """
        logger.info("=" * 80)
        logger.info("ULTRA-ADVANCED FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        df = df.copy()
        initial_cols = len(df.columns)
        
        # 1. Feature Interactions (most powerful signals come from combinations)
        df = self._add_feature_interactions(df)
        
        # 2. Market Microstructure Features
        df = self._add_microstructure_features(df)
        
        # 3. Statistical Arbitrage Signals
        df = self._add_stat_arb_signals(df)
        
        # 4. Regime-Conditional Features
        df = self._add_regime_conditional_features(df)
        
        # 5. Sequential Pattern Features
        df = self._add_sequential_patterns(df)
        
        # 6. Volatility-Adjusted Signals
        df = self._add_volatility_adjusted_signals(df)
        
        # 7. Cross-Asset Relative Strength (if we had other assets, but we can use cross-timeframe)
        df = self._add_relative_strength_features(df)
        
        # 8. Momentum Quality Indicators
        df = self._add_momentum_quality(df)
        
        final_cols = len(df.columns)
        logger.info(f"Added {final_cols - initial_cols} ultra-advanced features")
        
        return df
    
    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create powerful feature interactions.
        Many trading signals only work when multiple conditions are met.
        """
        logger.info("Adding feature interactions...")
        
        # Key features to interact
        if 'volatility_24' in df.columns and 'momentum_consensus' in df.columns:
            # High volatility + directional consensus = strong signal
            df['vol_momentum_signal'] = df['volatility_24'] * df['momentum_consensus']
        
        if 'short_term_momentum' in df.columns and 'long_term_momentum' in df.columns:
            # Trend alignment
            df['trend_alignment_strength'] = df['short_term_momentum'] * df['long_term_momentum']
            df['trend_alignment_sign'] = np.sign(df['trend_alignment_strength'])
        
        # Volatility × Persistence
        if 'volatility_24' in df.columns:
            persist_cols = [col for col in df.columns if 'persistence' in col]
            for persist_col in persist_cols[:3]:  # Top 3 persistence features
                df[f'vol_×_{persist_col}'] = df['volatility_24'] * df[persist_col]
        
        # Acceleration × Coherence
        if 'momentum_accel_24h' in df.columns:
            coherence_cols = [col for col in df.columns if 'coherence' in col]
            for coh_col in coherence_cols:
                df[f'accel_×_{coh_col}'] = df['momentum_accel_24h'] * df[coh_col]
        
        # Divergence × Volatility (divergences matter more in volatile markets)
        divergence_cols = [col for col in df.columns if 'divergence' in col]
        if 'volatility_24' in df.columns and divergence_cols:
            for div_col in divergence_cols:
                df[f'vol_weighted_{div_col}'] = df['volatility_24'] * df[div_col]
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Market microstructure features - price action patterns.
        """
        logger.info("Adding market microstructure features...")
        
        if 'price' not in df.columns:
            return df
        
        price = df['price']
        
        # Price momentum at different scales
        for window in [3, 5, 10, 20]:
            df[f'price_momentum_{window}'] = price.diff(window)
            
            # Momentum consistency (what % of bars moved in same direction?)
            price_changes = price.diff().rolling(window).apply(lambda x: (np.sign(x) == np.sign(x[-1])).sum() / len(x))
            df[f'momentum_consistency_{window}'] = price_changes
        
        # Higher high, lower low patterns
        df['hh_count_10'] = price.rolling(10).apply(lambda x: (x[-1] > x[:-1]).sum())
        df['ll_count_10'] = price.rolling(10).apply(lambda x: (x[-1] < x[:-1]).sum())
        df['hh_ll_ratio'] = df['hh_count_10'] / (df['ll_count_10'] + 1)
        
        # Price distance from recent extremes
        df['dist_from_high_20'] = (price.rolling(20).max() - price) / price
        df['dist_from_low_20'] = (price - price.rolling(20).min()) / price
        df['extreme_position'] = df['dist_from_low_20'] / (df['dist_from_high_20'] + df['dist_from_low_20'] + 1e-10)
        
        # Candlestick-like patterns (even though we don't have OHLC, we can approximate)
        # Using rolling windows to get "body" and "range"
        for window in [3, 5]:
            body = price.diff(window).abs()  # Approximate body
            range_ = price.rolling(window).max() - price.rolling(window).min()  # Approximate range
            df[f'body_range_ratio_{window}'] = body / (range_ + 1e-10)
        
        return df
    
    def _add_stat_arb_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Statistical arbitrage signals - mean reversion vs momentum.
        """
        logger.info("Adding statistical arbitrage signals...")
        
        if 'price' not in df.columns:
            return df
        
        price = df['price']
        
        # Z-scores at multiple timeframes (already have some, add more)
        for window in [5, 10, 20, 50, 100]:
            rolling_mean = price.rolling(window).mean()
            rolling_std = price.rolling(window).std()
            df[f'zscore_{window}'] = (price - rolling_mean) / (rolling_std + 1e-10)
            
            # Mean reversion signal (high z-score = expect reversion down)
            df[f'mean_reversion_signal_{window}'] = -df[f'zscore_{window}']
        
        # Bollinger Band position
        for window in [20, 50]:
            sma = price.rolling(window).mean()
            std = price.rolling(window).std()
            upper_band = sma + 2 * std
            lower_band = sma - 2 * std
            df[f'bb_position_{window}'] = (price - lower_band) / (upper_band - lower_band + 1e-10)
            
            # Bollinger squeeze (low volatility = big move coming)
            df[f'bb_squeeze_{window}'] = (upper_band - lower_band) / sma
        
        # RSI-like momentum oscillator
        for window in [7, 14, 21]:
            delta = price.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # RSI divergence from price
            df[f'rsi_price_divergence_{window}'] = df[f'rsi_{window}'].diff() - price.pct_change()
        
        # Momentum vs Mean Reversion Score
        # Combine multiple signals
        momentum_features = [col for col in df.columns if 'momentum' in col.lower() and 'consistency' not in col][:5]
        mr_features = [col for col in df.columns if 'reversion' in col or 'zscore' in col][:5]
        
        if momentum_features and mr_features:
            df['momentum_score'] = df[momentum_features].mean(axis=1)
            df['mean_reversion_score'] = df[mr_features].mean(axis=1)
            df['momentum_vs_mr'] = df['momentum_score'] - df['mean_reversion_score']
        
        return df
    
    def _add_regime_conditional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features that only activate in specific market regimes.
        Different signals work in trending vs ranging markets.
        """
        logger.info("Adding regime-conditional features...")
        
        # Identify trending vs ranging regime
        if 'volatility_24' in df.columns and 'price' in df.columns:
            # Trending: high directional movement, low chop
            price_trend = df['price'].diff(24).abs()  # 24-period trend strength
            df['regime_trending'] = (price_trend > df['price'].rolling(100).mean()) & (df['volatility_24'] > df['volatility_24'].rolling(50).mean())
            df['regime_ranging'] = ~df['regime_trending']
            
            # Conditional features: only active in specific regimes
            # In trending markets, momentum works better
            if 'momentum_consensus' in df.columns:
                df['momentum_if_trending'] = df['momentum_consensus'] * df['regime_trending'].astype(int)
            
            # In ranging markets, mean reversion works better
            if 'mean_reversion_score' in df.columns:
                df['mr_if_ranging'] = df['mean_reversion_score'] * df['regime_ranging'].astype(int)
        
        # Volatility regime
        if 'volatility_24' in df.columns:
            vol_ma = df['volatility_24'].rolling(50).mean()
            df['high_vol_regime'] = (df['volatility_24'] > vol_ma * 1.5).astype(int)
            df['low_vol_regime'] = (df['volatility_24'] < vol_ma * 0.5).astype(int)
            
            # Different features for different vol regimes
            # High vol: follow momentum
            if 'short_term_momentum' in df.columns:
                df['momentum_if_high_vol'] = df['short_term_momentum'] * df['high_vol_regime']
            
            # Low vol: expect breakout
            if 'bb_squeeze_20' in df.columns:
                df['breakout_if_low_vol'] = df['bb_squeeze_20'] * df['low_vol_regime']
        
        return df
    
    def _add_sequential_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-bar sequential patterns.
        E.g., "momentum has been positive for 3 bars and accelerating"
        """
        logger.info("Adding sequential pattern features...")
        
        # Momentum state sequences
        if 'momentum_state_24h_24h' in df.columns:
            momentum_state = df['momentum_state_24h_24h']
            
            # Count consecutive states
            for state in [1, 2, 3, 4]:  # Strong up, weakening up, potential bottom, strong down
                is_state = (momentum_state == state).astype(int)
                consecutive = is_state.groupby((is_state != is_state.shift()).cumsum()).cumcount() + 1
                df[f'consecutive_state_{state}'] = consecutive * is_state
        
        # Price pattern sequences
        if 'price' in df.columns:
            price_dir = np.sign(df['price'].diff())
            
            # Consecutive up/down bars
            df['consecutive_up'] = (price_dir == 1).astype(int).groupby((price_dir != 1).cumsum()).cumcount()
            df['consecutive_down'] = (price_dir == -1).astype(int).groupby((price_dir != -1).cumsum()).cumcount()
            
            # Alternating patterns (up-down-up-down = chop)
            df['alternating_pattern'] = (price_dir * price_dir.shift(1) < 0).astype(int).rolling(5).sum()
        
        # Derivative state sequences
        deriv_cols = [col for col in df.columns if col.startswith('deriv') and 'roc' not in col and 'lag' not in col and 'prime' not in col][:5]
        for deriv_col in deriv_cols:
            deriv_sign = np.sign(df[deriv_col])
            df[f'{deriv_col}_sign_consistency_5'] = (deriv_sign.rolling(5).sum().abs() / 5)  # 1 = all same direction
        
        return df
    
    def _add_volatility_adjusted_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust all signals by current volatility.
        In high vol, need bigger moves to be significant.
        """
        logger.info("Adding volatility-adjusted signals...")
        
        if 'volatility_24' not in df.columns:
            return df
        
        vol = df['volatility_24']
        vol_ma = vol.rolling(50).mean()
        vol_adjustment = vol / (vol_ma + 1e-10)
        
        # Volatility-adjusted returns
        if 'price' in df.columns:
            for period in [1, 3, 6, 12]:
                returns = df['price'].pct_change(period)
                df[f'vol_adjusted_return_{period}'] = returns / (vol_adjustment + 1e-10)
        
        # Volatility-adjusted momentum
        momentum_cols = [col for col in df.columns if 'momentum' in col and 'adjusted' not in col][:10]
        for mom_col in momentum_cols:
            df[f'{mom_col}_vol_adj'] = df[mom_col] / (vol_adjustment + 1e-10)
        
        return df
    
    def _add_relative_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-timeframe relative strength.
        Which timeframes are outperforming?
        """
        logger.info("Adding relative strength features...")
        
        # Get all derivatives
        deriv_cols = [col for col in df.columns if col.startswith('deriv') and 
                     'roc' not in col and 'lag' not in col and 'prime' not in col and 
                     'sign' not in col and '_' not in col[6:]]
        
        if len(deriv_cols) < 2:
            return df
        
        # Short-term vs long-term strength
        short_term = ['deriv15m', 'deriv30m', 'deriv1h', 'deriv2h']
        long_term = ['deriv24h', 'deriv3d', 'deriv7d']
        
        short_available = [col for col in short_term if col in df.columns]
        long_available = [col for col in long_term if col in df.columns]
        
        if short_available and long_available:
            short_avg = df[short_available].mean(axis=1)
            long_avg = df[long_available].mean(axis=1)
            
            df['short_vs_long_strength'] = short_avg - long_avg
            df['short_to_long_ratio'] = short_avg / (long_avg.abs() + 1e-10)
        
        # Which timeframe has strongest momentum?
        if deriv_cols:
            deriv_df = df[deriv_cols].abs()
            df['strongest_timeframe_idx'] = deriv_df.idxmax(axis=1).apply(lambda x: deriv_cols.index(x) if x in deriv_cols else 0)
            df['strongest_momentum_value'] = deriv_df.max(axis=1)
            df['momentum_concentration'] = df['strongest_momentum_value'] / (deriv_df.mean(axis=1) + 1e-10)
        
        return df
    
    def _add_momentum_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assess the QUALITY of momentum, not just direction.
        High-quality momentum is more likely to continue.
        """
        logger.info("Adding momentum quality indicators...")
        
        # Quality = consistency + magnitude + persistence + confirmation
        
        # 1. Consistency across timeframes
        if 'momentum_consensus' in df.columns and 'momentum_agreement' in df.columns:
            # High consensus, low disagreement = high quality
            df['momentum_quality_consistency'] = df['momentum_consensus'] * (1 / (df['momentum_agreement'] + 1))
        
        # 2. Magnitude of movement
        if 'price' in df.columns:
            price_change_10 = df['price'].pct_change(10).abs()
            price_change_ma = price_change_10.rolling(50).mean()
            df['momentum_quality_magnitude'] = price_change_10 / (price_change_ma + 1e-10)
        
        # 3. Persistence (already have this, but combine with others)
        persist_cols = [col for col in df.columns if 'persistence' in col]
        if persist_cols:
            df['momentum_quality_persistence'] = df[persist_cols].mean(axis=1)
        
        # 4. Volume proxy (use volatility as proxy since we don't have volume)
        if 'volatility_24' in df.columns:
            vol_increasing = df['volatility_24'] > df['volatility_24'].shift(3)
            df['momentum_quality_volume_proxy'] = vol_increasing.astype(int)
        
        # Combined quality score
        quality_components = [col for col in df.columns if 'momentum_quality' in col]
        if len(quality_components) > 1:
            # Normalize and combine
            for col in quality_components:
                df[f'{col}_normalized'] = (df[col] - df[col].rolling(100).mean()) / (df[col].rolling(100).std() + 1e-10)
            
            normalized_cols = [f'{col}_normalized' for col in quality_components]
            df['momentum_quality_score'] = df[normalized_cols].mean(axis=1)
        
        return df


def add_ultra_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all ultra-advanced features.
    
    Args:
        df: DataFrame with existing features
        
    Returns:
        DataFrame with ultra-advanced features added
    """
    engine = UltraAdvancedFeatureEngine()
    return engine.create_features(df)



