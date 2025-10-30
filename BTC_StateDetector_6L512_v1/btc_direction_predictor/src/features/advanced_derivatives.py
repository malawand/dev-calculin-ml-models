"""
Advanced derivative analysis for momentum and trend detection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AdvancedDerivativeAnalyzer:
    """
    Sophisticated analysis of derivatives, derivative primes, and their relationships
    to price action for improved trend prediction.
    """
    
    def __init__(self):
        # Define derivative timeframe groups
        self.short_term_derivs = ['deriv5m', 'deriv10m', 'deriv15m', 'deriv30m', 'deriv45m', 'deriv1h']
        self.medium_term_derivs = ['deriv2h', 'deriv4h', 'deriv8h', 'deriv12h', 'deriv16h', 'deriv24h']
        self.long_term_derivs = ['deriv48h', 'deriv3d', 'deriv4d', 'deriv5d', 'deriv6d', 'deriv7d', 'deriv14d', 'deriv30d']
        
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced derivative features.
        
        Args:
            df: DataFrame with raw derivatives and derivative primes
            
        Returns:
            DataFrame with advanced features added
        """
        logger.info("=" * 80)
        logger.info("ADVANCED DERIVATIVE ANALYSIS")
        logger.info("=" * 80)
        
        df = df.copy()
        initial_cols = len(df.columns)
        
        # 1. Velocity-Acceleration Alignment
        df = self._add_velocity_acceleration_alignment(df)
        
        # 2. Momentum Regime Detection
        df = self._add_momentum_regimes(df)
        
        # 3. Cross-Timeframe Divergences
        df = self._add_cross_timeframe_divergences(df)
        
        # 4. Momentum Persistence & Consistency
        df = self._add_momentum_persistence(df)
        
        # 5. Derivative Curvature (Jerk)
        df = self._add_derivative_curvature(df)
        
        # 6. Momentum Strength Indicators
        df = self._add_momentum_strength(df)
        
        # 7. Phase Cycle Analysis
        df = self._add_phase_cycle_features(df)
        
        # 8. Multi-Scale Momentum Coherence
        df = self._add_momentum_coherence(df)
        
        final_cols = len(df.columns)
        logger.info(f"Added {final_cols - initial_cols} advanced derivative features")
        
        return df
    
    def _add_velocity_acceleration_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze how velocity (deriv) and acceleration (deriv_prime) align.
        
        Key insights:
        - Both positive = strong uptrend (accelerating upward)
        - Velocity positive, acceleration negative = uptrend weakening
        - Velocity negative, acceleration positive = downtrend weakening (potential reversal)
        - Both negative = strong downtrend (accelerating downward)
        """
        logger.info("Adding velocity-acceleration alignment features...")
        
        # Find all derivative-prime pairs
        deriv_cols = [col for col in df.columns if 'deriv' in col and 'prime' not in col and 'roc' not in col and 'lag' not in col]
        
        for deriv_col in deriv_cols:
            if deriv_col not in df.columns:
                continue
                
            # Find corresponding prime derivatives
            base_name = deriv_col.replace('deriv', '')  # e.g., '24h' from 'deriv24h'
            
            # Look for prime derivatives with this base
            prime_cols = [col for col in df.columns if f'deriv{base_name}_prime' in col and 'sign' not in col and 'lag' not in col]
            
            for prime_col in prime_cols:
                # Extract the prime window (e.g., '24h' from 'deriv24h_prime24h')
                suffix = prime_col.replace(f'deriv{base_name}_prime', '')
                
                # Alignment score: +1 when both same sign, -1 when opposite, 0 when either is zero
                vel_sign = np.sign(df[deriv_col])
                accel_sign = np.sign(df[prime_col])
                alignment = vel_sign * accel_sign
                
                df[f'align_{base_name}_{suffix}'] = alignment
                
                # Momentum state classification
                # 1 = Strong Uptrend (both positive)
                # 2 = Weakening Uptrend (vel+, accel-)
                # 3 = Potential Bottom (vel-, accel+)
                # 4 = Strong Downtrend (both negative)
                # 0 = Neutral
                momentum_state = np.zeros(len(df))
                momentum_state[(vel_sign > 0) & (accel_sign > 0)] = 1  # Strong up
                momentum_state[(vel_sign > 0) & (accel_sign < 0)] = 2  # Weakening up
                momentum_state[(vel_sign < 0) & (accel_sign > 0)] = 3  # Potential reversal
                momentum_state[(vel_sign < 0) & (accel_sign < 0)] = 4  # Strong down
                
                df[f'momentum_state_{base_name}_{suffix}'] = momentum_state
                
                # Alignment strength (magnitude-weighted)
                df[f'align_strength_{base_name}_{suffix}'] = df[deriv_col] * df[prime_col]
        
        return df
    
    def _add_momentum_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify current momentum regime across multiple timeframes.
        """
        logger.info("Adding momentum regime features...")
        
        # For key timeframes, determine if momentum is building, peaking, fading, or reversing
        key_derivs = ['deriv1h', 'deriv4h', 'deriv24h', 'deriv3d', 'deriv7d']
        
        for deriv in key_derivs:
            if deriv not in df.columns:
                continue
            
            base = deriv.replace('deriv', '')
            
            # Rate of change of derivative (is momentum accelerating?)
            df[f'momentum_accel_{base}'] = df[deriv].diff()
            
            # Momentum trend (smoothed direction)
            df[f'momentum_trend_{base}'] = df[deriv].rolling(window=3, min_periods=1).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0,
                raw=False
            )
            
            # Momentum volatility (consistency)
            df[f'momentum_vol_{base}'] = df[deriv].rolling(window=12, min_periods=1).std()
        
        # Multi-timeframe momentum consensus
        consensus_cols = [f'momentum_trend_{d.replace("deriv", "")}' for d in key_derivs if d in df.columns]
        if consensus_cols:
            df['momentum_consensus'] = df[consensus_cols].mean(axis=1)
            df['momentum_agreement'] = df[consensus_cols].std(axis=1)  # Low = high agreement
        
        return df
    
    def _add_cross_timeframe_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect divergences between short, medium, and long-term derivatives.
        
        Divergences often signal trend changes.
        """
        logger.info("Adding cross-timeframe divergence features...")
        
        # Get available derivatives in each category
        short_available = [d for d in self.short_term_derivs if d in df.columns]
        medium_available = [d for d in self.medium_term_derivs if d in df.columns]
        long_available = [d for d in self.long_term_derivs if d in df.columns]
        
        # Calculate average momentum for each timeframe group
        if short_available:
            df['short_term_momentum'] = df[short_available].mean(axis=1)
            df['short_term_momentum_sign'] = np.sign(df['short_term_momentum'])
        
        if medium_available:
            df['medium_term_momentum'] = df[medium_available].mean(axis=1)
            df['medium_term_momentum_sign'] = np.sign(df['medium_term_momentum'])
        
        if long_available:
            df['long_term_momentum'] = df[long_available].mean(axis=1)
            df['long_term_momentum_sign'] = np.sign(df['long_term_momentum'])
        
        # Divergence indicators
        if 'short_term_momentum_sign' in df.columns and 'long_term_momentum_sign' in df.columns:
            # Short vs Long divergence (-1 = opposite, 0 = one neutral, 1 = aligned)
            df['short_long_divergence'] = df['short_term_momentum_sign'] - df['long_term_momentum_sign']
            
            # Magnitude of divergence
            df['short_long_div_magnitude'] = abs(df['short_term_momentum'] - df['long_term_momentum'])
        
        if 'short_term_momentum_sign' in df.columns and 'medium_term_momentum_sign' in df.columns:
            df['short_medium_divergence'] = df['short_term_momentum_sign'] - df['medium_term_momentum_sign']
            df['short_medium_div_magnitude'] = abs(df['short_term_momentum'] - df['medium_term_momentum'])
        
        # Trend convergence/divergence score
        if all(col in df.columns for col in ['short_term_momentum', 'medium_term_momentum', 'long_term_momentum']):
            # Positive when all moving same direction, negative when diverging
            all_momentum = df[['short_term_momentum_sign', 'medium_term_momentum_sign', 'long_term_momentum_sign']]
            df['momentum_convergence'] = all_momentum.std(axis=1) * -1  # Negative std = more convergence
        
        return df
    
    def _add_momentum_persistence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Measure how long momentum has persisted in current direction.
        Longer persistence may indicate stronger trends or exhaustion.
        """
        logger.info("Adding momentum persistence features...")
        
        key_derivs = ['deriv1h', 'deriv4h', 'deriv24h', 'deriv3d', 'deriv7d']
        
        for deriv in key_derivs:
            if deriv not in df.columns:
                continue
            
            base = deriv.replace('deriv', '')
            sign = np.sign(df[deriv])
            
            # Count consecutive periods in same direction
            persistence = (sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1)
            df[f'persistence_{base}'] = persistence
            
            # Weighted persistence (magnitude * duration)
            df[f'weighted_persistence_{base}'] = persistence * abs(df[deriv])
        
        return df
    
    def _add_derivative_curvature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Third derivative (jerk) - rate of change of acceleration.
        Helps detect inflection points.
        """
        logger.info("Adding derivative curvature (jerk) features...")
        
        # For key derivative primes, calculate their rate of change
        prime_cols = [col for col in df.columns if 'prime' in col and 'sign' not in col and 'lag' not in col]
        
        for prime_col in prime_cols[:10]:  # Limit to top 10 to avoid feature explosion
            # Jerk = change in acceleration
            df[f'{prime_col}_jerk'] = df[prime_col].diff()
            
            # Sign of jerk (is acceleration increasing or decreasing?)
            df[f'{prime_col}_jerk_sign'] = np.sign(df[f'{prime_col}_jerk'])
        
        return df
    
    def _add_momentum_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Magnitude-weighted momentum indicators.
        Not just direction, but HOW STRONG the momentum is.
        """
        logger.info("Adding momentum strength indicators...")
        
        # Get available derivatives
        deriv_cols = [col for col in df.columns if col.startswith('deriv') and 
                     'prime' not in col and 'roc' not in col and 'lag' not in col and 
                     '_' not in col[6:]]  # Exclude derived features
        
        if deriv_cols:
            # Overall momentum magnitude
            df['total_momentum_magnitude'] = df[deriv_cols].abs().sum(axis=1)
            
            # Directional momentum (positive - negative)
            positive_momentum = df[deriv_cols].apply(lambda x: x.clip(lower=0)).sum(axis=1)
            negative_momentum = df[deriv_cols].apply(lambda x: x.clip(upper=0)).abs().sum(axis=1)
            
            df['net_momentum'] = positive_momentum - negative_momentum
            df['momentum_ratio'] = positive_momentum / (negative_momentum + 1e-10)  # Avoid div by zero
        
        # Momentum concentration (is momentum focused in one direction or scattered?)
        if len(deriv_cols) > 1:
            df['momentum_concentration'] = df[deriv_cols].std(axis=1) / (df[deriv_cols].abs().mean(axis=1) + 1e-10)
        
        return df
    
    def _add_phase_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify where we are in the momentum cycle.
        """
        logger.info("Adding phase cycle features...")
        
        key_derivs = ['deriv4h', 'deriv24h', 'deriv3d']
        
        for deriv in key_derivs:
            if deriv not in df.columns:
                continue
            
            base = deriv.replace('deriv', '')
            
            # Normalized position in cycle (using z-score on rolling window)
            rolling_mean = df[deriv].rolling(window=48, min_periods=1).mean()
            rolling_std = df[deriv].rolling(window=48, min_periods=1).std()
            df[f'cycle_position_{base}'] = (df[deriv] - rolling_mean) / (rolling_std + 1e-10)
            
            # Is momentum at extreme? (potential reversal zones)
            df[f'at_extreme_{base}'] = (abs(df[f'cycle_position_{base}']) > 2).astype(int)
        
        return df
    
    def _add_momentum_coherence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Measure how coherent momentum is across multiple scales.
        High coherence = strong directional conviction.
        Low coherence = market indecision or chop.
        """
        logger.info("Adding multi-scale momentum coherence...")
        
        # Get derivatives at different scales
        scales = {
            'micro': ['deriv15m', 'deriv30m', 'deriv1h'],
            'short': ['deriv2h', 'deriv4h', 'deriv8h'],
            'medium': ['deriv12h', 'deriv24h', 'deriv48h'],
            'long': ['deriv3d', 'deriv7d', 'deriv14d']
        }
        
        for scale_name, scale_derivs in scales.items():
            available = [d for d in scale_derivs if d in df.columns]
            if len(available) < 2:
                continue
            
            # Direction coherence (how many agree on direction?)
            signs = df[available].apply(np.sign)
            df[f'{scale_name}_coherence'] = signs.mean(axis=1).abs()  # 1 = all agree, 0 = split
            
            # Magnitude coherence (similar strengths vs one dominant?)
            magnitudes = df[available].abs()
            df[f'{scale_name}_magnitude_std'] = magnitudes.std(axis=1)
        
        return df


def add_advanced_derivative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all advanced derivative features.
    
    Args:
        df: DataFrame with basic derivatives and derivative primes
        
    Returns:
        DataFrame with advanced features added
    """
    analyzer = AdvancedDerivativeAnalyzer()
    return analyzer.analyze(df)



