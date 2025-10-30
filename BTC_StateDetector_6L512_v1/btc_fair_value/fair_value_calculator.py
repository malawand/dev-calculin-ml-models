"""
Bitcoin Fair Value Calculator

Calculates empirical fair value, max, and min for Bitcoin based on:
1. Derivatives (velocity and acceleration of price)
2. Historical statistical bounds
3. Mean reversion targets
4. Volume-weighted averages

Goal: Determine if current price is overvalued, undervalued, or fair
"""
import numpy as np
import pandas as pd

class FairValueCalculator:
    """
    Calculates fair value of Bitcoin in the moment
    
    Uses multiple methods:
    - VWAP (Volume Weighted Average Price)
    - Bollinger Bands (statistical bounds)
    - Derivative-based fair value (velocity extrapolation)
    - Z-score normalization
    """
    
    def __init__(self):
        pass
    
    def calculate_vwap(self, df, period=240):
        """
        Volume Weighted Average Price
        
        Fair value = weighted average by volume
        """
        if len(df) < period or 'volume' not in df.columns:
            return None
        
        prices = df['price'].iloc[-period:].values
        volumes = df['volume'].iloc[-period:].values
        
        if volumes.sum() == 0:
            return np.mean(prices)
        
        vwap = np.sum(prices * volumes) / np.sum(volumes)
        return vwap
    
    def calculate_statistical_bounds(self, df, period=240, std_dev=2.0):
        """
        Bollinger Bands - statistical max/min
        
        Fair value = mean
        Max = mean + (std_dev * std)
        Min = mean - (std_dev * std)
        """
        if len(df) < period:
            return None
        
        prices = df['price'].iloc[-period:].values
        
        mean = np.mean(prices)
        std = np.std(prices)
        
        return {
            'fair': mean,
            'max': mean + (std_dev * std),
            'min': mean - (std_dev * std),
            'std': std
        }
    
    def calculate_derivative_fair_value(self, df):
        """
        Fair value based on velocity and acceleration
        
        Uses derivatives to calculate where price "should" be
        if current momentum continues
        """
        if len(df) < 60:
            return None
        
        # Get derivatives if available
        deriv_cols = [col for col in df.columns if 'deriv' in col.lower() and 'prime' not in col.lower()]
        
        if len(deriv_cols) == 0:
            # Calculate simple derivative
            prices = df['price'].values
            velocity = (prices[-1] - prices[-30]) / 30  # Per minute
            
            # Simple extrapolation
            fair_value = prices[-1] + (velocity * 15)  # 15 min ahead
            
            return {
                'fair': fair_value,
                'velocity': velocity,
                'method': 'simple'
            }
        
        # Use existing derivatives
        current_price = df['price'].iloc[-1]
        
        # Average velocity across timeframes
        velocities = []
        for col in deriv_cols[:5]:  # Use first 5
            if not df[col].isna().iloc[-1]:
                velocities.append(df[col].iloc[-1])
        
        if len(velocities) == 0:
            return None
        
        avg_velocity = np.mean(velocities)
        
        # Fair value = current price + (velocity * time horizon)
        time_horizon = 15  # 15 minutes
        fair_value = current_price + (avg_velocity * time_horizon)
        
        return {
            'fair': fair_value,
            'velocity': avg_velocity,
            'method': 'derivatives'
        }
    
    def calculate_zscore_range(self, df, period=240):
        """
        Z-score based fair value
        
        Shows how many standard deviations away from mean
        """
        if len(df) < period:
            return None
        
        prices = df['price'].iloc[-period:].values
        current = prices[-1]
        
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std == 0:
            return None
        
        zscore = (current - mean) / std
        
        return {
            'zscore': zscore,
            'fair': mean,
            'current': current,
            'interpretation': self._interpret_zscore(zscore)
        }
    
    def _interpret_zscore(self, zscore):
        """Interpret z-score"""
        if zscore > 2:
            return 'EXTREMELY_OVERVALUED'
        elif zscore > 1.5:
            return 'OVERVALUED'
        elif zscore > 0.5:
            return 'SLIGHTLY_OVERVALUED'
        elif zscore < -2:
            return 'EXTREMELY_UNDERVALUED'
        elif zscore < -1.5:
            return 'UNDERVALUED'
        elif zscore < -0.5:
            return 'SLIGHTLY_UNDERVALUED'
        else:
            return 'FAIR'
    
    def calculate_comprehensive_fair_value(self, df):
        """
        Comprehensive analysis combining all methods
        
        Returns empirical fair value, max, min, and current assessment
        """
        current_price = df['price'].iloc[-1]
        
        # Method 1: VWAP
        vwap_4h = self.calculate_vwap(df, period=240)
        
        # Method 2: Statistical bounds
        stats = self.calculate_statistical_bounds(df, period=240, std_dev=2.0)
        
        # Method 3: Derivative-based
        deriv_fair = self.calculate_derivative_fair_value(df)
        
        # Method 4: Z-score
        zscore_info = self.calculate_zscore_range(df, period=240)
        
        # Combine into consensus fair value
        fair_values = []
        
        if vwap_4h is not None:
            fair_values.append(vwap_4h)
        
        if stats is not None:
            fair_values.append(stats['fair'])
        
        if deriv_fair is not None:
            fair_values.append(deriv_fair['fair'])
        
        if zscore_info is not None:
            fair_values.append(zscore_info['fair'])
        
        if len(fair_values) == 0:
            return None
        
        # Consensus fair value (median to reduce outlier impact)
        consensus_fair = np.median(fair_values)
        
        # Calculate deviation from fair
        deviation_pct = ((current_price - consensus_fair) / consensus_fair) * 100
        
        # Determine max and min
        if stats is not None:
            empirical_max = stats['max']
            empirical_min = stats['min']
        else:
            # Fallback: ±2% from consensus
            empirical_max = consensus_fair * 1.02
            empirical_min = consensus_fair * 0.98
        
        # Position within range
        if empirical_max > empirical_min:
            position_pct = (current_price - empirical_min) / (empirical_max - empirical_min) * 100
        else:
            position_pct = 50.0
        
        # Assessment
        if current_price > empirical_max:
            assessment = 'OVERVALUED'
            expected_move = 'DOWN'
        elif current_price < empirical_min:
            assessment = 'UNDERVALUED'
            expected_move = 'UP'
        elif deviation_pct > 1.0:
            assessment = 'SLIGHTLY_OVERVALUED'
            expected_move = 'SLIGHT_DOWN'
        elif deviation_pct < -1.0:
            assessment = 'SLIGHTLY_UNDERVALUED'
            expected_move = 'SLIGHT_UP'
        else:
            assessment = 'FAIR'
            expected_move = 'NONE'
        
        return {
            'current_price': current_price,
            'fair_value': consensus_fair,
            'empirical_max': empirical_max,
            'empirical_min': empirical_min,
            'deviation_pct': deviation_pct,
            'position_in_range_pct': position_pct,
            'assessment': assessment,
            'expected_move': expected_move,
            'methods': {
                'vwap_4h': vwap_4h,
                'statistical_mean': stats['fair'] if stats else None,
                'statistical_max': stats['max'] if stats else None,
                'statistical_min': stats['min'] if stats else None,
                'derivative_fair': deriv_fair['fair'] if deriv_fair else None,
                'zscore': zscore_info['zscore'] if zscore_info else None,
                'zscore_interpretation': zscore_info['interpretation'] if zscore_info else None
            }
        }


