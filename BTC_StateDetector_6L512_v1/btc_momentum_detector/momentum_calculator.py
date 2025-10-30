"""
Momentum Calculator: Measures momentum strength, direction, and phase
Enhanced with volume analysis for better confirmation
"""
import numpy as np
import pandas as pd
from volume_analyzer import VolumeAnalyzer

class MomentumCalculator:
    def __init__(self):
        self.volume_analyzer = VolumeAnalyzer()
    
    def calculate_rate_of_change(self, prices, period):
        """Calculate rate of change over period"""
        if len(prices) < period:
            return 0.0
        return (prices[-1] - prices[-period]) / prices[-period]
    
    def calculate_acceleration(self, prices, period=30):
        """Calculate price acceleration (2nd derivative)"""
        if len(prices) < period * 2:
            return 0.0
        
        # Recent momentum
        recent_roc = self.calculate_rate_of_change(prices, period)
        
        # Earlier momentum
        earlier_prices = prices[:-period]
        if len(earlier_prices) >= period:
            earlier_roc = self.calculate_rate_of_change(earlier_prices, period)
        else:
            earlier_roc = 0.0
        
        # Acceleration = change in momentum
        acceleration = recent_roc - earlier_roc
        return acceleration
    
    def calculate_momentum_strength(self, df):
        """
        Calculate overall momentum strength (0-100)
        
        Combines:
        - Rate of change across multiple timeframes (40 points)
        - Acceleration (15 points)
        - Consistency (15 points)
        - Volume analysis (30 points) - ENHANCED!
        """
        if len(df) < 240:
            return 0.0
        
        prices = df['price'].values
        
        # Multi-timeframe rate of change
        roc_5m = abs(self.calculate_rate_of_change(prices, 5))
        roc_15m = abs(self.calculate_rate_of_change(prices, 15))
        roc_30m = abs(self.calculate_rate_of_change(prices, 30))
        roc_1h = abs(self.calculate_rate_of_change(prices, 60))
        roc_4h = abs(self.calculate_rate_of_change(prices, 240))
        
        # Weighted average (more weight to recent)
        avg_roc = (roc_5m * 0.3 + roc_15m * 0.25 + roc_30m * 0.2 + 
                   roc_1h * 0.15 + roc_4h * 0.1)
        
        # Acceleration
        accel = abs(self.calculate_acceleration(prices, 30))
        
        # Consistency: check if moves are in same direction
        returns_15m = np.diff(prices[-60:])  # Last hour in 1-min bars
        positive_moves = (returns_15m > 0).sum()
        total_moves = len(returns_15m)
        consistency = abs(2 * (positive_moves / total_moves) - 1) if total_moves > 0 else 0
        
        # ENHANCED VOLUME ANALYSIS
        volume_analysis = self.volume_analyzer.get_volume_analysis(df)
        volume_strength = volume_analysis['strength']  # 0-100
        
        # Adjust volume score based on alignment
        if volume_analysis['alignment'] == 'DIVERGENT':
            # Price/volume divergence = weak momentum (penalty)
            volume_strength *= 0.5
        elif volume_analysis['alignment'] == 'ALIGNED':
            # Aligned = boost confidence
            volume_strength *= 1.2
        
        # Combine into 0-100 score
        # ROC component (0-40)
        roc_score = min(avg_roc / 0.02, 1.0) * 40  # 2% move = max
        
        # Acceleration component (0-15)
        accel_score = min(accel / 0.01, 1.0) * 15  # 1% acceleration = max
        
        # Consistency component (0-15)
        consistency_score = consistency * 15
        
        # Volume component (0-30) - MUCH MORE WEIGHT
        volume_score = (volume_strength / 100) * 30
        
        total_score = roc_score + accel_score + consistency_score + volume_score
        
        return min(total_score, 100.0)
    
    def calculate_momentum_direction(self, df):
        """
        Calculate momentum direction
        
        Returns:
            'UP': Upward momentum
            'DOWN': Downward momentum
            'NONE': No clear direction
        """
        if len(df) < 240:
            return 'NONE'
        
        prices = df['price'].values
        
        # Multi-timeframe ROC (with direction)
        roc_15m = self.calculate_rate_of_change(prices, 15)
        roc_30m = self.calculate_rate_of_change(prices, 30)
        roc_1h = self.calculate_rate_of_change(prices, 60)
        roc_4h = self.calculate_rate_of_change(prices, 240)
        
        # Weighted vote (recent gets more weight)
        score = (
            (1 if roc_15m > 0 else -1) * 0.4 +
            (1 if roc_30m > 0 else -1) * 0.3 +
            (1 if roc_1h > 0 else -1) * 0.2 +
            (1 if roc_4h > 0 else -1) * 0.1
        )
        
        # Require clear majority
        if score > 0.3:
            return 'UP'
        elif score < -0.3:
            return 'DOWN'
        else:
            return 'NONE'
    
    def calculate_momentum_phase(self, df):
        """
        Determine momentum phase
        
        Returns:
            'BUILDING': Momentum starting to form
            'STRONG': Momentum established and sustained
            'FADING': Momentum losing strength
            'ABSENT': No momentum
        """
        if len(df) < 240:
            return 'ABSENT'
        
        prices = df['price'].values
        
        # Current momentum
        current_strength = self.calculate_momentum_strength(df)
        
        # Historical strength (30 mins ago)
        if len(df) >= 270:
            historical_df = df.iloc[:-30]
            historical_strength = self.calculate_momentum_strength(historical_df)
        else:
            historical_strength = current_strength
        
        # Acceleration
        accel = self.calculate_acceleration(prices, 30)
        
        # Classify phase
        if current_strength < 20:
            return 'ABSENT'
        
        elif current_strength >= 20 and current_strength < 50:
            # Low-medium strength
            if current_strength > historical_strength * 1.2 and accel > 0:
                return 'BUILDING'
            else:
                return 'FADING'
        
        elif current_strength >= 50 and current_strength < 75:
            # Medium-high strength
            if current_strength > historical_strength:
                return 'BUILDING'
            elif current_strength < historical_strength * 0.9:
                return 'FADING'
            else:
                return 'STRONG'
        
        else:  # current_strength >= 75
            # Very high strength
            if current_strength < historical_strength * 0.85:
                return 'FADING'
            else:
                return 'STRONG'
    
    def calculate_momentum_confidence(self, df):
        """
        Calculate confidence in momentum reading (0-1)
        
        Higher confidence when:
        - Multiple timeframes agree
        - Volume confirms
        - Price action is consistent
        - Momentum is clear (not choppy)
        """
        if len(df) < 240:
            return 0.0
        
        prices = df['price'].values
        
        # 1. Timeframe alignment
        roc_15m = self.calculate_rate_of_change(prices, 15)
        roc_30m = self.calculate_rate_of_change(prices, 30)
        roc_1h = self.calculate_rate_of_change(prices, 60)
        roc_4h = self.calculate_rate_of_change(prices, 240)
        
        # Check if all have same sign
        signs = [np.sign(roc_15m), np.sign(roc_30m), np.sign(roc_1h), np.sign(roc_4h)]
        alignment = abs(sum(signs)) / 4.0  # 1.0 = perfect alignment
        
        # 2. Consistency
        returns_1h = np.diff(prices[-60:])
        positive = (returns_1h > 0).sum()
        consistency = abs(2 * (positive / len(returns_1h)) - 1)
        
        # 3. Volume confirmation
        volume_conf = 0.5  # Default if no volume
        if 'volume' in df.columns and df['volume'].sum() > 0:
            recent_vol = df['volume'].iloc[-30:].mean()
            avg_vol = df['volume'].iloc[-240:].mean()
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                volume_conf = min(vol_ratio / 1.5, 1.0)  # 1.5x avg = full confidence
        
        # 4. Momentum clarity (not too weak, not too chaotic)
        strength = self.calculate_momentum_strength(df)
        clarity = min(strength / 60, 1.0) if strength < 60 else max(0, (100 - strength) / 40)
        
        # Combine
        confidence = (alignment * 0.35 + consistency * 0.25 + 
                     volume_conf * 0.25 + clarity * 0.15)
        
        return min(confidence, 1.0)
    
    def get_momentum_report(self, df):
        """
        Get comprehensive momentum analysis
        
        Returns dict with all momentum metrics INCLUDING VOLUME ANALYSIS
        """
        strength = self.calculate_momentum_strength(df)
        direction = self.calculate_momentum_direction(df)
        phase = self.calculate_momentum_phase(df)
        confidence = self.calculate_momentum_confidence(df)
        
        # Additional details
        prices = df['price'].values
        roc_15m = self.calculate_rate_of_change(prices, 15)
        roc_1h = self.calculate_rate_of_change(prices, 60)
        roc_4h = self.calculate_rate_of_change(prices, 240)
        accel = self.calculate_acceleration(prices, 30)
        
        # VOLUME ANALYSIS
        volume_analysis = self.volume_analyzer.get_volume_analysis(df)
        
        return {
            'strength': strength,
            'direction': direction,
            'phase': phase,
            'confidence': confidence,
            'roc_15m': roc_15m,
            'roc_1h': roc_1h,
            'roc_4h': roc_4h,
            'acceleration': accel,
            'current_price': prices[-1] if len(prices) > 0 else None,
            'volume': volume_analysis  # Full volume analysis
        }

