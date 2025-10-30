"""
Volume Analyzer: Advanced volume metrics to confirm momentum
Volume should increase with momentum - if not, it's a weak signal
"""
import numpy as np
import pandas as pd

class VolumeAnalyzer:
    def __init__(self):
        pass
    
    def calculate_volume_trend(self, df, period=60):
        """
        Calculate if volume is increasing or decreasing
        Returns: 'INCREASING', 'DECREASING', 'STABLE'
        """
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return 'UNKNOWN'
        
        if len(df) < period * 2:
            return 'UNKNOWN'
        
        recent_vol = df['volume'].iloc[-period:].mean()
        earlier_vol = df['volume'].iloc[-period*2:-period].mean()
        
        if earlier_vol == 0:
            return 'UNKNOWN'
        
        change = (recent_vol - earlier_vol) / earlier_vol
        
        if change > 0.15:  # 15% increase
            return 'INCREASING'
        elif change < -0.15:  # 15% decrease
            return 'DECREASING'
        else:
            return 'STABLE'
    
    def calculate_volume_spike(self, df, lookback=60, threshold=1.5):
        """
        Detect if there's a volume spike
        Returns: bool and spike_ratio
        """
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return False, 0.0
        
        if len(df) < lookback + 5:
            return False, 0.0
        
        recent_vol = df['volume'].iloc[-5:].mean()  # Last 5 minutes
        avg_vol = df['volume'].iloc[-lookback:].mean()
        
        if avg_vol == 0:
            return False, 0.0
        
        spike_ratio = recent_vol / avg_vol
        
        return spike_ratio > threshold, spike_ratio
    
    def calculate_volume_momentum_alignment(self, df):
        """
        Check if volume trend aligns with price momentum
        Strong signal: Price UP + Volume UP or Price DOWN + Volume DOWN
        Weak signal: Divergence (price up but volume down)
        
        Returns: 'ALIGNED', 'DIVERGENT', 'NEUTRAL'
        """
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return 'NEUTRAL'
        
        if len(df) < 120:
            return 'NEUTRAL'
        
        # Price direction
        recent_price = df['price'].iloc[-30:].mean()
        earlier_price = df['price'].iloc[-90:-60].mean()
        price_direction = 'UP' if recent_price > earlier_price else 'DOWN'
        
        # Volume direction
        volume_trend = self.calculate_volume_trend(df, period=30)
        
        if volume_trend == 'UNKNOWN':
            return 'NEUTRAL'
        
        # Check alignment
        if price_direction == 'UP' and volume_trend == 'INCREASING':
            return 'ALIGNED'  # Strong bullish
        elif price_direction == 'DOWN' and volume_trend == 'INCREASING':
            return 'ALIGNED'  # Strong bearish (panic selling)
        elif price_direction == 'UP' and volume_trend == 'DECREASING':
            return 'DIVERGENT'  # Weak move (fake breakout)
        elif price_direction == 'DOWN' and volume_trend == 'DECREASING':
            return 'DIVERGENT'  # Weak move (capitulation ending)
        else:
            return 'NEUTRAL'
    
    def calculate_volume_strength_score(self, df):
        """
        Calculate volume strength (0-100)
        Higher = more conviction in the move
        """
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return 0.0
        
        if len(df) < 240:
            return 0.0
        
        scores = []
        
        # 1. Volume relative to average (0-30 points)
        recent_vol = df['volume'].iloc[-30:].mean()
        avg_vol = df['volume'].iloc[-240:].mean()
        if avg_vol > 0:
            vol_ratio = recent_vol / avg_vol
            vol_score = min(vol_ratio / 2.0, 1.0) * 30  # 2x volume = full points
            scores.append(vol_score)
        
        # 2. Volume trend (0-25 points)
        trend = self.calculate_volume_trend(df, period=60)
        trend_score = {'INCREASING': 25, 'STABLE': 12.5, 'DECREASING': 0, 'UNKNOWN': 0}
        scores.append(trend_score.get(trend, 0))
        
        # 3. Volume spike (0-20 points)
        has_spike, spike_ratio = self.calculate_volume_spike(df)
        spike_score = min((spike_ratio - 1.0) / 2.0, 1.0) * 20 if has_spike else 0
        scores.append(spike_score)
        
        # 4. Volume-price alignment (0-25 points)
        alignment = self.calculate_volume_momentum_alignment(df)
        alignment_score = {'ALIGNED': 25, 'NEUTRAL': 12.5, 'DIVERGENT': 0}
        scores.append(alignment_score.get(alignment, 0))
        
        return sum(scores)
    
    def get_volume_analysis(self, df):
        """
        Get comprehensive volume analysis
        """
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return {
                'available': False,
                'strength': 0.0,
                'trend': 'UNKNOWN',
                'spike': False,
                'spike_ratio': 0.0,
                'alignment': 'NEUTRAL',
                'conviction': 'UNKNOWN'
            }
        
        strength = self.calculate_volume_strength_score(df)
        trend = self.calculate_volume_trend(df)
        has_spike, spike_ratio = self.calculate_volume_spike(df)
        alignment = self.calculate_volume_momentum_alignment(df)
        
        # Overall conviction assessment
        if strength > 70 and alignment == 'ALIGNED':
            conviction = 'VERY_HIGH'
        elif strength > 50 and alignment == 'ALIGNED':
            conviction = 'HIGH'
        elif strength > 30:
            conviction = 'MEDIUM'
        elif alignment == 'DIVERGENT':
            conviction = 'LOW'  # Warning sign
        else:
            conviction = 'WEAK'
        
        return {
            'available': True,
            'strength': strength,
            'trend': trend,
            'spike': has_spike,
            'spike_ratio': spike_ratio,
            'alignment': alignment,
            'conviction': conviction,
            'current_volume': df['volume'].iloc[-1] if len(df) > 0 else 0,
            'avg_volume': df['volume'].iloc[-60:].mean() if len(df) >= 60 else 0
        }


