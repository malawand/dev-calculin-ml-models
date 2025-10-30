"""
Momentum Trading Signals: Entry and exit based on momentum phases
"""
import numpy as np
from momentum_calculator import MomentumCalculator

class MomentumSignals:
    def __init__(self, 
                 min_strength_entry=35,
                 min_confidence_entry=0.5,
                 exit_strength_threshold=25,
                 trailing_stop_pct=0.005):
        
        self.min_strength_entry = min_strength_entry
        self.min_confidence_entry = min_confidence_entry
        self.exit_strength_threshold = exit_strength_threshold
        self.trailing_stop_pct = trailing_stop_pct
        self.calculator = MomentumCalculator()
    
    def generate_entry_signal(self, df):
        """
        Generate entry signal when momentum is building
        
        Returns:
            {
                'action': 'BUY' | 'SELL' | 'NONE',
                'confidence': float,
                'entry_price': float,
                'stop_loss': float,
                'target_price': float,
                'momentum_report': dict
            }
        """
        report = self.calculator.get_momentum_report(df)
        
        # No signal if below thresholds
        if report['strength'] < self.min_strength_entry:
            return self._no_signal(report)
        
        if report['confidence'] < self.min_confidence_entry:
            return self._no_signal(report)
        
        # No signal if no clear direction
        if report['direction'] == 'NONE':
            return self._no_signal(report)
        
        # Only enter on BUILDING or STRONG phases
        if report['phase'] not in ['BUILDING', 'STRONG']:
            return self._no_signal(report)
        
        current_price = report['current_price']
        
        # Calculate position sizing based on momentum strength
        # Stronger momentum = larger position (within limits)
        position_multiplier = min(report['strength'] / 100, 0.8) + 0.2  # 0.2 to 1.0
        
        # Entry based on direction
        if report['direction'] == 'UP':
            # BUY signal
            # Stop loss: below recent support or -1%
            stop_loss = current_price * 0.99
            
            # Target: based on momentum strength
            # Stronger momentum = higher target
            target_multiplier = 1.0 + (report['strength'] / 100 * 0.03)  # Up to 3%
            target_price = current_price * target_multiplier
            
            expected_return = (target_price - current_price) / current_price * 100
            risk = (current_price - stop_loss) / current_price * 100
            
            return {
                'action': 'BUY',
                'confidence': report['confidence'],
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'expected_return_pct': expected_return,
                'risk_pct': risk,
                'reward_risk_ratio': expected_return / risk if risk > 0 else 0,
                'position_multiplier': position_multiplier,
                'momentum_report': report,
                'reason': f"Momentum {report['phase']} with strength {report['strength']:.0f}"
            }
        
        elif report['direction'] == 'DOWN':
            # SELL signal
            # Stop loss: above recent resistance or +1%
            stop_loss = current_price * 1.01
            
            # Target: based on momentum strength
            target_multiplier = 1.0 - (report['strength'] / 100 * 0.03)  # Down to -3%
            target_price = current_price * target_multiplier
            
            expected_return = (current_price - target_price) / current_price * 100
            risk = (stop_loss - current_price) / current_price * 100
            
            return {
                'action': 'SELL',
                'confidence': report['confidence'],
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'expected_return_pct': expected_return,
                'risk_pct': risk,
                'reward_risk_ratio': expected_return / risk if risk > 0 else 0,
                'position_multiplier': position_multiplier,
                'momentum_report': report,
                'reason': f"Momentum {report['phase']} with strength {report['strength']:.0f}"
            }
        
        return self._no_signal(report)
    
    def check_exit_signal(self, df, position_type, entry_price):
        """
        Check if we should exit an open position
        
        Args:
            df: Price data
            position_type: 'LONG' or 'SHORT'
            entry_price: Price we entered at
        
        Returns:
            {
                'should_exit': bool,
                'reason': str,
                'exit_type': 'MOMENTUM_FADED' | 'TARGET' | 'STOP' | None
            }
        """
        report = self.calculator.get_momentum_report(df)
        current_price = report['current_price']
        
        # 1. Check if momentum has faded
        if report['phase'] == 'FADING':
            return {
                'should_exit': True,
                'reason': f"Momentum fading (strength: {report['strength']:.0f})",
                'exit_type': 'MOMENTUM_FADED',
                'momentum_report': report
            }
        
        # 2. Check if momentum reversed direction
        if position_type == 'LONG' and report['direction'] == 'DOWN':
            return {
                'should_exit': True,
                'reason': f"Momentum reversed to DOWN",
                'exit_type': 'MOMENTUM_REVERSED',
                'momentum_report': report
            }
        
        if position_type == 'SHORT' and report['direction'] == 'UP':
            return {
                'should_exit': True,
                'reason': f"Momentum reversed to UP",
                'exit_type': 'MOMENTUM_REVERSED',
                'momentum_report': report
            }
        
        # 3. Check if momentum dropped below threshold
        if report['strength'] < self.exit_strength_threshold:
            return {
                'should_exit': True,
                'reason': f"Momentum too weak (strength: {report['strength']:.0f} < {self.exit_strength_threshold})",
                'exit_type': 'MOMENTUM_WEAK',
                'momentum_report': report
            }
        
        # 4. Trailing stop (if in profit)
        if position_type == 'LONG':
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct > self.trailing_stop_pct:
                # We're in profit, use trailing stop
                trailing_stop = entry_price * (1 + profit_pct - self.trailing_stop_pct)
                if current_price < trailing_stop:
                    return {
                        'should_exit': True,
                        'reason': f"Trailing stop hit (profit: {profit_pct*100:.2f}%)",
                        'exit_type': 'TRAILING_STOP',
                        'momentum_report': report
                    }
        
        elif position_type == 'SHORT':
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct > self.trailing_stop_pct:
                # We're in profit, use trailing stop
                trailing_stop = entry_price * (1 - profit_pct + self.trailing_stop_pct)
                if current_price > trailing_stop:
                    return {
                        'should_exit': True,
                        'reason': f"Trailing stop hit (profit: {profit_pct*100:.2f}%)",
                        'exit_type': 'TRAILING_STOP',
                        'momentum_report': report
                    }
        
        # No exit signal
        return {
            'should_exit': False,
            'reason': None,
            'exit_type': None,
            'momentum_report': report
        }
    
    def _no_signal(self, report=None):
        """Return no trading signal"""
        return {
            'action': 'NONE',
            'confidence': 0.0,
            'entry_price': None,
            'stop_loss': None,
            'target_price': None,
            'expected_return_pct': 0.0,
            'risk_pct': 0.0,
            'reward_risk_ratio': 0.0,
            'position_multiplier': 0.0,
            'momentum_report': report,
            'reason': None
        }


