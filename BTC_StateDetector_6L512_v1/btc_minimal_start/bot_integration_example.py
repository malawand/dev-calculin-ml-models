#!/usr/bin/env python3
"""
Example Trading Bot Integration
Shows how to use the prediction model in your trading bot
"""

import time
import json
from datetime import datetime
from predict_live import LivePredictor

class SimpleTradingBot:
    """Example trading bot that uses the prediction model"""
    
    def __init__(self, initial_capital=10000, risk_per_trade=0.02):
        """
        Initialize bot
        
        Args:
            initial_capital: Starting capital in USD
            risk_per_trade: Risk per trade as fraction of capital (0.02 = 2%)
        """
        self.predictor = LivePredictor()
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.position = None
        self.trades = []
        
        print(f"🤖 Trading Bot Initialized")
        print(f"   Capital: ${self.capital:,.2f}")
        print(f"   Risk per trade: {self.risk_per_trade:.1%}")
        print()
    
    def calculate_position_size(self, probability, current_price):
        """
        Calculate position size based on Kelly Criterion (simplified)
        
        More confident predictions = larger positions (within risk limits)
        """
        # Base position size (1% of capital)
        base_size = self.capital * self.risk_per_trade
        
        # Adjust by confidence (probability distance from 0.5)
        confidence_multiplier = abs(probability - 0.5) * 4  # 0 to 2x
        adjusted_size = base_size * (1 + confidence_multiplier)
        
        # Convert USD to BTC
        btc_amount = adjusted_size / current_price
        
        return btc_amount
    
    def execute_trade(self, signal, prediction):
        """
        Execute a trade based on prediction
        
        In production, this would:
        1. Call exchange API
        2. Place order
        3. Set stop loss / take profit
        4. Update database
        """
        price = prediction['current_price']
        prob = prediction['probability_up']
        
        # Close existing position if opposite signal
        if self.position:
            if (self.position['type'] == 'LONG' and signal in ['SELL', 'STRONG_SELL']) or \
               (self.position['type'] == 'SHORT' and signal in ['BUY', 'STRONG_BUY']):
                self.close_position(price, prediction['timestamp'])
        
        # Open new position
        if not self.position:
            if signal in ['STRONG_BUY', 'BUY']:
                self.open_position('LONG', price, prob, prediction['timestamp'])
            elif signal in ['STRONG_SELL', 'SELL']:
                self.open_position('SHORT', price, prob, prediction['timestamp'])
    
    def open_position(self, position_type, price, probability, timestamp):
        """Open a new position"""
        # Calculate position size
        btc_amount = self.calculate_position_size(probability, price)
        position_value = btc_amount * price
        
        # Set stop loss and take profit
        if position_type == 'LONG':
            stop_loss = price * 0.99  # 1% stop loss
            take_profit = price * 1.02  # 2% take profit
        else:  # SHORT
            stop_loss = price * 1.01  # 1% stop loss
            take_profit = price * 0.98  # 2% take profit
        
        self.position = {
            'type': position_type,
            'entry_price': price,
            'btc_amount': btc_amount,
            'position_value': position_value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'probability': probability,
            'entry_time': timestamp
        }
        
        print(f"\n📊 {'🟢 LONG' if position_type == 'LONG' else '🔴 SHORT'} POSITION OPENED")
        print(f"   Time: {timestamp}")
        print(f"   Entry Price: ${price:,.2f}")
        print(f"   Amount: {btc_amount:.6f} BTC (${position_value:,.2f})")
        print(f"   Confidence: {probability:.2%}")
        print(f"   Stop Loss: ${stop_loss:,.2f}")
        print(f"   Take Profit: ${take_profit:,.2f}")
        print()
    
    def close_position(self, current_price, timestamp):
        """Close current position"""
        if not self.position:
            return
        
        entry_price = self.position['entry_price']
        btc_amount = self.position['btc_amount']
        position_value = self.position['position_value']
        
        # Calculate P&L
        if self.position['type'] == 'LONG':
            pnl_usd = (current_price - entry_price) * btc_amount
        else:  # SHORT
            pnl_usd = (entry_price - current_price) * btc_amount
        
        pnl_pct = (pnl_usd / position_value) * 100
        
        # Update capital
        self.capital += pnl_usd
        
        # Log trade
        trade = {
            'type': self.position['type'],
            'entry_price': entry_price,
            'exit_price': current_price,
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'btc_amount': btc_amount,
            'pnl_usd': pnl_usd,
            'pnl_pct': pnl_pct,
            'probability': self.position['probability']
        }
        self.trades.append(trade)
        
        print(f"\n📊 POSITION CLOSED")
        print(f"   Type: {self.position['type']}")
        print(f"   Entry: ${entry_price:,.2f} → Exit: ${current_price:,.2f}")
        print(f"   P&L: ${pnl_usd:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"   New Capital: ${self.capital:,.2f}")
        print()
        
        self.position = None
    
    def check_stop_loss_take_profit(self, current_price, timestamp):
        """Check if stop loss or take profit hit"""
        if not self.position:
            return
        
        if self.position['type'] == 'LONG':
            if current_price <= self.position['stop_loss']:
                print(f"🛑 STOP LOSS HIT")
                self.close_position(current_price, timestamp)
            elif current_price >= self.position['take_profit']:
                print(f"✅ TAKE PROFIT HIT")
                self.close_position(current_price, timestamp)
        else:  # SHORT
            if current_price >= self.position['stop_loss']:
                print(f"🛑 STOP LOSS HIT")
                self.close_position(current_price, timestamp)
            elif current_price <= self.position['take_profit']:
                print(f"✅ TAKE PROFIT HIT")
                self.close_position(current_price, timestamp)
    
    def print_summary(self):
        """Print trading summary"""
        if not self.trades:
            print("No trades executed yet.")
            return
        
        print("\n" + "=" * 70)
        print("                    📊 TRADING SUMMARY")
        print("=" * 70)
        
        total_pnl = sum(t['pnl_usd'] for t in self.trades)
        winning_trades = [t for t in self.trades if t['pnl_usd'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_usd'] <= 0]
        
        print(f"\nTotal Trades: {len(self.trades)}")
        print(f"Winning: {len(winning_trades)} ({len(winning_trades)/len(self.trades):.1%})")
        print(f"Losing: {len(losing_trades)} ({len(losing_trades)/len(self.trades):.1%})")
        print(f"\nInitial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total P&L: ${total_pnl:+,.2f} ({(total_pnl/self.initial_capital)*100:+.2f}%)")
        
        if winning_trades:
            avg_win = sum(t['pnl_usd'] for t in winning_trades) / len(winning_trades)
            print(f"\nAvg Win: ${avg_win:+,.2f}")
        
        if losing_trades:
            avg_loss = sum(t['pnl_usd'] for t in losing_trades) / len(losing_trades)
            print(f"Avg Loss: ${avg_loss:+,.2f}")
        
        if winning_trades and losing_trades:
            profit_factor = abs(sum(t['pnl_usd'] for t in winning_trades) / sum(t['pnl_usd'] for t in losing_trades))
            print(f"\nProfit Factor: {profit_factor:.2f}")
        
        print("=" * 70 + "\n")
    
    def run(self, check_interval_minutes=15, max_trades=None):
        """
        Run the trading bot
        
        Args:
            check_interval_minutes: How often to check for new signals
            max_trades: Maximum number of trades to execute (None = unlimited)
        """
        print("🚀 Starting Trading Bot...")
        print(f"   Check interval: {check_interval_minutes} minutes")
        print(f"   Max trades: {max_trades if max_trades else 'Unlimited'}")
        print(f"   Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Get prediction
                prediction = self.predictor.predict()
                signal = prediction['signal_strength']
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
                print(f"   Price: ${prediction['current_price']:,.2f}")
                print(f"   Signal: {signal} (Prob: {prediction['probability_up']:.2%})")
                
                # Check stop loss / take profit
                if self.position:
                    self.check_stop_loss_take_profit(
                        prediction['current_price'],
                        prediction['timestamp']
                    )
                
                # Execute trade if signal strong enough
                if signal in ['STRONG_BUY', 'BUY', 'STRONG_SELL', 'SELL']:
                    self.execute_trade(signal, prediction)
                
                # Check if max trades reached
                if max_trades and len(self.trades) >= max_trades:
                    print(f"\n✅ Reached max trades ({max_trades}). Stopping.")
                    break
                
                # Wait for next check
                time.sleep(check_interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
            
            # Close any open positions
            if self.position:
                final_prediction = self.predictor.predict()
                self.close_position(
                    final_prediction['current_price'],
                    final_prediction['timestamp']
                )
        
        finally:
            # Print summary
            self.print_summary()


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Example Trading Bot')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital (USD)')
    parser.add_argument('--risk', type=float, default=0.02, help='Risk per trade (0.02 = 2%)')
    parser.add_argument('--interval', type=int, default=15, help='Check interval (minutes)')
    parser.add_argument('--max-trades', type=int, help='Max number of trades')
    
    args = parser.parse_args()
    
    # Create and run bot
    bot = SimpleTradingBot(
        initial_capital=args.capital,
        risk_per_trade=args.risk
    )
    
    bot.run(
        check_interval_minutes=args.interval,
        max_trades=args.max_trades
    )


if __name__ == "__main__":
    main()



