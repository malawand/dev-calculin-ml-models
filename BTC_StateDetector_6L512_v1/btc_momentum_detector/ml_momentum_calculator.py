"""
ML-Enhanced Momentum Calculator

Instead of hand-crafted formulas, use ML to learn the TRUE momentum
from price action, volume, and derivatives.

Goal: More accurate momentum strength, direction, and phase detection
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

class MLMomentumCalculator:
    """
    Uses ML to calculate momentum more accurately than hand-crafted formulas
    
    Instead of:
        momentum = (ROC * 0.4) + (acceleration * 0.2) + (volume * 0.3) + (consistency * 0.1)
    
    ML learns:
        momentum = f(ROC, acceleration, volume, consistency, derivatives, price patterns, ...)
        
    By training on: "What actually happened next?"
    """
    
    def __init__(self):
        self.strength_model = None  # Predicts momentum strength (0-100)
        self.direction_model = None  # Predicts direction (-1, 0, 1)
        self.phase_model = None  # Predicts phase (BUILDING, STRONG, FADING, ABSENT)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_raw_features(self, df):
        """
        Extract comprehensive features for ML to learn from
        
        ML will learn which matter most and how to combine them
        """
        if len(df) < 300:
            return None
        
        features = {}
        prices = df['price'].values
        
        # Price features (multiple timeframes)
        for period in [5, 15, 30, 60, 120, 240]:
            if len(prices) >= period:
                # Rate of change
                roc = (prices[-1] - prices[-period]) / prices[-period]
                features[f'roc_{period}'] = roc
                
                # Volatility
                returns = np.diff(prices[-period:]) / prices[-period:-1]
                features[f'volatility_{period}'] = np.std(returns)
                
                # Trend strength (linear fit)
                x = np.arange(period)
                y = prices[-period:]
                if len(x) == len(y):
                    correlation = np.corrcoef(x, y)[0, 1]
                    features[f'trend_{period}'] = correlation
        
        # Acceleration (change in momentum)
        if len(prices) >= 60:
            roc_recent = (prices[-1] - prices[-30]) / prices[-30]
            roc_earlier = (prices[-30] - prices[-60]) / prices[-60]
            features['acceleration'] = roc_recent - roc_earlier
        
        # Price position relative to recent range
        if len(prices) >= 60:
            high = np.max(prices[-60:])
            low = np.min(prices[-60:])
            if high > low:
                features['price_position'] = (prices[-1] - low) / (high - low)
            else:
                features['price_position'] = 0.5
        
        # Moving average distances
        for period in [10, 20, 50, 100, 200]:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                features[f'distance_ma{period}'] = (prices[-1] - ma) / ma
        
        # Volume features (if available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            volumes = df['volume'].values
            
            for period in [15, 30, 60, 120, 240]:
                if len(volumes) >= period:
                    recent_vol = np.mean(volumes[-period//2:])
                    avg_vol = np.mean(volumes[-period:])
                    if avg_vol > 0:
                        features[f'volume_ratio_{period}'] = recent_vol / avg_vol
                    
                    # Volume trend
                    vol_trend = np.corrcoef(np.arange(period), volumes[-period:])[0, 1]
                    features[f'volume_trend_{period}'] = vol_trend
                    
                    # Price-volume correlation
                    if len(prices) >= period:
                        pv_corr = np.corrcoef(prices[-period:], volumes[-period:])[0, 1]
                        features[f'pv_correlation_{period}'] = pv_corr
        else:
            # Set volume features to 0 if not available
            for period in [15, 30, 60, 120, 240]:
                features[f'volume_ratio_{period}'] = 1.0
                features[f'volume_trend_{period}'] = 0.0
                features[f'pv_correlation_{period}'] = 0.0
        
        # Consistency (directional uniformity)
        if len(prices) >= 60:
            returns = np.diff(prices[-60:])
            positive_returns = (returns > 0).sum()
            features['consistency'] = abs(2 * (positive_returns / len(returns)) - 1)
        
        # Recent momentum vs longer-term
        if len(prices) >= 120:
            short_mom = (prices[-1] - prices[-30]) / prices[-30]
            long_mom = (prices[-1] - prices[-120]) / prices[-120]
            features['momentum_ratio'] = short_mom / long_mom if long_mom != 0 else 0
        
        # Derivative features (if available)
        derivative_cols = [col for col in df.columns if 'deriv' in col.lower()]
        for col in derivative_cols[:10]:  # Limit to 10 most recent
            if len(df[col].dropna()) > 0:
                features[f'{col}_last'] = df[col].iloc[-1]
        
        return features
    
    def train_on_historical_data(self, df):
        """
        Train ML models on historical data
        
        For each point in history:
        - Extract features at that moment
        - Look forward to see what ACTUALLY happened
        - Train ML to predict the "true" momentum
        
        True momentum = What happened in the next 1-4 hours
        """
        print("🤖 Training ML Momentum Calculator on historical data...")
        print()
        
        lookback = 300
        forward_periods = [15, 30, 60, 120, 240]  # Look forward 15min to 4h
        
        training_data = []
        
        print("   Extracting training samples...")
        for i in range(lookback, len(df) - max(forward_periods), 30):  # Sample every 30 bars
            if i % 5000 == 0:
                pct = (i / len(df)) * 100
                print(f"   Progress: {pct:.1f}%")
            
            # Get window
            df_window = df.iloc[i-lookback:i+1].copy()
            
            # Extract features
            features = self.extract_raw_features(df_window)
            if features is None:
                continue
            
            current_price = df['price'].iloc[i]
            
            # Calculate TRUE momentum by looking forward
            future_moves = []
            for forward in forward_periods:
                future_idx = i + forward
                if future_idx < len(df):
                    future_price = df['price'].iloc[future_idx]
                    move_pct = ((future_price - current_price) / current_price) * 100
                    future_moves.append(move_pct)
            
            if len(future_moves) < 3:
                continue
            
            # True momentum strength = average absolute move
            true_strength = np.mean(np.abs(future_moves))
            
            # True direction = sign of move
            avg_move = np.mean(future_moves)
            if avg_move > 0.2:
                true_direction = 1  # UP
            elif avg_move < -0.2:
                true_direction = -1  # DOWN
            else:
                true_direction = 0  # NONE
            
            # True phase based on acceleration of moves
            if len(future_moves) >= 3:
                early_move = future_moves[0]
                late_move = future_moves[-1]
                
                if true_strength < 0.3:
                    true_phase = 0  # ABSENT
                elif late_move > early_move and true_strength > 0.5:
                    true_phase = 1  # BUILDING
                elif true_strength > 1.0:
                    true_phase = 2  # STRONG
                else:
                    true_phase = 3  # FADING
            else:
                true_phase = 0
            
            training_data.append({
                'features': features,
                'true_strength': min(true_strength * 30, 100),  # Scale to 0-100
                'true_direction': true_direction,
                'true_phase': true_phase
            })
        
        print(f"\n   ✅ Collected {len(training_data):,} training samples")
        print()
        
        # Convert to arrays
        feature_names = sorted(list(training_data[0]['features'].keys()))
        X = []
        y_strength = []
        y_direction = []
        y_phase = []
        
        for sample in training_data:
            X.append([sample['features'].get(fn, 0) for fn in feature_names])
            y_strength.append(sample['true_strength'])
            y_direction.append(sample['true_direction'])
            y_phase.append(sample['true_phase'])
        
        X = np.array(X)
        y_strength = np.array(y_strength)
        y_direction = np.array(y_direction)
        y_phase = np.array(y_phase)
        
        # Handle NaN/inf in both X and y
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_strength = np.nan_to_num(y_strength, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Filter out any remaining invalid samples
        valid_mask = ~np.isnan(y_strength) & ~np.isinf(y_strength) & ~np.isnan(y_direction) & ~np.isnan(y_phase)
        X = X[valid_mask]
        y_strength = y_strength[valid_mask]
        y_direction = y_direction[valid_mask]
        y_phase = y_phase[valid_mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_strength_train, y_strength_test = y_strength[:split_idx], y_strength[split_idx:]
        y_direction_train, y_direction_test = y_direction[:split_idx], y_direction[split_idx:]
        y_phase_train, y_phase_test = y_phase[:split_idx], y_phase[split_idx:]
        
        # Train Model 1: Strength (regression 0-100)
        print("   Training Strength Model...")
        self.strength_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        self.strength_model.fit(X_train, y_strength_train)
        
        train_score = self.strength_model.score(X_train, y_strength_train)
        test_score = self.strength_model.score(X_test, y_strength_test)
        print(f"      Train R²: {train_score:.3f}")
        print(f"      Test R²:  {test_score:.3f}")
        
        # Calculate MAE
        y_pred_test = self.strength_model.predict(X_test)
        mae = np.mean(np.abs(y_pred_test - y_strength_test))
        print(f"      MAE:      {mae:.1f} points")
        print()
        
        # Train Model 2: Direction (classification -1, 0, 1)
        print("   Training Direction Model...")
        self.direction_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )
        self.direction_model.fit(X_train, y_direction_train)
        
        train_acc = self.direction_model.score(X_train, y_direction_train)
        test_acc = self.direction_model.score(X_test, y_direction_test)
        print(f"      Train Accuracy: {train_acc*100:.1f}%")
        print(f"      Test Accuracy:  {test_acc*100:.1f}%")
        print()
        
        # Train Model 3: Phase (classification)
        print("   Training Phase Model...")
        self.phase_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )
        self.phase_model.fit(X_train, y_phase_train)
        
        train_acc_phase = self.phase_model.score(X_train, y_phase_train)
        test_acc_phase = self.phase_model.score(X_test, y_phase_test)
        print(f"      Train Accuracy: {train_acc_phase*100:.1f}%")
        print(f"      Test Accuracy:  {test_acc_phase*100:.1f}%")
        print()
        
        self.is_trained = True
        self.feature_names = feature_names
        
        print("   ✅ ML Momentum Calculator trained successfully!")
        print()
        
        return {
            'strength': {'train_r2': train_score, 'test_r2': test_score, 'mae': mae},
            'direction': {'train_acc': train_acc, 'test_acc': test_acc},
            'phase': {'train_acc': train_acc_phase, 'test_acc': test_acc_phase}
        }
    
    def calculate_momentum(self, df):
        """
        Calculate momentum using trained ML models
        
        Returns: {
            'strength': 0-100,
            'direction': 'UP'/'DOWN'/'NONE',
            'phase': 'BUILDING'/'STRONG'/'FADING'/'ABSENT',
            'confidence': 0-1
        }
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train_on_historical_data() first")
        
        # Extract features
        features = self.extract_raw_features(df)
        if features is None:
            return None
        
        # Convert to array
        X = np.array([[features.get(fn, 0) for fn in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        strength = self.strength_model.predict(X_scaled)[0]
        strength = np.clip(strength, 0, 100)
        
        direction_idx = self.direction_model.predict(X_scaled)[0]
        direction_proba = self.direction_model.predict_proba(X_scaled)[0]
        direction = {-1: 'DOWN', 0: 'NONE', 1: 'UP'}[direction_idx]
        
        # Map direction_idx to proba index
        classes = self.direction_model.classes_
        proba_idx = np.where(classes == direction_idx)[0][0]
        direction_confidence = direction_proba[proba_idx]
        
        phase_idx = self.phase_model.predict(X_scaled)[0]
        phase_proba = self.phase_model.predict_proba(X_scaled)[0]
        phase = {0: 'ABSENT', 1: 'BUILDING', 2: 'STRONG', 3: 'FADING'}[phase_idx]
        phase_confidence = phase_proba[phase_idx]
        
        # Overall confidence
        confidence = (direction_confidence + phase_confidence) / 2
        
        return {
            'strength': strength,
            'direction': direction,
            'phase': phase,
            'confidence': confidence,
            'direction_confidence': direction_confidence,
            'phase_confidence': phase_confidence
        }
    
    def save(self, path):
        """Save trained models"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'strength_model.pkl', 'wb') as f:
            pickle.dump(self.strength_model, f)
        with open(path / 'direction_model.pkl', 'wb') as f:
            pickle.dump(self.direction_model, f)
        with open(path / 'phase_model.pkl', 'wb') as f:
            pickle.dump(self.phase_model, f)
        with open(path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(path / 'feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
    
    def load(self, path):
        """Load trained models"""
        path = Path(path)
        
        with open(path / 'strength_model.pkl', 'rb') as f:
            self.strength_model = pickle.load(f)
        with open(path / 'direction_model.pkl', 'rb') as f:
            self.direction_model = pickle.load(f)
        with open(path / 'phase_model.pkl', 'rb') as f:
            self.phase_model = pickle.load(f)
        with open(path / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open(path / 'feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        self.is_trained = True

