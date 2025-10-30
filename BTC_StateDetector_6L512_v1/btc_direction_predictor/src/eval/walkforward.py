"""
Walk-forward validation for time series
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Expanding window walk-forward validation"""
    
    def __init__(self, n_splits: int = 5):
        """
        Initialize validator.
        
        Args:
            n_splits: Number of time series splits
        """
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        
    def validate(self, model_class, X: np.ndarray, y: np.ndarray, 
                config: Dict, feature_names: List[str] = None) -> Dict:
        """
        Perform walk-forward validation.
        
        Args:
            model_class: Model class to instantiate
            X: Features
            y: Labels
            config: Configuration
            feature_names: Feature names for importance
            
        Returns:
            Dictionary with cv_scores and trained models
        """
        logger.info("=" * 80)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Splits: {self.n_splits}")
        logger.info(f"Total samples: {len(X)}")
        
        fold_metrics = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(X), 1):
            logger.info(f"\nFold {fold}/{self.n_splits}")
            logger.info(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features (fit on train only!)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = model_class(config)
            model.train(X_train_scaled, y_train, X_val_scaled, y_val)
            
            # Evaluate
            from .metrics import MetricsCalculator
            calc = MetricsCalculator()
            
            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled) if hasattr(model, 'predict_proba') else None
            
            metrics = calc.calculate_classification_metrics(y_val, y_pred, y_proba)
            fold_metrics.append(metrics)
            models.append(model)
            
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  MCC: {metrics['mcc']:.4f}")
            logger.info(f"  F1 (UP): {metrics['f1_up']:.4f}")
        
        # Aggregate results
        logger.info("\n" + "=" * 80)
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        cv_scores = {}
        for metric in fold_metrics[0].keys():
            values = [m[metric] for m in fold_metrics]
            cv_scores[f'{metric}_mean'] = np.mean(values)
            cv_scores[f'{metric}_std'] = np.std(values)
            
            logger.info(f"{metric:.<30} {cv_scores[f'{metric}_mean']:.4f} ± {cv_scores[f'{metric}_std']:.4f}")
        
        return {
            'cv_scores': cv_scores,
            'fold_metrics': fold_metrics,
            'models': models
        }
    
    def get_train_test_split(self, X: np.ndarray, y: np.ndarray, 
                            test_ratio: float = 0.2) -> Tuple:
        """
        Get train/test split for time series.
        
        Args:
            X: Features
            y: Labels
            test_ratio: Fraction for test set
            
        Returns:
            (X_train, X_test, y_train, y_test, train_idx, test_idx)
        """
        split_idx = int(len(X) * (1 - test_ratio))
        
        train_idx = np.arange(split_idx)
        test_idx = np.arange(split_idx, len(X))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, train_idx, test_idx


def walk_forward_validation(model_class, X: np.ndarray, y: np.ndarray, 
                           config: Dict, n_splits: int = 5) -> Dict:
    """Convenience function for walk-forward validation"""
    validator = WalkForwardValidator(n_splits=n_splits)
    return validator.validate(model_class, X, y, config)



