"""
Performance metrics for classification and trading
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for AUC)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_up'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['recall_up'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['f1_up'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # AUC if probabilities provided
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        # Confusion matrix elements
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
        
        return metrics
    
    @staticmethod
    def calculate_trading_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  returns: np.ndarray, fee_bps: float = 5) -> Dict[str, float]:
        """
        Calculate trading-specific metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (signals)
            returns: Actual returns for each period
            fee_bps: Trading fee in basis points
            
        Returns:
            Dictionary of trading metrics
        """
        metrics = {}
        
        # Convert fee to decimal
        fee = fee_bps / 10000.0
        
        # Strategy returns: only take positions when predicted UP (y_pred=1)
        strategy_returns = np.where(y_pred == 1, returns - fee, 0)
        
        # Hit rate (correctness when we trade)
        trades_mask = y_pred == 1
        if trades_mask.sum() > 0:
            metrics['hit_rate'] = accuracy_score(y_true[trades_mask], y_pred[trades_mask])
            metrics['avg_return_per_trade'] = returns[trades_mask].mean()
            metrics['num_trades'] = int(trades_mask.sum())
        else:
            metrics['hit_rate'] = 0.0
            metrics['avg_return_per_trade'] = 0.0
            metrics['num_trades'] = 0
        
        # Cumulative return
        metrics['cumulative_return'] = strategy_returns.sum()
        
        # Sharpe ratio (annualized, assuming 5-min bars)
        if strategy_returns.std() > 0:
            # 288 bars per day (5-min), 365 days per year
            metrics['sharpe_ratio'] = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(288 * 365)
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
        """Pretty print metrics"""
        logger.info("=" * 60)
        logger.info(f"{title}")
        logger.info("=" * 60)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key:.<30} {value:.4f}")
            else:
                logger.info(f"{key:.<30} {value}")
        
        logger.info("=" * 60)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                  returns_test: np.ndarray = None, config: Dict = None) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model with predict/predict_proba methods
        X_test: Test features
        y_test: Test labels
        returns_test: Test returns (for trading metrics)
        config: Configuration dict
        
    Returns:
        Dictionary of all metrics
    """
    calc = MetricsCalculator()
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Classification metrics
    metrics = calc.calculate_classification_metrics(y_test, y_pred, y_proba)
    
    # Trading metrics if returns provided
    if returns_test is not None:
        fee_bps = config['backtest']['fee_bps'] if config else 5
        trading_metrics = calc.calculate_trading_metrics(y_test, y_pred, returns_test, fee_bps)
        metrics.update(trading_metrics)
    
    return metrics



