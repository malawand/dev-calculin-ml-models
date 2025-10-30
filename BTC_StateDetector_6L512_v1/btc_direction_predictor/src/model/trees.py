"""
LightGBM model with hyperparameter search
"""
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, matthews_corrcoef
import numpy as np
import logging
from typing import Dict, Tuple
import joblib

logger = logging.getLogger(__name__)


class LightGBMClassifier:
    """LightGBM binary classifier with hyperparameter search"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        self.model = None
        self.best_params = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> 'LightGBMClassifier':
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Self
        """
        logger.info("Training LightGBM classifier...")
        
        # Base parameters
        base_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'force_col_wise': True,
            'random_state': 42
        }
        
        # Hyperparameter search if configured
        if self.config['modeling']['random_search_iters'] > 0:
            logger.info("Performing hyperparameter search...")
            self.model = self._hyperparameter_search(X_train, y_train, base_params)
        else:
            # Use default parameters
            params = {**base_params, 'num_leaves': 31, 'max_depth': 5, 'n_estimators': 200}
            self.model = lgb.LGBMClassifier(**params)
            
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='binary_logloss',
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
                )
            else:
                self.model.fit(X_train, y_train)
        
        logger.info("Training complete")
        return self
    
    def _hyperparameter_search(self, X_train: np.ndarray, y_train: np.ndarray, 
                               base_params: Dict) -> lgb.LGBMClassifier:
        """Perform random search for hyperparameters"""
        
        # Parameter distributions
        param_dist = {
            'max_depth': self.config['modeling']['lightgbm']['max_depth'],
            'num_leaves': self.config['modeling']['lightgbm']['num_leaves'],
            'learning_rate': self.config['modeling']['lightgbm']['learning_rate'],
            'n_estimators': self.config['modeling']['lightgbm']['n_estimators'],
            'min_child_samples': self.config['modeling']['lightgbm']['min_child_samples'],
        }
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        # MCC scorer
        mcc_scorer = make_scorer(matthews_corrcoef)
        
        # Random search
        model = lgb.LGBMClassifier(**base_params)
        
        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=self.config['modeling']['random_search_iters'],
            scoring=mcc_scorer,
            cv=tscv,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score (MCC): {search.best_score_:.4f}")
        
        self.best_params = search.best_params_
        
        return search.best_estimator_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        """Get feature importances"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        return dict(zip(feature_names, importances))
    
    def save(self, path: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params,
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'LightGBMClassifier':
        """Load model from disk"""
        data = joblib.load(path)
        instance = cls(data['config'])
        instance.model = data['model']
        instance.best_params = data['best_params']
        return instance


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray, config: Dict,
                   X_val: np.ndarray = None, y_val: np.ndarray = None) -> LightGBMClassifier:
    """Convenience function to train LightGBM"""
    classifier = LightGBMClassifier(config)
    return classifier.train(X_train, y_train, X_val, y_val)



