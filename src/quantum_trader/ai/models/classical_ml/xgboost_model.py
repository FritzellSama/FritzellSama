"""XGBoost Model for Trading - PRODUCTION"""
from decimal import Decimal
from typing import Dict, Any
from abc import ABC, abstractmethod
import os, logging
import numpy as np
import polars as pl

try:
    import xgboost as xgb
    import joblib
except ImportError:
    raise ImportError("XGBoost required")

logger = logging.getLogger(__name__)

class BaseMLModel(ABC):
    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None: pass
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray: pass
    @abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]: pass
    @abstractmethod
    def save(self, path: str) -> None: pass
    @abstractmethod
    def load(self, path: str) -> None: pass

class XGBoostModel(BaseMLModel):
    """Production XGBoost Model"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # XGBoost hyperparameters from config
        self.params = {
            'n_estimators': int(config.get('n_estimators', os.getenv('XGB_N_ESTIMATORS', '1000'))),
            'max_depth': int(config.get('max_depth', os.getenv('XGB_MAX_DEPTH', '6'))),
            'learning_rate': float(config.get('learning_rate', os.getenv('XGB_LR', '0.05'))),
            'subsample': float(config.get('subsample', os.getenv('XGB_SUBSAMPLE', '0.8'))),
            'colsample_bytree': float(config.get('colsample_bytree', os.getenv('XGB_COLSAMPLE', '0.8'))),
            'reg_alpha': float(config.get('reg_alpha', os.getenv('XGB_REG_ALPHA', '0.001'))),
            'reg_lambda': float(config.get('reg_lambda', os.getenv('XGB_REG_LAMBDA', '0.01'))),
            'tree_method': config.get('tree_method', os.getenv('XGB_TREE_METHOD', 'hist')),
            'random_state': config.get('random_state', 42),
            'n_jobs': -1
        }

        self.model = xgb.XGBClassifier(**self.params)
        self.is_trained = False

        self.logger.info(f"XGBoost initialized: n_estimators={self.params['n_estimators']}")

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        try:
            self.logger.info(f"Training XGBoost on {features.shape}")
            self.model.fit(features, labels.ravel())
            self.is_trained = True
            self.logger.info("XGBoost training completed")
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(features)

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        preds = self.predict(features)
        pred_classes = np.argmax(preds, axis=1)
        acc = float(np.mean(pred_classes == labels.ravel()))
        return {'accuracy': acc, 'n_samples': len(labels)}

    def get_feature_importance(self) -> pl.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        importance = self.model.feature_importances_
        return pl.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(importance))],
            'importance': [Decimal(str(imp)) for imp in importance]
        }).sort('importance', descending=True)

    def save(self, path: str) -> None:
        joblib.dump({'model': self.model, 'config': self.config, 'is_trained': self.is_trained}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data['model']
        self.config = data['config']
        self.is_trained = data['is_trained']
