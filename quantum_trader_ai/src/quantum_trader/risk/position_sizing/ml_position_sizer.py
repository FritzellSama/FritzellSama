"""
ML-Based Position Sizer
CRITICAL: Machine learning position sizing using ensemble models
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pickle
from pathlib import Path

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class MLPositionSizer:
    """
    Machine learning-based position sizing using RandomForest and XGBoost

    This class implements ML-based position sizing without requiring BaseMLModel.
    It uses sklearn's RandomForestRegressor and optionally XGBoost for predictions.
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize ML position sizer

        Args:
            model_type: Type of model to use ('random_forest', 'xgboost', 'ensemble')
        """
        self.config = get_config()
        self.model_type = model_type

        # Load configuration
        self.max_position_size_pct = self.config.get_decimal("risk", "position_sizing.max_position_size_pct")
        self.min_position_size_usd = self.config.get_decimal("risk", "position_sizing.min_position_size_usd")
        self.max_position_size_usd = self.config.get_decimal("risk", "position_limits.max_position_size_usd")

        # Model state
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False

        # Feature engineering settings
        self.lookback_periods = [5, 10, 20, 50]
        self.volatility_windows = [10, 20, 50]

        logger.info(
            f"MLPositionSizer initialized: model_type={model_type}, "
            f"max_size_pct={self.max_position_size_pct}, "
            f"min_size=${self.min_position_size_usd}"
        )

    async def train_model(
        self,
        historical_data: pl.DataFrame,
        target_column: str = "optimal_size",
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train ML model on historical data

        Args:
            historical_data: DataFrame with features and target variable
                Required columns: price, volume, volatility, returns, optimal_size
            target_column: Name of target column to predict
            validation_split: Fraction of data for validation

        Returns:
            Dictionary with training metrics:
            {
                'train_score': float,
                'val_score': float,
                'train_rmse': float,
                'val_rmse': float,
                'feature_importance': Dict[str, float]
            }
        """
        try:
            if historical_data.is_empty():
                logger.error("Cannot train model: empty historical data")
                raise ValueError("Historical data cannot be empty")

            logger.info(f"Training ML model with {len(historical_data)} samples")

            # Generate features
            features_df = await self.update_features(historical_data)

            if target_column not in features_df.columns:
                logger.error(f"Target column '{target_column}' not found in data")
                raise ValueError(f"Missing target column: {target_column}")

            # Prepare training data
            feature_columns = [col for col in features_df.columns
                             if col not in [target_column, 'timestamp', 'symbol']]

            X = features_df.select(feature_columns).to_numpy()
            y = features_df.select(target_column).to_numpy().flatten()

            # Remove any NaN or inf values
            valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) < 10:
                logger.error(f"Insufficient valid training samples: {len(X)}")
                raise ValueError("Need at least 10 valid samples for training")

            # Split train/validation
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Standardize features
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train model based on type
            if self.model_type == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                self.model.fit(X_train_scaled, y_train)

            elif self.model_type == "xgboost":
                try:
                    import xgboost as xgb
                    self.model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1
                    )
                    self.model.fit(X_train_scaled, y_train)
                except ImportError:
                    logger.warning("XGBoost not available, falling back to RandomForest")
                    from sklearn.ensemble import RandomForestRegressor
                    self.model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    self.model.fit(X_train_scaled, y_train)

            elif self.model_type == "ensemble":
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.ensemble import VotingRegressor

                rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                gb = GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42)

                self.model = VotingRegressor(
                    estimators=[('rf', rf), ('gb', gb)],
                    n_jobs=-1
                )
                self.model.fit(X_train_scaled, y_train)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Calculate metrics
            train_score = self.model.score(X_train_scaled, y_train)
            val_score = self.model.score(X_val_scaled, y_val)

            train_pred = self.model.predict(X_train_scaled)
            val_pred = self.model.predict(X_val_scaled)

            train_rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))
            val_rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))

            # Get feature importance
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = {
                    name: float(imp)
                    for name, imp in zip(feature_columns, importances)
                }
                # Sort by importance
                feature_importance = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )

            self.feature_names = feature_columns
            self.is_trained = True

            metrics = {
                'train_score': float(train_score),
                'val_score': float(val_score),
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'feature_importance': feature_importance,
                'n_features': len(feature_columns),
                'n_samples': len(X)
            }

            logger.info(
                f"Model trained successfully: "
                f"val_score={val_score:.4f}, val_rmse={val_rmse:.6f}, "
                f"n_features={len(feature_columns)}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error training ML model: {e}", exc_info=True)
            raise

    async def predict_optimal_size(
        self,
        current_data: pl.DataFrame,
        portfolio_value: Decimal,
        symbol: str
    ) -> Decimal:
        """
        Predict optimal position size using trained model

        Args:
            current_data: DataFrame with current market data and features
            portfolio_value: Current portfolio value
            symbol: Asset symbol

        Returns:
            Optimal position size as Decimal (percentage of portfolio)
        """
        try:
            if not self.is_trained or self.model is None:
                logger.warning("Model not trained, using default 2% position size")
                return Decimal("0.02")

            # Generate features for current data
            features_df = await self.update_features(current_data)

            if features_df.is_empty():
                logger.warning("No features generated, using default size")
                return Decimal("0.02")

            # Get latest row
            latest_features = features_df.sort("timestamp").tail(1)

            # Extract feature values
            X = latest_features.select(self.feature_names).to_numpy()

            # Check for invalid values
            if not np.isfinite(X).all():
                logger.warning("Invalid feature values, using default size")
                return Decimal("0.02")

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Predict
            predicted_size = float(self.model.predict(X_scaled)[0])

            # Convert to Decimal and apply limits
            predicted_size_pct = Decimal(str(predicted_size)).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )

            # Ensure within configured limits
            predicted_size_pct = max(
                Decimal("0"),
                min(predicted_size_pct, self.max_position_size_pct)
            )

            # Check minimum dollar amount
            predicted_size_usd = predicted_size_pct * portfolio_value
            if predicted_size_usd < self.min_position_size_usd:
                logger.debug(
                    f"Predicted size ${predicted_size_usd} below minimum, "
                    f"adjusting to ${self.min_position_size_usd}"
                )
                predicted_size_pct = self.min_position_size_usd / portfolio_value

            # Check maximum dollar amount
            if predicted_size_usd > self.max_position_size_usd:
                logger.warning(
                    f"Predicted size ${predicted_size_usd} exceeds maximum, "
                    f"capping at ${self.max_position_size_usd}"
                )
                predicted_size_pct = self.max_position_size_usd / portfolio_value

            logger.info(
                f"Predicted position size for {symbol}: {predicted_size_pct:.4%} "
                f"(${predicted_size_usd:,.2f})"
            )

            return predicted_size_pct

        except Exception as e:
            logger.error(f"Error predicting optimal size: {e}", exc_info=True)
            # Return safe default on error
            return Decimal("0.02")

    async def update_features(
        self,
        market_data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Generate features from market data for ML model

        Args:
            market_data: DataFrame with columns:
                ['timestamp', 'symbol', 'price', 'volume', 'returns']

        Returns:
            DataFrame with engineered features
        """
        try:
            if market_data.is_empty():
                logger.error("Cannot generate features: empty market data")
                raise ValueError("Market data cannot be empty")

            # Sort by timestamp
            df = market_data.sort("timestamp")

            # Ensure required columns exist
            required_cols = ['timestamp', 'price']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                raise ValueError(f"Missing columns: {missing_cols}")

            # Calculate returns if not present
            if 'returns' not in df.columns:
                df = df.with_columns(
                    (pl.col('price').pct_change()).alias('returns')
                )

            # Calculate volume features if volume exists
            if 'volume' in df.columns:
                df = df.with_columns([
                    pl.col('volume').rolling_mean(window_size=20).alias('volume_ma20'),
                    (pl.col('volume') / pl.col('volume').rolling_mean(window_size=20)).alias('volume_ratio')
                ])

            # Price-based features
            for period in self.lookback_periods:
                df = df.with_columns([
                    pl.col('price').rolling_mean(window_size=period).alias(f'price_ma{period}'),
                    (pl.col('price') / pl.col('price').rolling_mean(window_size=period) - 1).alias(f'price_dev{period}')
                ])

            # Volatility features
            for window in self.volatility_windows:
                df = df.with_columns([
                    pl.col('returns').rolling_std(window_size=window).alias(f'volatility_{window}d')
                ])

            # Return-based features
            df = df.with_columns([
                pl.col('returns').rolling_mean(window_size=10).alias('returns_ma10'),
                pl.col('returns').rolling_mean(window_size=20).alias('returns_ma20'),
                pl.col('returns').rolling_std(window_size=20).alias('returns_std20'),
                (pl.col('returns').rolling_mean(window_size=10) /
                 pl.col('returns').rolling_std(window_size=20)).alias('returns_sharpe')
            ])

            # Momentum features
            df = df.with_columns([
                (pl.col('price') / pl.col('price').shift(5) - 1).alias('momentum_5d'),
                (pl.col('price') / pl.col('price').shift(20) - 1).alias('momentum_20d'),
                (pl.col('price') / pl.col('price').shift(50) - 1).alias('momentum_50d')
            ])

            # Drawdown feature
            df = df.with_columns([
                (pl.col('price') / pl.col('price').rolling_max(window_size=50) - 1).alias('drawdown_50d')
            ])

            # Drop rows with NaN values
            df = df.drop_nulls()

            logger.info(
                f"Generated {len(df.columns)} features from market data: "
                f"{len(df)} valid samples"
            )

            return df

        except Exception as e:
            logger.error(f"Error updating features: {e}", exc_info=True)
            raise

    async def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk

        Args:
            filepath: Path to save model file
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("Cannot save model: model not trained")
                raise ValueError("Model must be trained before saving")

            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'config': {
                    'max_position_size_pct': float(self.max_position_size_pct),
                    'min_position_size_usd': float(self.min_position_size_usd),
                    'max_position_size_usd': float(self.max_position_size_usd)
                }
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            raise

    async def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk

        Args:
            filepath: Path to model file
        """
        try:
            if not Path(filepath).exists():
                logger.error(f"Model file not found: {filepath}")
                raise FileNotFoundError(f"Model file not found: {filepath}")

            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.is_trained = True

            logger.info(
                f"Model loaded from {filepath}: "
                f"type={self.model_type}, features={len(self.feature_names)}"
            )

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
