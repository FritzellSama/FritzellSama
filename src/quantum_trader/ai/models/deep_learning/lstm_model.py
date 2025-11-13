"""LSTM Model for Time Series Prediction - CRITICAL PRODUCTION CODE"""
from decimal import Decimal
from typing import Dict, Any, Optional
from datetime import datetime
import os, logging
from abc import ABC, abstractmethod
import numpy as np
import polars as pl

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(f"PyTorch required: {e}")

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

class LSTMModel(BaseMLModel, nn.Module):
    """Production LSTM for Trading with Attention"""

    def __init__(self, config: Dict) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.input_size = int(config.get('input_size', os.getenv('LSTM_INPUT_SIZE', '200')))
        self.hidden_size = int(config.get('hidden_size', os.getenv('LSTM_HIDDEN_SIZE', '256')))
        self.num_layers = int(config.get('num_layers', os.getenv('LSTM_NUM_LAYERS', '3')))
        self.dropout = float(config.get('dropout', os.getenv('LSTM_DROPOUT', '0.2')))
        self.output_size = int(config.get('output_size', os.getenv('LSTM_OUTPUT_SIZE', '3')))

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=False
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.output_size)
        )

        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.lr = float(config.get('learning_rate', os.getenv('LSTM_LR', '0.001')))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)
        self.is_trained = False

        self.logger.info(f"LSTM initialized: {self.num_layers} layers, hidden={self.hidden_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last hidden state
        final_hidden = attn_out[:, -1, :]

        # Output
        output = self.fc(final_hidden)
        return output

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        try:
            self.logger.info(f"Training LSTM on {features.shape} samples")

            X = torch.FloatTensor(features).to(self.device)
            y = torch.LongTensor(labels.ravel()).to(self.device)

            n_epochs = int(self.config.get('n_epochs', os.getenv('LSTM_N_EPOCHS', '100')))
            batch_size = int(self.config.get('batch_size', os.getenv('LSTM_BATCH_SIZE', '32')))

            for epoch in range(n_epochs):
                self.train_mode = True
                total_loss = 0

                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    batch_y = y[i:i+batch_size]

                    self.optimizer.zero_grad()
                    outputs = self.forward(batch_X)
                    loss = nn.functional.cross_entropy(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    total_loss += loss.item()

                if epoch % max(1, n_epochs // 10) == 0:
                    avg_loss = total_loss / (len(X) // batch_size + 1)
                    self.logger.debug(f"Epoch {epoch}/{n_epochs}: loss={avg_loss:.4f}")

            self.is_trained = True
            self.logger.info("LSTM training completed")

        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise RuntimeError(f"LSTM training error: {e}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        try:
            if not self.is_trained:
                raise RuntimeError("Model not trained")

            self.eval()
            with torch.no_grad():
                X = torch.FloatTensor(features).to(self.device)
                outputs = self.forward(X)
                probs = nn.functional.softmax(outputs, dim=1)
                return probs.cpu().numpy()
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction error: {e}")

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        try:
            predictions = self.predict(features)
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = labels.ravel()
            accuracy = float(np.mean(pred_classes == true_classes))
            return {'accuracy': accuracy, 'n_samples': len(labels)}
        except Exception as e:
            return {'error': str(e)}

    def save(self, path: str) -> None:
        try:
            torch.save({
                'model_state': self.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'config': self.config,
                'is_trained': self.is_trained
            }, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            raise IOError(f"Save error: {e}")

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.is_trained = checkpoint['is_trained']
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            raise IOError(f"Load error: {e}")
