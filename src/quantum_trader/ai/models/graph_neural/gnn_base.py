"""
Graph Neural Network Base for Market Relationship Modeling

CRITICAL: GNN for modeling asset relationships and market structure
"""

from decimal import Decimal
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import logging
from abc import ABC, abstractmethod
import numpy as np
import polars as pl

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError(f"PyTorch not installed: {e}")

logger = logging.getLogger(__name__)


class BaseMLModel(ABC):
    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass


class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: node features [num_nodes, in_features]
        # adj: adjacency matrix [num_nodes, num_nodes]
        support = self.linear(x)
        output = torch.mm(adj, support)
        return output


class GNNBase(BaseMLModel, nn.Module):
    """Base GNN Model for Trading"""
    
    def __init__(self, config: Dict) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration
        self.input_dim = int(config.get('input_dim', os.getenv('GNN_INPUT_DIM', '64')))
        self.hidden_dim = int(config.get('hidden_dim', os.getenv('GNN_HIDDEN_DIM', '128')))
        self.output_dim = int(config.get('output_dim', os.getenv('GNN_OUTPUT_DIM', '3')))
        self.n_layers = int(config.get('n_layers', os.getenv('GNN_N_LAYERS', '3')))
        self.dropout = float(config.get('dropout', os.getenv('GNN_DROPOUT', '0.2')))
        
        # Build layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GraphConvLayer(self.input_dim, self.hidden_dim))
        for _ in range(self.n_layers - 2):
            self.conv_layers.append(GraphConvLayer(self.hidden_dim, self.hidden_dim))
        self.conv_layers.append(GraphConvLayer(self.hidden_dim, self.output_dim))
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Training config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.learning_rate = float(config.get('learning_rate', os.getenv('GNN_LR', '0.001')))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        self.to(self.device)
        self.is_trained = False
        
        self.logger.info(f"GNN initialized: {self.n_layers} layers, hidden_dim={self.hidden_dim}")
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN"""
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x, adj)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Final layer (no activation)
        x = self.conv_layers[-1](x, adj)
        return x
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train GNN model"""
        try:
            self.logger.info(f"Training GNN on {features.shape[0]} samples")
            
            # Convert to tensors
            # features should be [node_features, adjacency_matrix]
            # For simplicity, assume features contains both
            x_tensor = torch.FloatTensor(features).to(self.device)
            y_tensor = torch.LongTensor(labels).to(self.device)
            
            # Simple training loop (in production, use proper batching)
            n_epochs = int(self.config.get('n_epochs', os.getenv('GNN_N_EPOCHS', '100')))
            
            for epoch in range(n_epochs):
                self.train_mode = True
                self.optimizer.zero_grad()
                
                # Forward (simplified - assumes adj matrix is identity for now)
                adj = torch.eye(x_tensor.shape[0]).to(self.device)
                output = self.forward(x_tensor, adj)
                
                # Loss
                loss = F.cross_entropy(output, y_tensor)
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                if epoch % max(1, n_epochs // 10) == 0:
                    self.logger.debug(f"Epoch {epoch}/{n_epochs}: loss={loss.item():.4f}")
            
            self.is_trained = True
            self.logger.info("GNN training completed")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise RuntimeError(f"GNN training error: {e}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        try:
            if not self.is_trained:
                raise RuntimeError("Model not trained")
            
            self.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(features).to(self.device)
                adj = torch.eye(x_tensor.shape[0]).to(self.device)
                output = self.forward(x_tensor, adj)
                
                # Softmax for probabilities
                probs = F.softmax(output, dim=1)
                return probs.cpu().numpy()
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction error: {e}")
    
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model"""
        try:
            predictions = self.predict(features)
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = labels.ravel() if labels.ndim > 1 else labels
            
            accuracy = float(np.mean(pred_classes == true_classes))
            
            return {
                'accuracy': accuracy,
                'n_samples': len(labels)
            }
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {'error': str(e)}
    
    def save(self, path: str) -> None:
        """Save model"""
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'is_trained': self.is_trained
            }, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            raise IOError(f"Model save error: {e}")
    
    def load(self, path: str) -> None:
        """Load model"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.config = checkpoint['config']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.is_trained = checkpoint['is_trained']
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Load failed: {e}")
            raise IOError(f"Model load error: {e}")
