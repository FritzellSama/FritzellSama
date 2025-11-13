"""
ML Model Training and Management System.

This module provides comprehensive model training, validation, and management
for the Quantum Trader AI trading platform.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
import os

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class ModelTrainer:
    """Production-ready ML model trainer with comprehensive training pipeline.

    Attributes:
        config: Configuration dictionary from config files
        model: ML model instance to train
        device: PyTorch device (cuda/cpu)
        optimizer: PyTorch optimizer
        criterion: Loss function
        best_loss: Best validation loss achieved
        patience_counter: Early stopping counter
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize model trainer.

        Args:
            config: Configuration dictionary containing:
                - training.batch_size
                - training.learning_rate
                - training.epochs
                - training.early_stopping_patience
                - training.validation_split
                - training.checkpoint_dir
                - training.device
        """
        self.config = config
        self._validate_config()

        self.batch_size = self.config["training"]["batch_size"]
        self.learning_rate = Decimal(str(self.config["training"]["learning_rate"]))
        self.epochs = self.config["training"]["epochs"]
        self.patience = self.config["training"]["early_stopping_patience"]
        self.validation_split = Decimal(str(self.config["training"]["validation_split"]))
        self.checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])
        self.device = torch.device(self.config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.best_loss = Decimal("inf")
        self.patience_counter = 0

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("model_trainer_initialized",
                   device=str(self.device),
                   batch_size=self.batch_size,
                   learning_rate=str(self.learning_rate))

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        required_keys = [
            "training.batch_size",
            "training.learning_rate",
            "training.epochs",
            "training.early_stopping_patience",
            "training.validation_split",
            "training.checkpoint_dir"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

        if self.config["training"]["batch_size"] <= 0:
            raise ValueError("batch_size must be positive")

        if self.config["training"]["epochs"] <= 0:
            raise ValueError("epochs must be positive")

    def initialize_model(self, model: nn.Module) -> None:
        """Initialize model for training.

        Args:
            model: PyTorch model to train
        """
        self.model = model.to(self.device)

        optimizer_type = self.config["training"].get("optimizer", "adam")
        lr_float = float(self.learning_rate)

        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr_float,
                weight_decay=float(self.config["training"].get("weight_decay", 0.0001))
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr_float,
                momentum=float(self.config["training"].get("momentum", 0.9))
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        criterion_type = self.config["training"].get("criterion", "mse")
        if criterion_type.lower() == "mse":
            self.criterion = nn.MSELoss()
        elif criterion_type.lower() == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_type.lower() == "bce":
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_type}")

        logger.info("model_initialized",
                   optimizer=optimizer_type,
                   criterion=criterion_type,
                   parameters=sum(p.numel() for p in self.model.parameters()))

    def prepare_data(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders.

        Args:
            features: Feature array (samples, features)
            labels: Label array (samples,)

        Returns:
            Tuple of (train_loader, val_loader)

        Raises:
            ValueError: If data shapes are invalid
        """
        if len(features) != len(labels):
            raise ValueError(f"Feature and label lengths must match: {len(features)} != {len(labels)}")

        if len(features) == 0:
            raise ValueError("Cannot train on empty dataset")

        # Split data
        val_size = float(self.validation_split)
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=val_size,
            random_state=self.config["training"].get("random_seed", 42),
            shuffle=True
        )

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)

        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config["training"].get("num_workers", 4),
            pin_memory=True if self.device.type == "cuda" else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config["training"].get("num_workers", 4),
            pin_memory=True if self.device.type == "cuda" else False
        )

        logger.info("data_prepared",
                   train_samples=len(train_dataset),
                   val_samples=len(val_dataset),
                   feature_dim=features.shape[1])

        return train_loader, val_loader

    async def train_epoch(self, train_loader: DataLoader) -> Decimal:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss for epoch
        """
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise RuntimeError("Model not initialized. Call initialize_model first.")

        self.model.train()
        total_loss = Decimal("0")
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                data = data.to(self.device)
                target = target.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()

                # Gradient clipping
                max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.optimizer.step()

                total_loss += Decimal(str(loss.item()))
                num_batches += 1

                if batch_idx % self.config["training"].get("log_interval", 100) == 0:
                    logger.debug("training_batch",
                               batch=batch_idx,
                               loss=str(loss.item()))

            except Exception as e:
                logger.error("training_batch_failed",
                           batch=batch_idx,
                           error=str(e))
                raise

        avg_loss = total_loss / Decimal(str(num_batches)) if num_batches > 0 else Decimal("0")
        return avg_loss

    async def validate_epoch(self, val_loader: DataLoader) -> Decimal:
        """Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        if self.model is None or self.criterion is None:
            raise RuntimeError("Model not initialized. Call initialize_model first.")

        self.model.eval()
        total_loss = Decimal("0")
        num_batches = 0

        with torch.no_grad():
            for data, target in val_loader:
                try:
                    data = data.to(self.device)
                    target = target.to(self.device)

                    output = self.model(data)
                    loss = self.criterion(output, target)

                    total_loss += Decimal(str(loss.item()))
                    num_batches += 1

                except Exception as e:
                    logger.error("validation_batch_failed", error=str(e))
                    raise

        avg_loss = total_loss / Decimal(str(num_batches)) if num_batches > 0 else Decimal("0")
        return avg_loss

    async def train(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Complete training pipeline.

        Args:
            features: Feature array
            labels: Label array

        Returns:
            Training history dictionary with losses and metrics

        Raises:
            RuntimeError: If model not initialized
            ValueError: If data is invalid
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model first.")

        logger.info("training_started",
                   samples=len(features),
                   features=features.shape[1],
                   epochs=self.epochs)

        # Prepare data
        train_loader, val_loader = self.prepare_data(features, labels)

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "epochs_completed": 0
        }

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=float(self.config["training"].get("lr_reduce_factor", 0.5)),
            patience=self.config["training"].get("lr_patience", 5),
            verbose=True
        )

        for epoch in range(self.epochs):
            try:
                # Train epoch
                train_loss = await self.train_epoch(train_loader)

                # Validate epoch
                val_loss = await self.validate_epoch(val_loader)

                # Update scheduler
                scheduler.step(float(val_loss))

                # Record history
                history["train_loss"].append(str(train_loss))
                history["val_loss"].append(str(val_loss))
                history["learning_rates"].append(str(self.optimizer.param_groups[0]['lr']))
                history["epochs_completed"] = epoch + 1

                logger.info("epoch_completed",
                          epoch=epoch + 1,
                          train_loss=str(train_loss),
                          val_loss=str(val_loss))

                # Early stopping check
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter = 0

                    # Save best model
                    await self.save_checkpoint(epoch, val_loss, is_best=True)
                else:
                    self.patience_counter += 1

                    if self.patience_counter >= self.patience:
                        logger.info("early_stopping_triggered",
                                  epoch=epoch + 1,
                                  patience=self.patience)
                        break

                # Regular checkpoint
                if (epoch + 1) % self.config["training"].get("checkpoint_interval", 10) == 0:
                    await self.save_checkpoint(epoch, val_loss, is_best=False)

            except Exception as e:
                logger.error("epoch_failed", epoch=epoch, error=str(e))
                raise

        logger.info("training_completed",
                   epochs=history["epochs_completed"],
                   best_loss=str(self.best_loss))

        return history

    async def save_checkpoint(
        self,
        epoch: int,
        val_loss: Decimal,
        is_best: bool = False
    ) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model not initialized")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': str(val_loss),
            'best_loss': str(self.best_loss),
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info("best_model_saved", epoch=epoch, val_loss=str(val_loss))

    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.best_loss = Decimal(checkpoint['best_loss'])

        logger.info("checkpoint_loaded",
                   path=checkpoint_path,
                   epoch=checkpoint['epoch'])

    def evaluate_model(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Decimal]:
        """Evaluate model performance.

        Args:
            features: Feature array
            labels: Label array

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(features).to(self.device)
            y_tensor = torch.FloatTensor(labels).to(self.device)

            predictions = self.model(X_tensor)

            # Calculate metrics
            mse = nn.MSELoss()(predictions, y_tensor)
            mae = nn.L1Loss()(predictions, y_tensor)

            metrics = {
                "mse": Decimal(str(mse.item())),
                "mae": Decimal(str(mae.item())),
                "rmse": Decimal(str(np.sqrt(mse.item())))
            }

        logger.info("model_evaluated", metrics={k: str(v) for k, v in metrics.items()})

        return metrics
