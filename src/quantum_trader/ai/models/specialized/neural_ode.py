"""
Neural Ordinary Differential Equations for Time Series Prediction.

This module implements Neural ODEs for continuous-time modeling of
financial time series data.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch import Tensor
from structlog import get_logger

logger = get_logger(__name__)


class ODEFunc(nn.Module):
    """ODE function network.

    Neural network that defines the derivative function for the ODE solver.

    Attributes:
        net: Neural network layers
        nonlinearity: Activation function
    """

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        """Initialize ODE function.

        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super(ODEFunc, self).__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                layers.append(nn.Tanh())
                layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*layers)
        self.nonlinearity = nn.Tanh()

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Forward pass of ODE function.

        Args:
            t: Time tensor
            x: State tensor

        Returns:
            Derivative dx/dt
        """
        return self.net(x)


class ODESolver:
    """Numerical ODE solver using Euler or RK4 methods.

    Attributes:
        method: Integration method ('euler' or 'rk4')
        step_size: Integration step size
    """

    def __init__(self, method: str = "rk4", step_size: float = 0.1) -> None:
        """Initialize ODE solver.

        Args:
            method: Integration method ('euler' or 'rk4')
            step_size: Integration step size

        Raises:
            ValueError: If method is not supported
        """
        if method not in ["euler", "rk4"]:
            raise ValueError(f"Unsupported integration method: {method}")

        self.method = method
        self.step_size = step_size

        logger.debug("ode_solver_initialized",
                    method=method,
                    step_size=step_size)

    def integrate(
        self,
        func: ODEFunc,
        y0: Tensor,
        t: Tensor
    ) -> Tensor:
        """Integrate ODE from y0 over time points t.

        Args:
            func: ODE function
            y0: Initial state
            t: Time points

        Returns:
            Solution tensor
        """
        if self.method == "euler":
            return self._euler_integrate(func, y0, t)
        else:
            return self._rk4_integrate(func, y0, t)

    def _euler_integrate(
        self,
        func: ODEFunc,
        y0: Tensor,
        t: Tensor
    ) -> Tensor:
        """Euler integration method.

        Args:
            func: ODE function
            y0: Initial state
            t: Time points

        Returns:
            Solution tensor
        """
        y = y0
        ys = [y0]

        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            dy = func(t[i], y)
            y = y + dy * dt
            ys.append(y)

        return torch.stack(ys, dim=0)

    def _rk4_integrate(
        self,
        func: ODEFunc,
        y0: Tensor,
        t: Tensor
    ) -> Tensor:
        """Runge-Kutta 4th order integration.

        Args:
            func: ODE function
            y0: Initial state
            t: Time points

        Returns:
            Solution tensor
        """
        y = y0
        ys = [y0]

        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]

            k1 = func(t[i], y)
            k2 = func(t[i] + dt / 2, y + dt * k1 / 2)
            k3 = func(t[i] + dt / 2, y + dt * k2 / 2)
            k4 = func(t[i] + dt, y + dt * k3)

            y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            ys.append(y)

        return torch.stack(ys, dim=0)


class NeuralODE(nn.Module):
    """Neural ODE model for time series prediction.

    Implements continuous-time neural network using ODE dynamics
    for financial time series forecasting.

    Attributes:
        config: Configuration dictionary
        input_dim: Input feature dimension
        hidden_dim: Hidden state dimension
        output_dim: Output dimension
        encoder: Input encoder network
        ode_func: ODE function network
        decoder: Output decoder network
        solver: ODE solver
    """

    def __init__(
        self,
        config: Dict[str, Any],
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        """Initialize Neural ODE model.

        Args:
            config: Configuration dictionary containing:
                - models.neural_ode.num_layers
                - models.neural_ode.integration_method
                - models.neural_ode.step_size
                - models.neural_ode.time_steps
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            output_dim: Output prediction dimension
        """
        super(NeuralODE, self).__init__()

        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self._validate_config()

        model_config = self.config["models"]["neural_ode"]
        num_layers = model_config["num_layers"]
        integration_method = model_config["integration_method"]
        step_size = model_config["step_size"]
        self.time_steps = model_config["time_steps"]

        # Encoder: input -> hidden state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ODE function
        self.ode_func = ODEFunc(hidden_dim, num_layers)

        # Decoder: hidden state -> output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

        # ODE solver
        self.solver = ODESolver(method=integration_method, step_size=step_size)

        logger.info("neural_ode_initialized",
                   input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   output_dim=output_dim,
                   method=integration_method)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing
        """
        required_keys = [
            "models.neural_ode.num_layers",
            "models.neural_ode.integration_method",
            "models.neural_ode.step_size",
            "models.neural_ode.time_steps"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """Forward pass through Neural ODE.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            t: Time points (optional, default creates uniform time points)

        Returns:
            Predicted output tensor (batch_size, time_steps, output_dim)
        """
        batch_size = x.size(0)

        # Encode input to initial hidden state
        # Take last time step as initial condition
        x_last = x[:, -1, :]
        h0 = self.encoder(x_last)

        # Create time points if not provided
        if t is None:
            t = torch.linspace(0, 1, self.time_steps, device=x.device)

        # Solve ODE for each sample in batch
        outputs = []
        for i in range(batch_size):
            h_trajectory = self.solver.integrate(self.ode_func, h0[i], t)
            outputs.append(h_trajectory)

        h_batch = torch.stack(outputs, dim=0)

        # Decode hidden states to predictions
        predictions = self.decoder(h_batch)

        return predictions

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions on input data.

        Args:
            x: Input array (batch_size, seq_len, input_dim)

        Returns:
            Predictions array (batch_size, time_steps, output_dim)
        """
        self.eval()

        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            predictions = self.forward(x_tensor)
            return predictions.cpu().numpy()


class NeuralODETrader:
    """Production-ready Neural ODE trading model.

    Wrapper class for training and deploying Neural ODE models
    for trading applications.

    Attributes:
        config: Configuration dictionary
        model: Neural ODE model
        device: PyTorch device
        optimizer: Optimizer
        criterion: Loss function
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Neural ODE trader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._validate_config()

        self.device = torch.device(
            self.config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model: Optional[NeuralODE] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None

        logger.info("neural_ode_trader_initialized", device=str(self.device))

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config is missing
        """
        required_keys = [
            "models.neural_ode",
            "training"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

    def initialize_model(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        """Initialize Neural ODE model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension
        """
        self.model = NeuralODE(
            self.config,
            input_dim,
            hidden_dim,
            output_dim
        ).to(self.device)

        # Initialize optimizer
        lr = float(self.config["training"]["learning_rate"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=float(self.config["training"].get("weight_decay", 0.0001))
        )

        # Initialize loss function
        self.criterion = nn.MSELoss()

        logger.info("model_initialized",
                   input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   output_dim=output_dim,
                   parameters=sum(p.numel() for p in self.model.parameters()))

    async def train_step(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray
    ) -> Decimal:
        """Perform single training step.

        Args:
            x_batch: Input batch
            y_batch: Target batch

        Returns:
            Loss value

        Raises:
            RuntimeError: If model not initialized
        """
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise RuntimeError("Model not initialized. Call initialize_model first.")

        self.model.train()

        try:
            # Convert to tensors
            x_tensor = torch.FloatTensor(x_batch).to(self.device)
            y_tensor = torch.FloatTensor(y_batch).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x_tensor)

            # Calculate loss
            loss = self.criterion(predictions, y_tensor)

            # Backward pass
            loss.backward()

            # Gradient clipping
            max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer.step()

            return Decimal(str(loss.item()))

        except Exception as e:
            logger.error("training_step_failed", error=str(e))
            raise

    async def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            x: Input data

        Returns:
            Predictions

        Raises:
            RuntimeError: If model not initialized
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        try:
            predictions = self.model.predict(x)
            return predictions

        except Exception as e:
            logger.error("prediction_failed", error=str(e))
            raise

    def save_model(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Save path

        Raises:
            RuntimeError: If model not initialized
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)

        logger.info("model_saved", path=path)

    def load_model(self, path: str, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Load model from disk.

        Args:
            path: Model path
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension

        Raises:
            FileNotFoundError: If model file not found
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.initialize_model(input_dim, hidden_dim, output_dim)

        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        logger.info("model_loaded", path=path)
