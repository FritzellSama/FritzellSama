"""Continual Learning for Adaptive Models."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List
import polars as pl
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

@dataclass
class ContinualLearningConfig:
    """Configuration for continual learning."""
    memory_size: int = 1000
    rehearsal_batch_size: int = 32
    update_frequency: int = 100
    ewc_lambda: Decimal = Decimal("0.5")

class ExperienceReplayBuffer:
    """Buffer for experience replay."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []

    def add(self, experience: Dict):
        """Add experience to buffer."""
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample from buffer."""
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

class ContinualLearner:
    """Continual learning with experience replay."""

    def __init__(self, model: nn.Module, config: ContinualLearningConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.update_count = 0

        logger.info("Initialized continual learner", extra={"timestamp": datetime.now(timezone.utc).isoformat()})

    async def update(
        self,
        new_data_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, Decimal]:
        """Update model with new data and replay."""
        # Add new data to memory
        for row in new_data_df.iter_rows(named=True):
            features = [row[col] for col in feature_cols]
            label = row[label_col]
            self.memory.add({"features": features, "label": label})

        # Train on new data + replay
        total_loss = Decimal("0")
        num_batches = 0

        # Get replay samples
        replay_samples = self.memory.sample(self.config.rehearsal_batch_size)

        if replay_samples:
            X = torch.tensor([[s["features"] for s in replay_samples]], dtype=torch.float32, device=self.device).squeeze(0)
            y = torch.tensor([s["label"] for s in replay_samples], dtype=torch.long, device=self.device)

            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss = Decimal(str(loss.item()))
            num_batches = 1

        self.update_count += 1

        logger.info("Continual learning update", extra={"update_count": self.update_count, "timestamp": datetime.now(timezone.utc).isoformat()})

        return {"loss": total_loss, "updates": Decimal(str(self.update_count))}
