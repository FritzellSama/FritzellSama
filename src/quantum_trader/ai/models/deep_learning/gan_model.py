"""GAN Model for Data Generation."""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, output_dim))
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
