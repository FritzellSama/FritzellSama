"""Time Series Transformer for Trading - PRODUCTION"""
from decimal import Decimal
from typing import Dict, Any
import os, logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        d_model = int(config.get('d_model', os.getenv('TRANSFORMER_D_MODEL', '512')))
        nhead = int(config.get('nhead', os.getenv('TRANSFORMER_NHEAD', '8')))
        num_layers = int(config.get('num_layers', os.getenv('TRANSFORMER_LAYERS', '6')))
        dim_feedforward = int(config.get('dim_feedforward', os.getenv('TRANSFORMER_FFN', '2048')))
        dropout = float(config.get('dropout', os.getenv('TRANSFORMER_DROPOUT', '0.1')))
        input_dim = int(config.get('input_dim', '200'))
        output_dim = int(config.get('output_dim', '3'))

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_dim)

        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.to(self.device)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.fc(x[:, -1, :])

    def save(self, path: str):
        torch.save({'model_state': self.state_dict(), 'config': self.config}, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state'])
