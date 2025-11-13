"""TD3 Agent for Trading - PRODUCTION RL"""
from decimal import Decimal
from typing import Dict, Any, Tuple
from datetime import datetime
import os, logging
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Normal
except ImportError:
    raise ImportError("PyTorch required")

logger = logging.getLogger(__name__)

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std

class TD3Agent:
    """Twin Delayed DDPG for Trading"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.state_dim = int(config.get('state_dim', os.getenv('TD3_STATE_DIM', '200')))
        self.action_dim = int(config.get('action_dim', os.getenv('TD3_ACTION_DIM', '1')))
        self.hidden_dim = int(config.get('hidden_dim', os.getenv('TD3_HIDDEN_DIM', '256')))
        
        self.gamma = float(config.get('gamma', os.getenv('TD3_GAMMA', '0.99')))
        self.tau = float(config.get('tau', os.getenv('TD3_TAU', '0.005')))
        self.alpha = float(config.get('alpha', os.getenv('TD3_ALPHA', '0.2')))
        self.lr = float(config.get('learning_rate', os.getenv('TD3_LR', '3e-4')))
        
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Networks
        self.policy = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.q1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.q2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.q1_target = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.q2_target = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.lr)
        
        self.logger.info(f"TD3Agent initialized")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std = self.policy(state_tensor)
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = Normal(mean, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)
        
        return action.cpu().numpy()[0]
    
    def save(self, path: str) -> None:
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'config': self.config
        }, path)
        self.logger.info(f"Agent saved to {path}")
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.logger.info(f"Agent loaded from {path}")
