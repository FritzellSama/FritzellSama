"""Reinforcement learning agents module.

This module provides various RL agent implementations including DQN, PPO, A3C,
DDPG, and multi-agent variants for trading applications.

Attributes:
    __all__: Public API exports for the RL agents module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import RLAgent
    from .dqn import DQNAgent
    from .ppo import PPOAgent
    from .a3c import A3CAgent
    from .ddpg import DDPGAgent
    from .multi_agent import MultiAgentTrader

__all__ = [
    "RLAgent",
    "DQNAgent",
    "PPOAgent",
    "A3CAgent",
    "DDPGAgent",
    "MultiAgentTrader",
    "AgentFactory",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for RL agent implementations."""
    if name == "RLAgent":
        from .base import RLAgent
        return RLAgent
    elif name == "DQNAgent":
        from .dqn import DQNAgent
        return DQNAgent
    elif name == "PPOAgent":
        from .ppo import PPOAgent
        return PPOAgent
    elif name == "A3CAgent":
        from .a3c import A3CAgent
        return A3CAgent
    elif name == "DDPGAgent":
        from .ddpg import DDPGAgent
        return DDPGAgent
    elif name == "MultiAgentTrader":
        from .multi_agent import MultiAgentTrader
        return MultiAgentTrader
    elif name == "AgentFactory":
        from .factory import AgentFactory
        return AgentFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
