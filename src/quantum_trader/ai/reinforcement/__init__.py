"""Reinforcement learning module.

This module provides reinforcement learning implementations for trading agents,
including policy gradient methods, Q-learning variants, actor-critic methods,
and multi-agent reinforcement learning.

Attributes:
    __all__: Public API exports for the reinforcement learning module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agents import RLAgent, DQNAgent, PPOAgent, A3CAgent
    from .environments import TradingEnvironment
    from .policies import Policy
    from .value_functions import ValueFunction

__all__ = [
    "RLAgent",
    "DQNAgent",
    "PPOAgent",
    "A3CAgent",
    "TradingEnvironment",
    "Policy",
    "ValueFunction",
    "ReplayBuffer",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader AI"


def __getattr__(name: str):
    """Lazy import for reinforcement learning components."""
    if name == "RLAgent":
        from .agents import RLAgent
        return RLAgent
    elif name == "DQNAgent":
        from .agents import DQNAgent
        return DQNAgent
    elif name == "PPOAgent":
        from .agents import PPOAgent
        return PPOAgent
    elif name == "A3CAgent":
        from .agents import A3CAgent
        return A3CAgent
    elif name == "TradingEnvironment":
        from .environments import TradingEnvironment
        return TradingEnvironment
    elif name == "Policy":
        from .policies import Policy
        return Policy
    elif name == "ValueFunction":
        from .value_functions import ValueFunction
        return ValueFunction
    elif name == "ReplayBuffer":
        from .replay import ReplayBuffer
        return ReplayBuffer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
