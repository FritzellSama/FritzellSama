"""RL agent evaluation."""
from decimal import Decimal
from typing import Dict, List
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)

class RLEvaluator:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.n_episodes = config.get("n_episodes", 10)
        
    async def evaluate(self, agent, env) -> Dict[str, Decimal]:
        total_reward = Decimal("0")
        
        for episode in range(self.n_episodes):
            state = env.reset()
            done = False
            episode_reward = Decimal("0")
            
            while not done:
                action = agent.select_action(state, add_noise=False)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
                
            total_reward += episode_reward
            
        avg_reward = total_reward / self.n_episodes
        logger.info("RL evaluation complete", avg_reward=str(avg_reward))
        
        return {
            "avg_reward": avg_reward,
            "n_episodes": Decimal(str(self.n_episodes))
        }
