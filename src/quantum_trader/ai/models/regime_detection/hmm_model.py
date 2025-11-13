"""Hidden Markov Model."""
import logging
import numpy as np
from hmmlearn import hmm

logger = logging.getLogger(__name__)

class HMMModel:
    def __init__(self, n_states: int = 3):
        self.model = hmm.GaussianHMM(n_components=n_states, random_state=42)
    def fit(self, X: np.ndarray):
        self.model.fit(X)
        logger.info(f"Fitted HMM with {self.model.n_components} states")
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
