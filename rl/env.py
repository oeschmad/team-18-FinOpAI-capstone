import numpy as np
import gym
from gym import spaces
class AssetAllocationEnv(gym.Env):
    def __init__(self, asset_data, asset_volatility, risk_score=1.0):
        super(AssetAllocationEnv, self).__init__()
        self.asset_data = asset_data
        self.asset_volatility = np.array(asset_volatility)  # Use provided volatility
        self.risk_score = risk_score
        self.num_assets = asset_data.shape[1]
        self.current_step = 0
        # Observation space: current returns
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_assets,), dtype=np.float32
        )
        # Action space: weights across all assets must sum to 1
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_assets,), dtype=np.float32
        )
    def reset(self):
        self.current_step = 0
        return self.asset_data.iloc[self.current_step].values
    def step(self, action):
        assert np.isclose(np.sum(action), 1.0, atol=1e-2), "Weights must sum to 1"
        current_returns = self.asset_data.iloc[self.current_step].values
        portfolio_return = np.dot(current_returns, action)
        # Risk-adjusted reward (penalize high variance assets based on user's risk tolerance)
        adjusted_return = portfolio_return - self.risk_score * np.dot(action, self.asset_volatility)
        self.current_step += 1
        done = self.current_step >= len(self.asset_data) - 1
        next_state = self.asset_data.iloc[self.current_step].values
        return next_state, adjusted_return, done, {}
    def render(self, mode='human'):
        pass







