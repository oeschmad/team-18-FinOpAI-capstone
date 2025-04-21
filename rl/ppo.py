import numpy as np
class PPOAgent:
    def __init__(self, env, lr=0.01, gamma=0.99):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.n_assets = env.num_assets
    def choose_action(self, state):
        # Sample a random allocation (can be replaced with a neural network policy)
        action = np.random.dirichlet(np.ones(self.n_assets))
        return action
    def train(self, episodes=100):
        history = []
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            history.append(total_reward)
        return history






