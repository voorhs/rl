import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
import numpy as np
from ..agent import Agent


class TabularCEAgent(Agent):
    def __init__(self, action_space, state_space, lr):
        self.action_space = action_space
        self.state_space = state_space
        self.lr = lr
                
        self.init()
    
    def init(self):
        self.action_discrete = isinstance(self.action_space, Discrete)
        self.state_discrete = isinstance(self.state_space, Discrete)

        if not (self.state_discrete and self.action_discrete):
            raise ValueError('tabular agent requires discrete action and observation space')
        
        self.policy = np.ones((self.state_space.n, self.action_space.n)) / self.action_space.n

    def update(self, elite_states, elite_actions):
        new_policy = np.zeros([self.state_space.n, self.action_space.n])
        for s, a in zip(elite_states, elite_actions):
            new_policy[s, a] += 1
        new_policy[np.where(new_policy.sum(axis=1) == 0)] = 1
        new_policy = new_policy / new_policy.sum(axis=1, keepdims=True)

        self.policy = self.lr * new_policy + (1 - self.lr) * self.policy
    
    def predict(self, states):
        return np.array([self.policy[s] for s in states])
            

def get_env_and_agent(env_name, lr):
    env: Env = gym.make(env_name, render_mode="rgb_array").env

    env.reset()

    agent = TabularCEAgent(env.action_space, env.observation_space, lr)

    return env, agent

