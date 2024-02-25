import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
from ..agent import Agent


class DeepCEAgent(Agent):
    def __init__(self, network, action_space, state_space, noise):
        self.model = network
        self.action_space = action_space
        self.state_space = state_space
        self.noise = noise
        
        self.init()
    
    def init(self):
        self.action_discrete = isinstance(self.action_space, Discrete)
        self.state_discrete = isinstance(self.state_space, Discrete)

        if self.state_discrete:
            tmp = np.zeros(self.state_space.n, dtype=np.int_)
            tmp[0] = 1
            X_example = [tmp]
        else:
            X_example = [np.random.randn(self.state_space.shape[0])]

        if self.action_discrete:
            y_example = [0]
            classes = range(self.action_space.n)
            self.model.partial_fit(X_example, y_example, classes)
        else:
            y_example = [np.random.randn(*self.action_space.shape)]
            self.model.partial_fit(X_example, y_example)
        

    def update(self, elite_states, elite_actions):
        elite_states = self.prepare_inputs(elite_states)
        return self.model.partial_fit(elite_states, elite_actions)
    
    def predict(self, states):
        if self.noise:
            self.add_noise()
        states = self.prepare_inputs(states)
        if self.action_discrete:
            return self.model.predict_proba(states)
        return self.model.predict(states)

    def prepare_inputs(self, inputs):
        inputs = np.array(inputs)
        if self.state_discrete:
            one_hot_inputs = np.zeros((len(inputs), self.state_space.n))
            one_hot_inputs[np.arange(len(inputs)), inputs] = 1
            return one_hot_inputs
        return inputs
    
    def add_noise(self):
        for w in self.model.coefs_:
            w += np.random.normal(scale=.3, size=w.shape)
        
            

def get_env_and_agent(env_name, network: MLPClassifier, noise=False):
    env: Env = gym.make(env_name, render_mode="rgb_array").env

    env.reset()

    agent = DeepCEAgent(network, env.action_space, env.observation_space, noise)

    return env, agent
