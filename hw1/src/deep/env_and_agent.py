import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
from sklearn.neural_network import MLPClassifier
import numpy as np
from ..agent import Agent


class DeepCEAgent(Agent):
    def __init__(self, action_space, state_space):
        self.model = MLPClassifier(
            hidden_layer_sizes=(20, 20),
            activation="tanh",
        )
        self.action_space = action_space
        self.state_space = state_space
        
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
        else:
            y_example = [np.random.randn(self.action_space.shape[0])]
            classes = None

        self.model.partial_fit(X_example, y_example, classes)

    def update(self, elite_states, elite_actions):
        elite_states = self.prepare_inputs(elite_states)
        return self.model.partial_fit(elite_states, elite_actions)
    
    def predict(self, states):
        states = self.prepare_inputs(states)
        return self.model.predict_proba(states)

    def prepare_inputs(self, inputs):
        inputs = np.array(inputs)
        if self.state_discrete:
            one_hot_inputs = np.zeros((len(inputs), self.state_space.n))
            one_hot_inputs[np.arange(len(inputs)), inputs] = 1
            return one_hot_inputs
        return inputs
            

def get_env_and_agent(env_name):
    env: Env = gym.make(env_name, render_mode="rgb_array").env

    env.reset()

    agent = DeepCEAgent(env.action_space, env.observation_space)

    return env, agent
