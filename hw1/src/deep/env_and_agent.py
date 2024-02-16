import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
from sklearn.neural_network import MLPClassifier
import numpy as np
from ..agent import Agent


class DeepCEAgent(Agent, MLPClassifier):
    def update(self, elite_states, elite_actions):
        return self.partial_fit(elite_states, elite_actions)
    
    def predict(self, states):
        return self.predict_proba(states)


def get_env_and_agent(env_name):
    env: Env = gym.make(env_name, render_mode="rgb_array").env

    env.reset()

    action_dicrete = isinstance(env.action_space, Discrete)
    state_dicrete = isinstance(env.observation_space, Discrete)

    agent = DeepCEAgent(
        hidden_layer_sizes=(20, 20),
        activation="tanh",
    )

    if state_dicrete:
        tmp = np.zeros(env.observation_space.n, dtype=np.int_)
        tmp[0] = 1
        X_example = [tmp]
    else:
        X_example = [np.random.randn(env.observation_space.shape[0])]

    if action_dicrete:
        y_example = [0]
        classes = range(env.action_space.n)
    else:
        y_example = [np.random.randn()]
        classes = None

    agent.partial_fit(X_example, y_example, classes)

    return env, agent
