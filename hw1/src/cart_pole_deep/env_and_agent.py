import gymnasium as gym
from gymnasium import Env
from sklearn.neural_network import MLPClassifier
from ..agent import Agent


class DeepCEAgent(Agent, MLPClassifier):
    def update(self, elite_states, elite_actions):
        return self.partial_fit(elite_states, elite_actions)
    
    def predict(self, states):
        return self.predict_proba(states)


def get_env_and_agent():
    env: Env = gym.make("CartPole-v0", render_mode="rgb_array").env

    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    agent = DeepCEAgent(
        hidden_layer_sizes=(20, 20),
        activation="tanh",
    )

    X_example = [env.reset()[0]] * n_actions
    y_example = list(range(n_actions))
    classes = list(range(n_actions))

    agent.partial_fit(X_example, y_example, classes)

    return env, agent
