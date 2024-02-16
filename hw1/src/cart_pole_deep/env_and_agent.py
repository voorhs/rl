import gymnasium as gym
from sklearn.neural_network import MLPClassifier


def get_env_and_agent():
    env = gym.make("CartPole-v0", render_mode="rgb_array").env

    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    agent = MLPClassifier(
        hidden_layer_sizes=(20, 20),
        activation="tanh",
    )

    X_example = [env.reset()[0]] * n_actions
    y_example = list(range(n_actions))
    classes = list(range(n_actions))

    agent.partial_fit(X_example, y_example, classes)

    return env, agent
