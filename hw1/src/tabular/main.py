from .. import train
from .env_and_agent import get_env_and_agent
from ..session import generate_sessions


def experiment(env_name, n_iter, n_sessions, percentile, lr, mean_reward_to_win=None):
    env, agent = get_env_and_agent(env_name, lr)
    return train(n_iter, n_sessions, percentile, env, agent, generate_sessions, mean_reward_to_win)
