from functools import partial
from .. import train
from .env_and_agent import get_env_and_agent
from ..session import generate_sessions


def experiment(env_name, n_iter, n_sessions, percentile, lr, n_batches_reuse=1, mean_reward_to_win=None, mp=True, t_max=int(1e4)):
    env, agent = get_env_and_agent(env_name, lr)
    gen = partial(generate_sessions, mp=mp, t_max=t_max)
    return train(n_iter, n_sessions, percentile, env, agent, gen, n_batches_reuse, mean_reward_to_win)
