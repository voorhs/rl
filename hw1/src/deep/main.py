from .. import train
from .env_and_agent import get_env_and_agent
from ..session import generate_session


def experiment(env_name, n_iter, n_sessions, percentile):
    env, agent = get_env_and_agent(env_name)
    train(n_iter, n_sessions, percentile, env, agent, generate_session)
