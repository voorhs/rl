from src import train
from src.cart_pole_deep import generate_session, get_env_and_agent

def main(n_iter, n_sessions, percentile):
    env, agent = get_env_and_agent()
    train(n_iter, n_sessions, percentile, env, agent)
