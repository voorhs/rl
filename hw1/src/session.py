from multiprocessing import Pool
from functools import partial
import numpy as np
from gymnasium import Env
from .agent import Agent


def session(i, env: Env, agent: Agent, t_max=1000):
    """
    Play a single game using agent neural network.
    Terminate when game finishes or after :t_max: steps
    """
    n_actions = env.action_space.n
    states, actions = [], []
    total_reward = 0

    s, _ = env.reset()

    for t in range(t_max):

        # use agent to predict a vector of action probabilities for state :s:
        probs = agent.predict([s]).flatten()

        assert probs.shape == (n_actions,), "make sure probabilities are a vector (hint: np.reshape)"

        # use the probabilities you predicted to pick an action
        # sample proportionally to the probabilities, don't just take the most likely action
        a = np.random.choice(n_actions, p=probs)

        new_s, r, terminated, truncated, _ = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if terminated or truncated:
            break
    return states, actions, total_reward


def generate_sessions(n_sessions, env: Env, agent: Agent, t_max=1000):
    func = partial(session, env=env, agent=agent, t_max=t_max)
    pool = Pool(processes=4)
    return pool.map(func, list(range(n_sessions)))
