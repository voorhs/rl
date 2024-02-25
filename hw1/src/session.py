from multiprocessing import Pool
from functools import partial
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
from joblib import Parallel, delayed
from .agent import Agent


def session(i, env_name, agent: Agent, sigma=None, t_max=1000):
    """
    Play a single game using agent neural network.
    Terminate when game finishes or after :t_max: steps
    """
    env: Env = gym.make(env_name, render_mode="rgb_array").env

    action_discrete = isinstance(env.action_space, Discrete)
    if action_discrete:
        n_actions = env.action_space.n
    else:
        n_actions = env.action_space.shape[0]
    
    states, actions = [], []
    total_reward = 0

    s, _ = env.reset()

    for t in range(t_max):

        # use agent to predict a vector of action probabilities for state :s:
        probs = agent.predict([s]).flatten()

        assert probs.shape == (n_actions,), "make sure probabilities are a vector (hint: np.reshape)"

        # use the probabilities you predicted to pick an action
        # sample proportionally to the probabilities, don't just take the most likely action
        if not action_discrete:
            a = probs + np.random.normal(scale=sigma) 
        else:
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


def generate_sessions(n_sessions, env_name, agent: Agent, sigma=None, mp=True, t_max=1000):
    func = partial(session, env_name=env_name, agent=agent, t_max=t_max, sigma=sigma)
    if mp:
        return Parallel(n_jobs=4)(delayed(func)(i) for i in range(n_sessions))
    return map(func, list(range(n_sessions)))
