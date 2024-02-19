import numpy as np
from gymnasium import Env
from .utils import show_progress
from .agent import Agent


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you are confused, see examples below. Please don't assume that states are integers
    (they will become different later).
    """

    is_elite_indicators = (rewards_batch > np.percentile(rewards_batch, q=percentile))
    elite_states, elite_actions = [], []
    for states, actions, is_elite in zip(states_batch, actions_batch, is_elite_indicators):
        if not is_elite:
            continue
        elite_states.extend(states)
        elite_actions.extend(actions)

    return elite_states, elite_actions


def train(n_iter, n_sessions, percentile, env: Env, agent: Agent, generate_session, mean_reward_to_win):
    log = []
    for i in range(n_iter):
        states_batch, actions_batch, rewards_batch = [], [], []
        for _ in range(n_sessions):
            states, actions, total_reward = generate_session(env, agent)
            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(total_reward)

        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)

        agent.update(elite_states, elite_actions)

        show_progress(rewards_batch, log, percentile, reward_range=[0, np.max(rewards_batch)])

        if np.mean(rewards_batch) > mean_reward_to_win:
            print("You Win! You may stop training now via KeyboardInterrupt.")
