""" Environment wrappers. """
from collections import deque

import cv2
import numpy as np

import gym
import gym.spaces as spaces
from ale_py.env.gym import AtariEnv
from torch.utils.tensorboard import SummaryWriter

from env_batch import ParallelEnvBatch
cv2.ocl.setUseOpenCL(False)


class OldGym(gym.Wrapper):
    def __init__(self, env):
        # remove order enforcing wrapper
        super().__init__(env.unwrapped)

    def step(self, action):
        obs, rew, term, trank, info = self.env.step(action)
        return obs, rew, term or trank, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs


class EpisodicLife(gym.Wrapper):
    """ Sets done flag to true when agent dies. """

    def __init__(self, env):
        super(EpisodicLife, self).__init__(env)
        self.lives = 0
        self.real_done = True

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.real_done = done
        info["real_done"] = done
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            done = True
        self.lives = lives
        return obs, rew, done, info

    def reset(self, **kwargs):
        if self.real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireReset(gym.Wrapper):
    """ Makes fire action when reseting environment.

    Some environments are fixed until the agent makes the fire action,
    this wrapper makes this action so that the epsiode starts automatically.
    """

    def __init__(self, env):
        super(FireReset, self).__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        if len(action_meanings) < 3:
            raise ValueError(
                "env.unwrapped.get_action_meanings() must be of length >= 3"
                f"but is of length {len(action_meanings)}")
        if env.unwrapped.get_action_meanings()[1] != "FIRE":
            raise ValueError(
                "env.unwrapped.get_action_meanings() must have 'FIRE' "
                f"under index 1, but is {action_meanings}")

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


class StartWithRandomActions(gym.Wrapper):
    """ Makes random number of random actions at the beginning of each
    episode. """

    def __init__(self, env, max_random_actions=30):
        super(StartWithRandomActions, self).__init__(env)
        self.max_random_actions = max_random_actions
        self.real_done = True

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.real_done = info.get("real_done", True)
        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        if self.real_done:
            num_random_actions = np.random.randint(self.max_random_actions + 1)
            for _ in range(num_random_actions):
                obs, _, _, _ = self.env.step(self.env.action_space.sample())
            self.real_done = False
        return obs


class ImagePreprocessing(gym.ObservationWrapper):
    """ Preprocesses image-observations by possibly grayscaling and resizing. """

    def __init__(self, env, width=84, height=84, grayscale=True):
        super(ImagePreprocessing, self).__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        ospace = self.env.observation_space
        low, high, dtype = ospace.low.min(), ospace.high.max(), ospace.dtype
        if self.grayscale:
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                shape=(width, height),
                dtype=dtype,
            )
        else:
            obs_shape = (width, height) + self.observation_space.shape[2:]
            self.observation_space = spaces.Box(low=low, high=high,
                                                shape=obs_shape, dtype=dtype)

    def observation(self, observation):
        """ Performs image preprocessing. """
        if self.grayscale:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, (self.width, self.height),
                                 cv2.INTER_AREA)
        return observation


class MaxBetweenFrames(gym.ObservationWrapper):
    """ Takes maximum between two subsequent frames. """

    def __init__(self, env):
        if (isinstance(env.unwrapped, AtariEnv) and
                "NoFrameskip" not in env.spec.id):
            raise ValueError(
                "MaxBetweenFrames requires NoFrameskip in atari env id")
        super(MaxBetweenFrames, self).__init__(env)
        self.last_obs = None

    def observation(self, observation):
        obs = np.maximum(observation, self.last_obs)
        self.last_obs = observation
        return obs

    def reset(self, **kwargs):
        self.last_obs = self.env.reset()
        return self.last_obs


class QueueFrames(gym.ObservationWrapper):
    """ Queues specified number of frames together along new dimension. """

    def __init__(self, env, nframes, concat=False):
        super(QueueFrames, self).__init__(env)
        self.obs_queue = deque([], maxlen=nframes)
        self.concat = concat
        ospace = self.observation_space
        if self.concat:
            oshape = ospace.shape[:-1] + (ospace.shape[-1] * nframes,)
        else:
            oshape = ospace.shape + (nframes,)
        self.observation_space = spaces.Box(
            ospace.low.min(), ospace.high.max(), oshape, ospace.dtype)

    def observation(self, observation):
        self.obs_queue.append(observation)
        return (np.concatenate(self.obs_queue, -1) if self.concat
                else np.dstack(self.obs_queue))

    def reset(self, **kwargs):
        obs = self.env.reset()
        for _ in range(self.obs_queue.maxlen - 1):
            self.obs_queue.append(obs)
        return self.observation(obs)


class SkipFrames(gym.Wrapper):
    """ Performs the same action for several steps and returns the final result.
    """

    def __init__(self, env, nskip=4):
        super(SkipFrames, self).__init__(env)
        if (isinstance(env.unwrapped, AtariEnv) and
                "NoFrameskip" not in env.spec.id):
            raise ValueError("SkipFrames requires NoFrameskip in atari env id")
        self.nskip = nskip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.nskip):
            obs, rew, done, info = self.env.step(action)
            total_reward += rew
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipReward(gym.RewardWrapper):
    """ Modifes reward to be in {-1, 0, 1} by taking sign of it. """

    def reward(self, reward):
        return np.sign(reward)
    
class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height and normalization
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[1], old_shape[0]), dtype=np.float32)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0).astype(np.float32) / 255.0

class TensorboardSummaries(gym.Wrapper):
    """ Writes env summaries."""

    def __init__(self, env, prefix=None, running_mean_size=100, step_var=None):
        super(TensorboardSummaries, self).__init__(env)
        self.episode_counter = 0
        self.prefix = prefix or self.env.spec.id
        self.writer = SummaryWriter(f"logs/{self.prefix}")
        self.step_var = 0

        self.nenvs = getattr(self.env.unwrapped, "nenvs", 1)
        self.rewards = np.zeros(self.nenvs)
        self.had_ended_episodes = np.zeros(self.nenvs, dtype=bool)
        self.episode_lengths = np.zeros(self.nenvs)
        self.reward_queues = [deque([], maxlen=running_mean_size)
                              for _ in range(self.nenvs)]

    def should_write_summaries(self):
        """ Returns true if it's time to write summaries. """
        return np.all(self.had_ended_episodes)

    def add_summaries(self):
        """ Writes summaries. """
        self.writer.add_scalar(
            "Episodes/total_reward",
            np.mean([q[-1] for q in self.reward_queues]),
            self.step_var
        )
        self.writer.add_scalar(
            f"Episodes/reward_mean_{self.reward_queues[0].maxlen}",
            np.mean([np.mean(q) for q in self.reward_queues]),
            self.step_var
        )
        self.writer.add_scalar(
            "Episodes/episode_length",
            np.mean(self.episode_lengths),
            self.step_var
        )
        if self.had_ended_episodes.size > 1:
            self.writer.add_scalar(
                "Episodes/min_reward",
                min(q[-1] for q in self.reward_queues),
                self.step_var
            )
            self.writer.add_scalar(
                "Episodes/max_reward",
                max(q[-1] for q in self.reward_queues),
                self.step_var
            )
        self.episode_lengths.fill(0)
        self.had_ended_episodes.fill(False)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.rewards += rew
        self.episode_lengths[~self.had_ended_episodes] += 1

        info_collection = [info] if isinstance(info, dict) else info
        done_collection = [done] if isinstance(done, bool) else done
        done_indices = [i for i, info in enumerate(info_collection)
                        if info.get("real_done", done_collection[i])]
        for i in done_indices:
            if not self.had_ended_episodes[i]:
                self.had_ended_episodes[i] = True
            self.reward_queues[i].append(self.rewards[i])
            self.rewards[i] = 0

        self.step_var += self.nenvs
        if self.should_write_summaries():
            self.add_summaries()
        return obs, rew, done, info

    def reset(self, **kwargs):
        self.rewards.fill(0)
        self.episode_lengths.fill(0)
        self.had_ended_episodes.fill(False)
        return self.env.reset(**kwargs)

# magic for parallel launching of environments
class _thunk:
    def __init__(self, i, env_seed, env_id, **kwargs):
        self.env_id = env_id
        self.i = i
        self.env_seed = env_seed
        self.kwargs = kwargs

    def __call__(self):
        return nature_dqn_env(self.env_id, seed=self.env_seed, summaries=False, clip_reward=False, **self.kwargs)

def nature_dqn_env(env_id, nenvs=None, seed=None, summaries=True, monitor=False, clip_reward=True, episodic_life=True):
    """ Wraps env as in Nature DQN paper and creates parallel actors. """
    if "NoFrameskip" not in env_id:
        raise ValueError(f"env_id must have 'NoFrameskip' but is {env_id}")
    if nenvs is not None:
        if seed is None:
            seed = list(range(nenvs))
        if isinstance(seed, int):
            seed = [seed] * nenvs
        if len(seed) != nenvs:
            raise ValueError(f"seed has length {len(seed)} but must have "
                             f"length equal to nenvs which is {nenvs}")

        thunks = [_thunk(i, env_seed, env_id, episodic_life=episodic_life) for i, env_seed in enumerate(seed)]
        env = ParallelEnvBatch(thunks)

        if summaries:
            env = TensorboardSummaries(env, prefix=env_id)
        if clip_reward:
            env = ClipReward(env)
        return env

    env = OldGym(gym.make(env_id))
    env.seed(seed)
    env.metadata['render.modes'] = ['rgb_array']

    from ale_py.env import gym as ale_gym

    # Patch to allow rendering Atari games.
    # The AtariEnv's render method expects the mode to be in self._render_mode
    # (usually initialized with env.make) instead of taking mode as a param.
    _original_atari_render = ale_gym.AtariEnv.render


    def atari_render(self, mode = 'rgb_array'):
        original_render_mode = self._render_mode
        try:
            self._render_mode = mode
            return _original_atari_render(self)
        finally:
            self._render_mode = original_render_mode

    ale_gym.AtariEnv.render = atari_render
    
    
    if summaries:
        env = TensorboardSummaries(env)
    
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireReset(env)
    env = StartWithRandomActions(env, max_random_actions=30)
    
    if monitor:
        env = gym.wrappers.Monitor(env, directory="videos", video_callable=lambda episode_id: True,force=True)
    if episodic_life:
        env = EpisodicLife(env)
        
    env = MaxBetweenFrames(env)
    env = SkipFrames(env, 4)
    env = ImagePreprocessing(env, width=84, height=84, grayscale=True)
    env = QueueFrames(env, 4)
    env = ImageToPyTorch(env)
    if clip_reward:
        env = ClipReward(env)
    
    return env

