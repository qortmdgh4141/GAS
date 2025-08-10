import time
import ogbench
import gymnasium
import numpy as np

from O_utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""
    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
    
    def _reset_stats(self):
        """Reset episode statistics."""
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        """Step and log episodic metrics."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.reward_sum += reward
        self.episode_length += 1
        
        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time
            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0)
            else:
                info["episode"]["normalized_return"] = info["episode"]["return"] * 100.0
            info["episode"]["success"] = float(np.isclose(info["episode"]["normalized_return"], 100.0, rtol=1e-10))
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        """Reset environment and episode stats."""
        seed = kwargs.pop("seed", None)
        if seed is not None:  
            np.random.seed(seed)
            self.env.np_random = np.random.RandomState(seed)
            self.env._np_random_seed = seed  
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


def make_env_and_datasets(env_name, seed):
    """Make OGBench environment and datasets."""
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name, compact_dataset=False)
    env = EpisodeMonitor(env)
    observation, info = env.reset(seed=seed)
    
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)
    
    return env, train_dataset, val_dataset