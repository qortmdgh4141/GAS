import gym
import time
import numpy as np

from D_utils.kitchen_utils import kitchen_get_dataset


class d4rl_EpisodeMonitor(gym.ActionWrapper):
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
        observation, reward, done, info = self.env.step(action)
        self.reward_sum += reward
        self.episode_length += 1
        
        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time
            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (self.get_normalized_score(info["episode"]["return"]) * 100.0)
            else:
                info["episode"]["normalized_return"] = info["episode"]["return"] * 100.0
            info["episode"]["success"] = float(np.isclose(info["episode"]["normalized_return"], 100.0, rtol=1e-10))
        
        return observation, reward, done, info
    
    def reset(self, seed):
        """Reset environment and episode stats."""
        if seed is not None: 
            np.random.seed(seed)
            self.env.seed(seed)
        self._reset_stats()
        return self.env.reset()
    
    
def d4rl_make_env_and_dataset(env_name, seed):
    """Make kitchen-partial environment and dataset."""
    env = gym.make(env_name)
    env = d4rl_EpisodeMonitor(env)
    observation = env.reset(seed) 
    
    dataset = kitchen_get_dataset(env) 
    dataset = dataset.copy({'observations': dataset['observations'][:, :30], 'next_observations': dataset['next_observations'][:, :30]})
    
    return env, dataset


  