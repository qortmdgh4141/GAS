import d4rl
import numpy as np

from copy import deepcopy
from dm_control.mujoco import engine

from O_utils.datasets import Dataset


def kitchen_get_dataset(env):
    """Preprocess dataset."""
    dataset = d4rl.qlearning_dataset(env)
    dataset['observations'] = dataset['observations'][:, :30]
    dataset['next_observations'] = dataset['next_observations'][:, :30]
    
    lim = 1 - 1e-5
    dataset['actions'] = np.clip(dataset['actions'], -lim, lim)
    dataset['terminals'][-1] = 1
    non_last_idx = np.nonzero(~dataset['terminals'])[0]
    last_idx = np.nonzero(dataset['terminals'])[0]
    penult_idx = last_idx - 1
    new_dataset = dict()
    for k, v in dataset.items():
        if k == 'terminals':
            v[penult_idx] = 1
        new_dataset[k] = v[non_last_idx]
    dataset = new_dataset
    terminals = dataset['terminals'].copy()

    return Dataset.create(
        observations = dataset['observations'].astype(np.float32),
        next_observations = dataset['next_observations'].astype(np.float32),
        actions = dataset['actions'].astype(np.float32),
        terminals = terminals.astype(np.float32),
        )


def kitchen_set_obs_and_goal(env, env_name, dataset, task_id, seed):
    """Set initial observation and final goal."""
    assert task_id == 1, f"Unsupported task_id: {task_id} for env_name: {env_name}. Only task_id=1 is supported."
    observation = env.reset(seed)
    
    if env_name in ['kitchen-partial-v0']: 
        goal_obs = observation[30:][9:]
        observation = observation[:30]
        goal = deepcopy(dataset['observations'][121284])  
        goal[9:] = goal_obs
    else:
        raise NotImplementedError
    
    return env, observation, goal


def kitchen_render(kitchen_env, wh=64):
    """Render kitchen environment."""
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.8, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    img = np.ascontiguousarray(img)
    return img