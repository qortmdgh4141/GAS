import cv2
import jax
import numpy as np

from tqdm import trange
from collections import defaultdict

from D_utils.kitchen_utils import kitchen_set_obs_and_goal, kitchen_render


def evaluate_with_graph(agent, key_graph, env, env_name, task_id, eval_episodes, eval_video_episodes, 
                        seed, eval_on_cpu, eval_subgoal_threshold, eval_final_goal_threshold, config, recompute_paths_per_episode=False):         
    """
    Evaluate GAS in the environment.
    In OGBench environments, the final goal includes slight random noise in each episode.
    Empirically, we found little to no performance difference between two strategies:
    (1) recomputing the shortest path every episode
    (2) reusing a precomputed path for the same task_id
    
    By default, evaluate_with_graph() uses strategy (2), computing the shortest path once via precompute_shortest_paths_to_all_tasks().
    When using strategy (2), we recommend setting eval_final_goal_threshold >= 2 to allow agents to reach slightly perturbed final goals.
    If final goals vary significantly across episodes, or if GAS is extended to online RL, we recommend setting recompute_paths_per_episode=True to use strategy (1).
    """
    eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0]) if eval_on_cpu else agent
    get_phi_fn = eval_agent.get_phi         
    actor_fn = supply_rng(eval_agent.sample_actions, rng=jax.random.PRNGKey(seed))  
    
    stats = defaultdict(list)
    renders = []
    for i in trange(eval_episodes + eval_video_episodes, leave=False, desc=f"Task {task_id} Episodes"):
        step = 0
        render = []
        should_render = i >= eval_episodes
        eval_seed = seed + i  
        env, observation, goal, reward, done, goal_rendered = setup_task_env(env, env_name, task_id, should_render, seed=eval_seed)
        
        epsilon=1e-10
        phi_obs = np.array(get_phi_fn(observation))
        phi_goal = np.array(get_phi_fn(goal))
        final_goal_on = False
        # Optionally, recompute the shortest path every episode (strategy (1))
        if recompute_paths_per_episode:
            key_graph.precompute_shortest_paths_to_all_tasks({task_id: goal}, {task_id: phi_goal},)
        shortest_path = key_graph.get_shortest_path(task_id=task_id, source=phi_obs, force_closest=True)
        while not done:
            phi_obs = np.array(get_phi_fn(observation))
            if final_goal_on:
                cur_obs_goal = phi_goal
            else:
                cached_shortest_path = key_graph.get_shortest_path(task_id=task_id, source=phi_obs)
                if cached_shortest_path is not None:
                    shortest_path = cached_shortest_path  
                distances = np.linalg.norm(np.array(shortest_path) - phi_obs, axis=1) 
                valid_indices = np.where(distances <= eval_subgoal_threshold)[0]  
                cur_node_idx = valid_indices[-1] if len(valid_indices) > 0 else 0         
                if len(shortest_path) <= eval_final_goal_threshold:
                    final_goal_on = True
                    cur_obs_goal = phi_goal
                else:
                    cur_obs_goal = shortest_path[cur_node_idx]
                  
            skills = (cur_obs_goal - phi_obs) / (np.linalg.norm(cur_obs_goal - phi_obs) + epsilon)  
            action = actor_fn(observations=observation, goals=skills, temperature=0.0)
            action = np.clip(np.array(action), -1, 1)
            next_observation, reward, done, info = env_step(env, env_name, action)  
          
            step += 1
            if should_render and (step % 3 == 0 or done):
                frame = get_frame(env, env_name)
                render.append(frame)
            observation = next_observation
        add_to(stats, flatten(info))
        if should_render:
            renders.append(np.array(render))
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, renders


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)
    return wrapped


def setup_task_env(env, env_name, task_id, should_render, seed):
    """Reset the environment for a specific task."""
    if env_name in ['kitchen-partial-v0',]:
        env, observation, goal = kitchen_set_obs_and_goal(env, env_name, task_id, seed=seed)
        goal_rendered = None
    else:
        observation, info = env.reset(seed=seed, options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        if should_render:
            goal_rendered =  info.get('goal_rendered') 
            goal_rendered = resize_frame(goal_rendered)
        else:
            goal_rendered = None
    reward = 0
    done = False
    return env, observation, goal, reward, done, goal_rendered


def env_step(env, env_name, action):
    """Step the environment once."""
    if env_name in ['kitchen-partial-v0']:
        next_observation, reward, done, info = env.step(action)
        next_observation = next_observation[:30]
    else:
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return next_observation, reward, done, info


def resize_frame(frame):
    """Resize a frame"""
    resized_frame = cv2.resize(frame, (200, 200), interpolation=cv2.INTER_LINEAR)
    return resized_frame


def get_frame(env, env_name):
    """Render a frame from the environment."""
    if env_name in ['kitchen-partial-v0',]:
        frame = kitchen_render(env, wh=200)
    else:
        frame = np.ascontiguousarray(env.render())
        frame = resize_frame(frame)
    return frame


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)
                  
