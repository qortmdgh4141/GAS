import jax
import dataclasses
import numpy as np
import jax.numpy as jnp

from tqdm import tqdm
from typing import Any
from functools import partial
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """
    Randomly crop an image.
    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """
    Dataset class.
    This class supports both regular datasets (i.e., storing both observations and next_observations).
    """
    @classmethod
    def create(cls, freeze=True, **fields):
        """
        Create a dataset from the fields.
        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        return result
    
    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)
    
    
@dataclasses.dataclass
class GCDataset:
    """
    Dataset class for TDR.
    This class provides a method to sample a batch of transitions with goals (tdr_goals) from the dataset. 
    The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports random-cropping image augmentation.
    
    It reads the following keys from the config:
    - discount:               Discount factor for geometric sampling.
    - tdr_value_p_curgoal:    Probability of using the current state as the tdr value goal.
    - tdr_value_p_trajgoal:   Probability of using a future state in the same trajectory as the tdr value goal.
    - tdr_value_p_randomgoal: Probability of using a random state as the tdr value goal.
    - value_geom_sample:      Whether to use geometric sampling for future tdr value goals.
    - p_aug:                  Probability of applying image augmentation.
    
    Attributes:
        dataset: Dataset object.
        config:  Configuration dictionary.
    """
    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        assert self.terminal_locs[-1] == self.size - 1
        assert np.isclose(self.config['tdr_value_p_curgoal'] + self.config['tdr_value_p_trajgoal'] + self.config['tdr_value_p_randomgoal'], 1.0)
            
    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr, batch[key],)
    
    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        
    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)
        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            middle_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            middle_goal_idxs = np.round((np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))).astype(int)
        goal_idxs = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-6), middle_goal_idxs, random_goal_idxs)
        # Goals at the current state.
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)
        return goal_idxs
    
    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """
        Sample a batch of transitions with goals.
        This method samples a batch of transitions with goals (tdr_value_goals) from the dataset. 
        It also computes the 'tdr_rewards' and 'tdr_masks' based on the indices of the goals.
        
        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        batch = self.dataset.sample(batch_size, idxs)
        
        tdr_value_goal_idxs = self.sample_goals(idxs, self.config['tdr_value_p_curgoal'], self.config['tdr_value_p_trajgoal'], self.config['tdr_value_p_randomgoal'], self.config['tdr_value_geom_sample'],)
        batch['tdr_value_goals'] = self.get_observations(tdr_value_goal_idxs)
        successes = (idxs == tdr_value_goal_idxs).astype(float)
        batch['tdr_masks'] = 1.0 - successes
        batch['tdr_rewards'] = successes - 1.0
        
        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'tdr_value_goals'])
        
        return batch


@dataclasses.dataclass
class GASDataset:
    """
    Dataset class for low-level policy.
    This class provides a method to sample a batch of transitions with subgoals (low_goals) from the dataset. 
    The subgoals are selected based on a fixed temporal distance within the same trajectory.
    It also supports random-cropping image augmentation.
    
    It reads the following keys from the config:
    - batch_size: Batch size for process features.
    - discount: Discount factor for geometric sampling.
    - p_aug: Probability of applying image augmentation.
    - way_steps: Temporal Distance Threshold (H_TD).
    
    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
    """
    dataset: Dataset
    config: Any    
    
    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        assert self.terminal_locs[-1] == self.size - 1
        self.get_phi_fn = None
        self.f_s = None
        self.next_f_s = None
        self.waysteps_idx = None

    def build_waysteps_idx_by_distance(self):
        """Precompute subgoal indices for each state based on a fixed temporal distance threshold within the same trajectory."""
        all_idxs = np.arange(self.dataset['observations'].shape[0])
        all_final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, all_idxs)]
        waysteps_idx = np.zeros(len(self.f_s), dtype=np.int32)
        for i in tqdm(range(len(self.f_s)), desc="Computing Distance-based Waypoints"):
            traj_end = all_final_state_idxs[i]
            subarr = self.f_s[i:traj_end+1] - self.f_s[i]  
            distances = np.linalg.norm(subarr, axis=1)
            idxs = np.where(distances >= self.config['way_steps'])[0]
            j_found = i + idxs[0] if len(idxs) > 0 else traj_end
            waysteps_idx[i] = j_found
        return waysteps_idx
    
    def process_features(self, get_phi_fn):
        """Precompute feature embeddings for observations and next_observations, and TD-aware subgoal indices."""
        self.get_phi_fn = get_phi_fn
        self.f_s = np.concatenate([np.array(self.get_phi_fn(self.dataset['observations'][i:i + self.config['batch_size']]))
                                   for i in range(0, self.dataset['observations'].shape[0], self.config['batch_size'])], axis=0)
        self.next_f_s = np.concatenate([np.array(self.get_phi_fn(self.dataset['next_observations'][i:i + self.config['batch_size']]))
                                        for i in range(0, self.dataset['next_observations'].shape[0], self.config['batch_size'])], axis=0)
        self.waysteps_idx = self.build_waysteps_idx_by_distance()
        
    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr, batch[key],)
        
    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """
        Sample a batch of transitions with goals.
        This method samples a batch of transitions with goals (actor_goals) from the dataset. 
        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        batch = self.dataset.sample(batch_size, idxs)

        low_goal_idxs = self.waysteps_idx[idxs]
        offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)
        low_goal_idxs = np.minimum(idxs + offsets, low_goal_idxs)
        
        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                batch['actor_goals'] = jax.tree_util.tree_map(lambda arr: arr[low_goal_idxs], self.dataset['observations'])
                self.augment(batch, ['observations', 'next_observations', 'actor_goals'])
                batch['phi_obs'] = self.get_phi_fn(batch['observations']) 
                batch['phi_next_obs'] = self.get_phi_fn(batch['next_observations'])
                batch['phi_actor_goals'] = self.get_phi_fn(batch['actor_goals'])
            else:
                batch['phi_obs'] = jax.tree_util.tree_map(lambda arr: arr[idxs], self.f_s)
                batch['phi_next_obs'] = jax.tree_util.tree_map(lambda arr: arr[idxs], self.next_f_s)
                batch['phi_actor_goals'] = jax.tree_util.tree_map(lambda arr: arr[low_goal_idxs], self.f_s)
    
        return batch