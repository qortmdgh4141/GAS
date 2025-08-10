import os
import pickle
import numpy as np

from tqdm import tqdm
from collections import defaultdict


class KeyNodes(object):
    """Construct keynodes based on TE filtering and TD-aware clustering."""
    def __init__(self):
        self.way_steps = None
        self.te_threshold =  None
        self.efficiency_indices = None          
        self.nodes = None                       

    def construct_nodes(self, f_s, dones_float, way_steps, te_threshold):
        """Cluster high-TE states from trajectories."""
        self.way_steps = way_steps
        self.te_threshold = te_threshold
        trajectories, start_indices = split_trajectories(f_s, dones_float)
        
        self.efficiency_indices = collect_efficiency_states(trajectories, start_indices, self.way_steps, self.te_threshold)
        self.nodes = td_aware_clustering(f_s, self.efficiency_indices, self.way_steps)

    def save_keynodes(self, save_dir, filename="keynodes"):
        """
        Save the keynodes to a file.
        Args:
            save_dir: Directory to save the keynodes.
            filename: File name.
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{filename}.pkl")   
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"[KeyNodes] Saved to {save_path}")

    def load_keynodes(self, save_dir, filename="keynodes"):
        """
        Load the keynodes from a file.
        Args:
            save_dir: Path to the directory containing the saved keynodes.
            filename: File name.
        """
        load_path = os.path.join(save_dir, f"{filename}.pkl")  
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)
        print(f"[KeyNodes] Loaded from {load_path}")
        
        
def split_trajectories(f_s, dones_float):
    """Split trajectories."""
    trajectories = []
    start_indices = []
    start_idx = 0
    done_indices = np.where(dones_float == 1)[0]
    for end_idx in tqdm(done_indices, desc="Splitting Trajectories"):
        traj = f_s[start_idx:end_idx + 1]
        trajectories.append(traj)
        start_indices.append(start_idx)
        start_idx = end_idx + 1
    if start_idx < len(f_s): 
        trajectories.append(f_s[start_idx:])
        start_indices.append(start_idx)
    return trajectories, start_indices


def collect_efficiency_states(trajectories, start_indices, way_steps, te_threshold):
    """Collect high-TE states across trajectories."""
    global_efficiency_indices = []
    for traj_idx, traj in tqdm(enumerate(trajectories), desc="Collecting Efficiency States", total=len(trajectories)):
        traj_es_local = filter_low_efficiency_states(traj, way_steps, te_threshold) 
        traj_es_global = [start_indices[traj_idx] + es for es in traj_es_local]            
        global_efficiency_indices.extend(traj_es_global)
    if len(set(global_efficiency_indices))!=  len(global_efficiency_indices):
        raise AssertionError(f"Duplicate efficiency indices found! (original={ len(set(global_efficiency_indices))}, unique={len(global_efficiency_indices)})")    
    return global_efficiency_indices


def filter_low_efficiency_states(traj, way_steps, te_threshold):
    """Filter out low-TE states using temporal distance threshold."""
    num_points = len(traj)
    indices = np.arange(num_points)  
    local_efficiency_indices = np.ones(num_points, dtype=bool)  
    for i in range(num_points - way_steps):
        obs_t = traj[i]
        obs_t_plus_step = traj[i + way_steps]
        subarr_traj = traj[i + 1:]  
        distances_future = np.linalg.norm(subarr_traj - obs_t, axis=1)
        idxs_above = np.where(distances_future >= way_steps)[0]
        obs_t_plus_distance = traj[i + 1 + idxs_above[0]] if len(idxs_above) > 0 else traj[-1]
 
        vector_step = obs_t_plus_step - obs_t
        vector_distance = obs_t_plus_distance - obs_t 
        vector_step /= np.linalg.norm(vector_step) + 1e-10
        vector_distance /= np.linalg.norm(vector_distance) + 1e-10
        
        cosine_similarity = np.dot(vector_step, vector_distance)
        if cosine_similarity < te_threshold:
            local_efficiency_indices[i] = False
        
    return indices[local_efficiency_indices]


def td_aware_clustering(f_s, efficiency_indices, way_steps):
    """Perform clustering based on temporal distance threshold."""
    min_dist = way_steps / 2
    f_s_sub = f_s[efficiency_indices]
    stickers = np.zeros_like(f_s_sub)
    sticker_assignments = defaultdict(list)
    
    stickers[0] = f_s_sub[0]
    sticker_assignments[0].append(0)
    num_stickers = 1
    for i in tqdm(range(1, len(f_s_sub)), desc="TD-Aware Clustering"):
        dists = np.linalg.norm(f_s_sub[i] - stickers[:num_stickers], axis=-1)
        min_idx = np.argmin(dists)    
        if dists[min_idx] > min_dist:
            stickers[num_stickers] = f_s_sub[i]
            sticker_assignments[num_stickers].append(i)
            num_stickers += 1
        else:
            sticker_assignments[min_idx].append(i)
    stickers = stickers[:num_stickers]
    
    sticker_centers = np.zeros_like(stickers)
    for s_idx, assigned_list in sticker_assignments.items():
        assigned_points = f_s_sub[assigned_list]
        sticker_centers[s_idx] = assigned_points.mean(axis=0)    
    
    return sticker_centers