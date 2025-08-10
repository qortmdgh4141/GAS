import os
import jax
import pickle
import numpy as np
import networkx as nx
import jax.numpy as jnp

from tqdm import tqdm


class KeyGraph:
    """Construct keygraph for task planning."""
    def __init__(self):
        self.way_steps = None
        self.te_threshold = None        
        self.nodes = None
        self.graph = None
        
        self.base_node_cnt = 0 # number of nodes before adding task goals
        self.task_goal_dict       = {}  # task id → raw goal state
        self.task_node_dict       = {}  # task id → phi-goal node
        self.task_node_idx_dict   = {}  # task id → phi-goal node idx
        self.task_paths_dict      = {}  # task id → {node idx: [path idx list]}
        self.task_paths_dist_dict = {}  # task id → {node idx: distance}
        
    def construct_graph(self, key_nodes, batch_size):        
        """Construct a directed graph from key nodes."""    
        self.way_steps = key_nodes.way_steps
        self.te_threshold = key_nodes.te_threshold
        self.nodes = key_nodes.nodes
        
        self.base_node_cnt = len(self.nodes) 
        self.graph = nx.DiGraph()       
        for node_idx, node_pos in tqdm(enumerate(self.nodes), desc="Adding Nodes to Graph", total=len(self.nodes)):
            self.graph.add_node(node_idx, pos=node_pos)
        self.graph, pdist_matrix = add_distance_based_edges(self.graph, self.nodes, self.way_steps, batch_size)
        
        self.graph = nx.DiGraph(self.graph)  
        self.graph = connect_strongly_connected_components(self.graph, pdist_matrix)

        for u, v in tqdm(self.graph.edges, desc="Assigning Weights to Edges", total=len(self.graph.edges)):
            distance = np.linalg.norm(self.nodes[u] - self.nodes[v])
            self.graph[u][v]['weight'] = distance

    def precompute_shortest_paths_to_all_tasks(self, task_goal_dict, task_node_dict):
        """Precompute shortest paths from all nodes to each task goal."""
        self.clear_task_goals()
        self.task_goal_dict = task_goal_dict
        self.task_node_dict = task_node_dict
        for task_id, target in tqdm(self.task_node_dict.items(), desc="Precomputing shortest paths for tasks"):
            target_idx = self.graph.number_of_nodes()
            self.graph, self.nodes = add_target_node(self.graph, self.nodes, target, target_idx, self.way_steps)
            self.task_node_idx_dict[task_id] = target_idx
            self.task_paths_dict[task_id], self.task_paths_dist_dict[task_id] = compute_shortest_paths_to_target(self.graph, target_idx)

    def clear_task_goals(self):
        """Remove all task-related nodes and paths."""
        for idx in sorted(self.task_node_idx_dict.values(), reverse=True):
            if self.graph.has_node(idx):
                self.graph.remove_node(idx)
        self.nodes = self.nodes[:self.base_node_cnt]
        
        self.task_goal_dict.clear()
        self.task_node_dict.clear()
        self.task_node_idx_dict.clear()
        self.task_paths_dict.clear()
        self.task_paths_dist_dict.clear()
    
    def get_shortest_path(self, task_id, source, force_closest=False):
        """Get the precomputed shortest path to a task goal from a source point."""
        nodes = self.nodes
        shortest_paths = self.task_paths_dict[task_id]
        shortest_paths_dist = self.task_paths_dist_dict[task_id]
        
        sp_keys = list(shortest_paths.keys())
        start_distances = np.linalg.norm(nodes[sp_keys] - source, axis=1)
        valid_indices = np.where(start_distances <= self.way_steps)[0]
        if len(valid_indices) == 0:
            if force_closest:
                closest_index = np.argmin(start_distances)
                valid_indices = [closest_index]
            else:
                return None

        best_total_distance = float('inf')
        best_path = None
        for idx in valid_indices:
            path_key = sp_keys[idx]
            path_distance = shortest_paths_dist[path_key]
            total_distance = start_distances[idx] + path_distance
            if total_distance < best_total_distance:
                best_total_distance = total_distance
                best_path = shortest_paths[path_key]

        return nodes[best_path] 
        
    def save_keygraph(self, save_dir, filename="keygraph"):
        """
        Save the keygraph to a file.
        Args:
            save_dir: Directory to save the keygraph.
            filename: File name.
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{filename}.pkl")   
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"[KeyGraph] Saved to {save_path}")

    def load_keygraph(self, save_dir, filename="keygraph"):
        """
        Load the keygraph from a file.
        Args:
            save_dir: Path to the directory containing the saved keygraph.
            filename: File name.
        """
        load_path = os.path.join(save_dir, f"{filename}.pkl")
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)
        print(f"[KeyGraph] Loaded from {load_path}")
        
        
def compute_pdist_matrix(query_points, reference_points, batch_size):
    """Compute pairwise distance matrix."""
    num_queries = query_points.shape[0]
    num_references = reference_points.shape[0]
    distance_matrix = np.zeros((num_queries, num_references), dtype=np.float32)
    for start_idx in tqdm(range(0, num_queries, batch_size), desc="Calculating Pairwise Distances in Batches", leave=False):  
        end_idx = min(start_idx + batch_size, num_queries)
        batch = query_points[start_idx:end_idx]
        distances = np.array(pairwise_distances(query_points=batch, reference_points=reference_points))
        distance_matrix[start_idx:end_idx, :] = distances
    return distance_matrix


@jax.jit
def pairwise_distances(query_points, reference_points):
    """Compute pairwise distance."""
    return jnp.sqrt(jnp.sum((query_points[:, jnp.newaxis, :] - reference_points[jnp.newaxis, :, :]) ** 2, axis=-1))


def add_distance_based_edges(graph, nodes, cutoff, batch_size):
    """Add bidirectional edges based on temporal distance threshold."""
    pdist_matrix = compute_pdist_matrix(nodes, nodes, batch_size)
    np.fill_diagonal(pdist_matrix, np.inf)
    for i, row in tqdm(enumerate(pdist_matrix), total=len(pdist_matrix), desc="Adding Distance-Based Edges to Graph"):
        neighbors = np.where(row <= cutoff)[0]
        for j in neighbors:
            graph.add_edge(i, j)
    return graph, pdist_matrix
    
                             
def connect_strongly_connected_components(graph, pdist_matrix):
    """Connect disconnected components to ensure full graph connectivity."""
    components = list(nx.strongly_connected_components(graph))
    component_groups = [list(comp) for comp in components]
    with tqdm(total=len(component_groups), desc="Merging components") as pbar:
        while len(component_groups) > 1:
            main_nodes = component_groups[0]
            other_nodes = [node for comp in component_groups[1:] for node in comp]
            dist_matrix = pdist_matrix[np.ix_(main_nodes, other_nodes)]
            
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            main_node, other_node = main_nodes[min_idx[0]], other_nodes[min_idx[1]]
            graph.add_edge(main_node, other_node)
            graph.add_edge(other_node, main_node)
            
            for comp in component_groups[1:]:
                if other_node in comp:
                    component_groups[0].extend(comp)
                    component_groups.remove(comp)
                    break
            pbar.update(1)
            
    return graph


def add_target_node(graph, nodes, target, target_idx, cutoff): 
    """Add a goal node to the graph and connect it to nearby nodes."""
    graph.add_node(target_idx, pos=target)
    distances_to_target = np.linalg.norm(nodes - target, axis=1)
    min_dists = np.min(distances_to_target) * 1.2
    threshold = max(cutoff, min_dists)
    
    for i, dist in enumerate(distances_to_target):
        if dist < threshold:
            graph.add_edge(i, target_idx, weight=dist)
            graph.add_edge(target_idx, i, weight=dist)
    updated_nodes = np.vstack([nodes, target])
    
    return graph, updated_nodes
   

def compute_shortest_paths_to_target(graph, target_idx):
    """Compute shortest paths to the given goal node."""
    lengths, paths = nx.single_source_dijkstra(graph, source=target_idx, weight='weight')
    
    shortest_paths = {}
    shortest_paths_dist = {}
    for node_idx, path in paths.items():
        if node_idx == target_idx:
            continue
        shortest_paths[node_idx] = path[::-1]  
        shortest_paths_dist[node_idx] = lengths[node_idx]  
    
    return shortest_paths, shortest_paths_dist 