from O_utils.evaluation import setup_task_env

from K_utils.keynodes_utils import KeyNodes
from K_utils.keygraph_utils import KeyGraph


def build_keynodes(dataset, all_f_s, way_steps, te_threshold, save_dir):
    """Build keynodes."""
    key_nodes = KeyNodes()
    key_nodes.construct_nodes(all_f_s, dataset['terminals'], way_steps, te_threshold)     
         
    key_nodes.save_keynodes(save_dir)
    
    return key_nodes
  
  
def build_keygraph(env, env_name, dataset, key_nodes, batch_size, task_id_list, seed, get_phi_fn, save_dir):  
    """Build keygraph and precompute shortest paths to task goals."""
    key_graph = KeyGraph()
    key_graph.construct_graph(key_nodes, batch_size)
    
    task_goal_dict = {}
    task_node_dict = {}
    for task_id in task_id_list:
        env, observation, goal, reward, done, goal_rendered = setup_task_env(env, env_name, dataset, task_id, True, seed)
        task_goal_dict[task_id] = goal
        task_node_dict[task_id] = get_phi_fn(goal)    
    key_graph.precompute_shortest_paths_to_all_tasks(task_goal_dict, task_node_dict)
    
    key_graph.save_keygraph(save_dir)
            
    return key_graph
