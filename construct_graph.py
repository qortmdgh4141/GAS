import os
import sys
import platform

# Disable preallocation of GPU memory.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 

# Set the GPU index.
gpu_index = sys.argv[sys.argv.index('--gpu') + 1] if '--gpu' in sys.argv else "0" 
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
print(f"\033[38;5;208m{'=' * 14}\n Using GPU: {gpu_index}\n{'=' * 14}\033[0m")

# Set up EGL for rendering.
if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']
       
import random
import numpy as np

from absl import app, flags
from ml_collections import config_flags

from D_utils.d4rl_env_utils import d4rl_make_env_and_dataset

from O_utils.datasets import Dataset, GCDataset
from O_utils.env_utils import make_env_and_datasets
from O_utils.log_utils import get_exp_name, setup_save_directory

from K_utils.graph_builder import build_keynodes, build_keygraph

from M_utils.agents import agents_dict
from M_utils.flax_utils import restore_agent

# Flags for TD-aware graph construction (GAS Stage 2).
FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.') 
flags.DEFINE_string('env_name', 'antmaze-giant-stitch-v0', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('gpu', 0, 'GPU index')
flags.DEFINE_string('save_graph_dir', 'exp_graph/', 'Save directory.')

flags.DEFINE_float('te_threshold', 0.99, 'TE threshold.')

flags.DEFINE_string('tdr_path', None, 'Pretrained TDR path.') 

config_flags.DEFINE_config_file('agent_config', 'M_utils/agents/gas.py', lock_config=False) 


def main(_):
    # Set random seeds and load configuration.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    config = FLAGS.agent_config

    # Create the save directory path.
    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_graph_dir = setup_save_directory(exp_name, FLAGS.env_name, FLAGS.run_group, FLAGS.save_graph_dir)
    
    # Set up environment and dataset.
    if FLAGS.env_name in ['kitchen-partial-v0',]:
        env, train_dataset = d4rl_make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
        val_dataset = None
    else:
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, FLAGS.seed)
    train_gc_dataset = GCDataset(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_gc_dataset = GCDataset(Dataset.create(**val_dataset), config)
        
    # Initialize agent.
    example_batch = train_gc_dataset.sample(1)
    agent_class = agents_dict[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], config,)

    # Restore TDR.
    tdr_restore_path = os.path.dirname(FLAGS.tdr_path)
    tdr_restore_epoch = os.path.basename(FLAGS.tdr_path).split('_')[-1].split('.')[0]
    agent = restore_agent(agent, tdr_restore_path, tdr_restore_epoch)
    
    # Get TDR feature extractor.
    get_phi_fn = agent.get_phi
     
    # Set up evaluation tasks.
    if FLAGS.env_name in ['kitchen-partial-v0',]: 
        task_infos = [{'task_name': 'task1',}]
    else:   
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    task_id_list = list(range(1, len(task_infos) + 1))
    
    # Build graph.
    all_f_s  = np.concatenate([np.array(get_phi_fn(train_gc_dataset.dataset['observations'][i:i + config['batch_size']])) for i in range(0, train_gc_dataset.dataset['observations'].shape[0], config['batch_size'])], axis=0)    
    key_nodes = build_keynodes(train_gc_dataset.dataset, all_f_s, config['way_steps'], FLAGS.te_threshold, FLAGS.save_graph_dir)
    key_graph = build_keygraph(env, FLAGS.env_name, train_gc_dataset.dataset, key_nodes, config['batch_size'], task_id_list, FLAGS.seed, get_phi_fn, FLAGS.save_graph_dir)
                                 
if __name__ == '__main__':
    app.run(main)
