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
        
import time
import wandb
import random
import numpy as np

from tqdm import tqdm
from absl import app, flags
from ml_collections import config_flags

from D_utils.d4rl_env_utils import d4rl_make_env_and_dataset

from O_utils.datasets import Dataset, GASDataset
from O_utils.env_utils import make_env_and_datasets
from O_utils.log_utils import get_exp_name, setup_save_directory, setup_wandb, CsvLogger

from M_utils.agents import agents_dict
from M_utils.flax_utils import restore_agent, save_agent

# Flags for training low-level policy (GAS Stage 3).
FLAGS = flags.FLAGS

flags.DEFINE_string('run_policy_project', 'Debug', 'Run project.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('env_name', 'antmaze-giant-stitch-v0', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('gpu', 0, 'GPU index')
flags.DEFINE_string('save_policy_dir', 'exp_policy/', 'Save directory.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')

flags.DEFINE_string('tdr_path', None, 'Pretrained TDR path.') 

config_flags.DEFINE_config_file('agent_config', 'M_utils/agents/gas.py', lock_config=False) 


def main(_):
    # Set random seeds and load configuration.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    config = FLAGS.agent_config

    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_policy_dir = setup_save_directory(exp_name, FLAGS.env_name, FLAGS.run_group, FLAGS.save_policy_dir)
    setup_wandb(FLAGS.run_policy_project, FLAGS.run_group, exp_name)
    
    # Set up environment and dataset.
    if FLAGS.env_name in ['kitchen-partial-v0',]:
        env, train_dataset = d4rl_make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
        val_dataset = None    
    else:
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, FLAGS.seed)
    train_gas_dataset = GASDataset(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_gas_dataset = GASDataset(Dataset.create(**val_dataset), config)
        
    # Initialize agent.
    example_batch = train_gas_dataset.dataset.sample(1)
    agent_class = agents_dict[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], config,)
    
    # Restore TDR.
    tdr_restore_path = os.path.dirname(FLAGS.tdr_path)
    tdr_restore_epoch = os.path.basename(FLAGS.tdr_path).split('_')[-1].split('.')[0]
    agent = restore_agent(agent, tdr_restore_path, tdr_restore_epoch)

    # Get TDR feature extractor.
    get_phi_fn = agent.get_phi
    
    # Process features for training and validation datasets.
    train_gas_dataset.process_features(get_phi_fn)
    if val_dataset is not None:
        val_gas_dataset.process_features(get_phi_fn)
        
    # Train low-level policy.
    train_logger = CsvLogger(os.path.join(FLAGS.save_policy_dir, 'train.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm(range(1, FLAGS.train_steps + 1), desc="Training Policy", smoothing=0.1, dynamic_ncols=True):        
        # Update low-level policy.
        batch = train_gas_dataset.sample(config['batch_size'])
        agent, update_info = agent.critic_actor_update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_gas_dataset.sample(config['batch_size'])
                _, val_info = agent.total_critic_actor_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)
            
        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_policy_dir, i)
    train_logger.close()

if __name__ == '__main__':
    app.run(main)
