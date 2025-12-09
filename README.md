# Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning

<p align="left">
  <a href="https://www.arxiv.org/abs/2506.07744"><img src="https://img.shields.io/badge/Paper-arXiv-blueviolet?style=for-the-badge&logo=arxiv&logoColor=white"></a>
  <a href="https://qortmdgh4141.github.io/projects/GAS/"><img src="https://img.shields.io/badge/Project%20Page-Website-blueviolet?style=for-the-badge&logo=rocket&logoColor=white"></a>
  <a href="https://www.youtube.com/watch?v=6mxRlbn2_6s"><img src="https://img.shields.io/badge/Talk%20(10min)-YouTube-blueviolet?style=for-the-badge&logo=youtube"></a>  
</p>

:bell: We are happy to announce that GAS was accepted at **ICML 2025**. :bell:

<img src="https://qortmdgh4141.github.io/projects/GAS/media/figures/icml2025_gas_overview_v2.png" width="100%">
<img src="https://qortmdgh4141.github.io/projects/GAS/media/figures/icml2025_state_barplot.png" width="100%">


## Overview

This is the official implementation of **[Graph-Assisted Stitching](https://arxiv.org/abs/2506.07744)** (**GAS**)

See the [project page](https://qortmdgh4141.github.io/projects/GAS/) for more details.


## Requirements

* Python 3.9
* MuJoCo 3.1.6
* JAX >= 0.4.26 (CUDA 12 build)

  
## Installation

```
conda create --name gas python=3.9
conda activate gas
pip install -r requirements.txt --no-deps
pip install jax[cuda12]>=0.4.26 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install D4RL 
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
```

## Quick Start

We provide a 🚀[Colab notebook](https://colab.research.google.com/github/qortmdgh4141/GAS/blob/main/demo/GAS_demo.ipynb) for running pretrained GAS and visualizing trajectories.  

In the notebook, modify only the environment and task ID:
```python
# Select the environment.
ENV_NAME_LIST = ["antmaze-giant-navigate-v0", "antmaze-giant-stitch-v0", "antmaze-large-explore-v0", "scene-play-v0",
                 "visual-antmaze-giant-navigate-v0", "visual-antmaze-giant-stitch-v0", "visual-antmaze-large-explore-v0", "visual-scene-play-v0",]
ENV_NAME = ENV_NAME_LIST[0]  # Change the index to select the desired environment 🌍

# Select the task ID.
TASK_ID_LIST = [1, 2, 3, 4, 5]
TASK_ID = TASK_ID_LIST[0]  # Change the index to select the desired task 🎯
```
The following demos are example trajectories from pretrained GAS:
<details>
<summary>AntMaze Demo</summary>
<img src="demo/media/antmaze-demo.gif" width="720">  
</details>

<details>
<summary>Kitchen Demo</summary>
<img src="demo/media/kitchen-demo.gif" width="480">  
</details>

<details>
<summary>Scene Demo</summary>
<img src="demo/media/scene-demo.gif" width="480">  
</details>


## Pretrained Checkpoints

Official GAS checkpoints are available on our 🤗[HuggingFace repository](https://huggingface.co/qortmdgh4141/GAS).

We provide a `keygraph.pkl` (TD-aware Graph) and a `params_*.pkl` (TDR, Value/Critic, and Low-level Policy).

#### State-based Environments
| Environment | Graph | Policy |
| --- | --- | --- |
| antmaze-giant-navigate | [keygraph.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-giant-navigate/keygraph.pkl) | [params_1000000.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-giant-navigate/params_1000000.pkl) |
| antmaze-giant-stitch   | [keygraph.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-giant-stitch/keygraph.pkl)   | [params_1000000.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-giant-stitch/params_1000000.pkl) |
| antmaze-large-explore  | [keygraph.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-large-explore/keygraph.pkl) | [params_1000000.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/antmaze-large-explore/params_1000000.pkl) |
| scene-play             | [keygraph.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/scene-play/keygraph.pkl)            | [params_1000000.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/scene-play/params_1000000.pkl) |
| kitchen-partial        | [keygraph.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/kitchen-partial/keygraph.pkl)       | [params_500000.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/kitchen-partial/params_500000.pkl) |

#### Pixel-based Environments
| Environment | Graph | Policy |
| --- | --- | --- |
| visual-antmaze-giant-navigate | [keygraph.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/visual-antmaze-giant-navigate/keygraph.pkl) | [params_500000.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/visual-antmaze-giant-navigate/params_500000.pkl) |
| visual-antmaze-giant-stitch | [keygraph.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/visual-antmaze-giant-stitch/keygraph.pkl)   | [params_500000.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/visual-antmaze-giant-stitch/params_500000.pkl) |
| visual-antmaze-large-explore | [keygraph.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/visual-scene-play/keygraph.pkl)             | [params_500000.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/visual-scene-play/params_500000.pkl) |
| visual-scene-play | [keygraph.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/visual-scene-play/keygraph.pkl)             | [params_500000.pkl](https://huggingface.co/qortmdgh4141/GAS/resolve/main/visual-scene-play/params_500000.pkl) |

Alternatively, you can download programmatically via the Hugging Face Hub:

<details>
<summary><b>Click to expand programmatic download</b></summary>

```bash
pip install huggingface_hub
```

```bash
import os
from huggingface_hub import snapshot_download

ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

# Keep only the environments you want
envs = ["antmaze-giant-navigate", "antmaze-giant-stitch", "antmaze-large-explore", "scene-play", "kitchen-partial",] 
allow = [f"{e}/*" for e in envs]

snapshot_download(repo_id="qortmdgh4141/GAS", local_dir=ckpt_dir, allow_patterns=allow,)
```
</details>


## Training and Evaluation

The default hyperparameters in the code are set based on the `antmaze-giant-stitch` task:

```
# Stage 1: Pre-Training Temporal Distance Representation
python pretrain_tdr.py

# Stage 2: TD-aware Graph Construction
python construct_graph.py --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl

# Stage 3: Learning Low-level Policy
python train_policy.py --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl

# Stage 4: Task Planning and Execution
python evaluate_gas.py --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl
```

We provide the complete list of the exact command-line flags used to reproduce the main results in the paper:

<details>
<summary><b>Click to expand the full list of commands (state-based environments)</b></summary>

```bash
# GAS on antmaze-giant-navigate
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_antmaze-giant-navigate --env_name antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_antmaze-giant-navigate --env_name antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_antmaze-giant-navigate --env_name antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_antmaze-giant-navigate --env_name antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8

# GAS on antmaze-large-navigate
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_antmaze-large-navigate --env_name antmaze-large-navigate-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_antmaze-large-navigate --env_name antmaze-large-navigate-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_antmaze-large-navigate --env_name antmaze-large-navigate-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_antmaze-large-navigate --env_name antmaze-large-navigate-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8

# GAS on antmaze-medium-navigate
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_antmaze-medium-navigate --env_name antmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_antmaze-medium-navigate --env_name antmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_antmaze-medium-navigate --env_name antmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_antmaze-medium-navigate --env_name antmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8

# GAS on antmaze-giant-stitch
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_antmaze-giant-stitch --env_name antmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_antmaze-giant-stitch --env_name antmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_antmaze-giant-stitch --env_name antmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_antmaze-giant-stitch --env_name antmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8

# GAS on antmaze-large-stitch
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_antmaze-large-stitch --env_name antmaze-large-stitch-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_antmaze-large-stitch --env_name antmaze-large-stitch-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_antmaze-large-stitch --env_name antmaze-large-stitch-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_antmaze-large-stitch --env_name antmaze-large-stitch-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8

# GAS on antmaze-medium-stitch
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_antmaze-medium-stitch --env_name antmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_antmaze-medium-stitch --env_name antmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_antmaze-medium-stitch --env_name antmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_antmaze-medium-stitch --env_name antmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8

# GAS on antmaze-large-explore
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_antmaze-large-explore --env_name antmaze-large-explore-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_antmaze-large-explore --env_name antmaze-large-explore-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_antmaze-large-explore --env_name antmaze-large-explore-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_antmaze-large-explore --env_name antmaze-large-explore-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8

# GAS on antmaze-medium-explore
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_antmaze-medium-explore --env_name antmaze-medium-explore-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_antmaze-medium-explore --env_name antmaze-medium-explore-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_antmaze-medium-explore --env_name antmaze-medium-explore-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_antmaze-medium-explore --env_name antmaze-medium-explore-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8

# GAS on scene-play
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_scene-play --env_name scene-play-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48
python construct_graph.py --run_group EXP_scene-play --env_name scene-play-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48
python train_policy.py --run_policy_project EXP_policy --run_group EXP_scene-play --env_name scene-play-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_scene-play --env_name scene-play-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48

# GAS on kitchen-partial
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_kitchen-partial --env_name kitchen-partial-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 10.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48
python construct_graph.py --run_group EXP_kitchen-partial --env_name kitchen-partial-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.9 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 10.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48
python train_policy.py --run_policy_project EXP_policy --run_group EXP_kitchen-partial --env_name kitchen-partial-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 10.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_kitchen-partial --env_name kitchen-partial-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 1 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_500000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 10.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48
```
</details>

<details>
<summary><b>Click to expand the full list of commands (pixel-based environments)</b></summary>

```bash
# GAS on visual-antmaze-giant-navigate
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_visual-antmaze-giant-navigate --env_name visual-antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --agent_config.encoder impala_small --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_visual-antmaze-giant-navigate --env_name visual-antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.9 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_visual-antmaze-giant-navigate --env_name visual-antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_visual-antmaze-giant-navigate --env_name visual-antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 0 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8

# GAS on visual-antmaze-large-navigate
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_visual-antmaze-large-navigate --env_name visual-antmaze-large-navigate-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_visual-antmaze-large-navigate --env_name visual-antmaze-large-navigate-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.9 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_visual-antmaze-large-navigate --env_name visual-antmaze-large-navigate-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_visual-antmaze-large-navigate --env_name visual-antmaze-large-navigate-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 0 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8

# GAS on visual-antmaze-medium-navigate
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_visual-antmaze-medium-navigate --env_name visual-antmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_visual-antmaze-medium-navigate --env_name visual-antmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.9 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_visual-antmaze-medium-navigate --env_name visual-antmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_visual-antmaze-medium-navigate --env_name visual-antmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 0 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8

# GAS on visual-antmaze-giant-stitch
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_visual-antmaze-giant-stitch --env_name visual-antmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --agent_config.encoder impala_small --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_visual-antmaze-giant-stitch --env_name visual-antmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.9 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_visual-antmaze-giant-stitch --env_name visual-antmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_visual-antmaze-giant-stitch --env_name visual-antmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 0 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8

# GAS on visual-antmaze-large-stitch
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_visual-antmaze-large-stitch --env_name visual-antmaze-large-stitch-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_visual-antmaze-large-stitch --env_name visual-antmaze-large-stitch-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.9 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_visual-antmaze-large-stitch --env_name visual-antmaze-large-stitch-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_visual-antmaze-large-stitch --env_name visual-antmaze-large-stitch-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 0 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8

# GAS on visual-antmaze-medium-stitch
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_visual-antmaze-medium-stitch --env_name visual-antmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_visual-antmaze-medium-stitch --env_name visual-antmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.9 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_visual-antmaze-medium-stitch --env_name visual-antmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_visual-antmaze-medium-stitch --env_name visual-antmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 0 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8

# GAS on visual-antmaze-large-explore
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_visual-antmaze-large-explore --env_name visual-antmaze-large-explore-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.01 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_visual-antmaze-large-explore --env_name visual-antmaze-large-explore-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.9 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.01 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_visual-antmaze-large-explore --env_name visual-antmaze-large-explore-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.01 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_visual-antmaze-large-explore --env_name visual-antmaze-large-explore-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 0 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.01 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8

# GAS on visual-antmaze-medium-explore
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_visual-antmaze-medium-explore --env_name visual-antmaze-medium-explore-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.01 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python construct_graph.py --run_group EXP_visual-antmaze-medium-explore --env_name visual-antmaze-medium-explore-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.9 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.01 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python train_policy.py --run_policy_project EXP_policy --run_group EXP_visual-antmaze-medium-explore --env_name visual-antmaze-medium-explore-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.01 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_visual-antmaze-medium-explore --env_name visual-antmaze-medium-explore-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 0 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.01 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 8

# GAS on visual-scene-play
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_visual-scene-play --env_name visual-scene-play-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 24
python construct_graph.py --run_group EXP_visual-scene-play --env_name visual-scene-play-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.9 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 24
python train_policy.py --run_policy_project EXP_policy --run_group EXP_visual-scene-play --env_name visual-scene-play-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 500000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 24
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_visual-scene-play --env_name visual-scene-play-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 0 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_500000.pkl --agent_config.encoder impala_small --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 1.0 --agent_config.batch_size 256 --agent_config.p_aug 0.5 --agent_config.way_steps 24
```
</details>

## Additional Results

We release additional results on the `humanoidmaze-*` environments to facilitate future research. 

These results are not reported in the [ICML 2025 paper](https://arxiv.org/abs/2506.07744).

<details>
<summary><b>Additional results on humanoidmaze environments</b></summary>
<img src="https://github.com/qortmdgh4141/projects/raw/main/GAS/media/figures/humanoidmaze_additional_results_v3.png" width="75%" alt="additional humanoidmaze results">
</details>

<details>
<summary><b>Click to expand the full list of commands (state-based environments)</b></summary>

```bash
# GAS on humanoidmaze-giant-navigate
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_humanoidmaze-giant-navigate --env_name humanoidmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python construct_graph.py --run_group EXP_humanoidmaze-giant-navigate --env_name humanoidmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python train_policy.py --run_policy_project EXP_policy --run_group EXP_humanoidmaze-giant-navigate --env_name humanoidmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_humanoidmaze-giant-navigate --env_name humanoidmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32

# GAS on humanoidmaze-large-navigate
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_humanoidmaze-large-navigate --env_name humanoidmaze-large-navigate-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python construct_graph.py --run_group EXP_humanoidmaze-large-navigate --env_name humanoidmaze-large-navigate-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python train_policy.py --run_policy_project EXP_policy --run_group EXP_humanoidmaze-large-navigate --env_name humanoidmaze-large-navigate-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_humanoidmaze-large-navigate --env_name humanoidmaze-large-navigate-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32

# GAS on humanoidmaze-medium-navigate
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_humanoidmaze-medium-navigate --env_name humanoidmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python construct_graph.py --run_group EXP_humanoidmaze-medium-navigate --env_name humanoidmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python train_policy.py --run_policy_project EXP_policy --run_group EXP_humanoidmaze-medium-navigate --env_name humanoidmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_humanoidmaze-medium-navigate --env_name humanoidmaze-medium-navigate-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32

# GAS on humanoidmaze-giant-stitch
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_humanoidmaze-giant-stitch --env_name humanoidmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python construct_graph.py --run_group EXP_humanoidmaze-giant-stitch --env_name humanoidmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python train_policy.py --run_policy_project EXP_policy --run_group EXP_humanoidmaze-giant-stitch --env_name humanoidmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_humanoidmaze-giant-stitch --env_name humanoidmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32

# GAS on humanoidmaze-large-stitch
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_humanoidmaze-large-stitch --env_name humanoidmaze-large-stitch-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python construct_graph.py --run_group EXP_humanoidmaze-large-stitch --env_name humanoidmaze-large-stitch-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python train_policy.py --run_policy_project EXP_policy --run_group EXP_humanoidmaze-large-stitch --env_name humanoidmaze-large-stitch-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_humanoidmaze-large-stitch --env_name humanoidmaze-large-stitch-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32

# GAS on humanoidmaze-medium-stitch
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_humanoidmaze-medium-stitch --env_name humanoidmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python construct_graph.py --run_group EXP_humanoidmaze-medium-stitch --env_name humanoidmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.99 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python train_policy.py --run_policy_project EXP_policy --run_group EXP_humanoidmaze-medium-stitch --env_name humanoidmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_humanoidmaze-medium-stitch --env_name humanoidmaze-medium-stitch-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 49 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.95 --agent_config.alpha 0.1 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 32
```
</details>


## Repository Structure

High-level overview of the file-tree:

<details>
<summary><b>Utilities</b></summary>

+ `D_utils/` - D4RL utilities (kitchen environment)  
+ `K_utils/` - Keygraph utilities (TD-aware Graph)  
+ `M_utils/` - Model utilities (TDR, Value/Critic, Low-level Policy)  
+ `O_utils/` - Offline RL utilities (antmaze/scene environments, dataset, evaluation, logging)  

</details>

<details>
<summary><b>Main Scripts</b></summary>

+ `pretrain_tdr.py`     - Stage 1: Pre-Training Temporal Distance Representation  
+ `construct_graph.py`  - Stage 2: TD-aware Graph Construction  
+ `train_policy.py`     - Stage 3: Learning Low-level Policy  
+ `evaluate_gas.py`     - Stage 4: Task Planning and Execution  

</details>

<details>
<summary><b>Others</b></summary>

+ `GAS_demo.ipynb`      - Colab Demo  
+ `requirements.txt`    - Python Dependencies  
+ `LICENSE`             - MIT License  
+ `README.md`           - You are here!  

</details>


## Acknowledgments

This codebase is inspired by or partly uses code from the following repositories:
- [D4RL](https://github.com/Farama-Foundation/D4RL) for the dataset structure and the kitchen environment.
- [OGBench](https://github.com/seohongpark/ogbench) for the dataset structure and the antmaze, scene environments.
- [HIQL](https://github.com/seohongpark/HIQL) and [HILP](https://github.com/seohongpark/HILP) for JAX-based implementations of RL algorithms.

Special thanks to [Seohong Park](https://seohong.me/) for providing a JAX-based HHILP implementation and for helpful discussions.


## Citation

```bibtex
@inproceedings{gas_baek2025,
    title={Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning},
    author={Seungho Baek and Taegeon Park and Jongchan Park and Seungjun Oh and Yusung Kim},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025},
}
```  
