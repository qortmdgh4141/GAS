# Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning
<p align="left">
  <a href="https://www.arxiv.org/abs/2506.07744"><img src="https://img.shields.io/badge/Paper-arXiv-blueviolet?style=for-the-badge&logo=arxiv&logoColor=white"></a>
  <a href="https://qortmdgh4141.github.io/projects/GAS/"><img src="https://img.shields.io/badge/Project%20Page-Website-blueviolet?style=for-the-badge&logo=github"></a>
  <a href="https://www.youtube.com/watch?v=6mxRlbn2_6s"><img src="https://img.shields.io/badge/Talk%20(10min)-YouTube-blueviolet?style=for-the-badge&logo=youtube"></a>
</p>

## Overview
This is the official implementation of **[Graph-Assisted Stitching](https://www.arxiv.org/abs/2506.07744/)** (**GAS**)

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

## Examples 
### State-Based Environmnets
* Stage 1: Pre-Training Temporal Distance Representation
```
# GAS on antmaze-giant-navigate
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_antmaze-giant-navigate --env_name antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on antmaze-giant-stitch
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_antmaze-giant-stitch --env_name antmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on antmaze-large-explore
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_antmaze-large-explore --env_name antmaze-large-explore-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on scene-play
python pretrain_tdr.py --run_tdr_project EXP_tdr --run_group EXP_scene-play --env_name scene-play-v0 --seed 0 --gpu 0 --save_tdr_dir EXP_tdr/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48
```

* Stage 2: TD-aware Graph Construction
```
# GAS on antmaze-giant-navigate
python construct_graph.py --run_group EXP_antmaze-giant-navigate --env_name antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.999 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on antmaze-giant-stitch
python construct_graph.py --run_group EXP_antmaze-giant-stitch --env_name antmaze-giant-stitch-v0 --seed 0 --gpu 0  --save_graph_dir EXP_graph/ --te_threshold 0.999 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on antmaze-large-explore
python construct_graph.py --run_group EXP_antmaze-large-explore --env_name antmaze-large-explore-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.999 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on scene-play
python construct_graph.py --run_group EXP_scene-play --env_name scene-play-v0 --seed 0 --gpu 0 --save_graph_dir EXP_graph/ --te_threshold 0.999 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48
```

* Stage 3: Learning Low-level Policy
```
# GAS on antmaze-giant-navigate
python train_policy.py --run_policy_project EXP_policy --run_group EXP_antmaze-giant-navigate --env_name antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on antmaze-giant-stitch
python train_policy.py --run_policy_project EXP_policy --run_group EXP_antmaze-giant-stitch --env_name antmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on antmaze-large-explore
python train_policy.py --run_policy_project EXP_policy --run_group EXP_antmaze-large-explore --env_name antmaze-large-explore-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on scene-play
python train_policy.py --run_policy_project EXP_policy --run_group EXP_scene-play --env_name scene-play-v0 --seed 0 --gpu 0 --save_policy_dir EXP_policy/ --train_steps 1000000 --log_interval 5000 --save_interval 100000 --tdr_path PATH_TO_TDR_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48
```

* Stage 4: Task Planning and Execution
```
# GAS on antmaze-giant-navigate
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_antmaze-giant-navigate --env_name antmaze-giant-navigate-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 4 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on antmaze-giant-stitch
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_antmaze-giant-stitch --env_name antmaze-giant-stitch-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 4 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.995 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on antmaze-large-explore
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_antmaze-large-explore --env_name antmaze-large-explore-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 4 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 0.01 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 8
# GAS on scene-play
python evaluate_gas.py --run_eval_project EXP_eval --run_group EXP_scene-play --env_name scene-play-v0 --seed 0 --gpu 0 --save_eval_dir EXP_eval/ --eval_on_cpu 1 --eval_episodes 4 --eval_video_episodes 1 --eval_final_goal_threshold 2 --keygraph_path PATH_TO_KEYGRAPH_CHECKPOINT/keygraph.pkl --policy_path PATH_TO_POLICY_CHECKPOINT/params_1000000.pkl --agent_config.encoder not_used --agent_config.discount 0.99 --agent_config.tdr_expectile 0.999 --agent_config.alpha 1.0 --agent_config.batch_size 1024 --agent_config.p_aug 0.0 --agent_config.way_steps 48
```

## License

MIT


## Acknowledgments

This codebase is inspired by or partly uses code from the following repositories:
- [D4RL](https://github.com/Farama-Foundation/D4RL) for the dataset structure and the kitchen environment.
- [OGBench](https://github.com/Farama-Foundation/D4RL) for the dataset structure and the antmaze, scene environments.
- [HIQL](https://github.com/seohongpark/HIQL) and [HILP](https://github.com/seohongpark/HILP) for JAX-based implementations of RL algorithms.

Special thanks to [Seohong Park](https://seohong.me/) for providing a JAX-based Hierarchical HILP implementation and for helpful discussions.

## Citation
```bibtex
@inproceedings{gas_baek2025,
    title={Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning},
    author={Seungho Baek and Taegeon Park and Jongchan Park and Seungjun Oh and Yusung Kim},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025},
    }
}
```  
