# Migration Guide

This guide maps the original internal project layout to a cleaner public GitHub layout.

## Detected original code files

### Train_qwen
- core/action_mapping.py
- core/instruction_generator.py
- core/model.py
- core/train.py
- scripts/count_trajectories.py
- scripts/zero2.json
- train_qwen3vl.sh

### Dagger
- scripts/collect_dagger.py
- scripts/train_after_dagger.py
- collect_dagger.sh
- train_after_dagger.sh

### RL_residual
- scripts/airsim_env.py
- scripts/residual_action_head_policy.py
- scripts/train_ppo.py
- train_residual.sh

### Val
- metrics/metric.py
- scripts/closed_loop_airsim.py
- scripts/closed_loop_rl_residual.py
- scripts/evaluate_results.py
- evaluate.sh
- validate_seen.sh
- validate_rl_residual_seen.sh

### Executor
- core/logger.py
- core/SimServerTool.py
- core/TrajectoryExecutor.py
- core/settings/30001/settings.json ... core/settings/30030/settings.json

## Recommended target paths

- `src/uav_pe/models/qwen_policy.py`
- `src/uav_pe/models/action_mapping.py`
- `src/uav_pe/data/instruction_generator.py`
- `src/uav_pe/training/train_il.py`
- `src/uav_pe/training/collect_dagger.py`
- `src/uav_pe/training/train_dagger.py`
- `src/uav_pe/models/residual_policy.py`
- `src/uav_pe/envs/airsim_env.py`
- `src/uav_pe/training/train_ppo.py`
- `src/uav_pe/evaluation/metrics.py`
- `src/uav_pe/evaluation/closed_loop_eval.py`
- `src/uav_pe/evaluation/evaluate_results.py`
- `src/uav_pe/executor/trajectory_executor.py`
- `src/uav_pe/envs/sim_server.py`
- `src/uav_pe/utils/logger.py`

## Naming conventions

- Use lowercase snake_case filenames.
- Separate `models`, `training`, `evaluation`, `envs`, and `utils`.
- Avoid keeping `.git`, `__pycache__`, logs, checkpoints, and dataset files in the public repo.
