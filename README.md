# anti-UAV 

This repository is a GitHub-style reorganized template for a anti-UAV project with the following stages inferred from the original code package:

- Imitation learning with Qwen-VL style policy training
- DAgger data collection and supervised refinement
- Residual reinforcement learning with PPO
- Closed-loop evaluation in AirSim
- Execution utilities and multi-port simulator settings


## Resources

- **Platform Download (Hugging Face)**: [low-altitude-hunter-env](https://huggingface.co/datasets/shaoqiang668/low-altitude-hunter-env)

### Dataset Download (Kaggle)
- [Low-Altitude-Hunter Env Part 1](https://www.kaggle.com/datasets/shaoqiang668/low-altitude-hunter-env-part1)
- [Low-Altitude-Hunter Env Part 2](https://www.kaggle.com/datasets/shaoqiang668/low-altitude-hunter-env-part2)
- [Low-Altitude-Hunter Env Part 3](https://www.kaggle.com/datasets/shaoqiang668/low-altitude-hunter-env-part3)
- [Low-Altitude-Hunter Env Part 4](https://www.kaggle.com/datasets/shaoqiang668/low-altitude-hunter-env-part4)
- [Low-Altitude-Hunter Env Part 5](https://www.kaggle.com/datasets/shaoqiang668/low-altitude-hunter-env-part5)
- [Low-Altitude-Hunter Env Part 6](https://www.kaggle.com/datasets/shaoqiang668/low-altitude-hunter-env-part6)
- [Low-Altitude-Hunter Env Part 7](https://www.kaggle.com/datasets/shaoqiang668/low-altitude-hunter-env-part7)
- [Low-Altitude-Hunter Env Part 8](https://www.kaggle.com/datasets/shaoqiang668/low-altitude-hunter-env-part8)
- [Low-Altitude-Hunter Env Part 9](https://www.kaggle.com/datasets/shaoqiang668/low-altitude-hunter-env-part9)
- [Low-Altitude-Hunter Env Part 10](https://www.kaggle.com/datasets/shaoqiang668/low-altitude-hunter-env-part10)



## Current status

This package was produced from an uploaded `RAR` archive whose full source files could not be extracted in the current environment because the required RAR backend is unavailable. To still make the project immediately usable as a public repository scaffold, this package includes:

- a clean open-source style directory layout
- unified script entry points
- config templates
- packaging files
- a migration guide mapping the original file layout to the recommended public layout

To create a full source-preserving public repo, re-upload the same project as a `.zip` archive and the source files can be moved into this structure directly.

## Recommended layout

```text
/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── setup.py
├── configs/
├── docs/
├── scripts/
├── src/uav_pe/
│   ├── models/
│   ├── data/
│   ├── envs/
│   ├── executor/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── tools/
└── examples/
```

## Original structure that was detected

- `Train_qwen/`
- `Dagger/`
- `RL_residual/`
- `Val/`
- `Executor/`

## Suggested mapping

| Original | Recommended |
|---|---|
| `Train_qwen/core/model.py` | `src/uav_pe/models/qwen_policy.py` |
| `Train_qwen/core/action_mapping.py` | `src/uav_pe/models/action_mapping.py` |
| `Train_qwen/core/instruction_generator.py` | `src/uav_pe/data/instruction_generator.py` |
| `Train_qwen/core/train.py` | `src/uav_pe/training/train_il.py` |
| `Dagger/scripts/collect_dagger.py` | `src/uav_pe/training/collect_dagger.py` |
| `Dagger/scripts/train_after_dagger.py` | `src/uav_pe/training/train_dagger.py` |
| `RL_residual/scripts/airsim_env.py` | `src/uav_pe/envs/airsim_env.py` |
| `RL_residual/scripts/residual_action_head_policy.py` | `src/uav_pe/models/residual_policy.py` |
| `RL_residual/scripts/train_ppo.py` | `src/uav_pe/training/train_ppo.py` |
| `Val/metrics/metric.py` | `src/uav_pe/evaluation/metrics.py` |
| `Val/scripts/closed_loop_airsim.py` | `src/uav_pe/evaluation/closed_loop_eval.py` |
| `Val/scripts/evaluate_results.py` | `src/uav_pe/evaluation/evaluate_results.py` |
| `Executor/core/TrajectoryExecutor.py` | `src/uav_pe/executor/trajectory_executor.py` |
| `Executor/core/SimServerTool.py` | `src/uav_pe/envs/sim_server.py` |
| `Executor/core/logger.py` | `src/uav_pe/utils/logger.py` |

## Installation

```bash
conda create -n uav_pe python=3.10 -y
conda activate uav_pe
pip install -r requirements.txt
pip install -e .
```

## Training entry points

```bash
bash scripts/train_il.sh
bash scripts/collect_dagger.sh
bash scripts/train_dagger.sh
bash scripts/train_rl.sh
```

## Evaluation

```bash
bash scripts/evaluate_seen.sh
bash scripts/evaluate_rl_residual_seen.sh
```

## What should not be committed

The original package included development artifacts that should be removed before publishing:

- `.git/`
- `__pycache__/`
- `logs/`
- checkpoints and model weights
- dataset files
- temporary outputs

## Next step for a full conversion

Re-upload the project as a `.zip` file instead of `.rar`. Then the source can be moved into this scaffold and a fully cleaned public repository can be generated directly.
