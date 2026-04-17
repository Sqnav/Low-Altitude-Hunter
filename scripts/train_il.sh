#!/bin/bash

if [ -z "$CUDA_HOME" ]; then
    if [ -f "/usr/local/cuda-12.1/bin/nvcc" ]; then
        export CUDA_HOME=/usr/local/cuda-12.1
    elif [ -f "/usr/local/cuda/bin/nvcc" ]; then
        export CUDA_HOME=/usr/local/cuda
    elif [ -f "/usr/bin/nvcc" ]; then
        export CUDA_HOME=/usr
    fi
fi
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/bin" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi

export DS_LOG_LEVEL=ERROR DS_LAUNCHER_LOG_LEVEL=ERROR DS_CONFIG_PRINT=0
export TRANSFORMERS_VERBOSITY=error TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export ACCELERATE_LOG_LEVEL=ERROR ACCELERATE_DISABLE_RICH=1
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
template_dir="$root_dir/UAV-Pursuit-Evasion-github-template"
base_model_path="$root_dir/Qwen3-VL-2B-Instruct"
output_dir="$root_dir/work_dirs/qwen3vl-uav-nonumeric"
mkdir -p "$output_dir"

export PYTHONPATH="$root_dir:$PYTHONPATH"

[ "$1" == "--from_scratch" ] && rm -rf $output_dir/checkpoint-* && echo "Deleted all checkpoints"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || source "/opt/anaconda3/etc/profile.d/conda.sh"
conda activate ysq_qwen

trajectory_range="1-450"
scene_list="City_1,City_2,City_3,City_4,City_5,City_6,City_7,City_8,City_9,City_10,City_11,City_12,City_13,City_14,City_15,City_16,City_17,City_18,City_19,City_20,City_21,City_22,City_23,City_24,City_25,City_26,City_27,City_28,City_29,City_30"
save_training_inputs=false

use_swanlab=false
swanlab_project="Qwen3VL-UAV-Training"
swanlab_experiment_name="qwen3vl-uav-$(date +%Y%m%d-%H%M%S)"
swanlab_workspace=""
swanlab_api_key="1BWz76qt6gpB13YRrygMZ"
swanlab_log_dir="$template_dir/work_dirs/swanlab_logs"

mkdir -p "$swanlab_log_dir"

if [ "$use_swanlab" = "true" ]; then
    export SWANLAB_NO_INTERACTIVE=1
    if [ -n "$swanlab_api_key" ]; then
        export SWANLAB_API_KEY="$swanlab_api_key"
    fi
    if [ -n "$swanlab_log_dir" ]; then
        export SWANLAB_LOG_DIR="$swanlab_log_dir"
        export SWANLAB_DIR="$swanlab_log_dir"
    fi
fi

python3 "$template_dir/tools/count_trajectories.py" \
    "$root_dir/Dataset" \
    "$scene_list" \
    "${trajectory_range:-}"

deepspeed \
    --include localhost:1 \
    --master_port 29101 \
    "$template_dir/src/uav_pe/training/train_il.py" \
    --model_name_or_path "$base_model_path" \
    --data_path "$root_dir/Dataset" \
    --dataset_path "$root_dir/Dataset" \
    --output_dir "$output_dir" \
    --scene_list "$scene_list" \
    --deepspeed "$template_dir/configs/zero2.json" \
    --model_max_length 2048 \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --optim "adamw_torch" \
    --report_to none \
    --save_training_inputs "$save_training_inputs" \
    --include_target_vel true \
    --include_prev_action true \
    --use_numeric_encoder true \
    --use_backbone true \
    --yaw_loss_weight 1.5 \
    --use_swanlab "$use_swanlab" \
    --swanlab_project "$swanlab_project" \
    --swanlab_experiment_name "$swanlab_experiment_name" \
    ${swanlab_workspace:+--swanlab_workspace "$swanlab_workspace"} \
    ${trajectory_range:+--trajectory_range "$trajectory_range"}
