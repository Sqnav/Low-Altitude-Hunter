SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
root_dir="${root_dir:-$(cd "$TEMPLATE_ROOT/.." && pwd)}"
dataset_root="$root_dir/Dataset"
scene_ids="City_1,City_2,City_3,City_4,City_5,City_6,City_7,City_8,City_9,City_10,City_11,City_12,City_13,City_14,City_15,City_16,City_17,City_18,City_19,City_20,City_21,City_22,City_23,City_24,City_25,City_26,City_27"
gpu_ids="1"
trajectory_range="1-450"
trajectory_chunk_size=25
model_path="$root_dir/work_dirs/qwen3vl-uav-numeric-all-dagger"
base_model_path="$root_dir/Qwen3-VL-2B-Instruct"
sim_server_host="127.0.0.1"
sim_server_port=30000

learning_rate=1e-5
n_steps=512
batch_size=32
save_path="$root_dir/work_dirs/RL_residual/residual_ppo.zip"
mkdir -p "$(dirname "$save_path")"
log_std_init="-3.5"
residual_scale="0.1"
critic_warmup_steps="10000"
target_kl="0.01"

save_every_n_steps=5000

resume_training=true
resume_from=""
training_state_path=""
save_execution_data=false
execution_data_dir="$root_dir/work_dirs/rl_execution_data"
mkdir -p "$execution_data_dir"

reward_progress_scale="1.0"
max_steps=""
include_target_vel=true
include_prev_action=true

use_numeric_encoder=true
use_backbone=true

use_swanlab=true
swanlab_project="RL-residual"
swanlab_experiment_name="ppo-$(date +%Y%m%d-%H%M%S)"
swanlab_workspace=""
swanlab_api_key="1BWz76qt6gpB13YRrygMZ"
swanlab_log_dir="$root_dir/work_dirs/swanlab_logs"

mkdir -p "$swanlab_log_dir"

if [ "$use_swanlab" = "true" ]; then
    export SWANLAB_NO_INTERACTIVE=1
    if [ -n "$swanlab_api_key" ]; then
        export SWANLAB_API_KEY="$swanlab_api_key"
    fi
    [ -n "$swanlab_log_dir" ] && export SWANLAB_LOG_DIR="$swanlab_log_dir" && export SWANLAB_DIR="$swanlab_log_dir"
fi

cd "$root_dir"
export PYTHONPATH="$root_dir:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RL_DEBUG_ACTION=1

swanlab_flag="--use_swanlab"
swanlab_exp_arg="--swanlab_experiment_name"
[ "$use_swanlab" != "true" ] && swanlab_flag="--no_swanlab" && swanlab_exp_arg=""

base_model_args=(--base_model_path "$base_model_path")

IFS=',' read -r -a gpu_ids_arr <<< "$gpu_ids"
if [ "${#gpu_ids_arr[@]}" -eq 0 ]; then
    echo "Error: gpu_ids is empty"
    exit 1
fi
gpu_id="${gpu_ids_arr[0]}"

cleanup() {
    echo ""
    echo "[train_airsim_ppo] Interrupt received; terminating training process..."
    kill $(jobs -p) 2>/dev/null
    exit 130
}
trap cleanup INT TERM

max_steps_arg=""
if [ -n "$max_steps" ]; then
    max_steps_arg="--max_steps $max_steps"
fi
target_kl_arg=""
if [ -n "$target_kl" ] && [ "$target_kl" != "0" ]; then
    target_kl_arg="--target_kl $target_kl"
fi

resume_args=()
if [ "$resume_training" = "true" ]; then
    resume_args+=(--resume)
    [ -n "$resume_from" ] && resume_args+=(--resume_from "$resume_from")
    [ -n "$training_state_path" ] && resume_args+=(--training_state_path "$training_state_path")
fi

echo "Starting PPO training: scene_ids=${scene_ids} GPU=${gpu_id}"
CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u "$root_dir/UAV-Pursuit-Evasion-github-template/src/uav_pe/training/train_ppo.py" \
    --scene_ids "$scene_ids" \
    --trajectory_range "$trajectory_range" \
    --trajectory_chunk_size "$trajectory_chunk_size" \
    --dataset_root "$dataset_root" \
    --sim_server_host "$sim_server_host" \
    --sim_server_port "$sim_server_port" \
    --gpu_id "$gpu_id" \
    --model_path "$model_path" \
    "${base_model_args[@]}" \
    --learning_rate "$learning_rate" \
    --n_steps "$n_steps" \
    --batch_size "$batch_size" \
    --save_path "$save_path" \
    --save_models "true" \
    --save_every_n_steps "$save_every_n_steps" \
    --save_execution_data "$save_execution_data" \
    --execution_data_dir "$execution_data_dir" \
    --reward_progress_scale "$reward_progress_scale" \
    --critic_warmup_steps "$critic_warmup_steps" \
    $max_steps_arg \
    $target_kl_arg \
    --log_std_init "$log_std_init" \
    --residual_scale "$residual_scale" \
    --include_target_vel true \
    --include_prev_action true \
    --use_numeric_encoder true \
    --use_backbone true \
    --swanlab_project "$swanlab_project" \
    ${swanlab_exp_arg:+$swanlab_exp_arg "$swanlab_experiment_name"} \
    $swanlab_flag \
    "${resume_args[@]}"
