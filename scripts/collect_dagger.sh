SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
root_dir="${root_dir:-$(cd "$TEMPLATE_ROOT/.." && pwd)}"
dataset_root="$root_dir/Dataset"
scene_ids="City_1,City_2,City_3,City_4,City_5,City_6,City_7,City_8,City_9,City_10,City_11,City_12,City_13,City_14,City_15,City_16,City_17,City_18,City_19,City_20,City_21,City_22,City_23,City_24,City_25,City_26,City_27"
gpu_ids="0"
trajectory_range="1-450"

use_numeric_encoder=true
use_backbone=true

model_path="${DAGGER_MODEL_PATH:-$root_dir/work_dirs/qwen3vl-uav-numeric-all}"
base_model_path=""
base_model_path="$root_dir/Qwen3-VL-2B-Instruct"
sim_server_host="127.0.0.1"
sim_server_port=30000

expert_rounds=1

if [ -n "$EXPERT_RATIO" ]; then
    expert_ratio_list="$EXPERT_RATIO"
else
    expert_ratio_list=""
    if [ "$expert_rounds" -le 1 ]; then
        expert_ratio_list="0.5"
    else
        for ((i=0; i<expert_rounds; i++)); do
            num=$((expert_rounds - 1 - i))
            den=$((expert_rounds - 1))
            ratio=$(python3 - <<EOF
num = $num
den = $den
print(f"{num/den:.2f}")
EOF
)
            if [ -z "$expert_ratio_list" ]; then
                expert_ratio_list="$ratio"
            else
                expert_ratio_list="$expert_ratio_list,$ratio"
            fi
        done
    fi
fi

output_root_dir="$root_dir/Dagger_dataset"
if [ -n "$DAGGER_COLLECT_SUBDIR" ]; then
    output_dataset_dir="$output_root_dir/$DAGGER_COLLECT_SUBDIR"
else
    output_dataset_dir="$output_root_dir"
fi

max_steps=""
max_steps_ratio="1"
include_target_vel=true
include_prev_action=true

cd "$root_dir"
export PYTHONPATH="$root_dir:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DAgger_DEBUG_GT="${DAgger_DEBUG_GT:-1}"
export DAgger_DEBUG_DIST="${DAgger_DEBUG_DIST:-1}"

base_model_args=()
[ -n "$base_model_path" ] && base_model_args=(--base_model_path "$base_model_path")

max_steps_arg=""
if [ -n "$max_steps" ]; then
    max_steps_arg="--max_steps $max_steps"
elif [ -n "$max_steps_ratio" ]; then
    max_steps_arg="--max_steps_ratio $max_steps_ratio"
fi

IFS=',' read -r -a scene_ids_arr <<< "$scene_ids"
IFS=',' read -r -a gpu_ids_arr <<< "$gpu_ids"
if [ "${#gpu_ids_arr[@]}" -eq 0 ]; then
    echo "Error: gpu_ids is empty"
    exit 1
fi
if [ "${#scene_ids_arr[@]}" -eq 0 ]; then
    echo "Error: scene_ids is empty"
    exit 1
fi

export DAGGER_MULTI_WORKER=1

cleanup_dagger() {
    echo ""
    echo "[collect_dagger] Interrupt received; terminating background collection processes..."
    kill $(jobs -p) 2>/dev/null
    exit 130
}
trap cleanup_dagger INT TERM

echo "Starting DAgger multi-GPU collection: scenes=${#scene_ids_arr[@]} GPUs=${gpu_ids} traj_range=${trajectory_range} expert_ratios=${expert_ratio_list} model=${model_path} -> ${output_dataset_dir}"

num_gpus=${#gpu_ids_arr[@]}
for gpu_index in $(seq 0 $((num_gpus - 1))); do
    gpu_id="${gpu_ids_arr[$gpu_index]}"
    physical_gpu_id="$gpu_id"
    logical_gpu_id="0"
    scene_ids_for_gpu=""
    for ((i = gpu_index; i < ${#scene_ids_arr[@]}; i += num_gpus)); do
        sid="${scene_ids_arr[$i]}"
        if [ -z "$scene_ids_for_gpu" ]; then
            scene_ids_for_gpu="$sid"
        else
            scene_ids_for_gpu="$scene_ids_for_gpu,$sid"
        fi
    done
    if [ -z "$scene_ids_for_gpu" ]; then
        echo "GPU=${gpu_id} has no assigned scenes; skipping"
        continue
    fi
    echo "Starting DAgger collection: GPU=${gpu_id} scenes=${scene_ids_for_gpu}"
    PROGRESS_RANK="$gpu_index" CUDA_VISIBLE_DEVICES="$physical_gpu_id" python3 -u "$root_dir/UAV-Pursuit-Evasion-github-template/src/uav_pe/training/collect_dagger.py" \
        --scene_ids "$scene_ids_for_gpu" \
        --trajectory_range "$trajectory_range" \
        --dataset_root "$dataset_root" \
        --output_dataset_dir "$output_dataset_dir" \
        --sim_server_host "$sim_server_host" \
        --sim_server_port "$sim_server_port" \
        --gpu_id "$logical_gpu_id" \
        --sim_gpu_id "$physical_gpu_id" \
        --model_path "$model_path" \
        --expert_ratio_list "$expert_ratio_list" \
        --use_numeric_encoder "$use_numeric_encoder" \
        --use_backbone "$use_backbone" \
        "${base_model_args[@]}" \
        $max_steps_arg \
        --include_target_vel "$include_target_vel" \
        --include_prev_action "$include_prev_action" &
done
wait

if command -v python3 &>/dev/null; then
    python3 -c "
import msgpackrpc
try:
    c = msgpackrpc.Client(msgpackrpc.Address('$sim_server_host', $sim_server_port), timeout=30)
    c.call('close_scenes', '$sim_server_host')
    c.close()
except Exception:
    pass
" 2>/dev/null || true
fi

echo "DAgger multi-GPU collection finished"
