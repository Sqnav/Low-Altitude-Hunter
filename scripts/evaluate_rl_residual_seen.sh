#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

root_dir="${root_dir:-$PROJECT_ROOT/..}"
dataset_root="$root_dir/Dataset"

scene_ids="City_28,City_29,City_30"
gpu_ids="1"
trajectory_range="1-500"
max_steps=""
save_results=true

sim_server_host="127.0.0.1"
sim_server_port=30000

success_dist_thresh_m=10

include_target_vel=true
include_prev_action=true

use_numeric_encoder=true
use_backbone=true

il_model_path="$root_dir/work_dirs/qwen3vl-uav-numeric-all-dagger"

ppo_model_path="$root_dir/work_dirs/RL_residual/residual_ppo.zip"

base_model_path="$root_dir/Qwen3-VL-2B-Instruct"

skip_existing=true

ppo_deterministic=true

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --scene_ids) scene_ids="$2"; shift; shift ;;
    --gpu_ids) gpu_ids="$2"; shift; shift ;;
    --trajectory_range) trajectory_range="$2"; shift; shift ;;
    --max_steps) max_steps="$2"; shift; shift ;;
    --success_dist_thresh_m) success_dist_thresh_m="$2"; shift; shift ;;
    --include_target_vel) include_target_vel="$2"; shift; shift ;;
    --include_prev_action) include_prev_action="$2"; shift; shift ;;
    --use_numeric_encoder) use_numeric_encoder="$2"; shift; shift ;;
    --use_backbone) use_backbone="$2"; shift; shift ;;
    --il_model_path) il_model_path="$2"; shift; shift ;;
    --ppo_model_path) ppo_model_path="$2"; shift; shift ;;
    --base_model_path) base_model_path="$2"; shift; shift ;;
    --skip_existing) skip_existing="$2"; shift; shift ;;
    --ppo_deterministic) ppo_deterministic="$2"; shift; shift ;;
    --no_save_results) save_results=false; shift ;;
    *) echo "Unknown arg: $1"; shift ;;
  esac
done

MODEL_DIR_NAME="$(basename "$ppo_model_path")"
MODEL_DIR_NAME="${MODEL_DIR_NAME%.zip}"
MODEL_DIR_NAME="${MODEL_DIR_NAME%.pth}"

CLOSED_LOOP_SCRIPT="$root_dir/UAV-Pursuit-Evasion-github-template/src/uav_pe/evaluation/closed_loop_eval.py"
if [ ! -f "$CLOSED_LOOP_SCRIPT" ]; then
  echo "Error: not found: $CLOSED_LOOP_SCRIPT"
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTHONPATH="$root_dir:$PYTHONPATH"
export DAGGER_MULTI_WORKER=1

cleanup() {
  echo ""
  echo "[validate_rl_residual_seen] Interrupt received; terminating background validation processes..."
  kill $(jobs -p) 2>/dev/null
  exit 130
}
trap cleanup INT TERM

save_results_arg="--save_results"
{ [ "$save_results" = false ] || [ "$save_results" = "False" ]; } && save_results_arg="--no_save_results"

max_steps_arg=""
if [ -n "$max_steps" ]; then
  max_steps_arg="--max_steps $max_steps"
fi

success_dist_arg=(--success_dist_thresh_m "$success_dist_thresh_m")
det_arg=(--ppo_deterministic "$ppo_deterministic")

IFS=',' read -r -a scene_ids_arr <<< "$scene_ids"
IFS=',' read -r -a gpu_ids_arr <<< "$gpu_ids"

if [ "${#gpu_ids_arr[@]}" -eq 0 ]; then
  echo "Error: gpu_ids is empty"
  exit 1
fi

num_gpus=${#gpu_ids_arr[@]}
export VAL_SEEN_TOTAL_GPUS="$num_gpus"

for gi in "${!gpu_ids_arr[@]}"; do
  gpu_id="${gpu_ids_arr[$gi]}"
  (
    assigned_scenes=()
    idx=0
    for scene_id in "${scene_ids_arr[@]}"; do
      if (( idx % num_gpus == gi )); then
        assigned_scenes+=("$scene_id")
      fi
      idx=$((idx + 1))
    done
    if [ "${#assigned_scenes[@]}" -eq 0 ]; then
      exit 0
    fi

    scene_ids_csv=$(IFS=,; echo "${assigned_scenes[*]}")
    echo "Starting RL_residual closed-loop validation: scenes=${scene_ids_csv} traj_range=${trajectory_range} GPU=${gpu_id} (progress_rank=${gi})"

    PROGRESS_RANK="$gi" CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u "$CLOSED_LOOP_SCRIPT" \
      --scene_ids "$scene_ids_csv" \
      --trajectory_range "$trajectory_range" \
      --dataset_root "$dataset_root" \
      --sim_server_host "$sim_server_host" \
      --sim_server_port "$sim_server_port" \
      --gpu_id 0 \
      --il_model_path "$il_model_path" \
      --ppo_model_path "$ppo_model_path" \
      --base_model_path "$base_model_path" \
      --include_target_vel "$include_target_vel" \
      --include_prev_action "$include_prev_action" \
      --use_numeric_encoder "$use_numeric_encoder" \
      --use_backbone "$use_backbone" \
      ${skip_existing:+--skip_existing} \
      $max_steps_arg \
      $save_results_arg \
      ${success_dist_arg[@]} \
      ${det_arg[@]}
  ) &
done

wait

root_results_dir="$root_dir/Val/results/$MODEL_DIR_NAME/seen"
if [ -d "$root_results_dir" ]; then
  echo ""
  echo "[validate_rl_residual_seen] Validation finished; evaluating SR/OSR/NE/SPL: $root_results_dir"
  cd "$root_dir" && python3 -u "$root_dir/UAV-Pursuit-Evasion-github-template/src/uav_pe/evaluation/evaluate_results.py" --root_dir "$root_results_dir"
  echo "[validate_rl_residual_seen] Evaluation finished"
else
  echo "[validate_rl_residual_seen] Warning: results dir not found: $root_results_dir"
fi

