#!/bin/bash
closed_loop=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
root_dir="${root_dir:-$(cd "$TEMPLATE_ROOT/.." && pwd)}"
dataset_root="$root_dir/Dataset"
scene_ids="City_1,City_2,City_3,City_4,City_5,City_6,City_7,City_8,City_9,City_10,City_11,City_12,City_13,City_14,City_15,City_16,City_17,City_18,City_19,City_20,City_21,City_22,City_23,City_24,City_25,City_26,City_27"
gpu_ids="0,2,3"
model_path="$root_dir/work_dirs/qwen3vl-uav-numeric-all-dagger"
model_dir_name="$(basename "$model_path")"
base_model_path="$root_dir/Qwen3-VL-2B-Instruct"
trajectory_range="451-500"
max_steps=""
save_results=true
sim_server_host="127.0.0.1"
sim_server_port=30000
success_dist_thresh_m=10
include_target_vel=true
include_prev_action=true
use_numeric_encoder=true
use_backbone=true

cd "$root_dir"
export PYTHONPATH="$root_dir:$PYTHONPATH"
export DAGGER_MULTI_WORKER=1

cleanup() {
    echo ""
    echo "[validate_seen] Interrupt received; terminating background validation processes..."
    kill $(jobs -p) 2>/dev/null
    exit 130
}

trap cleanup INT TERM

save_results_arg="--save_results"
{ [ "$save_results" = false ] || [ "$save_results" = "False" ]; } && save_results_arg="--no_save_results"

if [ "$closed_loop" = true ] || [ "$closed_loop" = "True" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    max_steps_arg=""
    if [ -n "$max_steps" ]; then
        max_steps_arg="--max_steps $max_steps"
    fi
    success_dist_arg=(--success_dist_thresh_m "$success_dist_thresh_m")
    CLOSED_LOOP_SCRIPT="$root_dir/UAV-Pursuit-Evasion-github-template/src/uav_pe/evaluation/closed_loop_eval.py"
    if [ ! -f "$CLOSED_LOOP_SCRIPT" ]; then
        echo "Error: not found: $CLOSED_LOOP_SCRIPT"
        exit 1
    fi
    base_model_args=()
    [ -n "$base_model_path" ] && base_model_args=(--base_model_path "$base_model_path")

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
            echo "Starting closed-loop validation: scenes=${scene_ids_csv} traj_range=${trajectory_range} GPU=${gpu_id} (progress_rank=${gi})"
            PROGRESS_RANK="$gi" CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u "$CLOSED_LOOP_SCRIPT"                 --scene_ids "$scene_ids_csv"                 --trajectory_range "$trajectory_range"                 --dataset_root "$dataset_root"                 --sim_server_host "$sim_server_host"                 --sim_server_port "$sim_server_port"                 --gpu_id 0                 --model_path "$model_path"                 "${base_model_args[@]}"                 --include_target_vel "$include_target_vel"                 --include_prev_action "$include_prev_action"                 --use_numeric_encoder "$use_numeric_encoder"                 --use_backbone "$use_backbone"                 --skip_existing                 "${success_dist_arg[@]}"                 $max_steps_arg                 $save_results_arg
        ) &
    done

    wait
    if command -v python3 &>/dev/null; then
        python3 -c "
import msgpackrpc

scene_ids_str = '$scene_ids'
scene_ids = [s.strip() for s in scene_ids_str.split(',') if s.strip()]

try:
    c = msgpackrpc.Client(msgpackrpc.Address('$sim_server_host', $sim_server_port), timeout=30)
    c.call('close_scenes', '$sim_server_host', scene_ids)
    c.close()
except Exception:
    pass
" 2>/dev/null || true
    fi
else
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    OFFLINE_SCRIPT="$root_dir/Val/scripts/offline_validate_policy.py"
    if [ ! -f "$OFFLINE_SCRIPT" ]; then
        echo "Error: not found: $OFFLINE_SCRIPT"
        exit 1
    fi
    max_frames_arg=""
    [ -n "$max_steps" ] && max_frames_arg="--max_frames $max_steps"
    base_model_args=()
    [ -n "$base_model_path" ] && base_model_args=(--base_model_path "$base_model_path")

    IFS=',' read -r -a scene_ids_arr <<< "$scene_ids"
    IFS=',' read -r -a gpu_ids_arr <<< "$gpu_ids"
    if [ "${#gpu_ids_arr[@]}" -eq 0 ]; then
        echo "Error: gpu_ids is empty"
        exit 1
    fi
    num_gpus=${#gpu_ids_arr[@]}

    for gi in "${!gpu_ids_arr[@]}"; do
        gpu_id="${gpu_ids_arr[$gi]}"
        (
            idx=0
            for scene_id in "${scene_ids_arr[@]}"; do
                if (( idx % num_gpus != gi )); then
                    idx=$((idx + 1))
                    continue
                fi

                if [[ "$trajectory_range" == *-* ]]; then
                    tr_start="${trajectory_range%-*}"
                    tr_end="${trajectory_range#*-}"
                else
                    tr_start="$trajectory_range"
                    tr_end="$trajectory_range"
                fi
                resume_start=""
                for ((i=tr_start; i<=tr_end; i++)); do
                    traj_name=$(printf "trajectory_%04d" "$i")
                    traj_dir="$root_dir/Val/results/$model_dir_name/seen/$scene_id/$traj_name"
                    if [ ! -f "$traj_dir/uav_trajectory.json" ]; then
                        resume_start="$i"
                        break
                    fi
                done
                if [ -z "$resume_start" ]; then
                    echo "Scene=${scene_id} already completed for [$tr_start,$tr_end]; skipping."
                    idx=$((idx + 1))
                    continue
                fi
                if [ "$resume_start" -eq "$tr_end" ]; then
                    tr_chunk="$resume_start"
                else
                    tr_chunk="${resume_start}-${tr_end}"
                fi

                echo "Starting offline validation: scene=${scene_id} traj_range=${tr_chunk} GPU=${gpu_id}"
                CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u "$OFFLINE_SCRIPT" \
                    --scene_id "$scene_id" \
                    --trajectory_range "$tr_chunk" \
                    --dataset_root "$dataset_root" \
                    --model_path "$model_path" \
                    "${base_model_args[@]}" \
                    --gpu_id 0 \
                    --include_target_vel "$include_target_vel" \
                    --include_prev_action "$include_prev_action" \
                    --use_numeric_encoder "$use_numeric_encoder" \
                    --use_backbone "$use_backbone" \
                    $save_results_arg \
                    $max_frames_arg

                idx=$((idx + 1))
            done
        ) &
    done

    wait
fi

root_results_dir="$root_dir/Val/results/$model_dir_name/seen"
if [ -d "$root_results_dir" ]; then
    echo ""
    echo "[validate_seen] Validation finished; evaluating SR/OSR/NE/SPL: $root_results_dir"
    cd "$root_dir" && python3 -u "$root_dir/UAV-Pursuit-Evasion-github-template/src/uav_pe/evaluation/evaluate_results.py" --root_dir "$root_results_dir"
    echo "[validate_seen] Evaluation finished"
else
    echo "[validate_seen] Results dir not found; skipping evaluation: $root_results_dir"
fi
