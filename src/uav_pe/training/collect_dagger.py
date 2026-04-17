#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as R
import msgpackrpc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Executor.core import TrajectoryExecutor
from Train_qwen.core.action_mapping import norm_action_to_physical, physical_action_to_norm
from Train_qwen.core.instruction_generator import generate_system_prompt, generate_user_prompt
from Train_qwen.core.train import save_instruction_jsons_for_dataset
from Val.scripts.offline_validate_policy import load_model_like_validate
from Val.scripts.closed_loop_airsim_test import policy_step
from RL.scripts.airsim_env import AirSimUAVTrainEnv
def close_scenes_rpc(sim_server_host, sim_server_port, scene_ids=None):
    try:
        c = msgpackrpc.Client(
            msgpackrpc.Address(sim_server_host, sim_server_port),
            timeout=30,
        )
        if scene_ids is not None and len(scene_ids) > 0:
            c.call("close_scenes", sim_server_host, list(scene_ids))
        else:
            c.call("close_scenes", sim_server_host)
        c.close()
    except Exception:
        pass


def make_executor(scene_id, trajectory_name, sim_server_host, sim_server_port, gpu_id):
    return TrajectoryExecutor(
        scene_id=scene_id,
        sim_server_host=sim_server_host,
        sim_server_port=sim_server_port,
        gpu_id=gpu_id,
        scene_index=1,
        uav_vehicle_name="Drone_1",
        target_object_name="UAV1",
        target_asset_name=None,
        target_object_scale=(1.0, 1.0, 1.0),
        uav_speed=5.0,
        target_speed=3.0,
        camera_name="0",
        auto_start_scene=True,
        deterministic_step_mode=True,
    )


def quaternion_to_euler(quat_w, quat_x, quat_y, quat_z):
    rot = R.from_quat([quat_x, quat_y, quat_z, quat_w])
    euler = rot.as_euler("xyz", degrees=False)
    return {"roll": float(euler[0]), "pitch": float(euler[1]), "yaw": float(euler[2])}


def to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        return obj.item()
    return obj


def parse_trajectory_range(s: str) -> list:
    if not s or not str(s).strip():
        return []
    s = str(s).strip()
    ids = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            try:
                lo, hi = int(lo_str.strip()), int(hi_str.strip())
                ids.extend(range(lo, hi + 1) if lo <= hi else range(hi, lo + 1))
            except ValueError:
                continue
        else:
            try:
                ids.append(int(part))
            except ValueError:
                continue
    return sorted(set(ids))


def is_trajectory_complete(output_dataset_dir: Path, scene_id: str, trajectory_name: str) -> bool:
    traj_dir = output_dataset_dir / scene_id / trajectory_name
    uav_json = traj_dir / "uav_trajectory.json"
    target_json = traj_dir / "target_trajectory.json"
    rgb_dir = traj_dir / "rgb"
    if not uav_json.exists() or not target_json.exists() or not rgb_dir.is_dir():
        return False
    try:
        with open(uav_json, "r", encoding="utf-8") as f:
            uav_data = json.load(f)
        num_frames = int(uav_data.get("num_frames", 0))
        if num_frames <= 0:
            return False
        for i in range(num_frames):
            if not (rgb_dir / f"frame_{i:05d}.png").exists():
                return False
        return True
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def run_dagger(args, model=None, processor=None, executor=None, close_executor=True):
    if th.cuda.is_available():
        try:
            logical_gpu = int(getattr(args, "gpu_id", 0))
        except Exception:
            logical_gpu = 0
        device = th.device(f"cuda:{logical_gpu}")
    else:
        device = th.device("cpu")
    base_model_path = (getattr(args, "base_model_path", None) or "").strip() or None
    if base_model_path is None:
        base_model_path = str(PROJECT_ROOT / "Qwen3-VL-2B-Instruct")

    if model is None or processor is None:
        _use_backbone = True
        _use_numeric = True
        model, processor = load_model_like_validate(
            model_path=args.model_path,
            base_model_path=base_model_path,
            device=device,
            use_numeric_encoder=_use_numeric,
            use_backbone=_use_backbone,
        )
    else:
        base_model_path = base_model_path or str(PROJECT_ROOT / "Qwen3-VL-2B-Instruct")
    inc_vel = True
    inc_prev = True

    for p in model.parameters():
        p.requires_grad = False

    model_dict = {
        "model": model,
        "processor": processor or getattr(model, "processor", None),
        "device": device,
        "checkpoint_dir": str(Path(args.model_path)),
        "base_model_path": base_model_path,
        "generate_system_prompt": generate_system_prompt,
        "generate_user_prompt": generate_user_prompt,
        "include_target_vel": inc_vel,
        "include_prev_action": inc_prev,
    }

    if executor is None:
        sim_gpu_id = getattr(args, "sim_gpu_id", None)
        if sim_gpu_id is None:
            sim_gpu_id = getattr(args, "gpu_id", 0)
        executor = make_executor(
            scene_id=args.scene_id,
            trajectory_name=args.trajectory_name,
            sim_server_host=args.sim_server_host,
            sim_server_port=args.sim_server_port,
            gpu_id=int(sim_gpu_id),
        )

    trajectory_range = getattr(args, "trajectory_range", None)
    if isinstance(trajectory_range, str) and not trajectory_range.strip():
        trajectory_range = None
    env = AirSimUAVTrainEnv(
        model_dict=model_dict,
        dataset_root=Path(args.dataset_root),
        scene_id=args.scene_id,
        trajectory_name=args.trajectory_name,
        executor=executor,
        max_vel=getattr(args, "max_vel", 5.0),
        max_yaw_rate=getattr(args, "max_yaw_rate", 45.0),
        yaw_scale=1.0,
        reward_progress_scale=1.0,
        trajectory_range=trajectory_range,
        reward_type="progress",
        reward_r_level=10.0,
        max_steps=getattr(args, "max_steps", None),
        max_steps_ratio=getattr(args, "max_steps_ratio", None),
        use_gt_action_loss=True,
        gt_action_loss_weight=0.0,
    )

    max_vel = getattr(args, "max_vel", 5.0)
    max_yaw_rate = getattr(args, "max_yaw_rate", 45.0)
    beta = getattr(args, "expert_ratio", 0.4)
    output_dataset_dir = Path(getattr(args, "output_dataset_dir", str(PROJECT_ROOT / "Dagger" / "collected_dataset")))
    scene_id = args.scene_id
    trajectory_name = env.trajectory_name
    dataset_dir = output_dataset_dir / scene_id / trajectory_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = dataset_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=args.seed)
    merged_trajectory_data = []
    target_trajectory_airsim = []
    last_a_phys = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    step_count = 0
    target_asset_name = getattr(env, "_target_asset_name", "UAV1")
    last_target_airsim = None

    max_steps = getattr(env, "max_steps", None)
    try:
        from tqdm import tqdm  # type: ignore
        _progress_rank = os.environ.get("PROGRESS_RANK", "0")
        _tqdm_position = int(_progress_rank) if str(_progress_rank).isdigit() else 0
        step_pbar = tqdm(
            total=max_steps,
            desc=f"DAgger:{scene_id}/{trajectory_name}",
            ncols=120,
            position=_tqdm_position,
            leave=True,
        )
    except Exception:
        step_pbar = None

    while True:
        rgb_img, depth_img = executor.get_camera_images()
        if rgb_img is not None:
            rgb_path = rgb_dir / f"frame_{step_count:05d}.png"
            if isinstance(rgb_img, np.ndarray):
                cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            else:
                from PIL import Image
                cv2.imwrite(str(rgb_path), cv2.cvtColor(np.array(rgb_img.convert("RGB")), cv2.COLOR_RGB2BGR))

        uav_state = executor.get_uav_state()
        cur_pos = uav_state["position"]  # AirSim NED
        uav_pos_world = np.array([float(cur_pos[0]), float(cur_pos[1]), float(-cur_pos[2])])
        quat = uav_state["orientation"]
        uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        uav_euler = quaternion_to_euler(uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z)

        next_idx = min(step_count + 1, len(env.target_traj_airsim) - 1)
        next_target_airsim = env.target_traj_airsim[next_idx]
        next_target_world = np.array([
            float(next_target_airsim[0]),
            float(next_target_airsim[1]),
            float(-next_target_airsim[2]),
        ])
        target_trajectory_airsim.append({
            "x": next_target_world[0],
            "y": next_target_world[1],
            "z": next_target_world[2],
        })

        rel_pos_airsim = np.asarray(next_target_airsim, dtype=np.float32) - np.asarray(cur_pos, dtype=np.float32)
        dist_to_target = float(np.linalg.norm(rel_pos_airsim))

        is_capture_frame = dist_to_target < 5.0
        if is_capture_frame:
            frame_data = {
                "frame_idx": step_count,
                "uav_position": {"x": uav_pos_world[0], "y": uav_pos_world[1], "z": uav_pos_world[2]},
                "uav_orientation_quaternion": {"w": uav_quat_w, "x": uav_quat_x, "y": uav_quat_y, "z": uav_quat_z},
                "uav_orientation_euler": uav_euler,
                "target_position": {"x": next_target_world[0], "y": next_target_world[1], "z": next_target_world[2]},
                "distance": dist_to_target,
            }
        else:
            a_expert_phys = env.compute_gt_phys_action()
            a_expert_norm = physical_action_to_norm(a_expert_phys, max_vel, max_yaw_rate)
            if isinstance(a_expert_norm, np.ndarray):
                a_expert_norm = np.asarray(a_expert_norm, dtype=np.float32).reshape(4)
            else:
                a_expert_norm = a_expert_norm.detach().cpu().numpy().astype(np.float32).reshape(4)
            a_expert_phys_arr = norm_action_to_physical(a_expert_norm, max_vel, max_yaw_rate)
            if isinstance(a_expert_phys_arr, np.ndarray):
                a_expert_phys_arr = np.asarray(a_expert_phys_arr, dtype=np.float32).reshape(4)
            else:
                a_expert_phys_arr = a_expert_phys_arr.detach().cpu().numpy().astype(np.float32).reshape(4)
            frame_data = {
                "frame_idx": step_count,
                "uav_position": {"x": uav_pos_world[0], "y": uav_pos_world[1], "z": uav_pos_world[2]},
                "uav_orientation_quaternion": {"w": uav_quat_w, "x": uav_quat_x, "y": uav_quat_y, "z": uav_quat_z},
                "uav_orientation_euler": uav_euler,
                "target_position": {"x": next_target_world[0], "y": next_target_world[1], "z": next_target_world[2]},
                "velocity_in_body_frame": {
                    "x": last_a_phys[0],
                    "y": last_a_phys[1],
                    "z": last_a_phys[2],
                },
                "yaw_rate": float(last_a_phys[3]),
                "distance": dist_to_target,
                "gt_velocity_in_body_frame": {
                    "x": float(a_expert_phys_arr[0]),
                    "y": float(a_expert_phys_arr[1]),
                    "z": float(a_expert_phys_arr[2]),
                },
                "gt_yaw_rate": float(a_expert_phys_arr[3]),
                "gt_action_norm": [float(a_expert_norm[0]), float(a_expert_norm[1]), float(a_expert_norm[2]), float(a_expert_norm[3])],
            }
        merged_trajectory_data.append(frame_data)

        if is_capture_frame:
            if step_pbar is not None:
                step_pbar.set_postfix(dist=f"{dist_to_target:.1f}m", act="(end)", refresh=False)
            break

        if getattr(model, "use_numeric_encoder", False):
            obs_for_policy = {
                "rgb": rgb_img,
                "uav_position_airsim": [float(cur_pos[0]), float(cur_pos[1]), float(cur_pos[2])],
                "target_position_airsim": [
                    float(next_target_airsim[0]),
                    float(next_target_airsim[1]),
                    float(next_target_airsim[2]),
                ],
                "target_position_airsim_prev": (
                    None
                    if last_target_airsim is None
                    else [
                        float(last_target_airsim[0]),
                        float(last_target_airsim[1]),
                        float(last_target_airsim[2]),
                    ]
                ),
                "quaternion": [uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z],
                "previous_action": [
                    float(last_a_phys[0]),
                    float(last_a_phys[1]),
                    float(last_a_phys[2]),
                    float(last_a_phys[3]),
                ],
                "is_first_frame": (step_count == 0),
            }
            with th.no_grad():
                a_pi_norm, _ = policy_step(model_dict, obs_for_policy, debug_save_path=None)
            a_pi = np.asarray(a_pi_norm, dtype=np.float32).reshape(4)
        else:
            obs_np = np.asarray(obs, dtype=np.float32)
            head_dtype = next(model.action_head.parameters()).dtype
            obs_t = th.from_numpy(obs_np).to(device=device, dtype=head_dtype).unsqueeze(0)
            with th.no_grad():
                raw_pi = model.action_head(obs_t)
                a_pi = th.tanh(raw_pi).squeeze(0).cpu().numpy().astype(np.float32)
        if np.random.rand() < beta:
            a_exec = a_expert_norm.copy()
        else:
            a_exec = a_pi
        a_exec = np.clip(a_exec, -1.0, 1.0).astype(np.float32)
        last_a_phys = norm_action_to_physical(a_exec, max_vel, max_yaw_rate)
        if isinstance(last_a_phys, np.ndarray):
            last_a_phys = np.asarray(last_a_phys, dtype=np.float32).reshape(4)
        else:
            last_a_phys = last_a_phys.detach().cpu().numpy().astype(np.float32).reshape(4)
        last_target_airsim = np.asarray(next_target_airsim, dtype=np.float32).reshape(3)

        next_obs, r, term, trunc, info = env.step(a_exec)
        obs = next_obs
        step_count += 1

        if step_pbar is not None:
            step_pbar.update(1)
            step_pbar.set_postfix(
                dist=f"{dist_to_target:.1f}m",
                act=f"({last_a_phys[0]:.1f},{last_a_phys[1]:.1f},{last_a_phys[2]:.1f},{last_a_phys[3]:.1f})",
                refresh=False,
            )

        if bool(term or trunc):
            rgb_img, depth_img = executor.get_camera_images()
            if rgb_img is not None:
                rgb_path = rgb_dir / f"frame_{step_count:05d}.png"
                if isinstance(rgb_img, np.ndarray):
                    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                else:
                    from PIL import Image
                    cv2.imwrite(str(rgb_path), cv2.cvtColor(np.array(rgb_img.convert("RGB")), cv2.COLOR_RGB2BGR))
            uav_state = executor.get_uav_state()
            cur_pos = uav_state["position"]
            uav_pos_world = np.array([float(cur_pos[0]), float(cur_pos[1]), float(-cur_pos[2])])
            quat = uav_state["orientation"]
            uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            uav_euler = quaternion_to_euler(uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z)
            next_idx = min(step_count, len(env.target_traj_airsim) - 1)
            next_target_airsim = env.target_traj_airsim[next_idx]
            next_target_world = np.array([
                float(next_target_airsim[0]), float(next_target_airsim[1]), float(-next_target_airsim[2]),
            ])
            target_trajectory_airsim.append({
                "x": next_target_world[0], "y": next_target_world[1], "z": next_target_world[2],
            })
            rel_pos_airsim = np.asarray(next_target_airsim, dtype=np.float32) - np.asarray(cur_pos, dtype=np.float32)
            dist_to_target = float(np.linalg.norm(rel_pos_airsim))
            frame_data = {
                "frame_idx": step_count,
                "uav_position": {"x": uav_pos_world[0], "y": uav_pos_world[1], "z": uav_pos_world[2]},
                "uav_orientation_quaternion": {"w": uav_quat_w, "x": uav_quat_x, "y": uav_quat_y, "z": uav_quat_z},
                "uav_orientation_euler": uav_euler,
                "target_position": {"x": next_target_world[0], "y": next_target_world[1], "z": next_target_world[2]},
                "distance": dist_to_target,
            }
            merged_trajectory_data.append(frame_data)
            if step_pbar is not None:
                step_pbar.set_postfix(dist=f"{dist_to_target:.1f}m", act="(end)", refresh=False)
            break

    if step_pbar is not None:
        step_pbar.close()

    try:
        if executor is not None and getattr(executor, "client", None) is not None:
            obj_name = getattr(executor, "target_object_name", None)
            if isinstance(obj_name, str) and obj_name:
                try:
                    executor.client.simDestroyObject(obj_name)
                    try:
                        executor.client.simContinueForFrames(1)
                    except Exception:
                        pass
                except Exception:
                    try:
                        existing = executor.client.simListSceneObjects(obj_name + ".*")
                        for n in existing or []:
                            try:
                                executor.client.simDestroyObject(n)
                            except Exception:
                                pass
                        try:
                            executor.client.simContinueForFrames(1)
                        except Exception:
                            pass
                    except Exception:
                        pass
    except Exception:
        pass

    num_frames = len(merged_trajectory_data)

    uav_traj_path = dataset_dir / "uav_trajectory.json"
    with open(uav_traj_path, "w", encoding="utf-8") as f:
        json.dump(to_json_serializable({
            "num_frames": num_frames,
            "target_asset_name": target_asset_name,
            "trajectory": merged_trajectory_data,
        }), f, indent=2, ensure_ascii=False)

    target_traj_path = dataset_dir / "target_trajectory.json"
    with open(target_traj_path, "w", encoding="utf-8") as f:
        json.dump(to_json_serializable({
            "num_frames": num_frames,
            "target_trajectory_airsim": target_trajectory_airsim,
        }), f, indent=2, ensure_ascii=False)

    list_data_dict = [
        {"json": f"{scene_id}/{trajectory_name}/uav_trajectory.json", "frame": i}
        for i in range(num_frames)
    ]
    save_instruction_jsons_for_dataset(
        str(output_dataset_dir),
        list_data_dict,
        include_target_vel=inc_vel,
        include_prev_action=inc_prev,
    )

    if close_executor:
        try:
            executor.disconnect()
        except Exception:
            pass
        executor = None
    return model, processor, executor


def parse_args():
    p = argparse.ArgumentParser(description="DAgger data collection: rollout with expert mixing; save Dataset-format outputs")
    p.add_argument("--model_path", type=str, default=str(PROJECT_ROOT / "work_dirs" / "qwen3vl-uav-1"))
    p.add_argument("--base_model_path", type=str, default="")
    p.add_argument("--scene_id", type=str, default="City_1", help="Scene id for single-scene mode")
    p.add_argument("--scene_ids", type=str, default="", help="Comma-separated scene ids for multi-scene sequential run")
    p.add_argument("--trajectory_name", type=str, default="trajectory_0001")
    p.add_argument("--trajectory_range", type=str, default="", help='e.g. "1-1" or "1-5"')
    p.add_argument("--dataset_root", type=str, default=str(PROJECT_ROOT / "Dataset"))
    p.add_argument("--output_dataset_dir", type=str, default=str(PROJECT_ROOT / "Dagger" / "collected_dataset"), help="Output root dir (same structure as Dataset)")
    p.add_argument("--sim_server_host", type=str, default="127.0.0.1")
    p.add_argument("--sim_server_port", type=int, default=30000)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--sim_gpu_id", type=int, default=None)
    p.add_argument("--max_vel", type=float, default=5.0)
    p.add_argument("--max_yaw_rate", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--max_steps_ratio", type=float, default=None)
    p.add_argument("--expert_ratio", type=float, default=0.4, help="Expert ratio beta: per-step probability to use expert action")
    p.add_argument("--expert_ratio_list", type=str, default="", help="Comma-separated expert ratios to run sequentially in one process")
    p.add_argument("--include_target_vel", type=str, default="true")
    p.add_argument("--include_prev_action", type=str, default="true")
    p.add_argument("--use_numeric_encoder", type=str, default="true")
    p.add_argument("--use_backbone", type=str, default="true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    traj_ids = parse_trajectory_range(getattr(args, "trajectory_range", ""))
    if traj_ids:
        trajectory_names = [f"trajectory_{tid:04d}" for tid in traj_ids]
        args.trajectory_range = None
    else:
        trajectory_names = [args.trajectory_name]

    total_rounds_env = os.environ.get("DAGGER_TOTAL_ROUNDS")
    round_idx_env = os.environ.get("DAGGER_ROUND_IDX")
    try:
        total_rounds = int(total_rounds_env) if total_rounds_env is not None else 1
        round_idx = int(round_idx_env) if round_idx_env is not None else 1
    except ValueError:
        total_rounds, round_idx = 1, 1
    n_traj = len(trajectory_names)
    if total_rounds > 1 and n_traj > 0:
        if round_idx < 1:
            round_idx = 1
        if round_idx > total_rounds:
            round_idx = total_rounds
        start = n_traj * (round_idx - 1) // total_rounds
        end = n_traj * round_idx // total_rounds
        sub_trajectory_names = trajectory_names[start:end]
        if not sub_trajectory_names:
            print(f"[DAgger] Round {round_idx}/{total_rounds}: no trajectories assigned; exiting.")
            sys.exit(0)
        print(f"[DAgger] Round {round_idx}/{total_rounds}: {sub_trajectory_names[0]} ~ {sub_trajectory_names[-1]} ({len(sub_trajectory_names)} of {n_traj})")
        trajectory_names = sub_trajectory_names

    ratio_list_str = (getattr(args, "expert_ratio_list", "") or "").strip()
    ratio_list = []
    if ratio_list_str:
        for part in ratio_list_str.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ratio_list.append(float(part))
            except ValueError:
                continue
    if not ratio_list:
        ratio_list = [getattr(args, "expert_ratio", 0.4)]

    base_output_root = Path(getattr(args, "output_dataset_dir", str(PROJECT_ROOT / "Dagger" / "collected_dataset")))
    scene_ids_str = (getattr(args, "scene_ids", "") or "").strip()
    if scene_ids_str:
        scene_ids_list = [s.strip() for s in scene_ids_str.split(",") if s.strip()]
    else:
        scene_ids_list = [args.scene_id]

    num_ratios = len(ratio_list)
    traj_chunks = []
    if num_ratios > 0 and trajectory_names:
        n = len(trajectory_names)
        for i in range(num_ratios):
            start = n * i // num_ratios
            end = n * (i + 1) // num_ratios
            traj_chunks.append(trajectory_names[start:end])
    else:
        traj_chunks = [trajectory_names] if trajectory_names else []

    if len(trajectory_names) > 1:
        print(f"[DAgger] Trajectory range: {trajectory_names[0]} ~ {trajectory_names[-1]} ({len(trajectory_names)})")
        for i, beta in enumerate(ratio_list):
            chunk = traj_chunks[i] if i < len(traj_chunks) else []
            if chunk:
                print(f"[DAgger] beta={beta} trajectories: {chunk[0]} ~ {chunk[-1]} ({len(chunk)})")
    if len(scene_ids_list) > 1:
        print(f"[DAgger] Multi-scene sequential run: {scene_ids_list[0]} ~ {scene_ids_list[-1]} ({len(scene_ids_list)} scenes)")

    model, processor = None, None
    shared_executor = None
    previous_scene_id = None
    try:
        for scene_id in scene_ids_list:
            if previous_scene_id is not None:
                close_scenes_rpc(args.sim_server_host, args.sim_server_port, [previous_scene_id])
            args.scene_id = scene_id
            if len(scene_ids_list) > 1:
                print(f"[DAgger] Starting scene {scene_id}")

            for idx, beta in enumerate(ratio_list):
                output_root = base_output_root
                args.expert_ratio = beta
                args.output_dataset_dir = str(output_root)
                chunk = traj_chunks[idx] if idx < len(traj_chunks) else []
                if not chunk:
                    continue
                print(f"[DAgger] beta={beta} trajectories={len(chunk)} output_dir={output_root}")

                for traj_name in chunk:
                    if is_trajectory_complete(output_root, args.scene_id, traj_name):
                        print(f"[DAgger] Skipping complete trajectory: {args.scene_id}/{traj_name} (beta={beta})")
                        continue
                    args.trajectory_name = traj_name
                    model, processor, shared_executor = run_dagger(
                        args,
                        model=model,
                        processor=processor,
                        executor=shared_executor,
                        close_executor=False,
                    )
                print(f"[DAgger] Finished beta={beta}")

            if shared_executor is not None:
                try:
                    shared_executor.disconnect()
                except Exception:
                    pass
                shared_executor = None
            previous_scene_id = scene_id

        print(f"[DAgger] Done. betas={ratio_list}")
    finally:
        if shared_executor is not None:
            try:
                shared_executor.disconnect()
            except Exception:
                pass
        if os.environ.get("DAGGER_MULTI_WORKER") == "1":
            pass
        else:
            close_scenes_rpc(args.sim_server_host, args.sim_server_port, None)
