#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
import cv2
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
import numpy as np
from PIL import Image
import msgpackrpc
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from Executor.core import TrajectoryExecutor  # type: ignore
from Train_qwen.core.action_mapping import norm_action_to_physical  # type: ignore
from Train_qwen.core.step0_debug_utils import (  # type: ignore
    get_action_head_input_from_backbone_outputs,
    save_step0_action_head_input,
)
from Val.scripts.offline_validate_policy import load_model_like_validate
MAX_VEL = 5.0
MAX_YAW_RATE = 45.0
SUCCESS_DIST_THRESH_M = 5.0
_drone_detector_model: Any = None
_drone_detector_device: str = "cpu"
def _get_drone_detector(device: str):
    global _drone_detector_model, _drone_detector_device
    if _drone_detector_model is not None and _drone_detector_device == device:
        return _drone_detector_model
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    dino_root = PROJECT_ROOT / "Detector" / "GroundingDINO"
    if str(dino_root) not in sys.path:
        sys.path.insert(0, str(dino_root))
    from groundingdino.util.inference import load_model as load_dino_model, predict
    import groundingdino.datasets.transforms as T
    config_path = PROJECT_ROOT / "Detector" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    ckpt_path = PROJECT_ROOT / "Detector" / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"
    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"GroundingDINO not found: {config_path} or {ckpt_path}")
    _drone_detector_model = load_dino_model(str(config_path), str(ckpt_path), device=device)
    _drone_detector_device = device
    return _drone_detector_model
def _detect_drone_in_rgb(
    rgb_image: np.ndarray,
    device: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> bool:
    try:
        if rgb_image is None:
            return False
        if hasattr(rgb_image, "convert"):
            pil_img = rgb_image.convert("RGB")
        else:
            pil_img = Image.fromarray(np.asarray(rgb_image).astype(np.uint8)).convert("RGB")
        dino_root = PROJECT_ROOT / "Detector" / "GroundingDINO"
        if str(dino_root) not in sys.path:
            sys.path.insert(0, str(dino_root))
        import groundingdino.datasets.transforms as T_dino
        from groundingdino.util.inference import predict
        transform = T_dino.Compose([
            T_dino.RandomResize([800], max_size=1333),
            T_dino.ToTensor(),
            T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(pil_img, None)
        model = _get_drone_detector(device)
        boxes, logits, phrases = predict(
            model=model,
            image=image_transformed,
            caption="Drone",
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
        return boxes is not None and len(boxes) > 0
    except Exception as e:
        print(f"[closed-loop] Drone detection error: {e}")
        return False
def world_to_airsim_pos(p_world: np.ndarray) -> np.ndarray:
    p_world = np.asarray(p_world, dtype=np.float32).reshape(3,)
    return np.array([p_world[0], p_world[1], -p_world[2]], dtype=np.float32)
def airsim_to_world_pos(p_airsim: np.ndarray) -> np.ndarray:
    p_airsim = np.asarray(p_airsim, dtype=np.float32).reshape(3,)
    return np.array([p_airsim[0], p_airsim[1], -p_airsim[2]], dtype=np.float32)
def body_z_airsim_to_saved(z_body_airsim: float) -> float:
    return float(-z_body_airsim)
def get_param_norm(model, norm_type=2.0):
    params = [p.data for p in model.parameters()]
    device = params[0].device
    total = torch.zeros((), device=device)
    for p in params:
        total += p.norm(norm_type) ** norm_type
    return total.pow(1.0 / norm_type).item()
def parse_trajectory_range(trajectory_range: str) -> List[int]:
    s = trajectory_range.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        start = int(a.strip())
        end = int(b.strip())
        return list(range(start, end + 1))
    return [int(s)]
def trajectory_numbers_to_names(numbers: List[int]) -> List[str]:
    return [f"trajectory_{n:04d}" for n in numbers]

def load_uav_and_target_trajectories(
    dataset_root: Path,
    scene_id: str,
    trajectory_name: str,
) -> Tuple[np.ndarray, List[np.ndarray], str]:
    traj_dir = dataset_root / scene_id / trajectory_name
    uav_json = traj_dir / "uav_trajectory.json"
    target_json = traj_dir / "target_trajectory.json"
    if not uav_json.exists():
        raise FileNotFoundError(f"UAV trajectory file not found: {uav_json}")
    if not target_json.exists():
        raise FileNotFoundError(f"Target trajectory file not found: {target_json}")
    with uav_json.open("r", encoding="utf-8") as f:
        uav_data = json.load(f)
    with target_json.open("r", encoding="utf-8") as f:
        target_data = json.load(f)
    if "trajectory" not in uav_data or len(uav_data["trajectory"]) == 0:
        raise ValueError(f"No valid UAV trajectory data in {uav_json}")
    first_frame = uav_data["trajectory"][0]
    if "uav_position" not in first_frame:
        raise ValueError(f"{uav_json}: frame 0 missing 'uav_position'")
    uav_pos = first_frame["uav_position"]
    uav_start_world = np.array(
        [float(uav_pos["x"]), float(uav_pos["y"]), float(uav_pos["z"])],
        dtype=np.float32,
    )
    uav_start_airsim = np.array(
        [uav_start_world[0], uav_start_world[1], -uav_start_world[2]],
        dtype=np.float32,
    )
    target_asset_name = uav_data.get("target_asset_name", "UAV1")
    if not isinstance(target_asset_name, str):
        target_asset_name = str(target_asset_name)
    if "target_trajectory_airsim" not in target_data:
        raise ValueError(f"{target_json}: missing 'target_trajectory_airsim'")
    target_traj_list: List[np.ndarray] = []
    for p in target_data["target_trajectory_airsim"]:
        x = float(p["x"])
        y = float(p["y"])
        z_world = float(p["z"])
        target_traj_list.append(
            np.array([x, y, -z_world], dtype=np.float32)
        )
    return uav_start_airsim, target_traj_list, target_asset_name

def load_model(
    model_path: str,
    base_model_path: str | Path | None = None,
    device: Any = None,
    use_numeric_encoder: bool = False,
    use_backbone: bool = True,
) -> Dict[str, Any]:
    if base_model_path is None:
        base_model_path = str(PROJECT_ROOT / "Qwen3-VL-2B-Instruct")
    else:
        base_model_path = str(base_model_path)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_like_validate(
        model_path=model_path,
        base_model_path=base_model_path,
        device=device,
        use_numeric_encoder=use_numeric_encoder,
        use_backbone=use_backbone,
    )
    print("base_model_path =", base_model_path)
    print("model class     =", type(model.backbone))
    print("name_or_path    =", getattr(model.backbone.config, "_name_or_path", None))
    p = next(model.backbone.parameters()).detach().flatten()[:5].cpu()
    print("first5 params   =", p)
    from Train_qwen.core.instruction_generator import generate_system_prompt, generate_user_prompt
    return {
        "model": model,
        "processor": processor or getattr(model, "processor", None),
        "generate_system_prompt": generate_system_prompt,
        "generate_user_prompt": generate_user_prompt,
        "device": device,
        "checkpoint_dir": str(Path(model_path)),
        "base_model_path": base_model_path,
    }
def policy_step(
    model_dict: Dict[str, Any],
    obs: Dict[str, Any],
    debug_save_path: str | Path | None = None,
) -> Tuple[np.ndarray, str]:
    import torch
    from PIL import Image
    from Train_qwen.core.instruction_generator import compute_instruction_numeric_state
    
    model = model_dict['model']
    processor = model_dict.get('processor') or getattr(model, "processor", None)
    device = model_dict['device']
    if processor is None:
        raise ValueError("processor is None: check load_model() return or model.processor")
    rgb_img = obs.get('rgb')
    if rgb_img is None:
        raise ValueError("obs missing 'rgb'")
    if isinstance(rgb_img, np.ndarray):
        rgb_img = Image.fromarray(rgb_img).convert("RGB")
    elif not isinstance(rgb_img, Image.Image):
        raise ValueError(f"Unsupported rgb_img type: {type(rgb_img)}")
    uav_pos = obs.get("uav_position_airsim")
    target_pos = obs.get("target_position_airsim")
    quat = obs.get("quaternion")
    prev_action = obs.get("previous_action")
    if uav_pos is None or target_pos is None or quat is None or prev_action is None:
        raise ValueError(
            "obs must include uav_position_airsim, target_position_airsim, quaternion, previous_action"
        )
    system_prompt = model_dict["generate_system_prompt"]()
    _inc_vel = model_dict.get("include_target_vel", True)
    _inc_prev = model_dict.get("include_prev_action", True)
    user_text = model_dict["generate_user_prompt"](
        uav_position_airsim=uav_pos,
        target_position_airsim=target_pos,
        quaternion=quat,
        prev_action=prev_action,
        target_position_airsim_prev=obs.get("target_position_airsim_prev"),
        dt=1.0,
        include_target_vel=_inc_vel,
        include_prev_action=_inc_prev,
        is_first_frame=obs.get("is_first_frame", False),
    )
    
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb_img},
                {"type": "text", "text": user_text}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    inputs = processor(
        text=[text],
        images=[rgb_img],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    debug_path = None
    if debug_save_path is not None:
        try:
            debug_path = Path(debug_save_path)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            ids = inputs["input_ids"][0].tolist()
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump({"text": text, "input_ids": ids}, f, indent=2, ensure_ascii=False)
            print(f"[closed_loop] Saved step0 debug JSON: {debug_path}")
        except Exception as e:
            print(f"[closed_loop] Failed to save step0 debug JSON: {e}")
            debug_path = None
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    if debug_path is not None:
        try:
            with torch.no_grad():
                backbone_outputs = model.backbone(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                action_head_input = get_action_head_input_from_backbone_outputs(
                    backbone_outputs.hidden_states[-1],
                    inputs["attention_mask"],
                    backbone_outputs.hidden_states[-1].device,
                )
            hidden_path = debug_path.parent / "step0_hidden.npy"
            save_step0_action_head_input(
                hidden_path,
                action_head_input,
                log_prefix="[closed_loop]",
            )
        except Exception as e:
            print(f"[closed_loop] Failed to save step0 action_head input: {e}")
    pv = inputs.get("pixel_values")
    if pv is None:
        raise ValueError(
            "[closed_loop] processor did not return pixel_values; check obs['rgb'] and processor call"
        )
    if pv.numel() == 0 or (pv.abs().sum().item() == 0):
        import warnings
        warnings.warn(
            "[closed_loop] pixel_values are all zeros or empty; the model likely received no valid image input",
            UserWarning,
            stacklevel=2,
        )
    with torch.no_grad():
        model_kwargs = {}
        if getattr(model, "use_numeric_encoder", False):
            num_vals = compute_instruction_numeric_state(
                uav_position_airsim=uav_pos,
                target_position_airsim=target_pos,
                quaternion=quat,
                prev_action=prev_action,
                target_position_airsim_prev=obs.get("target_position_airsim_prev"),
                dt=1.0,
            )
            first_param = next(model.backbone.parameters())
            num_state = torch.tensor(
                [list(num_vals)],
                dtype=first_param.dtype,
                device=device,
            )
            model_kwargs["num_state"] = num_state
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            pixel_values=inputs.get("pixel_values").to(device) if inputs.get("pixel_values") is not None else None,
            image_grid_thw=inputs.get("image_grid_thw").to(device) if inputs.get("image_grid_thw") is not None else None,
            action=None,
            traj_id=None,
            **model_kwargs,
        )
        norm_action = outputs.action.cpu().numpy()
        if len(norm_action.shape) > 1:
            norm_action = norm_action[0]
        return norm_action.astype(np.float32), user_text
def apply_action_to_uav(
    executor: TrajectoryExecutor,
    uav_state: Dict[str, Any],
    action: np.ndarray,
) -> None:
    from scipy.spatial.transform import Rotation as R  # type: ignore
    import airsim  # type: ignore
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != 4:
        raise ValueError(f"action must have shape (4,), got {action.shape}")
    vx, vy, vz, yaw_rate = float(action[0]), float(action[1]), float(action[2]), float(action[3])
    cur_pos = uav_state["position"].astype(np.float32)  # AirSim world (NED)
    cur_quat = uav_state["orientation"]                 # [w, x, y, z]
    rotation = R.from_quat([cur_quat[1], cur_quat[2], cur_quat[3], cur_quat[0]])  # [x,y,z,w]
    vel_body = np.array([vx, vy, vz], dtype=np.float32)
    delta_world = rotation.apply(vel_body)
    target_pos = cur_pos + delta_world
    euler = rotation.as_euler('xyz', degrees=False)
    target_yaw = float(euler[2]) + float(np.deg2rad(yaw_rate))
    quat = airsim.to_quaternion(0, 0, target_yaw)
    last_ok, last_verify, last_err, last_err_xy, last_err_z = False, None, float("inf"), float("inf"), float("inf")
    for outer in range(2):
        tol_xy = 0.5 if outer == 0 else 2.0
        tol_z = 0.5 if outer == 0 else 2.0
        ok, verify_pos, pos_err, err_xy, err_z = executor._set_vehicle_pose_paused(  # type: ignore[attr-defined]
            float(target_pos[0]),
            float(target_pos[1]),
            float(target_pos[2]),
            quat,
            retries=3,
            tol_xy=tol_xy,
            tol_z=tol_z,
        )
        last_ok, last_verify, last_err, last_err_xy, last_err_z = ok, verify_pos, pos_err, err_xy, err_z
        if ok:
            break
    if not last_ok:
        raise RuntimeError(
            f"Failed to set UAV pose after retries: target({target_pos[0]:.2f},{target_pos[1]:.2f},{target_pos[2]:.2f}), "
            f"actual({last_verify[0]:.2f},{last_verify[1]:.2f},{last_verify[2]:.2f}), "
            f"err={last_err:.2f}m (XY:{last_err_xy:.2f}m, Z:{last_err_z:.2f}m)"
        )
def print_available_assets(executor: TrajectoryExecutor) -> None:
    import airsim  # type: ignore

    print("\n" + "="*60)
    print("Checking available assets...")
    print("="*60)

    client = executor.client
    if client is None:
        print("Warning: AirSim client not connected; cannot list assets")
        return

    available_assets = []
    try:
        if hasattr(client, 'simListAssets'):
            assets = client.simListAssets()
            if assets:
                available_assets = list(assets)
                print(f"Found {len(available_assets)} assets via simListAssets:")
                for asset in sorted(available_assets):
                    print(f"  - {asset}")
                print()
    except Exception as e:
        print(f"  simListAssets unavailable: {e}")

    try:
        scene_objects = client.simListSceneObjects()
        if scene_objects:
            print(f"Total scene objects: {len(scene_objects)}")
            uav_objects = [obj for obj in scene_objects if obj.startswith('UAV')]
            if uav_objects:
                print(f"Found UAV objects: {len(uav_objects)}")
                asset_candidates = set()
                for obj in uav_objects:
                    parts = obj.split('_')
                    if parts[0].startswith('UAV'):
                        asset_candidates.add(parts[0])
                if asset_candidates:
                    print(f"Candidate asset names: {sorted(asset_candidates)}")
            print()
    except Exception as e:
        print(f"  Failed to list scene objects: {e}")

    print("Testing common asset names (UAV1-UAV20)...")
    test_assets = []
    test_position = airsim.Pose(
        airsim.Vector3r(0, 0, -100),
        airsim.to_quaternion(0, 0, 0)
    )
    test_scale = airsim.Vector3r(1.0, 1.0, 1.0)

    was_paused = False
    try:
        was_paused = client.simGetVehiclePose(executor.uav_vehicle_name) is not None
    except:
        pass

    try:
        client.simPause(True)
    except:
        pass

    try:
        for i in range(1, 21):
            asset_name = f"UAV{i}"
            test_object_name = f"TEST_ASSET_{i}_{int(time.time() * 1000) % 1000000}"

            try:
                success = client.simSpawnObject(
                    test_object_name,
                    asset_name,
                    test_position,
                    test_scale,
                    physics_enabled=False,
                    is_blueprint=False
                )
                
                if success:
                    test_assets.append(asset_name)
                    try:
                        client.simDestroyObject(test_object_name)
                        client.simContinueForFrames(1)
                    except Exception as del_e:
                        pass
            except Exception as spawn_e:
                pass
    finally:
        try:
            client.simPause(was_paused)
        except:
            pass

    if test_assets:
        print(f"Found {len(test_assets)} available assets by spawning:")
        for asset in sorted(test_assets):
            print(f"  - {asset}")
    else:
        print("  No available assets found (UAV1-UAV20)")

    print("="*60)
    print()
def run_closed_loop_test(
    scene_id: str,
    trajectory_name: str,
    dataset_root: str,
    sim_server_host: str,
    sim_server_port: int,
    gpu_id: int,
    model_path: str,
    model: Dict[str, Any] | None = None,
    max_steps: int | None = None,
    save_results: bool = True,
    output_dir: str | None = None,
    executor: TrajectoryExecutor | None = None,
    close_executor: bool = True,
    include_target_vel: bool = True,
    include_prev_action: bool = True,
    debug_verbose: bool = False,
    base_model_path: str | None = None,
    tqdm_position: int | None = None,
    success_dist_thresh_m: float | None = None,
) -> None:
    dataset_root_path = Path(dataset_root)

    traj_dir = dataset_root_path / scene_id / trajectory_name
    uav_traj_file = traj_dir / "uav_trajectory.json"

    if not uav_traj_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {uav_traj_file}")
    if model is None:
        print("[closed_loop] Loading model (backbone + LoRA + action_head)...")
        sys.stdout.flush()
        model = load_model(model_path, base_model_path=base_model_path)
        print("[closed_loop] Model loaded.")
        sys.stdout.flush()
    model["include_target_vel"] = include_target_vel
    model["include_prev_action"] = include_prev_action
    external_executor = executor is not None
    if executor is None:
        print("[closed_loop] Connecting simulator (SimServer {}:{})...".format(sim_server_host, sim_server_port))
        sys.stdout.flush()
        executor = TrajectoryExecutor(
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
        print("[closed_loop] Simulator connected.")
        sys.stdout.flush()

    uav_traj, target_traj = executor.load_trajectory(str(uav_traj_file))
    with open(uav_traj_file, "r", encoding="utf-8") as f:
        uav_data = json.load(f)
    target_asset_name = uav_data.get("target_asset_name")
    if target_asset_name:
        executor.target_asset_name = str(target_asset_name)
        executor._target_asset_name_explicitly_set = True

    executor._prepare_target_object()  # type: ignore[attr-defined]

    executor._initialize_simulation(uav_traj, target_traj)  # type: ignore[attr-defined]

    target_json = traj_dir / "target_trajectory.json"
    if not target_json.exists():
        raise FileNotFoundError(f"Target trajectory file not found: {target_json}")

    with target_json.open("r", encoding="utf-8") as f:
        target_data = json.load(f)

    if "target_trajectory_airsim" not in target_data:
        raise ValueError(f"Missing 'target_trajectory_airsim' field in {target_json}")

    target_traj_airsim: List[np.ndarray] = []
    for p in target_data["target_trajectory_airsim"]:
        x = float(p["x"])
        y = float(p["y"])
        z_world = float(p["z"])
        target_traj_airsim.append(
            np.array([x, y, -z_world], dtype=np.float32)
        )

    if len(target_traj_airsim) < 2:
        raise ValueError("Target trajectory length must be >= 2")
    movement_traj_airsim: List[np.ndarray] = target_traj_airsim[1:]
    init_uav_state = executor.get_uav_state()
    init_target_pos = target_traj_airsim[0]
    init_rel_pos = init_target_pos - init_uav_state["position"]
    init_quat = init_uav_state["orientation"]
    init_rel_pos_body = executor._airsim_to_body_frame(  # type: ignore[attr-defined]
        init_rel_pos,
        init_quat[0], init_quat[1], init_quat[2], init_quat[3]
    )
    if save_results:
        if output_dir is None:
            model_dir_name = Path(model_path).name
            output_dir = str(
                PROJECT_ROOT
                / "Val"
                / "results"
                / model_dir_name
                / "seen"
                / scene_id
                / trajectory_name
            )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        rgb_dir = output_path / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
        rgb_dir = None

    merged_trajectory_data: List[Dict[str, Any]] = []

    def quaternion_to_euler(quat_w, quat_x, quat_y, quat_z):
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w])
        euler = rotation.as_euler('xyz', degrees=False)
        return {
            "roll": float(euler[0]),
            "pitch": float(euler[1]),
            "yaw": float(euler[2])
        }

    def append_trajectory_data(
        frame_idx: int,
        uav_state: Dict[str, Any],
        target_pos: np.ndarray,
        obs: Dict[str, Any],
        action: np.ndarray | None,
        system_prompt: str,
        user_prompt_text: str,
    ) -> None:
        cur_pos_airsim = np.asarray(uav_state["position"], dtype=np.float32).reshape(3,)
        uav_pos_world = airsim_to_world_pos(cur_pos_airsim)
        uav_quat = uav_state["orientation"]  # [w, x, y, z]
        uav_quat_w = float(uav_quat[0])
        uav_quat_x = float(uav_quat[1])
        uav_quat_y = float(uav_quat[2])
        uav_quat_z = float(uav_quat[3])
        frame_data = {
            "frame_idx": frame_idx,
            "uav_position": {
                "x": float(uav_pos_world[0]),
                "y": float(uav_pos_world[1]),
                "z": float(uav_pos_world[2]),
            },
            "uav_orientation_quaternion": {
                "w": uav_quat_w,
                "x": uav_quat_x,
                "y": uav_quat_y,
                "z": uav_quat_z
            }
        }

        if target_pos is not None:
            uav_pos_airsim = np.asarray(uav_state["position"], dtype=np.float32).reshape(3,)
            target_pos_airsim = np.asarray(target_pos, dtype=np.float32).reshape(3,)
            rel_pos_airsim = target_pos_airsim - uav_pos_airsim
            rel_pos_body = executor._airsim_to_body_frame(  # type: ignore[attr-defined]
                rel_pos_airsim,
                uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z
            )
            uav_pos_world_save = airsim_to_world_pos(uav_pos_airsim)
            target_pos_world = airsim_to_world_pos(target_pos_airsim)
            rel_pos_world = target_pos_world - uav_pos_world_save
            frame_data["target_position"] = {
                "x": float(target_pos_world[0]),
                "y": float(target_pos_world[1]),
                "z": float(target_pos_world[2]),
            }
            frame_data["relative_position_world"] = {
                "x": float(rel_pos_world[0]),
                "y": float(rel_pos_world[1]),
                "z": float(rel_pos_world[2]),
            }
            frame_data["target_position_in_body_frame"] = {
                "x": float(rel_pos_body[0]),
                "y": float(rel_pos_body[1]),
                "z": float(rel_pos_body[2]),
            }
            frame_data["distance"] = float(np.linalg.norm(rel_pos_airsim))
        else:
            frame_data["target_position"] = None
            frame_data["relative_position_world"] = None
            frame_data["target_position_in_body_frame"] = None
            frame_data["distance"] = None
        frame_data["model_input"] = {
            "rgb_path": f"rgb/frame_{frame_idx:05d}.png" if save_results and output_path is not None else None,
            "depth_path": None,
            "target_position_in_body_frame": frame_data.get("target_position_in_body_frame"),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt_text,
        }
        if action is not None:
            frame_data["model_output_action"] = {
                "vx": float(action[0]),
                "vy": float(action[1]),
                "vz": float(action[2]),
                "yaw_rate": float(action[3]),
            }
        else:
            frame_data["model_output_action"] = None
            frame_data["no_policy_action"] = True

        merged_trajectory_data.append(frame_data)

    uav_steps = max(0, len(uav_data.get("trajectory", [])) - 1)
    L_target = len(movement_traj_airsim)
    L = min(uav_steps, L_target)
    if L < 1:
        raise ValueError(
            f"Invalid trajectory steps: uav_steps={uav_steps}, target_movement_len={L_target}, trajectory={trajectory_name}"
        )
    if max_steps is not None:
        num_steps = min(L, max_steps)
    else:
        num_steps = L
    cap_dist_m = (
        float(success_dist_thresh_m)
        if success_dist_thresh_m is not None
        else float(SUCCESS_DIST_THRESH_M)
    )
    trajectory_collided = False
    collision_step = None
    trajectory_captured = False
    try:
        try:
            from tqdm import tqdm  # type: ignore
            tqdm_kw: Dict[str, Any] = {
                "desc": f"{scene_id}/{trajectory_name}",
                "ncols": 120,
                "leave": True,
            }
            if tqdm_position is not None:
                tqdm_kw["position"] = tqdm_position
            step_iter = tqdm(range(num_steps), **tqdm_kw)
        except Exception:
            step_iter = range(num_steps)
        last_phys_action = (0.0, 0.0, 0.0, 0.0)
        last_target_pos_airsim = None
        last_user_prompt_text = ""
        for step_idx in step_iter:
            target_pos_cmd = movement_traj_airsim[step_idx]
            executor.move_target_object(target_pos_cmd)
            target_pos_from_airsim = executor.get_object_position()
            if target_pos_from_airsim is None:
                raise RuntimeError(
                    f"[closed_loop] Step {step_idx}: failed to get target position from AirSim (get_object_position returned None), target: {executor.target_object_name}"
                )
            target_pos = np.asarray(target_pos_from_airsim, dtype=np.float32).reshape(3,)
            uav_state = executor.get_uav_state()
            rgb_img, depth_img = executor.get_camera_images()
            if rgb_img is None:
                raise RuntimeError(
                    f"[closed_loop] Step {step_idx}: get_camera_images() returned RGB=None; cannot feed the model"
                )
            if save_results and output_path is not None:
                executor.save_frame_data(step_idx, rgb_img, depth_img, str(output_path))
            quat = uav_state["orientation"]  # [w, x, y, z]
            uav_pos = uav_state["position"]
            obs: Dict[str, Any] = {
                "rgb": rgb_img,
                "uav_position_airsim": [float(uav_pos[0]), float(uav_pos[1]), float(uav_pos[2])],
                "target_position_airsim": [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])],
                "target_position_airsim_prev": list(last_target_pos_airsim) if last_target_pos_airsim is not None else None,
                "quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
                "previous_action": list(last_phys_action),
                "is_first_frame": (step_idx == 0),
            }
            norm_action, user_prompt_text = policy_step(model, obs, debug_save_path=None)
            last_user_prompt_text = user_prompt_text
            phys_action = np.asarray(
                norm_action_to_physical(norm_action, MAX_VEL, MAX_YAW_RATE),
                dtype=np.float32
            ).reshape(4)
            if save_results:
                system_prompt_for_save = (
                    model["generate_system_prompt"]()
                    if "generate_system_prompt" in model
                    else model.get("system_prompt", "")
                )
                append_trajectory_data(
                    step_idx, uav_state, target_pos, obs, phys_action,
                    system_prompt_for_save, user_prompt_text
                )
            apply_action_to_uav(executor, uav_state, phys_action)
            last_phys_action = (float(phys_action[0]), float(phys_action[1]), float(phys_action[2]), float(phys_action[3]))
            last_target_pos_airsim = (float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
            executor._step_if_needed(1)  # type: ignore[attr-defined]
            uav_state_after = executor.get_uav_state()
            if uav_state_after.get("has_collided", False):
                trajectory_collided = True
                collision_step = step_idx
                msg = f"[closed_loop] Collision detected at frame {step_idx}; stopping trajectory"
                if hasattr(step_iter, "write"):
                    step_iter.write(msg)
                else:
                    print(msg)
                break
            uav_pos_after = np.asarray(uav_state_after["position"], dtype=np.float32).reshape(3,)
            tgt_pos_airsim = np.asarray(target_pos, dtype=np.float32).reshape(3,)
            dist_after = float(np.linalg.norm(tgt_pos_airsim - uav_pos_after))
            if dist_after < cap_dist_m:
                rgb_after, depth_after = executor.get_camera_images()
                detector_device = "cpu"
                if _detect_drone_in_rgb(rgb_after, detector_device, box_threshold=0.3, text_threshold=0.25):
                    trajectory_captured = True
                    frame_idx = step_idx + 1
                    if save_results and output_path is not None:
                        try:
                            executor.save_frame_data(frame_idx, rgb_after, depth_after, str(output_path))
                        except Exception as e:
                            print(f"[closed_loop] Failed to save captured-frame image: {e}")
                    if save_results:
                        try:
                            system_prompt_for_save = (
                                model["generate_system_prompt"]()
                                if "generate_system_prompt" in model
                                else model.get("system_prompt", "")
                            )
                        except Exception:
                            system_prompt_for_save = model.get("system_prompt", "")
                        append_trajectory_data(
                            frame_idx,
                            uav_state_after,
                            tgt_pos_airsim,
                            obs,
                            phys_action,
                            system_prompt_for_save,
                            user_prompt_text,
                        )
                    msg = (
                        f"[closed_loop] Captured: scene={scene_id} trajectory={trajectory_name} "
                        f"frame {frame_idx} dist {dist_after:.2f}m < {cap_dist_m}m and drone detected; stopping early"
                    )
                    if hasattr(step_iter, "write"):
                        step_iter.write(msg)
                    else:
                        print(msg)
                    break
            uav_pos_airsim = np.asarray(uav_state["position"], dtype=np.float32).reshape(3,)
            tgt_pos_airsim = np.asarray(target_pos, dtype=np.float32).reshape(3,)
            rel_pos_airsim = tgt_pos_airsim - uav_pos_airsim
            dist = float(np.linalg.norm(rel_pos_airsim))
            if hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix(
                    act=f"({float(phys_action[0]):.1f},{float(phys_action[1]):.1f},{float(phys_action[2]):.1f},{float(phys_action[3]):.1f})",
                    dist=f"{dist:.1f}m",
                    refresh=False,
                )
        else:
            if save_results and num_steps > 0:
                terminal_idx = num_steps
                uav_state_term = executor.get_uav_state()
                tp_term = executor.get_object_position()
                if tp_term is None:
                    raise RuntimeError(
                        f"[closed_loop] Terminal frame: failed to get target position from AirSim, target: {executor.target_object_name}"
                    )
                target_pos_term = np.asarray(tp_term, dtype=np.float32).reshape(3,)
                rgb_term, depth_term = executor.get_camera_images()
                if rgb_term is None:
                    raise RuntimeError(
                        "[closed_loop] Terminal frame: get_camera_images() returned RGB=None; cannot save last frame"
                    )
                if output_path is not None:
                    executor.save_frame_data(terminal_idx, rgb_term, depth_term, str(output_path))
                try:
                    system_prompt_term = (
                        model["generate_system_prompt"]()
                        if "generate_system_prompt" in model
                        else model.get("system_prompt", "")
                    )
                except Exception:
                    system_prompt_term = model.get("system_prompt", "")
                append_trajectory_data(
                    terminal_idx,
                    uav_state_term,
                    target_pos_term,
                    {},
                    None,
                    system_prompt_term,
                    last_user_prompt_text,
                )
    finally:
        if save_results and output_path is not None and len(merged_trajectory_data) > 0:
            try:
                uav_traj_path = output_path / "uav_trajectory.json"
                temp_uav_path = output_path / "uav_trajectory.json.tmp"
                with open(temp_uav_path, 'w', encoding='utf-8') as f:
                    payload = {
                        "num_frames": len(merged_trajectory_data),
                        "target_asset_name": executor.target_asset_name,
                        "trajectory": merged_trajectory_data,
                        "collided": trajectory_collided,
                    }
                    if trajectory_collided and collision_step is not None:
                        payload["collision_step"] = collision_step
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                temp_uav_path.replace(uav_traj_path)
                frames_json_path = output_path / "frames.json"
                try:
                    frames_for_metric = []
                    for d in merged_trajectory_data:
                        up = d.get("uav_position")
                        if isinstance(up, dict):
                            frames_for_metric.append({
                                "uav_position_world": [
                                    float(up.get("x", 0.0)),
                                    float(up.get("y", 0.0)),
                                    float(up.get("z", 0.0)),
                                ]
                            })
                    original_trajectory_for_metric = [
                        {"position": airsim_to_world_pos(p).tolist()}
                        for p in target_traj_airsim
                    ]
                    frames_payload = {
                        "captured": trajectory_captured,
                        "frames": frames_for_metric,
                        "original_trajectory": original_trajectory_for_metric,
                    }
                    with open(frames_json_path, "w", encoding="utf-8") as f:
                        json.dump(frames_payload, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                except Exception as e:
                    print(f"Warning: failed to save frames.json: {e}")
                try:
                    uav_traj_plot: List[List[float]] = []
                    target_traj_plot: List[List[float]] = []
                    for frame_data in merged_trajectory_data:
                        uav_pos = frame_data.get("uav_position")
                        if isinstance(uav_pos, dict):
                            uav_traj_plot.append([
                                float(uav_pos.get("x", 0.0)),
                                float(uav_pos.get("y", 0.0)),
                                float(uav_pos.get("z", 0.0)),
                            ])
                        tgt_pos = frame_data.get("target_position")
                        if isinstance(tgt_pos, dict):
                            target_traj_plot.append([
                                float(tgt_pos.get("x", 0.0)),
                                float(tgt_pos.get("y", 0.0)),
                                float(tgt_pos.get("z", 0.0)),
                            ])
                    uav_gt_traj_plot: List[List[float]] = []
                    try:
                        gt_uav_path = dataset_root_path / scene_id / trajectory_name / "uav_trajectory.json"
                        if gt_uav_path.exists():
                            with open(gt_uav_path, "r", encoding="utf-8") as f:
                                gt_uav_data = json.load(f)
                            for frame_data in (gt_uav_data.get("trajectory") or []):
                                up = frame_data.get("uav_position")
                                if isinstance(up, dict):
                                    uav_gt_traj_plot.append([
                                        float(up.get("x", 0.0)),
                                        float(up.get("y", 0.0)),
                                        float(up.get("z", 0.0)),
                                    ])
                    except Exception:
                        uav_gt_traj_plot = []
                except Exception:
                    pass
            except Exception as e:
                print(f"Warning: failed to save trajectory files: {e}")

        if close_executor:
            try:
                executor.disconnect()
            except Exception:
                pass
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-loop testing in AirSim for one or more trajectories (UAV actions from a trained model; target follows dataset trajectory)."
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Single scene id (e.g., City_1). Mutually exclusive with --scene_ids.",
    )
    parser.add_argument(
        "--scene_ids",
        type=str,
        default=None,
        help="Comma-separated scene ids (e.g., City_1,City_2). Mutually exclusive with --scene_id.",
    )
    parser.add_argument(
        "--trajectory_name",
        type=str,
        default=None,
        help="Single trajectory dir name (e.g., trajectory_0001). Mutually exclusive with --trajectory_range.",
    )
    parser.add_argument(
        "--trajectory_range",
        type=str,
        default=None,
        help="Trajectory range like 1-5 or 1. Mutually exclusive with --trajectory_name.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(PROJECT_ROOT / "Dataset"),
        help="Dataset root directory (default: <project_root>/Dataset).",
    )
    parser.add_argument(
        "--sim_server_host",
        type=str,
        default="127.0.0.1",
        help="SimServer RPC host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--sim_server_port",
        type=int,
        default=30000,
        help="SimServer RPC port (default: 30000).",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU id for this scene (default: 0).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Trained model path (see load_model() for the loading logic).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max steps (optional). If omitted, run min(uav_steps, target_steps).",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        default=True,
        help="Whether to save results (default: True).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (optional). Default: Val/results/<model>/seen/<scene>/<trajectory>.",
    )
    parser.add_argument(
        "--include_target_vel",
        type=str,
        default="true",
        help="Whether the user prompt includes target velocity line: true/false.",
    )
    parser.add_argument(
        "--include_prev_action",
        type=str,
        default="true",
        help="Whether the user prompt includes previous action line: true/false.",
    )
    parser.add_argument(
        "--use_numeric_encoder",
        type=str,
        default="false",
        help="Whether to use numeric encoder: true/false.",
    )
    parser.add_argument(
        "--use_backbone",
        type=str,
        default="true",
        help="Whether to use backbone. If false, run action head from num_state only.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Base model dir (default: <project_root>/Qwen3-VL-2B-Instruct).",
    )
    parser.add_argument(
        "--debug_verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip scene/trajectory if uav_trajectory.json already exists under the default output dir.",
    )
    parser.add_argument(
        "--success_dist_thresh_m",
        type=float,
        default=None,
        help=(
            "Capture distance threshold (meters): after stepping, if UAV-target distance is below this value and the drone is detected, mark as captured. "
            f"If omitted, uses default {SUCCESS_DIST_THRESH_M}m."
        ),
    )
    parser.add_argument(
        "--no_save_results",
        action="store_false",
        dest="save_results",
        help="Do not save results.",
    )
    return parser.parse_args()
def _parse_scene_list(args: argparse.Namespace) -> List[str]:
    has_single = bool((args.scene_id or "").strip())
    has_multi = bool((args.scene_ids or "").strip())
    if has_single and has_multi:
        raise SystemExit("Error: --scene_id and --scene_ids are mutually exclusive")
    if not has_single and not has_multi:
        raise SystemExit("Error: must specify --scene_id or --scene_ids")
    if has_multi:
        scene_list = [s.strip() for s in str(args.scene_ids).split(",") if s.strip()]
        if not scene_list:
            raise SystemExit("Error: parsed --scene_ids is empty")
        return scene_list
    return [str(args.scene_id).strip()]
def _default_result_dir(model_path: str, scene_id: str, trajectory_name: str) -> Path:
    return PROJECT_ROOT / "Val" / "results" / Path(model_path).name / "seen" / scene_id / trajectory_name
def _should_skip_existing_result(
    *,
    scene_id: str,
    trajectory_name: str,
    model_path: str,
    output_dir: str | None,
    skip_existing: bool,
) -> bool:
    if not skip_existing:
        return False
    if output_dir is not None:
        return False
    result_dir = _default_result_dir(model_path, scene_id, trajectory_name)
    return (result_dir / "uav_trajectory.json").exists()
def _close_scene(sim_server_host: str, sim_server_port: int, scene_id: str) -> None:
    try:
        socket_client = msgpackrpc.Client(
            msgpackrpc.Address(sim_server_host, sim_server_port),
            timeout=30,
        )
        socket_client.call("close_scenes", sim_server_host, [scene_id])
        socket_client.close()
    except Exception:
        pass
def main() -> None:
    args = parse_args()
    scene_list = _parse_scene_list(args)
    if args.trajectory_range is None and args.trajectory_name is None:
        raise SystemExit("Error: must specify --trajectory_name or --trajectory_range")
    if args.trajectory_range is not None and args.trajectory_name is not None:
        raise SystemExit("Error: --trajectory_name and --trajectory_range are mutually exclusive")
    _progress_rank = os.environ.get("PROGRESS_RANK", "0")
    _tqdm_position = int(_progress_rank) if str(_progress_rank).isdigit() else 0
    _inc_vel = str(getattr(args, "include_target_vel", "true")).lower() in ("true", "1", "yes")
    _inc_prev = str(getattr(args, "include_prev_action", "true")).lower() in ("true", "1", "yes")
    _use_numeric = str(getattr(args, "use_numeric_encoder", "false")).lower() in ("true", "1", "yes")
    _use_backbone = str(getattr(args, "use_backbone", "true")).lower() in ("true", "1", "yes")
    base_model_path = (getattr(args, "base_model_path", None) or "").strip() or None
    gpu_id = int(args.gpu_id) if isinstance(args.gpu_id, str) and str(args.gpu_id).isdigit() else int(args.gpu_id)
    if _tqdm_position == 0:
        print("[closed_loop] Loading model (backbone + LoRA + action_head)...")
        sys.stdout.flush()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shared_model = load_model(
        args.model_path,
        base_model_path=base_model_path,
        device=device,
        use_numeric_encoder=_use_numeric,
        use_backbone=_use_backbone,
    )
    if _tqdm_position == 0:
        print("[closed_loop] Model loaded; reusing it for all scenes/trajectories in this process.")
        sys.stdout.flush()
    shared_model["include_target_vel"] = _inc_vel
    shared_model["include_prev_action"] = _inc_prev
    total_run = 0
    total_skip = 0
    if args.trajectory_range is not None:
        numbers = parse_trajectory_range(args.trajectory_range)
        names = trajectory_numbers_to_names(numbers)
        for scene_idx, scene_id in enumerate(scene_list, start=1):
            scene_names: List[str] = []
            for trajectory_name in names:
                if _should_skip_existing_result(
                    scene_id=scene_id,
                    trajectory_name=trajectory_name,
                    model_path=args.model_path,
                    output_dir=args.output_dir,
                    skip_existing=bool(getattr(args, "skip_existing", False)),
                ):
                    total_skip += 1
                    print(f"[closed_loop] Skipping existing result: {scene_id}/{trajectory_name}")
                    continue
                scene_names.append(trajectory_name)
            if not scene_names:
                print(f"[closed_loop] Scene {scene_id} already completed for this trajectory_range; skipping scene.")
                _close_scene(args.sim_server_host, args.sim_server_port, scene_id)
                continue
            shared_executor: TrajectoryExecutor | None = None
            try:
                print(f"[closed_loop] Starting scene {scene_id} ({scene_idx}/{len(scene_list)}), {len(scene_names)} trajectories")
                sys.stdout.flush()
                shared_executor = TrajectoryExecutor(
                    scene_id=scene_id,
                    sim_server_host=args.sim_server_host,
                    sim_server_port=args.sim_server_port,
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
                for trajectory_name in scene_names:
                    run_closed_loop_test(
                        scene_id=scene_id,
                        trajectory_name=trajectory_name,
                        dataset_root=args.dataset_root,
                        sim_server_host=args.sim_server_host,
                        sim_server_port=args.sim_server_port,
                        gpu_id=gpu_id,
                        model_path=args.model_path,
                        model=shared_model,
                        max_steps=args.max_steps,
                        save_results=args.save_results,
                        output_dir=args.output_dir,
                        executor=shared_executor,
                        close_executor=False,
                        include_target_vel=_inc_vel,
                        include_prev_action=_inc_prev,
                        debug_verbose=getattr(args, "debug_verbose", False),
                        base_model_path=base_model_path,
                        tqdm_position=_tqdm_position,
                        success_dist_thresh_m=getattr(
                            args, "success_dist_thresh_m", None
                        ),
                    )
                    total_run += 1
                print(f"[closed_loop] Scene {scene_id} finished, ran {len(scene_names)} trajectories.")
            finally:
                if shared_executor is not None:
                    try:
                        shared_executor.disconnect()
                    except Exception:
                        pass
                _close_scene(args.sim_server_host, args.sim_server_port, scene_id)
    else:
        for scene_idx, scene_id in enumerate(scene_list, start=1):
            if _should_skip_existing_result(
                scene_id=scene_id,
                trajectory_name=args.trajectory_name,
                model_path=args.model_path,
                output_dir=args.output_dir,
                skip_existing=bool(getattr(args, "skip_existing", False)),
            ):
                total_skip += 1
                print(f"[closed_loop] Skipping existing result: {scene_id}/{args.trajectory_name}")
                _close_scene(args.sim_server_host, args.sim_server_port, scene_id)
                continue
            try:
                print(f"[closed_loop] Starting scene {scene_id} ({scene_idx}/{len(scene_list)})")
                run_closed_loop_test(
                    scene_id=scene_id,
                    trajectory_name=args.trajectory_name,
                    dataset_root=args.dataset_root,
                    sim_server_host=args.sim_server_host,
                    sim_server_port=args.sim_server_port,
                    gpu_id=gpu_id,
                    model_path=args.model_path,
                    model=shared_model,
                    max_steps=args.max_steps,
                    save_results=args.save_results,
                    output_dir=args.output_dir,
                    include_target_vel=_inc_vel,
                    include_prev_action=_inc_prev,
                    debug_verbose=getattr(args, "debug_verbose", False),
                    base_model_path=base_model_path,
                    tqdm_position=_tqdm_position,
                    success_dist_thresh_m=getattr(
                        args, "success_dist_thresh_m", None
                    ),
                )
                total_run += 1
            finally:
                _close_scene(args.sim_server_host, args.sim_server_port, scene_id)
if __name__ == "__main__":
    main()
