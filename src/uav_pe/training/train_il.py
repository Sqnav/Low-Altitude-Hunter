#!/usr/bin/env python

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import transformers
transformers.logging.set_verbosity_error()

import json
import torch
import numpy as np
import logging
import warnings
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint

from Train_qwen.core.instruction_generator import generate_system_prompt, generate_user_prompt, compute_instruction_numeric_state

try:
    import swanlab
    from swanlab.integration.transformers import SwanLabCallback
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    if int(os.environ.get('LOCAL_RANK', -1)) <= 0:
        print("Warning: swanlab is not installed; experiment tracking will be skipped. Install via 'pip install swanlab'.")

from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from PIL import Image

from Train_qwen.core.action_mapping import norm_action_to_physical

try:
    from .model import (
        UAVQwen3VLModel,
        ModelArguments,
        trajectory_balanced_mean
    )
except ImportError:
    from Train_qwen.core.model import (
        UAVQwen3VLModel,
        ModelArguments,
        trajectory_balanced_mean
    )

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*tokenizer has new PAD/BOS/EOS tokens.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The tokenizer has new PAD/BOS/EOS tokens.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*model config and generation config.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Updated tokens.*", category=UserWarning)

if "TQDM_DISABLE" not in os.environ:
    os.environ["TQDM_DISABLE"] = "0"

torch.manual_seed(42)
np.random.seed(42)

@dataclass
class DataArguments:
    data_path: str = field(default="./Dataset")
    dataset_path: str = field(default="./Dataset")
    lazy_preprocess: bool = field(default=True)
    input_prompt: str = field(default=None)
    refine_prompt: str = field(default=None)
    trajectory_range: Optional[str] = field(default=None)
    scene_list: List[str] = field(default_factory=lambda: ["City_1"])
    include_target_vel: str = field(default="true", metadata={"help": "Whether prompts include the 'Target vel (m/s)' line (passed from train_qwen3vl.sh)."})
    include_prev_action: str = field(default="true", metadata={"help": "Whether prompts include the 'Ego previous action' line (passed from train_qwen3vl.sh)."})

@dataclass
class TrainingArguments(TrainingArguments):
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")
    save_training_inputs: bool = field(default=False)
    disable_tqdm: bool = field(default=False, metadata={"help": "Disable tqdm progress bar"})
    use_swanlab: bool = field(default=True, metadata={"help": "Enable SwanLab experiment tracking"})
    swanlab_project: Optional[str] = field(default=None, metadata={"help": "SwanLab project name"})
    swanlab_experiment_name: Optional[str] = field(default=None, metadata={"help": "SwanLab experiment name"})
    swanlab_workspace: Optional[str] = field(default=None, metadata={"help": "SwanLab workspace"})
    sign_loss_weight: float = field(default=1.0, metadata={"help": "Sign-consistency loss weight for vx/vy/vz/yaw: adds softplus(-pred*gt) when > 0"})
    yaw_loss_weight: float = field(default=2.0, metadata={"help": "Relative weight for yaw dimension in regression loss"})
    sign_loss_eps: float = field(default=1e-1, metadata={"help": "Sign loss threshold: apply only when |gt| > eps"})

class UAVQwen3VLDataset(Dataset):
    def __init__(self, data_path: str, dataset_path: str, processor, data_args: DataArguments, model_max_length: int = 2048, use_numeric_encoder: bool = False):
        super().__init__()
        self.data_path = data_path
        self.dataset_path = dataset_path
        self.processor = processor
        self.data_args = data_args
        self.model_max_length = model_max_length
        self.use_numeric_encoder = use_numeric_encoder
        
        self.is_main_process = int(os.environ.get('LOCAL_RANK', -1)) <= 0
        scene_list = None
        if data_args.scene_list:
            raw = data_args.scene_list if isinstance(data_args.scene_list, list) else [data_args.scene_list]
            scene_list = []
            for s in raw:
                scene_list.extend([x.strip() for x in str(s).split(",") if x.strip()])
        
        trajectory_range_dict = {}
        if data_args.trajectory_range:
            trajectory_range_dict = self._parse_trajectory_range(data_args.trajectory_range, scene_list)
            if trajectory_range_dict and self.is_main_process:
                print(f"Trajectory range: {trajectory_range_dict}")
        
        dataset_dir = Path(data_path)
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {data_path}")
        
        self.list_data_dict = self._scan_trainset_local(dataset_dir, scene_list, trajectory_range_dict)
        random.shuffle(self.list_data_dict)
        
        if self.is_main_process:
            print(f"Loaded {len(self.list_data_dict)} training samples")
    
    def _parse_trajectory_range(self, trajectory_range_str: str, scene_list: Optional[List[str]] = None) -> Dict[str, tuple]:
        result = {}
        if not trajectory_range_str:
            return result
        
        if ':' in trajectory_range_str:
            for part in trajectory_range_str.split(','):
                part = part.strip()
                if ':' in part:
                    scene_id, range_str = part.split(':', 1)
                    scene_id = scene_id.strip()
                    range_str = range_str.strip()
                    
                    if '-' in range_str:
                        start_str, end_str = range_str.split('-', 1)
                        try:
                            start_idx = int(start_str.strip())
                            end_idx = int(end_str.strip())
                            result[scene_id] = (start_idx, end_idx)
                        except ValueError:
                            if self.is_main_process:
                                print(f"Warning: cannot parse trajectory range '{range_str}', skipping")
                    else:
                        try:
                            idx = int(range_str.strip())
                            result[scene_id] = (idx, idx)
                        except ValueError:
                            if self.is_main_process:
                                print(f"Warning: cannot parse trajectory id '{range_str}', skipping")
        else:
            if '-' in trajectory_range_str:
                start_str, end_str = trajectory_range_str.split('-', 1)
                try:
                    start_idx = int(start_str.strip())
                    end_idx = int(end_str.strip())
                    if scene_list:
                        for scene in scene_list:
                            result[scene] = (start_idx, end_idx)
                except ValueError:
                    if self.is_main_process:
                        print(f"Warning: cannot parse trajectory range '{trajectory_range_str}'")
            else:
                try:
                    idx = int(trajectory_range_str.strip())
                    if scene_list:
                        for scene in scene_list:
                            result[scene] = (idx, idx)
                except ValueError:
                    if self.is_main_process:
                        print(f"Warning: cannot parse trajectory id '{trajectory_range_str}'")
        
        return result
    
    def _should_include_trajectory(self, traj_name: str, scene_name: str, trajectory_range_dict: Dict[str, tuple]) -> bool:
        if not trajectory_range_dict or scene_name not in trajectory_range_dict:
            return True
        
        start_idx, end_idx = trajectory_range_dict[scene_name]
        
        if traj_name.startswith('trajectory_'):
            try:
                traj_num_str = traj_name.replace('trajectory_', '')
                traj_num = int(traj_num_str)
                return start_idx <= traj_num <= end_idx
            except ValueError:
                return False
        
        return False
    
    def _scan_trainset_local(self, dataset_dir: Path, scene_list: Optional[List[str]] = None, trajectory_range_dict: Optional[Dict[str, tuple]] = None) -> List[Dict]:
        list_data_dict = []
        
        for scene_dir in sorted(dataset_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            if scene_list is not None and scene_dir.name not in scene_list:
                continue
            
            for traj_dir in sorted(scene_dir.iterdir()):
                if not traj_dir.is_dir() or not traj_dir.name.startswith('trajectory_'):
                    continue
                
                if trajectory_range_dict and not self._should_include_trajectory(traj_dir.name, scene_dir.name, trajectory_range_dict):
                    continue
                
                instruction_path = traj_dir / "instruction.json"
                traj_file = traj_dir / "uav_trajectory.json"
                if not traj_file.exists():
                    traj_file = traj_dir / "trajectory.json"
                if not traj_file.exists():
                    continue
                
                rel_path = traj_file.relative_to(Path(self.dataset_path))
                rel_path_str = str(rel_path).replace('\\', '/')
                
                try:
                    if instruction_path.exists():
                        with open(instruction_path, 'r', encoding='utf-8') as f:
                            local_samples = json.load(f)
                        for local_sample in local_samples:
                            global_sample = {
                                "json": rel_path_str,
                                "frame": local_sample["frame"]
                            }
                            if "conversations" in local_sample:
                                global_sample["conversations"] = local_sample["conversations"]
                            list_data_dict.append(global_sample)
                    else:
                        with open(traj_file, 'r', encoding='utf-8') as f:
                            traj_data = json.load(f)
                        trajectory_list = traj_data.get("trajectory", [])
                        num_frames = traj_data.get("num_frames", len(trajectory_list))
                        if num_frames <= 0:
                            continue
                        for frame_idx in range(num_frames):
                            list_data_dict.append({"json": rel_path_str, "frame": frame_idx})
                except Exception as e:
                    if int(os.environ.get('LOCAL_RANK', -1)) <= 0:
                        print(f"Warning: error processing {traj_dir}: {e}")
                    continue
        
        return list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempt, max_attempt = 0, 15
        
        while attempt < max_attempt:
            try:
                current_idx = (i + attempt) % len(self.list_data_dict)
                infos = self.list_data_dict[current_idx]
                traj_dir = os.path.join(self.dataset_path, *infos['json'].split('/')[:-1])
                json_path = os.path.join(self.dataset_path, infos['json'])
                frame_num = infos['frame']
                path_parts = infos['json'].replace('\\', '/').split('/')
                scene_id = path_parts[0] if len(path_parts) >= 1 else None
                trajectory_name = path_parts[1] if len(path_parts) >= 2 else None

                with open(json_path, 'r') as f:
                    trajectory_data = json.load(f)
                
                current_frame_idx = frame_num
                trajectory_list = trajectory_data.get('trajectory', [])
                num_frames = trajectory_data.get('num_frames', len(trajectory_list))
                
                if current_frame_idx >= num_frames - 1:
                    attempt += 1
                    continue
                
                image_path = os.path.join(traj_dir, 'rgb', f'frame_{frame_num:05d}.png')
                if not os.path.exists(image_path):
                    attempt += 1
                    continue
                
                image = Image.open(image_path).convert('RGB')

                trajectory_list = trajectory_data.get('trajectory', [])
                num_frames = trajectory_data.get('num_frames', len(trajectory_list))
                current_frame_data = trajectory_list[current_frame_idx]
                uav_pos_saved = current_frame_data.get('uav_position')
                target_pos_saved = current_frame_data.get('target_position')
                quat = current_frame_data.get('uav_orientation_quaternion')
                if not uav_pos_saved or not target_pos_saved or not quat:
                    attempt += 1
                    continue
                uav_pos = {
                    "x": float(uav_pos_saved["x"]),
                    "y": float(uav_pos_saved["y"]),
                    "z": -float(uav_pos_saved["z"]),
                }
                target_pos_airsim = {
                    "x": float(target_pos_saved["x"]),
                    "y": float(target_pos_saved["y"]),
                    "z": -float(target_pos_saved["z"]),
                }
                if current_frame_idx >= 1:
                    prev_frame = trajectory_list[current_frame_idx - 1]
                    v = prev_frame.get("velocity_in_body_frame") or {}
                    yr = prev_frame.get("yaw_rate", 0.0)
                    prev_action = (float(v.get("x", 0)), float(v.get("y", 0)), float(v.get("z", 0)), float(yr))
                    tp_prev = prev_frame.get("target_position")
                    target_pos_airsim_prev = (
                        {"x": float(tp_prev["x"]), "y": float(tp_prev["y"]), "z": -float(tp_prev["z"])}
                        if tp_prev else None
                    )
                else:
                    prev_action = (0.0, 0.0, 0.0, 0.0)
                    target_pos_airsim_prev = None

                system_prompt = generate_system_prompt(
                    scene_id=scene_id,
                    trajectory_name=trajectory_name,
                    frame_idx=current_frame_idx,
                    num_frames=num_frames,
                )
                _inc_vel = str(getattr(self.data_args, "include_target_vel", "true")).lower() in ("true", "1", "yes")
                _inc_prev = str(getattr(self.data_args, "include_prev_action", "true")).lower() in ("true", "1", "yes")
                user_text = generate_user_prompt(
                    uav_position_airsim=uav_pos,
                    target_position_airsim=target_pos_airsim,
                    quaternion=quat,
                    prev_action=prev_action,
                    target_position_airsim_prev=target_pos_airsim_prev,
                    dt=1.0,
                    include_target_vel=_inc_vel,
                    include_prev_action=_inc_prev,
                    is_first_frame=(current_frame_idx == 0),
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
                            {"type": "image", "image": image},
                            {"type": "text", "text": user_text}
                        ]
                    }
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                
                if 'velocity_in_body_frame' in current_frame_data and current_frame_data['velocity_in_body_frame'] is not None:
                    vel = current_frame_data['velocity_in_body_frame']
                    yaw_rate = current_frame_data.get('yaw_rate', 0.0)
                    waypoint = np.array([vel['x'], vel['y'], vel['z'], yaw_rate])
                else:
                    raise ValueError(f"Current frame (frame_idx={current_frame_idx}) is missing velocity_in_body_frame")
                
                traj_name = os.path.basename(traj_dir)
                traj_id = hash(traj_name) % (2**31)
                out = {
                    "text": text,
                    "image": image,
                    "action": torch.tensor(waypoint, dtype=torch.float32),
                    "traj_id": torch.tensor(traj_id, dtype=torch.long),
                }
                if self.use_numeric_encoder:
                    num_vals = compute_instruction_numeric_state(
                        uav_position_airsim=uav_pos,
                        target_position_airsim=target_pos_airsim,
                        quaternion=quat,
                        prev_action=prev_action,
                        target_position_airsim_prev=target_pos_airsim_prev,
                        dt=1.0,
                    )
                    out["num_state"] = torch.tensor(num_vals, dtype=torch.float32)
                return out

            except Exception as e:
                attempt += 1
                if attempt >= max_attempt:
                    raise RuntimeError(f"Failed to load valid sample after {max_attempt} attempts. Last error: {e}")


def _build_messages_for_frame(
    trajectory_list: List[Dict],
    current_frame_idx: int,
    num_frames: int,
    scene_id: Optional[str],
    trajectory_name: Optional[str],
    dataset_path: str,
    traj_dir: Path,
    include_target_vel: bool = True,
    include_prev_action: bool = True,
) -> Optional[List[Dict]]:
    if current_frame_idx >= num_frames - 1:
        return None
    current_frame_data = trajectory_list[current_frame_idx]
    uav_pos_saved = current_frame_data.get("uav_position")
    target_pos_saved = current_frame_data.get("target_position")
    quat = current_frame_data.get("uav_orientation_quaternion")
    if not uav_pos_saved or not target_pos_saved or not quat:
        return None
    uav_pos = {
        "x": float(uav_pos_saved["x"]),
        "y": float(uav_pos_saved["y"]),
        "z": -float(uav_pos_saved["z"]),
    }
    target_pos_airsim = {
        "x": float(target_pos_saved["x"]),
        "y": float(target_pos_saved["y"]),
        "z": -float(target_pos_saved["z"]),
    }
    if current_frame_idx >= 1:
        prev_frame = trajectory_list[current_frame_idx - 1]
        v = prev_frame.get("velocity_in_body_frame") or {}
        yr = prev_frame.get("yaw_rate", 0.0)
        prev_action = (float(v.get("x", 0)), float(v.get("y", 0)), float(v.get("z", 0)), float(yr))
        tp_prev = prev_frame.get("target_position")
        target_pos_airsim_prev = (
            {"x": float(tp_prev["x"]), "y": float(tp_prev["y"]), "z": -float(tp_prev["z"])}
            if tp_prev else None
        )
    else:
        prev_action = (0.0, 0.0, 0.0, 0.0)
        target_pos_airsim_prev = None
    system_prompt = generate_system_prompt(
        scene_id=scene_id,
        trajectory_name=trajectory_name,
        frame_idx=current_frame_idx,
        num_frames=num_frames,
    )
    user_text = generate_user_prompt(
        uav_position_airsim=uav_pos,
        target_position_airsim=target_pos_airsim,
        quaternion=quat,
        prev_action=prev_action,
        target_position_airsim_prev=target_pos_airsim_prev,
        dt=1.0,
        include_target_vel=include_target_vel,
        include_prev_action=include_prev_action,
        is_first_frame=(current_frame_idx == 0),
    )
    image_rel = f"rgb/frame_{current_frame_idx:05d}.png"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image", "image": image_rel}, {"type": "text", "text": user_text}]},
    ]
    return messages


def save_instruction_jsons_for_dataset(
    dataset_path: str,
    list_data_dict: List[Dict],
    include_target_vel: bool = True,
    include_prev_action: bool = True,
) -> None:
    from collections import defaultdict
    dataset_dir = Path(dataset_path)
    by_traj = defaultdict(list)
    for item in list_data_dict:
        json_path = item.get("json", "")
        frame = item.get("frame", 0)
        by_traj[json_path].append(frame)
    for json_path, frames in by_traj.items():
        traj_dir = dataset_dir / Path(json_path).parent
        json_file = dataset_dir / json_path
        if not json_file.exists():
            continue
        path_parts = json_path.replace("\\", "/").split("/")
        scene_id = path_parts[0] if len(path_parts) >= 1 else None
        trajectory_name = path_parts[1] if len(path_parts) >= 2 else None
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                trajectory_data = json.load(f)
        except Exception:
            continue
        trajectory_list = trajectory_data.get("trajectory", [])
        num_frames = trajectory_data.get("num_frames", len(trajectory_list))
        out_entries = []
        for frame in sorted(set(frames)):
            messages = _build_messages_for_frame(
                trajectory_list=trajectory_list,
                current_frame_idx=frame,
                num_frames=num_frames,
                scene_id=scene_id,
                trajectory_name=trajectory_name,
                dataset_path=dataset_path,
                traj_dir=traj_dir,
                include_target_vel=include_target_vel,
                include_prev_action=include_prev_action,
            )
            if messages is not None:
                out_entries.append({"frame": frame, "messages": messages})
        out_entries.sort(key=lambda x: x["frame"])
        instruction_path = traj_dir / "instruction.json"
        try:
            with open(instruction_path, "w", encoding="utf-8") as f:
                json.dump(out_entries, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if int(os.environ.get("LOCAL_RANK", -1)) <= 0:
                logging.warning("Failed to write instruction.json %s: %s", instruction_path, e)


class ActionErrorCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None:
            return
        
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        if local_rank > 0:
            return
        
        if any(key.startswith('action_error/') for key in logs.keys()):
            pass


class ActionErrorTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._action_error_accumulator = {
            "vx_mae": [],
            "vy_mae": [],
            "vz_mae": [],
            "yaw_mae": [],
            "vx_mse": [],
            "vy_mse": [],
            "vz_mse": [],
            "yaw_mse": [],
        }
    
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        model_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "action": inputs.get("action"),
            "traj_id": inputs.get("traj_id"),
        }
        if "num_state" in inputs and inputs.get("num_state") is not None:
            model_inputs["num_state"] = inputs["num_state"]

        is_missing_text = (model_inputs["input_ids"] is None) and ("inputs_embeds" not in inputs)
        is_missing_image = (model_inputs["pixel_values"] is None) and ("pixel_values" not in inputs)
        if is_missing_text and is_missing_image:
            params = [p for p in model.parameters() if p.requires_grad]
            if len(params) > 0:
                zero_loss = sum(p.sum() * 0 for p in params)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                zero_loss = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

            if return_outputs:
                return (zero_loss, None)
            return zero_loss

        if "inputs_embeds" in inputs and model_inputs["input_ids"] is None:
            model_inputs["inputs_embeds"] = inputs["inputs_embeds"]

        outputs = model(**model_inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else None

        if hasattr(outputs, "action") and "action" in inputs:
            norm_pred = outputs.action
            gt_action = inputs["action"]
            max_vel = getattr(model, "max_speed", 5.0)
            max_yaw_rate = getattr(model, "max_yaw_rate", 45.0)
            pred_action = norm_action_to_physical(norm_pred, max_vel, max_yaw_rate)

            if pred_action.device != gt_action.device:
                gt_action = gt_action.to(pred_action.device)
            if pred_action.dtype != gt_action.dtype:
                gt_action = gt_action.to(pred_action.dtype)

            with torch.no_grad():
                mae = torch.abs(pred_action - gt_action).mean(dim=0)
                mse = torch.pow(pred_action - gt_action, 2).mean(dim=0)

                self._action_error_accumulator["vx_mae"].append(mae[0].item())
                self._action_error_accumulator["vy_mae"].append(mae[1].item())
                self._action_error_accumulator["vz_mae"].append(mae[2].item())
                self._action_error_accumulator["yaw_mae"].append(mae[3].item())
                self._action_error_accumulator["vx_mse"].append(mse[0].item())
                self._action_error_accumulator["vy_mse"].append(mse[1].item())
                self._action_error_accumulator["vz_mse"].append(mse[2].item())
                self._action_error_accumulator["yaw_mse"].append(mse[3].item())

                if not hasattr(outputs, "action_errors"):
                    outputs.action_errors = {}
                outputs.action_errors = {
                    "vx_mae": mae[0].item(),
                    "vy_mae": mae[1].item(),
                    "vz_mae": mae[2].item(),
                    "yaw_mae": mae[3].item(),
                    "vx_mse": mse[0].item(),
                    "vy_mse": mse[1].item(),
                    "vz_mse": mse[2].item(),
                    "yaw_mse": mse[3].item(),
                }

        if return_outputs:
            loss_components: Dict[str, float] = {}
            if loss is not None:
                loss_components["loss/main"] = float(loss.item())
            if hasattr(outputs, "aux_losses") and isinstance(outputs.aux_losses, dict):
                for k, v in outputs.aux_losses.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        loss_components[f"loss/{k}"] = float(v.item())
            self._last_loss_components = loss_components
            return (loss, outputs) if loss is not None else (None, outputs)
        return loss
    
    def log(self, logs: Dict[str, float], start_time: float | None = None) -> None:
        extra_losses: Dict[str, float] = getattr(self, "_last_loss_components", {}) or {}
        for k, v in extra_losses.items():
            if k not in logs and v is not None:
                logs[k] = v

        formatted_logs: Dict[str, float] = {}
        for k, v in logs.items():
            if isinstance(v, float):
                if abs(v) >= 1e-3:
                    formatted_logs[k] = round(v, 4)
                else:
                    formatted_logs[k] = float(f"{v:.4e}")
            else:
                formatted_logs[k] = v

        super().log(formatted_logs, start_time=start_time)


def train():
    local_rank_env = int(os.environ.get('LOCAL_RANK', -1))
    
    if local_rank_env > 0:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TQDM_DISABLE"] = "1"
    else:
        os.environ["TQDM_DISABLE"] = "0"
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    if local_rank_env > 0:
        logging.basicConfig(level=logging.ERROR, format='%(message)s', force=True)
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
    logger = logging.getLogger(__name__)

    logger.info("Loading Qwen3VL model...")
    model = UAVQwen3VLModel(
        model_name_or_path=model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        freeze_backbone=model_args.freeze_backbone,
        use_numeric_encoder=True,
        use_backbone=True,
    )
    model.sign_loss_weight = getattr(training_args, "sign_loss_weight", 0.0)
    model.yaw_loss_weight = getattr(training_args, "yaw_loss_weight", 1.5)
    model.sign_loss_eps = getattr(training_args, "sign_loss_eps", 1e-3)
    try:
        if hasattr(model, "backbone") and getattr(model.backbone, "config", None) is not None:
            model.backbone.config.use_cache = False
    except Exception:
        pass

    processor = model.processor

    logger.info("Preparing dataset...")
    train_dataset = UAVQwen3VLDataset(
        data_path=data_args.data_path,
        dataset_path=data_args.dataset_path,
        processor=processor,
        data_args=data_args,
        model_max_length=model_args.model_max_length,
        use_numeric_encoder=True,
    )

    if local_rank_env <= 0:
        _inc_vel = True
        _inc_prev = True
        logger.info("Writing generated messages back to instruction.json (Qwen format)...")
        save_instruction_jsons_for_dataset(
            data_args.dataset_path,
            train_dataset.list_data_dict,
            include_target_vel=_inc_vel,
            include_prev_action=_inc_prev,
        )
    
    if training_args.lora_enable:
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if local_rank_env <= 0:
            model.print_trainable_parameters()
    
    logger.info("Action head is enabled")
    for param in model.action_head.parameters():
        param.requires_grad = True
    if getattr(model, "use_numeric_encoder", False) and hasattr(model, "numeric_encoder"):
        for param in model.numeric_encoder.parameters():
            param.requires_grad = True
        logger.info("numeric_encoder is enabled")
    from transformers import default_data_collator

    def custom_collate_fn(examples):
        if len(examples) == 0:
            return {}

        if "text" in examples[0] and "image" in examples[0]:
            texts = [ex["text"] for ex in examples]
            images = [ex["image"] for ex in examples]

            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model_args.model_max_length,
            )

            actions = torch.stack([ex["action"] for ex in examples])
            traj_ids = torch.stack([ex["traj_id"] for ex in examples])
            batch = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"],
                "action": actions,
                "traj_id": traj_ids,
            }
            if "num_state" in examples[0]:
                batch["num_state"] = torch.stack([ex["num_state"] for ex in examples])
            return batch

        return default_data_collator(examples)
    
    data_collator = custom_collate_fn

    training_args.disable_tqdm = False
    
    callbacks = []
    if training_args.use_swanlab and SWANLAB_AVAILABLE:
        if training_args.swanlab_project:
            os.environ["SWANLAB_PROJ_NAME"] = training_args.swanlab_project
        if training_args.swanlab_workspace:
            os.environ["SWANLAB_WORKSPACE"] = training_args.swanlab_workspace
        
        transformers_version = transformers.__version__
        try:
            from packaging import version
            use_native_integration = version.parse(transformers_version) >= version.parse("4.50.0")
        except:
            use_native_integration = False
        
        if use_native_integration:
            if training_args.report_to is None or training_args.report_to == []:
                training_args.report_to = ["swanlab"]
            elif "swanlab" not in training_args.report_to:
                if isinstance(training_args.report_to, list):
                    training_args.report_to.append("swanlab")
                else:
                    training_args.report_to = [training_args.report_to, "swanlab"]
            
            if training_args.swanlab_experiment_name:
                training_args.run_name = training_args.swanlab_experiment_name
            
            logger.info(f"Using native SwanLab integration (transformers {transformers_version})")
        else:
            swanlab_config = {
                "model_name": model_args.model_name_or_path,
                "model_max_length": model_args.model_max_length,
                "freeze_backbone": model_args.freeze_backbone,
                "lora_enable": training_args.lora_enable,
                "lora_r": training_args.lora_r,
                "lora_alpha": training_args.lora_alpha,
                "lora_dropout": training_args.lora_dropout,
                "num_train_epochs": training_args.num_train_epochs,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "learning_rate": training_args.learning_rate,
                "weight_decay": training_args.weight_decay,
                "warmup_ratio": training_args.warmup_ratio,
                "lr_scheduler_type": training_args.lr_scheduler_type,
                "dataset_size": len(train_dataset),
                "scene_list": data_args.scene_list,
            }
            
            swanlab_callback = SwanLabCallback(
                project=training_args.swanlab_project or "Qwen3VL-UAV-Training",
                experiment_name=training_args.swanlab_experiment_name,
                config=swanlab_config
            )
            callbacks.append(swanlab_callback)
            logger.info(f"Using SwanLab Callback integration (transformers {transformers_version})")
    elif training_args.use_swanlab and not SWANLAB_AVAILABLE:
        logger.warning("SwanLab is enabled but not installed; skipping tracking. Run: pip install swanlab")
    
    if training_args.use_swanlab and SWANLAB_AVAILABLE:
        action_error_callback = ActionErrorCallback()
        callbacks.append(action_error_callback)
        logger.info("Action error logging is enabled")
    
    trainer = ActionErrorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        callbacks=callbacks if callbacks else None,
    )

    last_checkpoint = None
    if training_args.output_dir and os.path.isdir(training_args.output_dir):
        try:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        except Exception:
            last_checkpoint = None

    if last_checkpoint is not None and local_rank_env <= 0:
        logger.info(f"Found existing checkpoint; resuming from {last_checkpoint}")

    logger.info("Starting training...")
    logger.info(f"Num training samples: {len(train_dataset)}")
    logger.info(f"Num epochs: {training_args.num_train_epochs}")
    logger.info(f"Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {training_args.max_steps if training_args.max_steps > 0 else 'computed from epochs'}")
    if training_args.use_swanlab and SWANLAB_AVAILABLE:
        logger.info("SwanLab tracking is enabled")
    logger.info("=" * 50)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info("Saving model...")
    trainer.save_model()
    if training_args.lora_enable:
        model.save_pretrained(training_args.output_dir)

    try:
        save_dir = training_args.output_dir
        action_head_path = os.path.join(save_dir, "action_head.pt")
        torch.save(model.action_head.state_dict(), action_head_path)
        logger.info(f"Saved action_head weights to: {action_head_path}")
        if getattr(model, "use_numeric_encoder", False) and hasattr(model, "numeric_encoder"):
            numeric_encoder_path = os.path.join(save_dir, "numeric_encoder.pt")
            torch.save(model.numeric_encoder.state_dict(), numeric_encoder_path)
            logger.info(f"Saved numeric_encoder weights to: {numeric_encoder_path}")
    except Exception as e:
        logger.warning(f"Failed to save action_head/numeric_encoder weights: {e}")


if __name__ == "__main__":
    train()
