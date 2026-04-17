#!/usr/bin/env python

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import transformers
import torch
import numpy as np
import logging
import warnings
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from PIL import Image

from transformers import TrainingArguments, HfArgumentParser, TrainerCallback
from peft import LoraConfig, get_peft_model
from peft import PeftModel

from Train_qwen.core.instruction_generator import generate_system_prompt, generate_user_prompt
from Train_qwen.core.instruction_generator import compute_instruction_numeric_state
from Train_qwen.core.action_mapping import norm_action_to_physical

try:
    from Train_qwen.core.model import (
        UAVQwen3VLModel,
        ModelArguments,
        trajectory_balanced_mean,
    )
except ImportError:
    from model import UAVQwen3VLModel, ModelArguments, trajectory_balanced_mean

from Train_qwen.core.train import (
    DataArguments as _DataArguments,
    TrainingArguments as _TrainingArguments,
    UAVQwen3VLDataset,
    save_instruction_jsons_for_dataset,
    ActionErrorTrainer,
    ActionErrorCallback,
)

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

@dataclass
class DataArguments(_DataArguments):
    dagger_data_path: str = field(default="", metadata={"help": "Dagger data root; merged with data_path. Empty => only data_path."})
    pretrained_checkpoint: str = field(default="", metadata={"help": "Pretrained checkpoint dir (adapter_*.safetensors + action_head.pt). Empty => train from base."})
    dataset_manifest_path: str = field(default="", metadata={"help": "Dataset manifest JSON path; preferred on resume."})


class MergedUAVDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_path: str,
        dagger_data_path: str,
        processor,
        data_args: DataArguments,
        use_numeric_encoder: bool = False,
        model_max_length: int = 2048,
    ):
        super().__init__()
        self.data_path = data_path
        self.dataset_path = dataset_path
        self.dagger_data_path = dagger_data_path.strip() if dagger_data_path else ""
        self.processor = processor
        self.data_args = data_args
        self.model_max_length = model_max_length
        self.is_main_process = int(os.environ.get("LOCAL_RANK", -1)) <= 0
        self.use_numeric_encoder = bool(use_numeric_encoder)

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
                print(f"[MergedDataset] Trajectory range: {trajectory_range_dict}")

        dataset_dir = Path(data_path)
        if not dataset_dir.exists():
            raise ValueError(f"Original dataset directory does not exist: {data_path}")

        list_orig = self._scan_one_root(dataset_dir, scene_list, trajectory_range_dict, str(dataset_path))
        list_dagger = []
        if self.dagger_data_path:
            dagger_dir = Path(self.dagger_data_path)
            if dagger_dir.exists():
                subdirs = [d for d in sorted(dagger_dir.iterdir()) if d.is_dir()]
                round_like = [d for d in subdirs if d.name.startswith("round_")]

                traj_mode = os.environ.get("DAGGER_TRAJ_MODE", "").lower()

                if round_like and traj_mode != "split":
                    rounds_up_to = os.environ.get("DAGGER_TRAIN_ROUNDS_UP_TO")
                    try:
                        up_to = int(rounds_up_to) if rounds_up_to else None
                    except ValueError:
                        up_to = None
                    for rd in sorted(round_like, key=lambda x: int(x.name.replace("round_", "") or "0")):
                        rnum = int(rd.name.replace("round_", "") or "0")
                        if up_to is not None and rnum > up_to:
                            continue
                        list_dagger.extend(
                            self._scan_one_root(rd, scene_list, trajectory_range_dict, str(rd))
                        )
                else:
                    list_dagger = self._scan_one_root(
                        dagger_dir, scene_list, trajectory_range_dict, self.dagger_data_path
                    )
                if self.is_main_process:
                    print(f"[MergedDataset] Original {len(list_orig)} samples, Dagger {len(list_dagger)} samples")
            elif self.is_main_process:
                print(f"[MergedDataset] Warning: Dagger dir not found; using only original data: {self.dagger_data_path}")

        self.list_data_dict = list_orig + list_dagger
        manifest_path = (getattr(self.data_args, "dataset_manifest_path", None) or "").strip()
        if manifest_path and os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    self.list_data_dict = json.load(f)
                if self.is_main_process:
                    print(f"[MergedDataset] Loaded manifest: {manifest_path}, size={len(self.list_data_dict)}")
            except Exception as e:
                if self.is_main_process:
                    print(f"[MergedDataset] Warning: failed to load manifest; rescanning and overwriting: {manifest_path}, err={e}")
        if not (manifest_path and os.path.exists(manifest_path)):
            rng = random.Random(int(getattr(self.data_args, "data_seed", 42)))
            rng.shuffle(self.list_data_dict)
            if manifest_path and self.is_main_process:
                try:
                    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump(self.list_data_dict, f, ensure_ascii=False, indent=2)
                    print(f"[MergedDataset] Saved manifest: {manifest_path}, size={len(self.list_data_dict)}")
                except Exception as e:
                    print(f"[MergedDataset] Warning: failed to save manifest: {manifest_path}, err={e}")
        if self.is_main_process:
            print(f"[MergedDataset] Total merged samples: {len(self.list_data_dict)}")

    def _parse_trajectory_range(self, trajectory_range_str: str, scene_list: Optional[List[str]] = None) -> Dict[str, tuple]:
        result = {}
        if not trajectory_range_str:
            return result
        if ":" in trajectory_range_str:
            for part in trajectory_range_str.split(","):
                part = part.strip()
                if ":" in part:
                    scene_id, range_str = part.split(":", 1)
                    scene_id = scene_id.strip()
                    range_str = range_str.strip()
                    if "-" in range_str:
                        try:
                            a, b = range_str.split("-", 1)
                            result[scene_id] = (int(a.strip()), int(b.strip()))
                        except ValueError:
                            pass
                    else:
                        try:
                            idx = int(range_str.strip())
                            result[scene_id] = (idx, idx)
                        except ValueError:
                            pass
        else:
            if "-" in trajectory_range_str:
                try:
                    a, b = trajectory_range_str.split("-", 1)
                    start_idx, end_idx = int(a.strip()), int(b.strip())
                    if scene_list:
                        for scene in scene_list:
                            result[scene] = (start_idx, end_idx)
                except ValueError:
                    pass
            else:
                try:
                    idx = int(trajectory_range_str.strip())
                    if scene_list:
                        for scene in scene_list:
                            result[scene] = (idx, idx)
                except ValueError:
                    pass
        return result

    def _should_include_trajectory(self, traj_name: str, scene_name: str, trajectory_range_dict: Dict[str, tuple]) -> bool:
        if not trajectory_range_dict or scene_name not in trajectory_range_dict:
            return True
        start_idx, end_idx = trajectory_range_dict[scene_name]
        if traj_name.startswith("trajectory_"):
            try:
                traj_num = int(traj_name.replace("trajectory_", ""))
                return start_idx <= traj_num <= end_idx
            except ValueError:
                return False
        return False

    def _scan_one_root(
        self,
        dataset_dir: Path,
        scene_list: Optional[List[str]],
        trajectory_range_dict: Optional[Dict[str, tuple]],
        base_path: str,
    ) -> List[Dict]:
        list_data_dict = []
        for scene_dir in sorted(dataset_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if scene_list is not None and scene_dir.name not in scene_list:
                continue
            for traj_dir in sorted(scene_dir.iterdir()):
                if not traj_dir.is_dir() or not traj_dir.name.startswith("trajectory_"):
                    continue
                if trajectory_range_dict and not self._should_include_trajectory(
                    traj_dir.name, scene_dir.name, trajectory_range_dict
                ):
                    continue
                instruction_path = traj_dir / "instruction.json"
                traj_file = traj_dir / "uav_trajectory.json"
                if not instruction_path.exists() or not traj_file.exists():
                    continue
                try:
                    with open(instruction_path, "r", encoding="utf-8") as f:
                        local_samples = json.load(f)
                    rel_path = traj_file.relative_to(dataset_dir)
                    rel_str = str(rel_path).replace("\\", "/")
                    for local_sample in local_samples:
                        global_sample = {
                            "json": rel_str,
                            "frame": local_sample["frame"],
                            "base_path": base_path,
                        }
                        if "conversations" in local_sample:
                            global_sample["conversations"] = local_sample["conversations"]
                        list_data_dict.append(global_sample)
                except Exception as e:
                    if self.is_main_process:
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
                base_path = infos.get("base_path", self.dataset_path)
                traj_dir = os.path.join(base_path, *infos["json"].replace("\\", "/").split("/")[:-1])
                json_path = os.path.join(base_path, infos["json"].replace("\\", "/"))
                frame_num = infos["frame"]
                path_parts = infos["json"].replace("\\", "/").split("/")
                scene_id = path_parts[0] if len(path_parts) >= 1 else None
                trajectory_name = path_parts[1] if len(path_parts) >= 2 else None

                with open(json_path, "r") as f:
                    trajectory_data = json.load(f)
                trajectory_list = trajectory_data.get("trajectory", [])
                num_frames = trajectory_data.get("num_frames", len(trajectory_list))
                if frame_num >= num_frames - 1:
                    attempt += 1
                    continue

                image_path = os.path.join(traj_dir, "rgb", f"frame_{frame_num:05d}.png")
                if not os.path.exists(image_path):
                    attempt += 1
                    continue
                image = Image.open(image_path).convert("RGB")

                current_frame_data = trajectory_list[frame_num]
                uav_pos_saved = current_frame_data.get("uav_position")
                target_pos_saved = current_frame_data.get("target_position")
                quat = current_frame_data.get("uav_orientation_quaternion")
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
                if frame_num >= 1:
                    prev_frame = trajectory_list[frame_num - 1]
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

                _inc_vel = True
                _inc_prev = True
                system_prompt = generate_system_prompt(
                    scene_id=scene_id,
                    trajectory_name=trajectory_name,
                    frame_idx=frame_num,
                    num_frames=num_frames,
                )
                user_text = generate_user_prompt(
                    uav_position_airsim=uav_pos,
                    target_position_airsim=target_pos_airsim,
                    quaternion=quat,
                    prev_action=prev_action,
                    target_position_airsim_prev=target_pos_airsim_prev,
                    dt=1.0,
                    include_target_vel=_inc_vel,
                    include_prev_action=_inc_prev,
                    is_first_frame=(frame_num == 0),
                )
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )

                if current_frame_data.get("gt_velocity_in_body_frame") is not None and current_frame_data.get("gt_yaw_rate") is not None:
                    vel = current_frame_data["gt_velocity_in_body_frame"]
                    yaw_rate = current_frame_data["gt_yaw_rate"]
                else:
                    vel = current_frame_data.get("velocity_in_body_frame") or {}
                    yaw_rate = current_frame_data.get("yaw_rate", 0.0)
                waypoint = np.array([
                    float(vel.get("x", 0)),
                    float(vel.get("y", 0)),
                    float(vel.get("z", 0)),
                    float(yaw_rate),
                ], dtype=np.float32)

                traj_id = hash(trajectory_name or "") % (2 ** 31)
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
        raise RuntimeError("unreachable")


def train():
    local_rank_env = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank_env > 0:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TQDM_DISABLE"] = "1"

    parser = HfArgumentParser((ModelArguments, DataArguments, _TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    try:
        setattr(data_args, "data_seed", int(getattr(training_args, "data_seed", 42)))
    except Exception:
        setattr(data_args, "data_seed", 42)
    setattr(data_args, "resume_from_checkpoint", (getattr(training_args, "resume_from_checkpoint", None) or "").strip())

    if local_rank_env > 0:
        logging.basicConfig(level=logging.ERROR, format="%(message)s", force=True)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logger = logging.getLogger(__name__)

    logger.info("Loading Qwen3VL model...")
    model = UAVQwen3VLModel(
        model_name_or_path=model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        freeze_backbone=model_args.freeze_backbone,
        use_numeric_encoder=True,
        use_backbone=getattr(model_args, "use_backbone", True),
    )
    model.sign_loss_weight = getattr(training_args, "sign_loss_weight", 0.0)
    try:
        if hasattr(model, "backbone") and getattr(model.backbone, "config", None) is not None:
            model.backbone.config.use_cache = False
    except Exception:
        pass
    processor = model.processor

    resume_ckpt = (getattr(training_args, "resume_from_checkpoint", None) or getattr(data_args, "resume_from_checkpoint", None) or "").strip()
    pretrained_checkpoint = (getattr(data_args, "pretrained_checkpoint", None) or "").strip()
    if resume_ckpt:
        logger.info("Found resume_from_checkpoint=%s; training state will resume from this checkpoint", resume_ckpt)
    else:
        if pretrained_checkpoint:
            ckpt_path = Path(pretrained_checkpoint)
            if ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
                logger.info("Continuing training from existing model (model weights only): %s", pretrained_checkpoint)
                model = PeftModel.from_pretrained(model, str(ckpt_path))
                action_head_file = ckpt_path / "action_head.pt"
                if action_head_file.exists():
                    try:
                        state_dict = torch.load(str(action_head_file), map_location="cpu", weights_only=True)
                    except TypeError:
                        state_dict = torch.load(str(action_head_file), map_location="cpu")
                    if state_dict and any(k.startswith("action_head.") for k in state_dict.keys()):
                        state_dict = {
                            k.replace("action_head.", "", 1): v
                            for k, v in state_dict.items()
                            if k.startswith("action_head.")
                        }
                    model.action_head.load_state_dict(state_dict, strict=True)
                    logger.info("Loaded action_head weights: %s", action_head_file)
                else:
                    logger.warning("action_head.pt not found under checkpoint; action_head remains randomly initialized")
            else:
                logger.warning("pretrained_checkpoint missing or not a PEFT dir; training from base: %s", pretrained_checkpoint)
                pretrained_checkpoint = ""
    if not pretrained_checkpoint and training_args.lora_enable:
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
    for p in model.action_head.parameters():
        p.requires_grad = True
    if getattr(model, "use_numeric_encoder", False) and hasattr(model, "numeric_encoder"):
        for p in model.numeric_encoder.parameters():
            p.requires_grad = True

    logger.info("Preparing merged dataset (original + Dagger)...")
    train_dataset = MergedUAVDataset(
        data_path=data_args.data_path,
        dataset_path=data_args.dataset_path,
        dagger_data_path=getattr(data_args, "dagger_data_path", "") or "",
        processor=processor,
        data_args=data_args,
        use_numeric_encoder=True,
        model_max_length=model_args.model_max_length,
    )

    if local_rank_env <= 0:
        _inc_vel = True
        _inc_prev = True
        from collections import defaultdict
        by_base = defaultdict(list)
        for item in train_dataset.list_data_dict:
            by_base[item["base_path"]].append({"json": item["json"], "frame": item["frame"]})
        for base_path, sub_list in by_base.items():
            logger.info("Writing instruction.json -> %s (%d entries)", base_path, len(sub_list))
            save_instruction_jsons_for_dataset(
                base_path,
                sub_list,
                include_target_vel=_inc_vel,
                include_prev_action=_inc_prev,
            )

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
            if "num_state" in examples[0] and examples[0].get("num_state") is not None:
                batch["num_state"] = torch.stack([ex["num_state"] for ex in examples])
            return batch
        from transformers import default_data_collator
        return default_data_collator(examples)

    callbacks = []
    if getattr(training_args, "use_swanlab", False):
        try:
            import swanlab
            from swanlab.integration.transformers import SwanLabCallback
            if getattr(training_args, "swanlab_project", None):
                os.environ["SWANLAB_PROJ_NAME"] = training_args.swanlab_project
            use_native_swanlab = False
            try:
                from packaging import version
                use_native_swanlab = version.parse(transformers.__version__) >= version.parse("4.50.0")
            except Exception:
                pass
            if use_native_swanlab:
                if training_args.report_to is None or training_args.report_to == [] or training_args.report_to == "none":
                    training_args.report_to = ["swanlab"]
                elif "swanlab" not in (training_args.report_to if isinstance(training_args.report_to, list) else [training_args.report_to]):
                    training_args.report_to = list(training_args.report_to) if isinstance(training_args.report_to, list) else [training_args.report_to]
                    training_args.report_to.append("swanlab")
                if getattr(training_args, "swanlab_experiment_name", None):
                    training_args.run_name = training_args.swanlab_experiment_name
                if local_rank_env <= 0:
                    logger.info("SwanLab tracking enabled (report_to=swanlab); metrics will be synced")
            else:
                swanlab_config = {
                    "model_name_or_path": model_args.model_name_or_path,
                    "dagger_data_path": getattr(data_args, "dagger_data_path", ""),
                    "pretrained_checkpoint": getattr(data_args, "pretrained_checkpoint", ""),
                    "lora_enable": training_args.lora_enable,
                    "lora_r": training_args.lora_r,
                    "num_train_epochs": training_args.num_train_epochs,
                    "per_device_train_batch_size": training_args.per_device_train_batch_size,
                    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                    "learning_rate": training_args.learning_rate,
                    "dataset_size": len(train_dataset),
                }
                swanlab_callback = SwanLabCallback(
                    project=getattr(training_args, "swanlab_project", None) or "Qwen3VL-UAV-Dagger",
                    experiment_name=getattr(training_args, "swanlab_experiment_name", None),
                    config=swanlab_config,
                )
                callbacks.append(swanlab_callback)
                if local_rank_env <= 0:
                    logger.info("SwanLab tracking enabled (Callback); metrics will be synced")
            callbacks.append(ActionErrorCallback())
        except ImportError as e:
            if local_rank_env <= 0:
                logger.warning("SwanLab enabled but not installed or import failed; skipping tracking: %s", e)

    trainer = ActionErrorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,
        tokenizer=processor.tokenizer,
        callbacks=callbacks if callbacks else None,
    )

    class SaveActionHeadCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            model_obj = kwargs.get("model", None)
            if model_obj is None:
                return control
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            try:
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model_obj.action_head.state_dict(), os.path.join(ckpt_dir, "action_head.pt"))
                if getattr(model_obj, "use_numeric_encoder", False) and hasattr(model_obj, "numeric_encoder"):
                    torch.save(model_obj.numeric_encoder.state_dict(), os.path.join(ckpt_dir, "numeric_encoder.pt"))
            except Exception as e:
                if local_rank_env <= 0:
                    logger.warning("Failed to save action_head/numeric_encoder to checkpoint: %s", e)
            return control

    trainer.add_callback(SaveActionHeadCallback())

    logger.info("Starting training (original + Dagger)...")
    logger.info("Num training samples: %d", len(train_dataset))
    logger.info("Num epochs: %s", training_args.num_train_epochs)
    logger.info("Batch size: %s", training_args.per_device_train_batch_size)
    if resume_ckpt:
        logger.info("Resuming from checkpoint: %s", resume_ckpt)
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    logger.info("Saving model...")
    trainer.save_model()
    if training_args.lora_enable:
        model.save_pretrained(training_args.output_dir)
    try:
        save_dir = training_args.output_dir
        action_head_path = os.path.join(save_dir, "action_head.pt")
        torch.save(model.action_head.state_dict(), action_head_path)
        logger.info("Saved action_head: %s", action_head_path)
        if getattr(model, "use_numeric_encoder", False) and hasattr(model, "numeric_encoder"):
            numeric_encoder_path = os.path.join(save_dir, "numeric_encoder.pt")
            torch.save(model.numeric_encoder.state_dict(), numeric_encoder_path)
            logger.info("Saved numeric_encoder: %s", numeric_encoder_path)
    except Exception as e:
        logger.warning("Failed to save action_head: %s", e)


if __name__ == "__main__":
    train()
