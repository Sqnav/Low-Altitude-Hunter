#!/usr/bin/env python3

import json
import random
import sys
from pathlib import Path

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Executor.core import TrajectoryExecutor
from Train_qwen.core.action_mapping import norm_action_to_physical
from Train_qwen.core.step0_debug_utils import (
    get_action_head_input_from_backbone_outputs,
)
from Train_qwen.core.instruction_generator import (
    compute_instruction_numeric_state,
)
from Val.scripts.closed_loop_airsim import (
    load_uav_and_target_trajectories,
    apply_action_to_uav,
)


class AirSimUAVTrainEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_dict,
        dataset_root,
        scene_id,
        trajectory_name,
        executor,
        max_vel=5.0,
        max_yaw_rate=45.0,
        yaw_scale=1.0,
        reward_dist_scale=0.1,
        reward_success=100.0,
        reward_collision=-100.0,
        success_dist_thresh=10.0,
        reward_progress_scale=1.0,
        trajectory_range=None,
        reward_type="progress",
        reward_r_level=10.0,
        max_steps=None,
        max_steps_ratio=None,
        use_gt_action_loss=False,
        gt_action_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.model_dict = model_dict
        self.executor = executor
        self.scene_id = scene_id
        self.trajectory_name = trajectory_name
        self.dataset_root = Path(dataset_root)
        self.trajectory_range = list(trajectory_range) if trajectory_range else None
        self._trajectory_range_ptr = 0
        self._trajectory_exhausted = False
        self.reward_type = str(reward_type).strip().lower() if reward_type else "progress"
        self.reward_r_level = float(reward_r_level)
        self.max_vel = max_vel
        self.max_yaw_rate = max_yaw_rate
        self.yaw_scale = yaw_scale
        self.reward_dist_scale = reward_dist_scale
        self.reward_success = reward_success
        self.reward_collision = reward_collision
        self.success_dist_thresh = success_dist_thresh
        self.reward_progress_scale = reward_progress_scale
        self._last_phys_action = (0.0, 0.0, 0.0, 0.0)
        self._last_target_pos_airsim = None
        self._backbone_weight_checked = False
        self._max_steps_cap = int(max_steps) if max_steps is not None else None
        self._max_steps_ratio = float(max_steps_ratio) if max_steps_ratio is not None else None
        self.use_gt_action_loss = bool(use_gt_action_loss)
        self.gt_action_loss_weight = float(gt_action_loss_weight)

        self._uav_start_airsim, self._target_traj_airsim, self._target_asset_name = (
            load_uav_and_target_trajectories(
                self.dataset_root, scene_id, trajectory_name
            )
        )
        self.target_traj_airsim = self._target_traj_airsim
        uav_steps = self._get_uav_steps_from_json(self.trajectory_name)
        target_steps = len(self.target_traj_airsim) - 1
        self.max_steps = min(uav_steps, target_steps)
        if self.max_steps < 1:
            raise ValueError(
                f"Trajectory is too short: uav_steps={uav_steps}, target_steps={target_steps}"
            )
        self._apply_max_steps_cap()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        model = model_dict["model"]
        config = getattr(model.backbone, "config", None)
        if config is not None:
            if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
                hidden_size = config.text_config.hidden_size
            elif hasattr(config, "hidden_size"):
                hidden_size = config.hidden_size
            else:
                hidden_size = 2048
        else:
            hidden_size = 2048
        use_numeric_encoder = bool(
            self.model_dict.get("use_numeric_encoder", getattr(model, "use_numeric_encoder", False))
        )

        self._append_prev_action_dim = 4
        if use_numeric_encoder:
            num_hidden_dim = getattr(model, "num_hidden_dim", None)
            if num_hidden_dim is None and hasattr(model, "numeric_encoder"):
                for m in reversed(list(model.numeric_encoder.modules())):
                    if isinstance(m, torch.nn.Linear):
                        num_hidden_dim = m.out_features
                        break
            if num_hidden_dim is None:
                num_hidden_dim = 128
            self._core_obs_dim = hidden_size + num_hidden_dim
        else:
            self._core_obs_dim = hidden_size

        self._hidden_size = self._core_obs_dim + self._append_prev_action_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._hidden_size,), dtype=np.float32
        )

        self.current_step = 0
        self._last_hidden = None
        self.il_action_head = self.model_dict.get("il_action_head", None)

    def _apply_max_steps_cap(self):
        base = self.max_steps
        if self._max_steps_cap is not None:
            self.max_steps = min(base, self._max_steps_cap)
        elif self._max_steps_ratio is not None:
            s = int((base + 1) * self._max_steps_ratio)
            s = max(1, min(base, s))
            self.max_steps = s

    def _get_uav_steps_from_json(self, trajectory_name: str) -> int:
        uav_traj_file = self.dataset_root / self.scene_id / trajectory_name / "uav_trajectory.json"
        with open(uav_traj_file, "r", encoding="utf-8") as f:
            uav_data = json.load(f)
        traj = uav_data.get("trajectory", [])
        return max(0, len(traj) - 1)

    def _get_obs_state(self):
        uav_state = self.executor.get_uav_state()
        next_idx = min(self.current_step + 1, len(self.target_traj_airsim) - 1)
        if self.current_step == 0:
            target_pos_from_sim = self.executor.get_object_position()
            if target_pos_from_sim is not None:
                target_pos_airsim = np.asarray(target_pos_from_sim, dtype=np.float32).reshape(3)
            else:
                target_pos_airsim = self.target_traj_airsim[next_idx]
        else:
            target_pos_airsim = self.target_traj_airsim[next_idx]
        rgb_img, depth_img = self.executor.get_camera_images()
        if isinstance(rgb_img, np.ndarray):
            pil_img = Image.fromarray(rgb_img).convert("RGB")
        elif isinstance(rgb_img, Image.Image):
            pil_img = rgb_img
        else:
            raise ValueError(f"Unsupported rgb_img type: {type(rgb_img)}")

        rel_pos_airsim = target_pos_airsim - uav_state["position"]
        quat = uav_state["orientation"]
        rel_pos_body = self.executor._airsim_to_body_frame(
            rel_pos_airsim, quat[0], quat[1], quat[2], quat[3]
        )

        model = self.model_dict["model"]
        processor = self.model_dict.get("processor") or getattr(model, "processor", None)
        if processor is None:
            raise ValueError("model_dict is missing processor")
        device = self.model_dict["device"]

        model.eval()
        model.backbone.eval()

        system_prompt = self.model_dict["generate_system_prompt"]()
        uav_pos = uav_state["position"]
        user_text = self.model_dict["generate_user_prompt"](
            uav_position_airsim=[float(uav_pos[0]), float(uav_pos[1]), float(uav_pos[2])],
            target_position_airsim=[float(target_pos_airsim[0]), float(target_pos_airsim[1]), float(target_pos_airsim[2])],
            quaternion=[float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            prev_action=list(self._last_phys_action),
            target_position_airsim_prev=list(self._last_target_pos_airsim) if self._last_target_pos_airsim is not None else None,
            dt=1.0,
            include_target_vel=self.model_dict.get("include_target_vel", True),
            include_prev_action=self.model_dict.get("include_prev_action", True),
            is_first_frame=(self._last_target_pos_airsim is None),
        )


        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = processor(
            text=[text],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        pv = inputs.get("pixel_values")
        if pv is None:
            raise ValueError(
                "[RL env] processor did not return pixel_values (empty image input to model). Check get_camera_images and processor usage."
            )
        if pv.numel() == 0 or (pv.abs().sum().item() == 0):
            import warnings
            warnings.warn(
                "[RL env] pixel_values are all zero or empty; model may not have received a valid image, outputs can be abnormal.",
                UserWarning,
                stacklevel=2,
            )
        model.eval()
        model.backbone.eval()
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

                if not torch.is_tensor(action_head_input):
                    action_head_input = torch.as_tensor(
                        action_head_input,
                        device=backbone_outputs.hidden_states[-1].device,
                        dtype=backbone_outputs.hidden_states[-1].dtype,
                    )
                else:
                    action_head_input = action_head_input.to(
                        device=backbone_outputs.hidden_states[-1].device,
                        dtype=backbone_outputs.hidden_states[-1].dtype,
                    )

                use_numeric_encoder = bool(
                    self.model_dict.get("use_numeric_encoder", getattr(model, "use_numeric_encoder", False))
                )

                if use_numeric_encoder:
                    num_state_tuple = compute_instruction_numeric_state(
                        uav_position_airsim=[float(uav_pos[0]), float(uav_pos[1]), float(uav_pos[2])],
                        target_position_airsim=[float(target_pos_airsim[0]), float(target_pos_airsim[1]), float(target_pos_airsim[2])],
                        quaternion=[float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
                        prev_action=list(self._last_phys_action),
                        target_position_airsim_prev=list(self._last_target_pos_airsim)
                        if self._last_target_pos_airsim is not None else None,
                        dt=1.0,
                    )
                    num_state_arr = np.asarray(num_state_tuple, dtype=np.float32).reshape(1, -1)
                    num_state_tensor = torch.from_numpy(num_state_arr).to(
                        device=backbone_outputs.hidden_states[-1].device,
                        dtype=backbone_outputs.hidden_states[-1].dtype,
                    )
                    num_feat = model.numeric_encoder(num_state_tensor)
                    fused = torch.cat([action_head_input.reshape(1, -1), num_feat], dim=-1).squeeze(0)
                    out_tensor = fused
                else:
                    out_tensor = action_head_input.reshape(-1)

            if torch.is_tensor(out_tensor):
                out = out_tensor.detach().float().cpu().numpy().astype(np.float32)
            else:
                out = np.asarray(out_tensor, dtype=np.float32).flatten()
        finally:
            del inputs
        core_out = out.astype(np.float32, copy=False)
        prev_action_feat = np.asarray(self._last_phys_action, dtype=np.float32).reshape(4)
        out = np.concatenate([core_out, prev_action_feat], axis=0).astype(np.float32)

        assert out.shape == (self._hidden_size,), (
            f"Observation shape does not match observation_space: {out.shape} vs ({self._hidden_size},)"
        )

        self._last_hidden = core_out.copy()
        return out

    def _get_gt_hidden_at_frame(self, frame_idx: int):
        if self._gt_trajectory_data is None or frame_idx >= len(self._gt_trajectory_data):
            return None
        frame_data = self._gt_trajectory_data[frame_idx]
        if not isinstance(frame_data, dict):
            return None
        rgb_path = self.dataset_root / self.scene_id / self.trajectory_name / "rgb" / f"frame_{frame_idx:05d}.png"
        if not rgb_path.exists():
            return None
        pil_img = Image.open(rgb_path).convert("RGB")

        model_input = frame_data.get("model_input")
        if model_input and "system_prompt" in model_input and "user_prompt" in model_input:
            system_prompt = model_input["system_prompt"]
            user_text = model_input["user_prompt"]
        else:
            uav_pos = frame_data.get("uav_position")
            target_pos = frame_data.get("target_position")
            quat = frame_data.get("uav_orientation_quaternion")
            if not uav_pos or not target_pos or not quat:
                return None
            uav_airsim = [float(uav_pos["x"]), float(uav_pos["y"]), -float(uav_pos["z"])]
            target_airsim = [float(target_pos["x"]), float(target_pos["y"]), -float(target_pos["z"])]
            quat_list = [float(quat.get("w", 1)), float(quat.get("x", 0)), float(quat.get("y", 0)), float(quat.get("z", 0))]
            prev_action = [0.0, 0.0, 0.0, 0.0]
            target_prev = None
            if frame_idx > 0:
                prev_frame = self._gt_trajectory_data[frame_idx - 1]
                if isinstance(prev_frame, dict):
                    mo = prev_frame.get("model_output_action")
                    if mo:
                        prev_action = [float(mo.get("vx", 0)), float(mo.get("vy", 0)), float(mo.get("vz", 0)), float(mo.get("yaw_rate", 0))]
                    tp = prev_frame.get("target_position")
                    if tp:
                        target_prev = [float(tp["x"]), float(tp["y"]), -float(tp["z"])]
            system_prompt = self.model_dict["generate_system_prompt"]()
            user_text = self.model_dict["generate_user_prompt"](
                uav_position_airsim=uav_airsim,
                target_position_airsim=target_airsim,
                quaternion=quat_list,
                prev_action=prev_action,
                target_position_airsim_prev=target_prev,
                dt=1.0,
                include_target_vel=self.model_dict.get("include_target_vel", True),
                include_prev_action=self.model_dict.get("include_prev_action", True),
                is_first_frame=(frame_idx == 0),
            )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": user_text}]},
        ]
        model = self.model_dict["model"]
        processor = self.model_dict.get("processor") or getattr(model, "processor", None)
        device = self.model_dict["device"]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text], images=[pil_img], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        model.eval()
        model.backbone.eval()
        try:
            with torch.no_grad():
                backbone_outputs = model.backbone(**inputs, output_hidden_states=True, return_dict=True)
                action_head_input = get_action_head_input_from_backbone_outputs(
                    backbone_outputs.hidden_states[-1],
                    inputs["attention_mask"],
                    backbone_outputs.hidden_states[-1].device,
                )
            action_head_input = get_action_head_input_from_backbone_outputs(
                backbone_outputs.hidden_states[-1],
                inputs["attention_mask"],
                backbone_outputs.hidden_states[-1].device,
            )

            if torch.is_tensor(action_head_input):
                out = action_head_input.detach().float().cpu().numpy().astype(np.float32).flatten()
            else:
                out = np.asarray(action_head_input, dtype=np.float32).flatten()
            if torch.is_tensor(out):
                out = out.detach().float().cpu().numpy().astype(np.float32)
            else:
                out = np.asarray(out, dtype=np.float32).flatten()
        finally:
            del inputs
        return out

    def _compute_gt_phys_action_impl(self, uav_state, target_pos_airsim):
        rel_pos_airsim = np.asarray(target_pos_airsim, dtype=np.float64) - np.asarray(uav_state["position"], dtype=np.float64)
        quat = uav_state["orientation"]
        rel_pos_body = self.executor._airsim_to_body_frame(
            rel_pos_airsim, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        )
        norm = float(np.linalg.norm(rel_pos_body) + 1e-8)
        dir_body = rel_pos_body / norm
        vel_body = dir_body * float(self.max_vel)
        yaw_err_rad = float(np.arctan2(rel_pos_body[1], rel_pos_body[0]))
        yaw_rate_deg = float(np.degrees(yaw_err_rad))
        yaw_rate_deg = float(np.clip(yaw_rate_deg, -float(self.max_yaw_rate), float(self.max_yaw_rate)))
        return np.asarray(
            [vel_body[0], vel_body[1], vel_body[2], yaw_rate_deg],
            dtype=np.float32,
        )

    def compute_gt_phys_action(self):
        uav_state = self.executor.get_uav_state()
        target_pos_cmd = self.target_traj_airsim[self.current_step + 1]
        return self._compute_gt_phys_action_impl(uav_state, target_pos_cmd)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        phys_action = norm_action_to_physical(action, self.max_vel, self.max_yaw_rate)
        phys_action = np.asarray(phys_action, dtype=np.float32).reshape(4)

        if getattr(self, "_trajectory_exhausted", False):
            new_state = self._get_obs_state()
            info = {
                "distance": 0.0,
                "step": self.current_step,
                "delta_dist": 0.0,
                "episode_success": False,
                "episode_collision": False,
                "phys_action": np.zeros(4, dtype=np.float32),
                "base_phys_action": np.zeros(4, dtype=np.float32),
                "residual_phys_action": np.zeros(4, dtype=np.float32),
                "trajectory_name": self.trajectory_name,
                "trajectory_exhausted": True,
                "gt_phys_action": np.zeros(4, dtype=np.float32),
            }
            return new_state, 0.0, True, False, info

        uav_state = self.executor.get_uav_state()
        target_pos_cmd = self.target_traj_airsim[self.current_step + 1]
        prev_dist = float(np.linalg.norm(target_pos_cmd - uav_state["position"]))

        base_phys_action = None
        if self.il_action_head is not None and self._last_hidden is not None:
            try:
                with torch.no_grad():
                    head_param = next(self.il_action_head.parameters())
                    obs_for_base = (
                        torch.as_tensor(self._last_hidden, device=head_param.device, dtype=head_param.dtype)
                        .reshape(1, -1)
                    )
                    base_norm = torch.tanh(self.il_action_head(obs_for_base)).float().reshape(-1)
                    base_norm_np = base_norm.detach().cpu().numpy().astype(np.float32)
                    base_phys_action = norm_action_to_physical(
                        base_norm_np, self.max_vel, self.max_yaw_rate
                    )
                    base_phys_action = np.asarray(base_phys_action, dtype=np.float32).reshape(4)
            except Exception:
                base_phys_action = phys_action.copy()
        else:
            base_phys_action = phys_action.copy()

        is_first_action = int(self.current_step) == 0
        extra_frame0 = None

        gt_phys_action = self._compute_gt_phys_action_impl(uav_state, target_pos_cmd)

        if is_first_action:
            residual_phys_action = (
                phys_action - base_phys_action if base_phys_action is not None else np.zeros_like(phys_action)
            )
            extra_frame0 = {
                "step": 0,
                "distance": prev_dist,
                "delta_dist": 0.0,
                "episode_success": False,
                "episode_collision": False,
                "phys_action": phys_action.copy(),
                "base_phys_action": base_phys_action.copy(),
                "residual_phys_action": residual_phys_action.copy(),
                "gt_phys_action": gt_phys_action.copy(),
                "trajectory_name": self.trajectory_name,
            }

        prev_phys_action = np.asarray(self._last_phys_action, dtype=np.float32)

        apply_action_to_uav(self.executor, uav_state, phys_action)
        self.executor.move_target_object(target_pos_cmd)
        self.executor._step_if_needed(1)
        self.current_step += 1
        self._last_phys_action = (float(phys_action[0]), float(phys_action[1]), float(phys_action[2]), float(phys_action[3]))
        self._last_target_pos_airsim = (float(target_pos_cmd[0]), float(target_pos_cmd[1]), float(target_pos_cmd[2]))

        new_state = self._get_obs_state()
        uav_state_now = self.executor.get_uav_state()
        dist = float(np.linalg.norm(target_pos_cmd - uav_state_now["position"]))

        delta_dist = float(prev_dist - dist)
        if not np.isfinite(delta_dist):
            delta_dist = 0.0
        delta_dist_for_reward = float(delta_dist)
        reward_progress_for_info = reward_smooth_penalty_for_info = None
        if delta_dist_for_reward >= 0:
            reward_progress = (delta_dist_for_reward * self.reward_progress_scale) / 5.0
        else:
            reward_progress = (delta_dist_for_reward * (2.0 * self.reward_progress_scale)) / 5.0
        reward = reward_progress
        cur_phys_action = np.asarray(phys_action, dtype=np.float32)
        act_diff = cur_phys_action - prev_phys_action
        smooth_penalty = (0.1 * float(np.linalg.norm(act_diff))) / 5.0
        if not np.isfinite(smooth_penalty):
            smooth_penalty = 0.0
        reward -= smooth_penalty
        reward_progress_for_info = reward_progress
        reward_smooth_penalty_for_info = smooth_penalty

        gt_action_loss_for_info = None
        if self.use_gt_action_loss and self.gt_action_loss_weight > 0.0:
            diff = np.asarray(phys_action, dtype=np.float32) - gt_phys_action
            gt_action_loss = float(np.linalg.norm(diff))
            reward -= self.gt_action_loss_weight * gt_action_loss
            gt_action_loss_for_info = gt_action_loss

        terminated = False
        truncated = False

        episode_success = False
        episode_collision = False
        if uav_state_now.get("has_collided", False):
            reward += float(self.reward_collision)
            terminated = True
            episode_collision = True
        elif dist < self.success_dist_thresh:
            reward += float(self.reward_success)
            terminated = True
            episode_success = True
        elif self.current_step >= self.max_steps:
            terminated = True

        delta_dist = prev_dist - dist
        info = {
            "distance": dist,
            "step": self.current_step,
            "delta_dist": delta_dist,
            "episode_success": episode_success,
            "episode_collision": episode_collision,
            "phys_action": phys_action.copy(),
            "base_phys_action": base_phys_action.copy(),
            "residual_phys_action": (
                (phys_action - base_phys_action).copy()
                if base_phys_action is not None
                else np.zeros_like(phys_action)
            ),
            "trajectory_name": self.trajectory_name,
        }
        if extra_frame0 is not None:
            info["extra_frame0"] = extra_frame0
        if reward_progress_for_info is not None:
            info["reward_progress"] = float(reward_progress_for_info)
        if reward_smooth_penalty_for_info is not None:
            info["reward_smooth_penalty"] = float(reward_smooth_penalty_for_info)
        if gt_action_loss_for_info is not None:
            info["gt_action_loss"] = float(gt_action_loss_for_info)
        info["gt_phys_action"] = gt_phys_action.copy()
        return new_state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        if self.trajectory_range:
            if self._trajectory_range_ptr < len(self.trajectory_range):
                self.trajectory_name = self.trajectory_range[self._trajectory_range_ptr]
                self._trajectory_range_ptr += 1
                self._trajectory_exhausted = False
            else:
                self._trajectory_exhausted = True
        else:
            self._trajectory_exhausted = False

        if self._trajectory_exhausted:
            self.current_step = 0
            self._last_phys_action = (0.0, 0.0, 0.0, 0.0)
            self._last_target_pos_airsim = self._last_target_pos_airsim if self._last_target_pos_airsim is not None else None
            return self._get_obs_state(), {}

        uav_traj_file = (
            self.dataset_root / self.scene_id / self.trajectory_name / "uav_trajectory.json"
        )
        uav_traj, target_traj = self.executor.load_trajectory(str(uav_traj_file))
        with open(uav_traj_file, "r", encoding="utf-8") as f:
            uav_data = json.load(f)
        if uav_data.get("target_asset_name"):
            self.executor.target_asset_name = uav_data["target_asset_name"]
            self.executor._target_asset_name_explicitly_set = True
        self.executor._prepare_target_object()
        self.executor._initialize_simulation(uav_traj, target_traj)
        self._uav_start_airsim, self._target_traj_airsim, self._target_asset_name = (
            load_uav_and_target_trajectories(
                self.dataset_root, self.scene_id, self.trajectory_name
            )
        )
        self.target_traj_airsim = self._target_traj_airsim
        uav_steps = max(0, len(uav_traj) - 1)
        target_steps = len(self.target_traj_airsim) - 1
        self.max_steps = min(uav_steps, target_steps)
        if self.max_steps < 1:
            raise ValueError(
                f"Trajectory too short for {self.trajectory_name}: uav_steps={uav_steps}, target_steps={target_steps}"
            )
        self._apply_max_steps_cap()
        self.current_step = 0
        self._last_phys_action = (0.0, 0.0, 0.0, 0.0)
        self._last_target_pos_airsim = None
        self.executor.move_target_object(self.target_traj_airsim[1])
        return self._get_obs_state(), {}


