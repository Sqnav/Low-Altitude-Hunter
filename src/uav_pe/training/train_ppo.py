#!/usr/bin/env python3

import argparse
import copy
import sys
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch as th
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from Code.RL_residual.scripts.residual_action_head_policy import ResidualActionHeadPolicy
from RL_residual.scripts.airsim_env import AirSimUAVTrainEnv
from Executor.core import TrajectoryExecutor
from Train_qwen.core.instruction_generator import generate_system_prompt, generate_user_prompt
from Val.scripts.offline_validate_policy import load_model_like_validate  # message/message，message key message

from peft import LoraConfig, get_peft_model

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


def _default_training_state_path(save_path: str) -> Path:
    return Path(save_path).resolve().parent / "rl_training_state.json"


def _find_latest_step_checkpoint(save_path: str) -> Optional[Path]:
    base = Path(save_path)
    parent = base.parent
    stem = base.stem
    suf = base.suffix
    if not parent.is_dir():
        return None
    import re

    pat = re.compile(rf"^{re.escape(stem)}_step_(\d+){re.escape(suf)}$")
    best: tuple[int, Path] | None = None
    for p in parent.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, p)
    return best[1] if best else None


def _resolve_resume_checkpoint(save_path: str, resume_from: str) -> Optional[Path]:
    rf = (resume_from or "").strip()
    if rf:
        p = Path(rf)
        return p if p.is_file() else None
    sp = Path(save_path)
    if sp.is_file():
        return sp
    latest = _find_latest_step_checkpoint(save_path)
    return latest if latest is not None and latest.is_file() else None


def load_training_state(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[RL] message {path}: {e}")
        return None


def save_training_state(
    path: Path,
    *,
    next_round_index: int,
    next_scene_index: int,
    num_timesteps: int,
    completed_traj_count: int,
    completed_display_step: int,
    scene_ids_list: list[str],
    trajectory_names: list[str],
    trajectory_chunk_size: int,
    resume_checkpoint_path: str = "",
    finished: bool = False,
) -> None:
    data = {
        "version": 1,
        "finished": bool(finished),
        "next_round_index": int(next_round_index),
        "next_scene_index": int(next_scene_index),
        "num_timesteps": int(num_timesteps),
        "completed_traj_count": int(completed_traj_count),
        "completed_display_step": int(completed_display_step),
        "scene_ids_list": list(scene_ids_list),
        "trajectory_names": list(trajectory_names),
        "trajectory_chunk_size": int(trajectory_chunk_size),
        "resume_checkpoint_path": str(resume_checkpoint_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _fingerprint_ok(state: dict, scene_ids_list: list[str], trajectory_names: list[str], chunk_size: int) -> bool:
    if state.get("scene_ids_list") != list(scene_ids_list):
        return False
    if state.get("trajectory_names") != list(trajectory_names):
        return False
    if int(state.get("trajectory_chunk_size", -1)) != int(chunk_size):
        return False
    return True


class SwanLabRLCallback(BaseCallback):

    def __init__(
        self,
        total_timesteps,
        scene_id,
        trajectory_name,
        total_trajectories: int = 0,
        use_swanlab=True,
        swanlab_project="RL-PPO-UAV",
        swanlab_experiment_name=None,
        verbose=0,
        tqdm_position=None,
        save_execution_data=False,
        execution_data_dir=None,
        critic_warmup_steps: int = 0,
        initial_completed_traj_count: int = 0,
        initial_display_step: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.scene_id = scene_id
        self.trajectory_name = trajectory_name
        self.use_swanlab = use_swanlab and SWANLAB_AVAILABLE
        self.swanlab_project = swanlab_project
        self.swanlab_experiment_name = swanlab_experiment_name
        self.tqdm_position = tqdm_position
        self.save_execution_data = bool(save_execution_data)
        self.execution_data_dir = execution_data_dir
        self._pbar = None
        self._last_reward = 0.0
        self._last_dist = 0.0
        self._last_step = 0
        self._last_max_steps = 1
        self._last_trajectory = "-"
        self._episode_rewards = []
        self._episode_distances = []
        self._n_episodes = 0
        self._n_success = 0
        self._n_collision = 0
        self._sum_episode_reward = 0.0
        self._current_episode_reward = 0.0
        self.critic_warmup_steps = int(max(0, critic_warmup_steps))
        self._postwarm_n_episodes = 0
        self._postwarm_n_success = 0
        self._postwarm_n_collision = 0
        self._postwarm_sum_episode_reward = 0.0
        self._swanlab_initialized = False  # message learn() message init message "already initialized"
        self._completed_steps = int(initial_display_step)
        self._completed_traj_count = int(initial_completed_traj_count)
        self._baseline_traj_for_eta = int(initial_completed_traj_count)

        self._display_step = int(initial_display_step)
        self._last_real_num_timesteps = 0
        self._scene_start_display_step = 0
        self._last_delta_step = 0

        self.total_trajectories = int(total_trajectories) if total_trajectories else 0

        self._run_start_ts = int(time.time())
        self._episode_frames = []
        self._trajectory_exhausted_logged = False

    def _on_training_start(self):
        if self.use_swanlab and SWANLAB_AVAILABLE and not self._swanlab_initialized:
            self._swanlab_initialized = True
            import os
            os.environ.setdefault("SWANLAB_NO_INTERACTIVE", "1")
            if not os.environ.get("SWANLAB_API_KEY"):
                os.environ.setdefault("SWANLAB_MODE", "local")
            else:
                os.environ.setdefault("SWANLAB_MODE", "cloud")
            config = {
                "total_timesteps": self.total_timesteps,
                "scene_id": self.scene_id,
                "trajectory_name": self.trajectory_name,
                "critic_warmup_steps": int(self.critic_warmup_steps),
            }
            swanlab.init(
                project=self.swanlab_project,
                experiment_name=self.swanlab_experiment_name,
                config=config,
                mode=os.environ.get("SWANLAB_MODE", "cloud"),
            )
        self._scene_start_display_step = int(self._completed_steps)
        self._display_step = int(self._completed_steps)
        self._last_real_num_timesteps = int(self.num_timesteps)

        total_traj = int(self.total_trajectories) if int(self.total_trajectories) > 0 else 1
        try:
            from tqdm import tqdm
            tqdm_kwargs = dict(
                total=total_traj,
                initial=int(self._completed_traj_count),
                unit="traj",
                dynamic_ncols=True,
                desc=f"{self.scene_id}/{self.trajectory_name}",
                ncols=120,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}",
            )
            if self.tqdm_position is not None:
                tqdm_kwargs["position"] = int(self.tqdm_position)
            self._pbar = tqdm(**tqdm_kwargs)
            self._pbar.set_postfix_str(
                f"{self._get_elapsed_and_remaining_str()} steps={int(self.num_timesteps)} frame=-/- action=- dist=-"
            )
        except ImportError:
            self._pbar = None

    def _format_hhmmss(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        s = int(seconds)
        hh = s // 3600
        mm = (s % 3600) // 60
        ss = s % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    def _get_elapsed_and_remaining_str(self) -> str:
        elapsed_s = float(time.time() - self._run_start_ts)
        completed_total = int(self._completed_traj_count)
        total = int(self.total_trajectories) if int(self.total_trajectories) > 0 else 0

        elapsed_str = self._format_hhmmss(elapsed_s)
        if total <= 0:
            return f"{elapsed_str} / --:--:--"

        remaining_traj = total - completed_total
        if remaining_traj <= 0:
            return f"{elapsed_str} / 00:00:00"

        done_this_session = completed_total - int(self._baseline_traj_for_eta)
        if done_this_session <= 0:
            return f"{elapsed_str} / --:--:--"

        remaining_s = float(remaining_traj) * elapsed_s / float(done_this_session)
        remain_str = self._format_hhmmss(remaining_s)
        return f"{elapsed_str} / {remain_str}"

    def _on_step(self):
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])
        rewards = self.locals.get("rewards", None)
        real_step = int(self.num_timesteps)
        delta_step = real_step - int(self._last_real_num_timesteps)
        if delta_step < 0:
            delta_step = 0
        self._display_step += delta_step
        self._last_real_num_timesteps = real_step
        self._last_delta_step = delta_step
        if rewards is not None:
            r = rewards[0] if isinstance(rewards, (list, np.ndarray)) else float(rewards)
            self._last_reward = r
            self._current_episode_reward += float(r)
            if self.use_swanlab and SWANLAB_AVAILABLE:
                swanlab.log({"rl/step_reward": float(self._current_episode_reward)}, step=self._display_step)
        info = (infos[0] if (infos and isinstance(infos, list)) else infos) if infos else {}

        if info and info.get("trajectory_exhausted", False):
            try:
                self._completed_steps = int(self._display_step)
                if self._pbar is not None:
                    self._pbar.set_postfix_str(
                        f"{self._get_elapsed_and_remaining_str()} steps={int(self.num_timesteps)} frame={self._last_step}/{self._last_max_steps} action=- dist={self._last_dist:.1f}"
                    )
                    self._pbar.refresh()
            except Exception:
                pass
            try:
                if self.model is not None:
                    self.model.stop_training = True
            except Exception:
                pass
            if not getattr(self, "_trajectory_exhausted_logged", False):
                self._trajectory_exhausted_logged = True
                print("[RL] trajectory_range exhausted: stop training for current scene.")
            return False

        if self.use_swanlab and SWANLAB_AVAILABLE and info:
            step_log = {}
            if "reward_progress" in info:
                step_log["rl/reward_progress"] = float(info["reward_progress"])
            if "reward_smooth_penalty" in info:
                step_log["rl/reward_smooth_penalty"] = float(info["reward_smooth_penalty"])
            if step_log:
                swanlab.log(step_log, step=self._display_step)
        if info:
            self._last_dist = info.get("distance", self._last_dist)
            self._last_step = info.get("step", self._last_step)
            self._episode_distances.append(self._last_dist)
            try:
                step_reward = (
                    float(rewards[0]) if isinstance(rewards, (list, np.ndarray)) else float(rewards)
                ) if rewards is not None else None
            except Exception:
                step_reward = None

            extra_frame0 = info.get("extra_frame0", None)

            frame_rec = {
                "num_timesteps": int(self.num_timesteps),
                "display_step": int(self._display_step),
                "step": int(info.get("step", self._last_step)),
                "distance": float(info.get("distance", self._last_dist)),
                "delta_dist": float(info.get("delta_dist", 0.0)),
                "reward_step": step_reward,
                "reward_progress": float(info.get("reward_progress", 0.0)) if "reward_progress" in info else None,
                "reward_smooth_penalty": float(info.get("reward_smooth_penalty", 0.0)) if "reward_smooth_penalty" in info else None,
                "trajectory_name": info.get("trajectory_name", self._last_trajectory),
                "episode_success": bool(info.get("episode_success", False)),
                "episode_collision": bool(info.get("episode_collision", False)),
            }

            if isinstance(extra_frame0, dict):
                extra_phys_action = extra_frame0.get("phys_action")
                extra_gt_phys_action = extra_frame0.get("gt_phys_action")
                frame0_rec = {
                    "num_timesteps": max(0, int(self.num_timesteps) - 1),
                    "display_step": int(self._display_step),
                    "step": int(extra_frame0.get("step", 0)),
                    "distance": float(extra_frame0.get("distance", self._last_dist)),
                    "delta_dist": float(extra_frame0.get("delta_dist", 0.0)),
                    "reward_step": None,
                    "reward_progress": None,
                    "reward_smooth_penalty": None,
                    "trajectory_name": extra_frame0.get(
                        "trajectory_name", info.get("trajectory_name", self._last_trajectory)
                    ),
                    "episode_success": bool(extra_frame0.get("episode_success", False)),
                    "episode_collision": bool(extra_frame0.get("episode_collision", False)),
                }

                if isinstance(extra_phys_action, (list, np.ndarray)) and len(extra_phys_action) >= 4:
                    frame0_rec["phys_action"] = [
                        float(extra_phys_action[0]),
                        float(extra_phys_action[1]),
                        float(extra_phys_action[2]),
                        float(extra_phys_action[3]),
                    ]
                else:
                    frame0_rec["phys_action"] = None

                extra_base_phys_action = extra_frame0.get("base_phys_action", None)
                if (
                    isinstance(extra_base_phys_action, (list, np.ndarray))
                    and len(extra_base_phys_action) >= 4
                ):
                    frame0_rec["base_phys_action"] = [
                        float(extra_base_phys_action[0]),
                        float(extra_base_phys_action[1]),
                        float(extra_base_phys_action[2]),
                        float(extra_base_phys_action[3]),
                    ]
                else:
                    frame0_rec["base_phys_action"] = None

                if isinstance(extra_gt_phys_action, (list, np.ndarray)) and len(extra_gt_phys_action) >= 4:
                    pass

                self._episode_frames.append(frame0_rec)

            phys = info.get("phys_action")
            if isinstance(phys, (list, np.ndarray)) and len(phys) >= 4:
                frame_rec["phys_action"] = [float(phys[0]), float(phys[1]), float(phys[2]), float(phys[3])]
            else:
                frame_rec["phys_action"] = None

            base_phys = info.get("base_phys_action", None)
            if isinstance(base_phys, (list, np.ndarray)) and len(base_phys) >= 4:
                frame_rec["base_phys_action"] = [
                    float(base_phys[0]),
                    float(base_phys[1]),
                    float(base_phys[2]),
                    float(base_phys[3]),
                ]
            else:
                frame_rec["base_phys_action"] = None

            self._episode_frames.append(frame_rec)
        done = dones and (dones[0] if isinstance(dones, (list, np.ndarray)) else dones)
        if done:
            if info:
                final_step = int(info.get("step", self._last_step))
                max_steps = int(getattr(self.training_env.envs[0], "max_steps", final_step))
                if self._pbar is not None and max_steps > 0 and final_step >= max_steps:
                    self._pbar.set_postfix_str(
                        f"{self._get_elapsed_and_remaining_str()} steps={int(self.num_timesteps)} frame={max_steps}/{max_steps} action=- base_action=- dist={self._last_dist:.1f}"
                    )
            self._n_episodes += 1
            self._completed_traj_count += 1
            if self._pbar is not None:
                if self.total_trajectories > 0:
                    self._pbar.n = min(self._completed_traj_count, self.total_trajectories)
                else:
                    self._pbar.n = self._completed_traj_count
                self._pbar.refresh()
            post_warmup = int(self.num_timesteps) >= int(self.critic_warmup_steps)
            if post_warmup and info:
                if info.get("episode_success", False):
                    self._postwarm_n_success += 1
                if info.get("episode_collision", False):
                    self._postwarm_n_collision += 1
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
            else:
                self._episode_rewards.append(self._current_episode_reward)
            self._sum_episode_reward += self._current_episode_reward
            if post_warmup:
                self._postwarm_n_episodes += 1
                self._postwarm_sum_episode_reward += self._current_episode_reward

            mean_reward = None
            if self._postwarm_n_episodes > 0:
                mean_reward = self._postwarm_sum_episode_reward / self._postwarm_n_episodes
                if info and info.get("episode_collision", False):
                    reason = "collision"
                elif info and info.get("episode_success", False):
                    reason = "success"
                else:
                    reason = "max_steps"
                final_dist = float(info.get("distance", self._last_dist)) if info else float(self._last_dist)
                print(
                    f"[RL] episode {self._n_episodes} done, "
                    f"mean_reward = {(mean_reward if mean_reward is not None else float('nan')):.4f} ({reason}), "
                    f"final_dist = {final_dist:.3f}m"
                    f" | global_num_timesteps={int(self.num_timesteps)}"
                    f" | traj_done={self._completed_traj_count}/{(self.total_trajectories if self.total_trajectories > 0 else '?')}"
                )

                if self.save_execution_data and self.execution_data_dir:
                    try:
                        out_dir = Path(self.execution_data_dir)
                        scene_dir = out_dir / self.scene_id
                        scene_dir.mkdir(parents=True, exist_ok=True)
                        traj_name = info.get("trajectory_name", self._last_trajectory) if info else self._last_trajectory
                        out_file = scene_dir / f"{traj_name}.json"
                        rec = {
                            "run_start_ts": self._run_start_ts,
                            "episode_idx": self._n_episodes,
                            "scene_id": self.scene_id,
                            "trajectory_name": traj_name,
                            "reason": reason,
                            "mean_reward": float(mean_reward),
                            "final_distance_m": float(final_dist),
                            "step": int(info.get("step", self._last_step)) if info else int(self._last_step),
                            "episode_success": bool(info.get("episode_success", False)) if info else False,
                            "episode_collision": bool(info.get("episode_collision", False)) if info else False,
                            "num_timesteps": int(self.num_timesteps),
                            "display_step": int(self._display_step),
                            "num_frames": int(len(self._episode_frames)),
                            "frames": self._episode_frames,
                        }
                        with open(out_file, "w", encoding="utf-8") as f:
                            json.dump(rec, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"[RL] message: {e}")
                self._episode_frames = []
            if self.use_swanlab and SWANLAB_AVAILABLE and self._postwarm_n_episodes > 0:
                success_rate = self._postwarm_n_success / self._postwarm_n_episodes
                collision_rate = self._postwarm_n_collision / self._postwarm_n_episodes
                swanlab.log(
                    {
                        "rl/mean_reward": float(mean_reward) if mean_reward is not None else None,
                        "rl/success_rate": float(success_rate),
                        "rl/collision_rate": float(collision_rate),
                        "rl/postwarm_episodes": int(self._postwarm_n_episodes),
                    },
                    step=self._completed_traj_count,
                )
            self._current_episode_reward = 0.0
        if info:
            self._last_max_steps = getattr(
                self.training_env.envs[0], "max_steps", self._last_step + 1
            )
            self._last_trajectory = info.get("trajectory_name", self._last_trajectory)
            phys = info.get("phys_action")
            if phys is not None and isinstance(phys, (list, np.ndarray)):
                act_str = f"({phys[0]:.2f},{phys[1]:.2f},{phys[2]:.2f},{phys[3]:.2f})"
            else:
                act_str = "-"

            base_phys = info.get("base_phys_action", None)
            if base_phys is not None and isinstance(base_phys, (list, np.ndarray)):
                base_act_str = f"({base_phys[0]:.2f},{base_phys[1]:.2f},{base_phys[2]:.2f},{base_phys[3]:.2f})"
            else:
                base_act_str = "-"

            if self._pbar is not None:
                traj_name = self._last_trajectory
                if traj_name.startswith("trajectory_"):
                    traj_num = traj_name.replace("trajectory_", "", 1)
                else:
                    traj_num = traj_name
                self._pbar.set_description(f"{self.scene_id}/{traj_num}", refresh=False)
                self._pbar.set_postfix_str(
                    f"{self._get_elapsed_and_remaining_str()} steps={int(self.num_timesteps)} frame={self._last_step}/{self._last_max_steps} action={act_str} base_action={base_act_str} dist={self._last_dist:.1f}"
                )

        return True

    def _on_training_end(self):
        self._completed_steps = int(self._display_step)  # message learn message“message”，message initial message
        if self._pbar is not None:
            self._pbar.close()


class PeriodicSaveCallback(BaseCallback):

    def __init__(
        self,
        save_path: str,
        save_every_n_steps: int = 0,
        verbose: int = 0,
        save_residual_head: bool = True,
    ):
        super().__init__(verbose)
        self.save_path = str(save_path)
        self.save_every_n_steps = int(save_every_n_steps or 0)
        self.save_residual_head = bool(save_residual_head)

    def _on_step(self) -> bool:
        if self.save_every_n_steps <= 0:
            return True
        t = int(self.num_timesteps)
        if t <= 0:
            return True
        if t % self.save_every_n_steps != 0:
            return True

        try:
            base = Path(self.save_path)
            ckpt_path = str(base.with_name(f"{base.stem}_step_{t}{base.suffix}"))
            self.model.save(ckpt_path)
            if self.save_residual_head and hasattr(self.model.policy, "get_residual_head_state_dict"):
                residual_head_path = base.parent / f"residual_head_step_{t}.pt"
                th.save(self.model.policy.get_residual_head_state_dict(), str(residual_head_path))
            if self.verbose > 0:
                print(f"[RL] checkpoint saved at step {t}: {ckpt_path}")
        except Exception as e:
            print(f"[RL] checkpoint save failed at step {t}: {e}")
        return True


class CriticWarmupCallback(BaseCallback):

    def __init__(self, warmup_steps: int, warmup_log_std: float = -6.0, verbose: int = 0):
        super().__init__(verbose)
        self.warmup_steps = int(max(0, warmup_steps))
        self.warmup_log_std = float(warmup_log_std)
        self._actor_frozen = False
        self._actor_params = None
        self._orig_residual_scale = None
        self._orig_log_std = None

    def _set_actor_requires_grad(self, requires_grad: bool) -> None:
        if self._actor_params is None and self.model is not None:
            policy = self.model.policy
            actor_params = []
            for name, param in policy.named_parameters():
                if name.startswith("residual_head.") or name == "log_std":
                    actor_params.append(param)
                elif name.startswith("action_net"):
                    actor_params.append(param)
            self._actor_params = actor_params
        if self._actor_params is None:
            return
        for p in self._actor_params:
            p.requires_grad = requires_grad

        policy = self.model.policy if self.model is not None else None
        if policy is not None and hasattr(policy, "_residual_scale"):
            if requires_grad is False and self._orig_residual_scale is None:
                self._orig_residual_scale = float(getattr(policy, "_residual_scale"))
            if requires_grad is False:
                setattr(policy, "_residual_scale", 0.0)
            else:
                if self._orig_residual_scale is not None:
                    setattr(policy, "_residual_scale", self._orig_residual_scale)

        if policy is not None and hasattr(policy, "log_std"):
            if requires_grad is False:
                if self._orig_log_std is None:
                    self._orig_log_std = policy.log_std.detach().clone()
                with th.no_grad():
                    policy.log_std.data.fill_(self.warmup_log_std)
            else:
                if self._orig_log_std is not None:
                    with th.no_grad():
                        policy.log_std.data.copy_(self._orig_log_std)
                    self._orig_log_std = None

    def _on_training_start(self) -> None:
        if self.warmup_steps <= 0 or self.model is None:
            return
        if self.num_timesteps < self.warmup_steps:
            self._set_actor_requires_grad(False)
            self._actor_frozen = True
            if self.verbose > 0:
                print(f"[RL] Critic warmup: message actor，message {self.warmup_steps} message critic")
        else:
            self._actor_frozen = False

    def _on_step(self) -> bool:
        if self._actor_frozen and self.num_timesteps >= self.warmup_steps:
            self._set_actor_requires_grad(True)
            self._actor_frozen = False
            if self.verbose > 0:
                print("[RL] Critic warmup message：message actor，message joint message")
        return True


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


def run_ppo_training(args):
    model_path = args.model_path
    scene_id = args.scene_id
    trajectory_name = args.trajectory_name
    dataset_root = Path(args.dataset_root)
    sim_server_host = args.sim_server_host
    sim_server_port = args.sim_server_port
    gpu_id = args.gpu_id
    total_timesteps = int(getattr(args, "total_timesteps", 50000))
    learning_rate = args.learning_rate
    n_steps = args.n_steps
    batch_size = args.batch_size
    save_path = args.save_path

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    base_model_path = (getattr(args, "base_model_path", None) or "").strip() or None
    if base_model_path is None:
        base_model_path = str(PROJECT_ROOT / "Qwen3-VL-2B-Instruct")
    use_numeric_encoder = True
    use_backbone = True
    model, processor = load_model_like_validate(
        model_path=model_path,
        base_model_path=base_model_path,
        device=device,
        use_numeric_encoder=use_numeric_encoder,
        use_backbone=use_backbone,
    )
    model.use_numeric_encoder = use_numeric_encoder
    model.use_backbone = use_backbone
    _inc_vel = True
    _inc_prev = True

    rl_lora_enable = str(getattr(args, "rl_lora_enable", "false")).lower() in ("true", "1", "yes")
    if rl_lora_enable:
        print("[RL] message RL message LoRA message（message LoRA + action_head，message）")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    base_head = getattr(model, "action_head", None)
    if base_head is not None:
        for p in base_head.parameters():
            p.requires_grad = False

    model_dict = {
        "model": model,
        "processor": processor or getattr(model, "processor", None),
        "device": device,
        "checkpoint_dir": str(Path(model_path)),
        "base_model_path": base_model_path,
        "generate_system_prompt": generate_system_prompt,
        "generate_user_prompt": generate_user_prompt,
        "include_target_vel": _inc_vel,
        "include_prev_action": _inc_prev,
        "use_numeric_encoder": use_numeric_encoder,
    }

    for p in model.parameters():
        p.requires_grad = False

    if rl_lora_enable:
        lora_param_count = 0
        for name, p in model.named_parameters():
            if ("lora_" in name) or ("lora_A" in name) or ("lora_B" in name):
                p.requires_grad = True
                lora_param_count += p.numel()
        print(f"[RL] RL-LoRA: message LoRA message = {lora_param_count}")
        for p in model.action_head.parameters():
            p.requires_grad = False
    else:
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.action_head.parameters():
            p.requires_grad = False

    il_action_head = copy.deepcopy(model.action_head)
    il_action_head.eval()
    for p in il_action_head.parameters():
        p.requires_grad = False
    model_dict["il_action_head"] = il_action_head

    scene_ids_list = getattr(args, "scene_ids_list", None) or [scene_id]
    trajectory_range = getattr(args, "trajectory_range", None)  # list[str] message None
    trajectory_chunk_size = int(getattr(args, "trajectory_chunk_size", 50) or 50)

    def _chunk_list(seq, chunk_size: int):
        if not seq:
            return []
        if chunk_size <= 0:
            return [list(seq)]
        return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]

    full_traj_list = trajectory_range if isinstance(trajectory_range, (list, tuple)) else None
    print(f"[RL] message: scenes={scene_ids_list}")

    ppo_model = None
    progress_callback = None
    callback = None
    use_swanlab = getattr(args, "use_swanlab", True) and SWANLAB_AVAILABLE
    if use_swanlab:
        print("[RL] SwanLab message")
    _progress_rank = os.environ.get("PROGRESS_RANK", "0")
    _tqdm_position = int(_progress_rank) if str(_progress_rank).isdigit() else 0
    save_models_enabled = str(getattr(args, "save_models", "true")).lower() in ("true", "1", "yes")
    save_every_n_steps = int(getattr(args, "save_every_n_steps", 0) or 0)
    critic_warmup_steps = int(getattr(args, "critic_warmup_steps", 0) or 0)

    trajectory_rounds = (
        _chunk_list(full_traj_list, trajectory_chunk_size) if full_traj_list else [None]
    )

    print(f"[RL] trajectory rounds = {len(trajectory_rounds)}，chunk_size={trajectory_chunk_size}")

    if full_traj_list:
        fingerprint_traj_names = list(full_traj_list)
    else:
        fingerprint_traj_names = [str(getattr(args, "trajectory_name", "trajectory_0001"))]
    ts_path_str = (getattr(args, "training_state_path", "") or "").strip()
    training_state_path = Path(ts_path_str) if ts_path_str else _default_training_state_path(save_path)
    resume_flag = bool(getattr(args, "resume", False))
    resume_from_arg = (getattr(args, "resume_from", "") or "").strip()
    start_ri, start_si = 0, 0
    resume_initial_traj = 0
    resume_initial_display = 0
    training_state: Optional[dict] = None
    resume_ckpt_path: Optional[Path] = None

    if resume_flag:
        training_state = load_training_state(training_state_path)
        if not training_state:
            raise FileNotFoundError(
                f"[RL] --resume message: {training_state_path}\n"
                f"     message，message。"
            )
        if training_state.get("finished"):
            print(f"[RL] message（{training_state_path}），message。")
            return
        if not _fingerprint_ok(
            training_state, scene_ids_list, fingerprint_traj_names, trajectory_chunk_size
        ):
            raise RuntimeError(
                "[RL] message：message scene_ids / trajectory_range / trajectory_chunk_size "
                "message。message，message。"
                f"\n  message: {training_state_path}"
            )
        start_ri = int(training_state.get("next_round_index", 0))
        start_si = int(training_state.get("next_scene_index", 0))
        resume_initial_traj = int(training_state.get("completed_traj_count", 0))
        resume_initial_display = int(training_state.get("completed_display_step", 0))
        ckpt = _resolve_resume_checkpoint(save_path, resume_from_arg)
        if ckpt is None:
            raise FileNotFoundError(
                "[RL] --resume message PPO checkpoint。"
                f" message: resume_from={resume_from_arg!r}, save_path={save_path}, "
                f"message *_step_* message。"
            )
        resume_ckpt_path = ckpt
        print(
            f"[RL] message: message round={start_ri}, scene_index={start_si} message；"
            f"checkpoint={resume_ckpt_path}；message={training_state_path}"
        )

    def _should_skip(ri: int, si: int) -> bool:
        return (ri < start_ri) or (ri == start_ri and si < start_si)

    if resume_flag and start_ri >= len(trajectory_rounds):
        print(
            f"[RL] message next_round_index={start_ri} message round message {len(trajectory_rounds)}，message。"
        )
        return

    for ri, trajectory_range_chunk in enumerate(trajectory_rounds):
        if full_traj_list:
            start_idx = ri * trajectory_chunk_size
            end_idx = min(len(full_traj_list), (ri + 1) * trajectory_chunk_size) - 1
            print(
                f"[RL] ===== Round {ri+1}/{len(trajectory_rounds)}："
                f"trajectories[{start_idx}..{end_idx}] ====="
            )

        for si, scene_name in enumerate(scene_ids_list):
            if _should_skip(ri, si):
                continue
            if si > 0:
                prev_scene = scene_ids_list[si - 1]
                print(f"[RL] switch scene: {prev_scene} -> {scene_name}")
            print(f"[RL] ===== message {scene_name} ({si+1}/{len(scene_ids_list)}) (round {ri+1}) =====")

            executor = make_executor(
                scene_id=scene_name,
                trajectory_name=trajectory_name,
                sim_server_host=sim_server_host,
                sim_server_port=sim_server_port,
                gpu_id=gpu_id,
            )

            def make_env(_executor=executor, _scene=scene_name, _traj_chunk=trajectory_range_chunk):
                return AirSimUAVTrainEnv(
                    model_dict=model_dict,
                    dataset_root=dataset_root,
                    scene_id=_scene,
                    trajectory_name=trajectory_name,
                    executor=_executor,
                    max_vel=args.max_vel,
                    max_yaw_rate=args.max_yaw_rate,
                    yaw_scale=1.0,
                    reward_progress_scale=getattr(args, "reward_progress_scale", 1.0),
                    trajectory_range=_traj_chunk,
                    reward_type=getattr(args, "reward_type", "progress"),
                    reward_r_level=getattr(args, "reward_r_level", 10.0),
                    max_steps=getattr(args, "max_steps", None),
                    max_steps_ratio=getattr(args, "max_steps_ratio", None),
                )

            env = DummyVecEnv([make_env])

            created_from_checkpoint = False
            if ppo_model is None:
                core_obs_dim = getattr(env.envs[0], "_core_obs_dim", None)
                append_prev_action_dim = getattr(env.envs[0], "_append_prev_action_dim", 4)
                if resume_flag and resume_ckpt_path is not None:
                    ppo_model = PPO.load(
                        str(resume_ckpt_path),
                        env=env,
                        device=device,
                    )
                    created_from_checkpoint = True
                    print(f"[RL] message checkpoint message PPO（message）: {resume_ckpt_path}")
                else:
                    ppo_model = PPO(
                        ResidualActionHeadPolicy,
                        env,
                        learning_rate=learning_rate,
                        n_steps=n_steps,
                        batch_size=batch_size,
                        gamma=args.gamma,
                        verbose=0,
                        device=device,
                        target_kl=getattr(args, "target_kl", None),
                        clip_range=0.1,
                        policy_kwargs=dict(
                            base_action_head=model_dict["model"].action_head,
                            residual_scale=getattr(args, "residual_scale", 0.1),
                            residual_head_arch=[256, 256],
                            value_head_arch=[256, 256],
                            log_std_init=getattr(args, "log_std_init", -3.0),
                            core_obs_dim=core_obs_dim,
                            append_prev_action_dim=append_prev_action_dim,
                        ),
                    )
                print("[RL] PPO policy message（requires_grad=True）:")
                for n, p in ppo_model.policy.named_parameters():
                    if p.requires_grad:
                        print(f"  trainable: {n}, shape={tuple(p.shape)}")

                if trajectory_range_chunk:
                    traj_display = f"chunk{ri+1}_{len(trajectory_range_chunk)}traj"
                else:
                    traj_display = ("multi" if full_traj_list else trajectory_name)
                progress_callback = SwanLabRLCallback(
                    total_timesteps=int(total_timesteps),
                    total_trajectories=(
                        (len(full_traj_list) * len(scene_ids_list))
                        if full_traj_list is not None
                        else len(scene_ids_list)
                    ),
                    scene_id=scene_name,
                    trajectory_name=traj_display,
                    use_swanlab=use_swanlab,
                    swanlab_project=getattr(args, "swanlab_project", None) or "RL-PPO-UAV",
                    swanlab_experiment_name=getattr(args, "swanlab_experiment_name", None),
                    tqdm_position=_tqdm_position,
                    save_execution_data=str(getattr(args, "save_execution_data", "false")).lower()
                    in ("true", "1", "yes"),
                    execution_data_dir=(getattr(args, "execution_data_dir", "") or ""),
                    critic_warmup_steps=critic_warmup_steps,
                    initial_completed_traj_count=resume_initial_traj,
                    initial_display_step=resume_initial_display,
                )
                cb_list = [progress_callback]
                if save_models_enabled and save_every_n_steps > 0:
                    cb_list.append(
                        PeriodicSaveCallback(
                            save_path=save_path,
                            save_every_n_steps=save_every_n_steps,
                            verbose=0,
                            save_residual_head=True,
                        )
                    )
                if critic_warmup_steps > 0:
                    cb_list.append(
                        CriticWarmupCallback(
                            critic_warmup_steps,
                            warmup_log_std=getattr(args, "warmup_log_std", -6.0),
                        )
                    )
                callback = CallbackList(cb_list) if len(cb_list) > 1 else cb_list[0]
            else:
                ppo_model.set_env(env)
                if progress_callback is not None:
                    progress_callback.scene_id = scene_name
                    if trajectory_range_chunk:
                        progress_callback.trajectory_name = f"chunk{ri+1}_{len(trajectory_range_chunk)}traj"
                    else:
                        progress_callback.trajectory_name = ("multi" if full_traj_list else trajectory_name)
                    progress_callback._trajectory_exhausted_logged = False

            ppo_model.stop_training = False
            reset_learn_timesteps = (ri == 0 and si == 0) and not created_from_checkpoint
            ppo_model.learn(
                total_timesteps=int(total_timesteps),
                callback=callback,
                reset_num_timesteps=reset_learn_timesteps,
            )

            n_scenes = len(scene_ids_list)
            n_rounds = len(trajectory_rounds)
            if si < n_scenes - 1:
                next_ri, next_si = ri, si + 1
            elif ri < n_rounds - 1:
                next_ri, next_si = ri + 1, 0
            else:
                next_ri, next_si = n_rounds, 0
            finished_all = next_ri >= n_rounds
            ckpt_for_state = (
                str(resume_ckpt_path) if resume_ckpt_path is not None else str(save_path)
            )
            save_training_state(
                training_state_path,
                next_round_index=next_ri,
                next_scene_index=next_si,
                num_timesteps=int(ppo_model.num_timesteps),
                completed_traj_count=int(progress_callback._completed_traj_count),
                completed_display_step=int(progress_callback._display_step),
                scene_ids_list=scene_ids_list,
                trajectory_names=fingerprint_traj_names,
                trajectory_chunk_size=trajectory_chunk_size,
                resume_checkpoint_path=ckpt_for_state,
                finished=finished_all,
            )
            if finished_all:
                print(f"[RL] message round message，message {training_state_path}（finished=true）")
            else:
                print(
                    f"[RL] message -> message: round_index={next_ri}, scene_index={next_si} | {training_state_path}"
                )

            try:
                executor.disconnect()
            except Exception:
                pass

            try:
                import msgpackrpc  # type: ignore
                c = msgpackrpc.Client(
                    msgpackrpc.Address(sim_server_host, sim_server_port), timeout=30
                )
                c.call("close_scenes", sim_server_host, [scene_name])
                c.close()
            except Exception:
                pass

    save_models = str(getattr(args, "save_models", "true")).lower() in ("true", "1", "yes")
    if save_models:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        residual_head_path = save_path.parent / "residual_head.pt"
        th.save(ppo_model.policy.get_residual_head_state_dict(), str(residual_head_path))
        print(f"[RL] message residual_head: {residual_head_path}")
        ppo_model.save(str(save_path))
        print(f"[RL] message PPO message: {save_path}")
    else:
        print("[RL] save_models=false：message")

    try:
        executor.disconnect()
    except Exception:
        pass


def parse_trajectory_range(s: str) -> list[int]:
    if not s:
        return []
    s = str(s).strip()
    if not s:
        return []
    ids: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            try:
                lo = int(lo_str)
                hi = int(hi_str)
            except ValueError:
                continue
            if lo <= hi:
                ids.extend(range(lo, hi + 1))
            else:
                ids.extend(range(hi, lo + 1))
        else:
            try:
                ids.append(int(part))
            except ValueError:
                continue
    return sorted(set(ids))

def parse_scene_ids(s: str) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO message AirSim message（message）"
    )
    parser.add_argument("--model_path", type=str, default=str(PROJECT_ROOT / "work_dirs" / "qwen3vl-uav"), help="SL message（checkpoint message，message Val/validate_seen.sh message）")
    parser.add_argument("--base_model_path", type=str, default="", help="Qwen3-VL message（message/Qwen3-VL-2B-Instruct，message Val message key message）")
    parser.add_argument("--scene_id", type=str, default="City_1", help="message ID")
    parser.add_argument("--scene_ids", type=str, default="", help='message，message "City_1,City_2,City_3"；message --scene_id')
    parser.add_argument("--trajectory_name", type=str, default="trajectory_0001", help="message（message）")
    parser.add_argument(
        "--trajectory_range",
        type=str,
        default="",
        help='message（message，message），message "1-3" message "1,3,5" message "1-3,5"；'
             'message trajectory_name message trajectory_range，message trajectory_range message。',
    )
    parser.add_argument(
        "--trajectory_chunk_size",
        type=int,
        default=50,
        help="message trajectory_range message round；<=0 message。",
    )
    parser.add_argument("--dataset_root", type=str, default=str(PROJECT_ROOT / "Dataset"), help="Dataset message")
    parser.add_argument("--sim_server_host", type=str, default="127.0.0.1", help="SimServer message")
    parser.add_argument("--sim_server_port", type=int, default=30000, help="SimServer message")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--total_timesteps", type=int, default=50000, help="PPO message")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="message")
    parser.add_argument("--n_steps", type=int, default=1024, help="PPO n_steps")
    parser.add_argument("--batch_size", type=int, default=32, help="PPO batch_size")
    parser.add_argument("--gamma", type=float, default=0.99, help="message")
    parser.add_argument("--max_vel", type=float, default=5.0, help="message (m/message)")
    parser.add_argument("--max_yaw_rate", type=float, default=45.0, help="message yaw message (message/message)")
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(PROJECT_ROOT / "RL" / "ppo_action_head_refined.zip"),
        help="PPO message（message .zip，message SB3 PPO.save message；message residual_head.pt）",
    )
    parser.add_argument("--reward_progress_scale", type=float, default=1.0, help="message")
    parser.add_argument("--reward_type", type=str, default="progress", choices=("progress", "hidden_sim", "mixed"), help="message: progress=message/message/message; hidden_sim=F_smessageF_wmessage; mixed=hidden_sim+message")
    parser.add_argument("--reward_r_level", type=float, default=10.0, help="hidden_sim message r_level")
    parser.add_argument("--rl_lora_enable", type=str, default="false", help="RL message backbone message LoRA message（true/false），message action_head")
    parser.add_argument("--target_kl", type=float, default=None, help="PPO KL message KL(new||old)≈target_kl，None message KL message")
    parser.add_argument("--critic_warmup_steps", type=int, default=0, help="message critic（message actor），0 message")
    parser.add_argument(
        "--warmup_log_std",
        type=float,
        default=-6.0,
        help="critic warmup message log_std，message（message --critic_warmup_steps>0 message）",
    )
    parser.add_argument("--max_steps", type=int, default=None, help="message（message Val --max_steps message）；message max_steps_ratio message")
    parser.add_argument("--max_steps_ratio", type=float, default=None, help="message=message*message（message Val --max_steps_ratio message）；max_steps message")
    parser.add_argument("--log_std_init", type=float, default=-3.0, help="message log_std，std=exp(log_std)")
    parser.add_argument("--residual_scale", type=float, default=0.1, help="Residual PPO message residual_scale（λ）：message a = a_base + λ * a_residual，message 0.05~0.2 message“message”")
    parser.add_argument("--use_swanlab", action="store_true", default=True, help="message SwanLab")
    parser.add_argument("--no_swanlab", action="store_false", dest="use_swanlab", help="message SwanLab")
    parser.add_argument("--swanlab_project", type=str, default="RL-PPO-UAV", help="SwanLab message")
    parser.add_argument("--swanlab_experiment_name", type=str, default=None, help="SwanLab message")
    parser.add_argument("--include_target_vel", type=str, default="true", help="message Target vel message（message train_qwen3vl.sh message），true/false")
    parser.add_argument("--include_prev_action", type=str, default="true", help="message Ego previous action message（message train_qwen3vl.sh message），true/false")
    parser.add_argument("--use_numeric_encoder", type=str, default="false", help="message（numeric encoder），message Train_qwen/Val message，true/false")
    parser.add_argument("--use_backbone", type=str, default="true", help="message backbone（true=message；false=message），message Train_qwen/Val message，true/false")
    parser.add_argument(
        "--save_models",
        type=str,
        default="true",
        choices=("true", "false", "1", "0", "yes", "no"),
        help="message（message true）。message scene message worker message。",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=0,
        help="message N message timesteps message checkpoint（message save_models=true message worker message）。message 0=message。",
    )
    parser.add_argument(
        "--save_execution_data",
        type=str,
        default="false",
        choices=("true", "false", "1", "0", "yes", "no"),
        help="message episode message（jsonl）。",
    )
    parser.add_argument(
        "--execution_data_dir",
        type=str,
        default=str(PROJECT_ROOT / "work_dirs" / "rl_execution_data"),
        help="message（--save_execution_data=true message）。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="message rl_training_state.json + PPO checkpoint message；scene_ids / trajectory_range / chunk message。",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="message PPO message；message save_path、message stem_step_* message。",
    )
    parser.add_argument(
        "--training_state_path",
        type=str,
        default="",
        help="message JSON message（next round/scene、message）；message --save_path message rl_training_state.json。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scene_ids = parse_scene_ids(getattr(args, "scene_ids", ""))
    if not scene_ids:
        scene_ids = [args.scene_id]
    args.scene_ids_list = scene_ids

    traj_ids = parse_trajectory_range(getattr(args, "trajectory_range", ""))
    if traj_ids:
        trajectory_names = [f"trajectory_{tid:04d}" for tid in traj_ids]
        run_args = copy.copy(args)
        run_args.trajectory_name = trajectory_names[0]  # message executor message env message
        run_args.trajectory_range = trajectory_names    # message env，reset message
        print(f"[RL] message: scene_ids={scene_ids}, trajectory_range=message {len(trajectory_names)} message，message")
        print(f"[RL] save_path = {run_args.save_path}")
        run_ppo_training(run_args)
        print(f"[RL] message\n")
    else:
        run_ppo_training(args)


