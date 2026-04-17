#!/usr/bin/env python3

MAX_SPEED_NORM = 5.0

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def physical_action_to_norm(phys_action, max_vel, max_yaw_rate):
    if _HAS_TORCH and isinstance(phys_action, torch.Tensor):
        a = phys_action.float().reshape(-1, 4)
        vel_scale = 1.0 / (float(max_vel) + 1e-12)
        yaw_scale = 1.0 / (float(max_yaw_rate) + 1e-12)
        out = torch.cat([a[:, :3] * vel_scale, a[:, 3:4] * yaw_scale], dim=-1)
        return torch.clamp(out, -1.0, 1.0).reshape(phys_action.shape)
    a = np.asarray(phys_action, dtype=np.float64).reshape(-1, 4)
    vel_scale = 1.0 / (float(max_vel) + 1e-12)
    yaw_scale = 1.0 / (float(max_yaw_rate) + 1e-12)
    out = np.concatenate([a[:, :3] * vel_scale, a[:, 3:4] * yaw_scale], axis=1)
    return np.clip(out, -1.0, 1.0).astype(np.float32).reshape(phys_action.shape)


def norm_action_to_physical(norm_action, max_vel, max_yaw_rate, eps=1e-8, max_speed_norm=None):
    cap = float(max_speed_norm) if max_speed_norm is not None else MAX_SPEED_NORM
    if _HAS_TORCH and isinstance(norm_action, torch.Tensor):
        return _norm_to_physical_torch(norm_action, float(max_vel), float(max_yaw_rate), cap)
    return _norm_to_physical_numpy(norm_action, float(max_vel), float(max_yaw_rate), cap)


def _norm_to_physical_numpy(norm_action, max_vel, max_yaw_rate, max_speed_norm):
    a = np.asarray(norm_action, dtype=np.float64)
    orig_shape = a.shape
    a = a.reshape(-1, 4)
    vel_xyz = a[:, :3] * max_vel
    v_norm = np.linalg.norm(vel_xyz, axis=1, keepdims=True)
    scale = np.where(v_norm > max_speed_norm, max_speed_norm / (v_norm + 1e-12), 1.0)
    vel_xyz = vel_xyz * scale
    yaw_rate = a[:, 3:4] * max_yaw_rate
    out = np.concatenate([vel_xyz, yaw_rate], axis=1).astype(np.float32)
    return out.reshape(orig_shape)


def _norm_to_physical_torch(norm_action, max_vel, max_yaw_rate, max_speed_norm):
    a = norm_action.float() if norm_action.dtype != torch.float32 else norm_action
    orig_shape = a.shape
    a = a.reshape(-1, 4)
    vel_xyz = a[:, :3] * max_vel
    v_norm = torch.linalg.norm(vel_xyz, dim=1, keepdim=True)
    scale = torch.where(v_norm > max_speed_norm, max_speed_norm / (v_norm + 1e-12), torch.ones_like(v_norm))
    vel_xyz = vel_xyz * scale
    yaw_rate = a[:, 3:4] * max_yaw_rate
    out = torch.cat([vel_xyz, yaw_rate], dim=-1)
    return out.reshape(orig_shape)
