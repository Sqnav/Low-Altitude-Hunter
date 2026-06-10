#!/usr/bin/env python3

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.spatial.transform import Rotation as R
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


NUM_STATE_HISTORY_LEN = 8
NUM_STATE_DIM = NUM_STATE_HISTORY_LEN * 3


def _as_vec3(v: Any, default: float = 0.0) -> Tuple[float, float, float]:
    if v is None:
        return (default, default, default)
    if isinstance(v, (list, np.ndarray)) and len(v) >= 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    if isinstance(v, dict):
        return (float(v.get("x", default)), float(v.get("y", default)), float(v.get("z", default)))
    return (default, default, default)


def _as_quat(q: Any) -> Tuple[float, float, float, float]:
    if q is None or (isinstance(q, (list, tuple)) and len(q) < 4):
        return (1.0, 0.0, 0.0, 0.0)
    if isinstance(q, (list, tuple, np.ndarray)):
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    if isinstance(q, dict):
        return (
            float(q.get("w", 1)),
            float(q.get("x", 0)),
            float(q.get("y", 0)),
            float(q.get("z", 0)),
        )
    return (1.0, 0.0, 0.0, 0.0)


def _as_prev_action(v: Any, default: float = 0.0) -> Tuple[float, float, float, float]:
    if v is None or (isinstance(v, (list, tuple)) and len(v) < 4):
        return (default, default, default, default)
    if isinstance(v, (list, tuple)):
        return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    return (default, default, default, default)


def _airsim_to_body_frame(
    vector_airsim: Union[List[float], np.ndarray, Dict[str, float]],
    quaternion: Union[List[float], Tuple[float, float, float, float], Dict[str, float]],
) -> Tuple[float, float, float]:
    if not _HAS_SCIPY:
        raise RuntimeError("instruction_generator requires scipy; please install via 'pip install scipy'.")
    x, y, z = _as_vec3(vector_airsim)
    vec = np.array([x, y, z], dtype=np.float64)
    qw, qx, qy, qz = _as_quat(quaternion)
    rotation = R.from_quat([qx, qy, qz, qw])
    body = rotation.inv().apply(vec)
    return (float(body[0]), float(body[1]), float(body[2]))


def _history_vecs_from_start(
    uav_history_airsim: Optional[Iterable[Any]],
    uav_start_airsim: Any,
    history_len: int,
) -> Tuple[float, ...]:
    history_len = max(1, int(history_len))
    if uav_history_airsim is None:
        history = []
    else:
        history = [np.array(_as_vec3(p), dtype=np.float64) for p in uav_history_airsim]

    if not history:
        history = [np.array(_as_vec3(uav_start_airsim), dtype=np.float64)]

    start = np.array(_as_vec3(uav_start_airsim), dtype=np.float64)
    if not np.all(np.isfinite(start)):
        start = history[0].copy()

    if len(history) >= history_len:
        selected = history[-history_len:]
    else:
        selected = [history[0]] * (history_len - len(history)) + history

    rel = [p - start for p in selected]
    return tuple(float(x) for p in rel for x in p.tolist())

def compute_instruction_numeric_state(
    uav_position_airsim: Union[List[float], np.ndarray, Dict[str, float]],
    target_position_airsim: Union[List[float], np.ndarray, Dict[str, float]],
    quaternion: Union[List[float], Tuple[float, float, float, float], Dict[str, float]],
    prev_action: Union[List[float], Tuple[float, float, float, float]],
    target_position_airsim_prev: Optional[Union[List[float], np.ndarray, Dict[str, float]]] = None,
    dt: float = 1.0,
    trajectory_history_airsim: Optional[Iterable[Any]] = None,
    target_history_airsim: Optional[Iterable[Any]] = None,
    uav_history_airsim: Optional[Iterable[Any]] = None,
    uav_start_airsim: Optional[Union[List[float], np.ndarray, Dict[str, float]]] = None,
    history_len: int = NUM_STATE_HISTORY_LEN,
) -> Tuple[float, ...]:
    """Return fixed-length history trajectory points in the UAV-start frame.

    The numeric branch uses the latest ``history_len`` target/trajectory points,
    ordered from oldest to newest. Positions are AirSim-frame coordinates shifted
    by the trajectory's first UAV position. Short histories are left-padded with
    the first available trajectory point, so the output shape is always
    ``history_len * 3``.
    """
    del quaternion, prev_action, target_position_airsim_prev, dt
    start = uav_start_airsim if uav_start_airsim is not None else uav_position_airsim
    if trajectory_history_airsim is not None:
        history = trajectory_history_airsim
    elif target_history_airsim is not None:
        history = target_history_airsim
    elif uav_history_airsim is not None:
        history = uav_history_airsim
    else:
        history = [target_position_airsim]
    return _history_vecs_from_start(history, start, history_len)

def generate_system_prompt(
    scene_id: Optional[str] = None,
    trajectory_name: Optional[str] = None,
    frame_idx: Optional[int] = None,
    num_frames: Optional[int] = None,
    **kwargs: Any,
) -> str:
    return (
        "You are a UAV visual pursuit agent operating in the Body Frame (X-Forward, Y-Right, Z-Down). "
        "Your task is to analyze the FPV image and text instructions to output actions for target interception while avoiding collisions."
    )




def _bucket_magnitude(val: float, small: float, medium: float) -> str:
    a = abs(val)
    if a < 1e-4:
        return "zero"
    if a < small:
        return "weak"
    if a < medium:
        return "medium"
    return "strong"


def _describe_position_axis(val: float, axis: str) -> str:
    mag = _bucket_magnitude(val, small=2.0, medium=10.0)

    if axis == "x":
        if mag == "zero":
            return "roughly at the same longitudinal position"
        direction = "ahead" if val > 0 else "behind"
    elif axis == "y":
        if mag == "zero":
            return "roughly centered laterally"
        direction = "to the right" if val > 0 else "to the left"
    elif axis == "z":
        if mag == "zero":
            return "roughly at the same altitude"
        direction = "below" if val > 0 else "above"
    else:
        return "at an unknown position"

    if mag == "weak":
        prefix = "slightly "
    elif mag == "medium":
        prefix = "moderately "
    else:  # strong
        prefix = "far "

    return prefix + direction


def _describe_velocity_axis(val: float, axis: str) -> str:
    mag = _bucket_magnitude(val, small=0.3, medium=2.0)
    if mag == "zero":
        if axis == "x":
            return "almost stationary in the forward/backward direction"
        if axis == "y":
            return "almost stationary laterally"
        if axis == "z":
            return "almost stationary in altitude"
        return "almost stationary"

    if axis == "x":
        direction = "forward" if val > 0 else "backward"
    elif axis == "y":
        direction = "to the right" if val > 0 else "to the left"
    elif axis == "z":
        direction = "downward" if val > 0 else "upward"
    else:
        direction = "in an unknown direction"

    if mag == "weak":
        prefix = "slowly "
    elif mag == "medium":
        prefix = "moderately "
    else:
        prefix = "rapidly "

    return prefix + f"moving {direction}"


def _describe_yaw_rate(val: float) -> str:
    mag = _bucket_magnitude(val, small=2.0, medium=10.0)
    if mag == "zero":
        return "almost no turning"

    direction = "turning right" if val > 0 else "turning left"
    if mag == "weak":
        prefix = "slowly "
    elif mag == "medium":
        prefix = "moderately "
    else:
        prefix = "quickly "

    return prefix + direction


def generate_user_prompt(
    uav_position_airsim: Union[List[float], np.ndarray, Dict[str, float]],
    target_position_airsim: Union[List[float], np.ndarray, Dict[str, float]],
    quaternion: Union[List[float], Tuple[float, float, float, float], Dict[str, float]],
    prev_action: Union[List[float], Tuple[float, float, float, float]],
    target_position_airsim_prev: Optional[Union[List[float], np.ndarray, Dict[str, float]]] = None,
    dt: float = 1.0,
    include_target_vel: bool = True,
    include_prev_action: bool = True,
    is_first_frame: bool = False,
    **kwargs: Any,
) -> str:
    uav = np.array(_as_vec3(uav_position_airsim))
    target = np.array(_as_vec3(target_position_airsim))

    rel_airsim = target - uav
    pos_body = _airsim_to_body_frame(rel_airsim, quaternion)
    pos_x, pos_y, pos_z = float(pos_body[0]), float(pos_body[1]), float(pos_body[2])

    if target_position_airsim_prev is not None and dt > 0:
        target_prev = np.array(_as_vec3(target_position_airsim_prev))
        disp_airsim = target - target_prev
        vel_airsim = disp_airsim / dt
        vel_body = _airsim_to_body_frame(vel_airsim, quaternion)
        vel_x, vel_y, vel_z = float(vel_body[0]), float(vel_body[1]), float(vel_body[2])
    else:
        vel_x, vel_y, vel_z = 0.0, 0.0, 0.0

    prev_vx, prev_vy, prev_vz, prev_yaw = _as_prev_action(prev_action, 0.0)

    pos_desc = (
        f"The target is {_describe_position_axis(pos_x, 'x')}, "
        f"{_describe_position_axis(pos_y, 'y')}, and "
        f"{_describe_position_axis(pos_z, 'z')} relative to the UAV."
    )

    vel_desc = (
        f"The target is {_describe_velocity_axis(vel_x, 'x')}, "
        f"{_describe_velocity_axis(vel_y, 'y')}, and "
        f"{_describe_velocity_axis(vel_z, 'z')}."
    )

    ego_desc = (
        f"The ego UAV was previously {_describe_velocity_axis(prev_vx, 'x')}, "
        f"{_describe_velocity_axis(prev_vy, 'y')}, and "
        f"{_describe_velocity_axis(prev_vz, 'z')}, with {_describe_yaw_rate(prev_yaw)}."
    )

    parts = [
        "Obs: FPV RGB.\n",
        "Frame: body (X forward, Y right, Z down).\n",
        pos_desc + "\n",
    ]

    if not is_first_frame:
        if include_target_vel:
            parts.append(vel_desc + "\n")
        if include_prev_action:
            parts.append(ego_desc + "\n")

    parts.append("Predict features useful for the next control action.")
    return "".join(parts)
