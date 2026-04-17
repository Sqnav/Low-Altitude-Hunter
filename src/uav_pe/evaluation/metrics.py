#!/usr/bin/env python

import os
import json
import numpy as np
import re
import tqdm
import argparse
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

OSR_DIST_THRESH_M = 10.0


def sort_key(filename):
    return int(re.search(r'\d+', filename).group())


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def list_traj_dirs_in_scene(analysis_path: str) -> List[str]:
    if not os.path.isdir(analysis_path):
        return []
    return [
        traj_dir
        for traj_dir in os.listdir(analysis_path)
        if os.path.isdir(os.path.join(analysis_path, traj_dir))
        and "record" not in traj_dir
        and "dino" not in traj_dir
    ]


def _extract_xyz_from_dict(d: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(d, dict):
        return None
    if not all(k in d for k in ("x", "y", "z")):
        return None
    try:
        return np.array([float(d["x"]), float(d["y"]), float(d["z"])], dtype=float)
    except Exception:
        return None


def _extract_xyz_from_list(v: Any) -> Optional[np.ndarray]:
    try:
        arr = np.array(v, dtype=float).reshape(-1)
        if arr.shape[0] < 3:
            return None
        return arr[:3]
    except Exception:
        return None


def _get_target_pos_from_frame_entry(frame_entry: Dict[str, Any]) -> Optional[np.ndarray]:
    candidate_keys = [
        "target_position_world",
        "target_position",
        "gt_target_position",
        "target_world_position",
    ]
    for k in candidate_keys:
        if k in frame_entry:
            val = frame_entry[k]
            if isinstance(val, dict):
                pt = _extract_xyz_from_dict(val)
            else:
                pt = _extract_xyz_from_list(val)
            if pt is not None:
                return pt
    return None


def _get_uav_pos_from_frame_entry(frame_entry: Dict[str, Any]) -> Optional[np.ndarray]:
    candidate_keys = [
        "uav_position_world",
        "uav_position",
        "position",
    ]
    for k in candidate_keys:
        if k in frame_entry:
            val = frame_entry[k]
            if isinstance(val, dict):
                pt = _extract_xyz_from_dict(val)
            else:
                pt = _extract_xyz_from_list(val)
            if pt is not None:
                return pt
    return None


def _get_target_pos_from_uav_traj_frame(frame_entry: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(frame_entry, dict):
        return None
    if "target_position" in frame_entry:
        return _extract_xyz_from_dict(frame_entry["target_position"])
    return None


def _get_uav_pos_from_uav_traj_frame(frame_entry: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(frame_entry, dict):
        return None
    if "uav_position" in frame_entry:
        return _extract_xyz_from_dict(frame_entry["uav_position"])
    return None


def _load_uav_trajectory_pairs(traj_full_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    uav_json = os.path.join(traj_full_path, "uav_trajectory.json")
    if not os.path.isfile(uav_json):
        return [], []

    try:
        data = load_json(uav_json)
        traj = data.get("trajectory") or []
        uav_list: List[np.ndarray] = []
        tgt_list: List[np.ndarray] = []
        for fr in traj:
            pu = _get_uav_pos_from_uav_traj_frame(fr)
            pt = _get_target_pos_from_uav_traj_frame(fr)
            if pu is not None and pt is not None:
                uav_list.append(pu)
                tgt_list.append(pt)
        return uav_list, tgt_list
    except Exception:
        return [], []


def read_initial_distance_m(traj_full_path: str) -> Optional[float]:
    uav_json = os.path.join(traj_full_path, "uav_trajectory.json")
    if not os.path.isfile(uav_json):
        return None
    try:
        data = load_json(uav_json)
        traj = data.get("trajectory") or []
        if not traj:
            return None
        fr0 = traj[0]
        if "distance" in fr0 and fr0["distance"] is not None:
            return float(fr0["distance"])
        u = fr0.get("uav_position") or {}
        t = fr0.get("target_position") or {}
        if u and t:
            pu = np.array(
                [float(u.get("x", 0)), float(u.get("y", 0)), float(u.get("z", 0))],
                dtype=float,
            )
            pt = np.array(
                [float(t.get("x", 0)), float(t.get("y", 0)), float(t.get("z", 0))],
                dtype=float,
            )
            return float(np.linalg.norm(pt - pu))
    except Exception:
        pass
    return None


def classify_trajectories_by_initial_distance(
    results_root: str, threshold_m: float = 250.0
) -> Dict[str, List[Tuple[str, str]]]:
    full: List[Tuple[str, str]] = []
    simple: List[Tuple[str, str]] = []
    complex_list: List[Tuple[str, str]] = []
    root_path = os.path.abspath(results_root)
    if not os.path.isdir(root_path):
        logging.warning(f"Results root does not exist: {root_path}")
        return {"full": [], "simple": [], "complex": []}

    for scene_id in sorted(os.listdir(root_path)):
        scene_path = os.path.join(root_path, scene_id)
        if not os.path.isdir(scene_path) or scene_id.startswith("."):
            continue
        for traj_dir in list_traj_dirs_in_scene(scene_path):
            full.append((scene_id, traj_dir))
            d = read_initial_distance_m(os.path.join(scene_path, traj_dir))
            if d is not None and d < threshold_m:
                simple.append((scene_id, traj_dir))
            else:
                complex_list.append((scene_id, traj_dir))

    return {"full": full, "simple": simple, "complex": complex_list}


def compute_ne_list_for_dirs(path: str, dirs: List[str]) -> List[float]:
    ne_list: List[float] = []

    for traj_dir in tqdm.tqdm(dirs, desc="Computing NE (per-traj)", leave=False):
        traj_full_path = os.path.join(path, traj_dir)

        uav_seq, tgt_seq = _load_uav_trajectory_pairs(traj_full_path)
        if len(uav_seq) > 0 and len(tgt_seq) > 0:
            n = min(len(uav_seq), len(tgt_seq))
            ne = float(np.linalg.norm(uav_seq[n - 1] - tgt_seq[n - 1]))
            ne_list.append(ne)
            continue

        frames_json_path = os.path.join(traj_full_path, 'frames.json')
        if os.path.exists(frames_json_path):
            try:
                frames_data = load_json(frames_json_path)
                frames = frames_data.get("frames") or []
                if len(frames) == 0:
                    logging.warning(f"Trajectory {traj_dir}: frames.json has no frames")
                    continue

                last_uav = None
                last_tgt = None
                for fr in reversed(frames):
                    if last_uav is None:
                        last_uav = _get_uav_pos_from_frame_entry(fr)
                    if last_tgt is None:
                        last_tgt = _get_target_pos_from_frame_entry(fr)
                    if last_uav is not None and last_tgt is not None:
                        break

                if last_uav is None or last_tgt is None:
                    logging.warning(f"Trajectory {traj_dir}: cannot read final UAV/target positions from frames.json")
                    continue

                ne = float(np.linalg.norm(last_uav - last_tgt))
                ne_list.append(ne)
                continue
            except Exception as e:
                logging.warning(f"Failed to read {traj_dir}/frames.json: {e}")
                continue

        logging.warning(f"Trajectory {traj_dir}: missing target info for NE")

    return ne_list


def calculate_ne(path, dirs):
    ne_list = compute_ne_list_for_dirs(path, dirs)
    if len(ne_list) == 0:
        logging.warning("No valid trajectories for NE")
        return 0.0

    avg_ne = float(np.mean(np.array(ne_list)))
    logging.info(f"Average Navigation Error (NE): {avg_ne:.2f}")
    return avg_ne


def compute_spl_ratios_for_dirs(path: str, dirs: List[str], success_dirs: Set[str]) -> List[float]:
    spl_list: List[float] = []

    for traj_dir in tqdm.tqdm(dirs, desc="Computing SPL (per-traj)", leave=False):
        if traj_dir not in success_dirs:
            spl_list.append(0)
            continue

        frames_json_path = os.path.join(path, traj_dir, 'frames.json')

        if os.path.exists(frames_json_path):
            frames_data = load_json(frames_json_path)

            if 'frames' in frames_data and len(frames_data['frames']) > 0:
                pred_length = 0.0
                pre_point = None
                for frame in frames_data['frames']:
                    if 'uav_position_world' in frame:
                        point = np.array(frame['uav_position_world'], dtype=float)
                        if pre_point is not None:
                            pred_length += float(np.linalg.norm(pre_point - point))
                        pre_point = point
                if pre_point is None:
                    logging.warning(f"Trajectory {traj_dir}: frames.json has no valid position data")
                    spl_list.append(0)
                    continue
            else:
                logging.warning(f"Trajectory {traj_dir}: frames.json has no frames")
                spl_list.append(0)
                continue

            if 'original_trajectory' in frames_data and len(frames_data['original_trajectory']) > 0:
                path_length = 0.0
                ori_data = frames_data['original_trajectory']
                for i in range(len(ori_data) - 1):
                    p1 = np.array(ori_data[i]['position'], dtype=float)
                    p2 = np.array(ori_data[i + 1]['position'], dtype=float)
                    path_length += float(np.linalg.norm(p2 - p1))
                path_length -= 20.0
            else:
                logging.warning(f"Trajectory {traj_dir}: frames.json missing original_trajectory")
                spl_list.append(0)
                continue
        else:
            log_dir = os.path.join(path, traj_dir, 'log')
            if not os.path.exists(log_dir):
                logging.warning(f"Trajectory {traj_dir}: missing both frames.json and log/ directory")
                spl_list.append(0)
                continue
            logs = sorted(os.listdir(log_dir), key=sort_key)
            if len(logs) == 0:
                logging.warning(f"Trajectory {traj_dir}: log/ directory is empty")
                spl_list.append(0)
                continue

            pred_length = 0.0
            pre_point = None
            for log in logs:
                log_data = load_json(os.path.join(log_dir, log))
                point = np.array(log_data["sensors"]['state']['position'], dtype=float)
                if pre_point is not None:
                    pred_length += float(np.linalg.norm(pre_point - point))
                pre_point = point

            ori_info_path = os.path.join(path, traj_dir, 'ori_info.json')
            if not os.path.exists(ori_info_path):
                logging.warning(f"Trajectory {traj_dir}: missing ori_info.json")
                spl_list.append(0)
                continue
            ori_info = load_json(ori_info_path)
            ori_data_path = os.path.join(ori_info['ori_traj_dir'], 'merged_data.json')
            if not os.path.exists(ori_data_path):
                logging.warning(f"Trajectory {traj_dir}: original trajectory file not found: {ori_data_path}")
                spl_list.append(0)
                continue
            ori_data = load_json(ori_data_path)['trajectory_raw_detailed']

            path_length = 0.0
            for i in range(len(ori_data) - 1):
                p1 = np.array(ori_data[i]['position'], dtype=float)
                p2 = np.array(ori_data[i + 1]['position'], dtype=float)
                path_length += float(np.linalg.norm(p2 - p1))
            path_length -= 20.0

        path_length = max(path_length, 0.0)
        spl = path_length / max(path_length, pred_length, 1e-8)
        spl = max(spl, 0.0)
        spl_list.append(spl)

    return spl_list


def calculate_spl(path, dirs, success_dirs):
    success_set = set(success_dirs)
    spl_list = compute_spl_ratios_for_dirs(path, dirs, success_set)

    if len(spl_list) == 0:
        logging.warning("No valid trajectories for SPL")
        return 0.0

    avg_spl = float(np.mean(np.array(spl_list))) * 100
    logging.info(f"Average Success Path Length (SPL): {avg_spl:.2f}%")
    return avg_spl


def calculate_metrics_pooled(
    root_dir: str, trajectory_items: List[Tuple[str, str]]
) -> Optional[Dict[str, Any]]:
    if not trajectory_items:
        logging.warning("calculate_metrics_pooled: empty trajectory list")
        return {
            "SR": 0.0,
            "OSR": 0.0,
            "NE": 0.0,
            "SPL": 0.0,
            "num_trajectories": 0,
        }

    by_scene: Dict[str, List[str]] = defaultdict(list)
    for scene_id, traj_dir in trajectory_items:
        by_scene[scene_id].append(traj_dir)

    weighted_sr = 0.0
    weighted_osr = 0.0
    total_n = 0
    all_ne: List[float] = []
    all_spl_ratios: List[float] = []

    for scene_id, traj_list in sorted(by_scene.items()):
        st = set(traj_list)
        r = calculate_metrics(root_dir, scene_id, traj_subset=st)
        if r is None:
            continue
        n = int(r.get("num_trajectories", 0))
        if n <= 0:
            continue
        total_n += n
        weighted_sr += float(r["SR"]) * n
        weighted_osr += float(r["OSR"]) * n

        analysis_path = os.path.join(root_dir, scene_id)
        all_dirs = list_traj_dirs_in_scene(analysis_path)
        dirs = [d for d in all_dirs if d in st]
        if not dirs:
            continue

        all_ne.extend(compute_ne_list_for_dirs(analysis_path, dirs))
        succ_set = set(r.get("success_dirs") or [])
        all_spl_ratios.extend(
            compute_spl_ratios_for_dirs(analysis_path, dirs, succ_set)
        )

    if total_n == 0:
        return None

    sr = weighted_sr / total_n
    osr = weighted_osr / total_n
    ne = float(np.mean(all_ne)) if len(all_ne) > 0 else 0.0
    spl = float(np.mean(all_spl_ratios)) * 100 if len(all_spl_ratios) > 0 else 0.0

    logging.info(
        f"[Pooled N={total_n}] SR: {sr:.2f}% OSR: {osr:.2f}% NE: {ne:.2f} SPL: {spl:.2f}%"
    )
    return {
        "SR": sr,
        "OSR": osr,
        "NE": ne,
        "SPL": spl,
        "num_trajectories": total_n,
    }


def calculate_metrics(
    root_dir: str,
    analysis_item: str,
    traj_subset: Optional[Set[str]] = None,
):
    analysis_path = os.path.join(root_dir, analysis_item)
    if not os.path.exists(analysis_path):
        logging.warning(f"Analysis path does not exist: {analysis_path}")
        return None

    logging.info(f"\nStarting analysis for: {analysis_item}")

    dirs = list_traj_dirs_in_scene(analysis_path)
    if traj_subset is not None:
        dirs = [d for d in dirs if d in traj_subset]

    total = len(dirs)
    success = 0
    oracle = 0
    success_dirs = []

    for traj_dir in dirs:
        traj_full_path = os.path.join(analysis_path, traj_dir)
        frames_json_path = os.path.join(traj_full_path, 'frames.json')

        is_success = False
        is_oracle = False

        if os.path.exists(frames_json_path):
            try:
                frames_data = load_json(frames_json_path)
                if 'captured' in frames_data:
                    is_success = bool(frames_data['captured'])

                uav_seq, tgt_seq = _load_uav_trajectory_pairs(traj_full_path)
                if len(uav_seq) > 0 and len(tgt_seq) > 0:
                    n = min(len(uav_seq), len(tgt_seq))
                    for i in range(n):
                        dist = float(np.linalg.norm(uav_seq[i] - tgt_seq[i]))
                        if dist < OSR_DIST_THRESH_M:
                            is_oracle = True
                            break
                else:
                    frames = frames_data.get('frames', [])
                    for fr in frames:
                        p_uav = _get_uav_pos_from_frame_entry(fr)
                        p_tgt = _get_target_pos_from_frame_entry(fr)
                        if p_uav is None or p_tgt is None:
                            continue
                        dist = float(np.linalg.norm(p_uav - p_tgt))
                        if dist < OSR_DIST_THRESH_M:
                            is_oracle = True
                            break
            except Exception:
                pass

        if not is_success and 'success' in traj_dir:
            is_success = True
            is_oracle = True
        elif not is_oracle and 'oracle' in traj_dir:
            is_oracle = True

        if is_success:
            success += 1
            success_dirs.append(traj_dir)
        if is_oracle:
            oracle += 1

    sr = success / (total + 1e-8) * 100
    logging.info(f"Success Rate (SR): {sr:.2f}%")

    osr = oracle / (total + 1e-8) * 100
    logging.info(f"Oracle Success Rate (OSR): {osr:.2f}%")

    ne = calculate_ne(analysis_path, dirs)

    spl = calculate_spl(analysis_path, dirs, success_dirs)

    results = {
        "SR": sr,
        "OSR": osr,
        "NE": ne,
        "SPL": spl,
        "num_trajectories": total,
        "success_dirs": success_dirs,
    }

    logging.info(f"\n=== Final Results for {analysis_item} ===")
    logging.info(f"SR:  {sr:.2f}%")
    logging.info(f"OSR: {osr:.2f}%")
    logging.info(f"NE:  {ne:.2f}")
    logging.info(f"SPL: {spl:.2f}%")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics: SR, OSR, NE, SPL")
    parser.add_argument('--root_dir', type=str, required=True,
                        help="The root directory of the dataset.")
    parser.add_argument('--analysis_item', type=str, required=True,
                        help="The analysis item name (experiment directory name).")

    args = parser.parse_args()

    calculate_metrics(args.root_dir, args.analysis_item)