#!/usr/bin/env python

import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple


def parse_trajectory_range(traj_range_str: str, scene_list: List[str]) -> Dict[str, Tuple[int, int]]:
    if not traj_range_str:
        return {}
    
    result = {}
    if ':' in traj_range_str:
        for part in traj_range_str.split(','):
            part = part.strip()
            if ':' in part:
                scene, range_part = part.split(':', 1)
                scene = scene.strip()
                if '-' in range_part:
                    start, end = map(int, range_part.split('-'))
                    result[scene] = (start, end)
    else:
        if '-' in traj_range_str:
            start, end = map(int, traj_range_str.split('-'))
            for scene in scene_list:
                result[scene] = (start, end)
        else:
            try:
                num = int(traj_range_str)
                for scene in scene_list:
                    result[scene] = (num, num)
            except ValueError:
                pass
    return result


def count_trajectories(dataset_dir: Path, scene_list: List[str], trajectory_range: str) -> Dict:
    traj_ranges = parse_trajectory_range(trajectory_range, scene_list)
    
    total_trajectories = 0
    trajectory_details = {}
    
    for scene_dir in sorted(dataset_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        scene_name = scene_dir.name
        if scene_list and scene_name not in scene_list:
            continue
        
        scene_traj_count = 0
        valid_traj_list = []
        
        for traj_dir in sorted(scene_dir.iterdir()):
            if not traj_dir.is_dir() or not traj_dir.name.startswith('trajectory_'):
                continue
            
            try:
                traj_num = int(traj_dir.name.replace('trajectory_', ''))
            except:
                continue
            
            if scene_name in traj_ranges:
                start, end = traj_ranges[scene_name]
                if traj_num < start or traj_num > end:
                    continue
            
            instruction_path = traj_dir / "instruction.json"
            traj_file = traj_dir / "uav_trajectory.json"
            if not traj_file.exists():
                traj_file = traj_dir / "trajectory.json"
            
            if instruction_path.exists() and traj_file.exists():
                scene_traj_count += 1
                valid_traj_list.append(traj_num)
        
        if scene_traj_count > 0:
            trajectory_details[scene_name] = {
                'count': scene_traj_count,
                'trajectories': sorted(valid_traj_list)
            }
            total_trajectories += scene_traj_count
    
    return {
        'total': total_trajectories,
        'details': trajectory_details
    }


def main():
    if len(sys.argv) < 4:
        print("Usage: python count_trajectories.py <dataset_dir> <scene_list> <trajectory_range>")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    scene_list_str = sys.argv[2]
    trajectory_range_str = sys.argv[3] if len(sys.argv) > 3 else ""
    
    scene_list = [s.strip() for s in scene_list_str.split(',')] if scene_list_str else []
    
    result = count_trajectories(dataset_dir, scene_list, trajectory_range_str)
    
    print("============================================================")
    print("Counting training trajectories...")
    print(f"Scenes: {', '.join(scene_list) if scene_list else 'ALL'}")
    if trajectory_range_str:
        print(f"Trajectory range: {trajectory_range_str}")
    print(f"Total trajectories: {result['total']}")
    print("\nPer-scene breakdown:")
    for scene, info in sorted(result['details'].items()):
        print(f"  {scene}: {info['count']} trajectories")
        if len(info['trajectories']) <= 20:
            print(f"    IDs: {info['trajectories']}")
        else:
            print(f"    IDs: {info['trajectories'][:10]} ... {info['trajectories'][-10:]}")
    print("============================================================")


if __name__ == "__main__":
    main()
