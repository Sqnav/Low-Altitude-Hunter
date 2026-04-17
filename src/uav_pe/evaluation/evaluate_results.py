#!/usr/bin/env python

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional

_val_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_val_root))

from metrics.metric import (
    calculate_metrics,
    calculate_metrics_pooled,
    classify_trajectories_by_initial_distance,
)


def evaluate_scene(root_dir, scene_id, output_file=None):
    print(f"\n{'='*60}")
    print(f"Evaluating scene: {scene_id}")
    print(f"{'='*60}")
    
    results = calculate_metrics(root_dir, scene_id)
    
    if results is None:
        print(f"Warning: failed to evaluate scene {scene_id} or no data found")
        return None
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metrics_out = dict(results)
        metrics_out.pop("success_dirs", None)
        output_data = {
            "scene_id": scene_id,
            "root_dir": root_dir,
            "metrics": metrics_out,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to: {output_path}")
    
    return results


def evaluate_all_scenes(root_dir, output_file=None):
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: results root directory does not exist: {root_dir}")
        return None
    
    scene_dirs = [d for d in root_path.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    
    if len(scene_dirs) == 0:
        print(f"Warning: no scene directories found under {root_dir}")
        return None
    
    print(f"\nFound {len(scene_dirs)} scene directories")
    
    all_results = {}
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        results = evaluate_scene(root_dir, scene_id)
        if results is not None:
            all_results[scene_id] = results
    
    if len(all_results) > 0:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        
        total_sr = sum(r['SR'] for r in all_results.values()) / len(all_results)
        total_osr = sum(r['OSR'] for r in all_results.values()) / len(all_results)
        total_ne = sum(r['NE'] for r in all_results.values()) / len(all_results)
        total_spl = sum(r['SPL'] for r in all_results.values()) / len(all_results)
        
        print(f"Mean SR:  {total_sr:.2f}%")
        print(f"Mean OSR: {total_osr:.2f}%")
        print(f"Mean NE:  {total_ne:.2f}")
        print(f"Mean SPL: {total_spl:.2f}%")
        
        all_results['_summary'] = {
            'SR': total_sr,
            'OSR': total_osr,
            'NE': total_ne,
            'SPL': total_spl,
            'num_scenes': len(all_results)
        }
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scenes_for_dump = {}
        for sid, r in all_results.items():
            if sid == "_summary" or not isinstance(r, dict):
                scenes_for_dump[sid] = r
                continue
            rr = dict(r)
            rr.pop("success_dirs", None)
            scenes_for_dump[sid] = rr

        output_data = {"root_dir": root_dir, "scenes": scenes_for_dump}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved all results to: {output_path}")
    
    return all_results


def evaluate_three_modes(
    root_dir: str,
    distance_threshold_m: float = 250.0,
    output_file: Optional[str] = None,
) -> dict:
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: results root directory does not exist: {root_dir}")
        return {}

    groups = classify_trajectories_by_initial_distance(
        str(root_path), threshold_m=distance_threshold_m
    )
    mode_labels = {
        "full": "Full",
        "simple": f"Simple (<{distance_threshold_m}m)",
        "complex": f"Complex (≥{distance_threshold_m}m or distance unknown)",
    }

    print(f"\n{'=' * 60}")
    print("Three-mode evaluation (grouped by frame-0 distance in uav_trajectory.json)")
    print(f"Results root: {root_dir}")
    print(f"Simple/complex threshold: {distance_threshold_m} m")
    print(f"  full:    {len(groups['full'])} trajectories")
    print(f"  simple:  {len(groups['simple'])} trajectories")
    print(f"  complex: {len(groups['complex'])} trajectories")
    print(f"{'=' * 60}\n")

    out: dict = {"root_dir": root_dir, "distance_threshold_m": distance_threshold_m, "modes": {}}

    for mode_key in ("full", "simple", "complex"):
        items = groups[mode_key]
        print(f"\n>>> {mode_labels[mode_key]} (N={len(items)})")
        if not items:
            out["modes"][mode_key] = {
                "label": mode_labels[mode_key],
                "SR": 0.0,
                "OSR": 0.0,
                "NE": 0.0,
                "SPL": 0.0,
                "num_trajectories": 0,
            }
            print("  (empty, skipped)")
            continue
        r = calculate_metrics_pooled(str(root_path), items)
        if r is None:
            out["modes"][mode_key] = {"label": mode_labels[mode_key], "error": "metrics_failed"}
            continue
        r_out = {k: v for k, v in r.items() if k != "success_dirs"}
        r_out["label"] = mode_labels[mode_key]
        out["modes"][mode_key] = r_out
        print(
            f"  SR:  {r['SR']:.2f}%  OSR: {r['OSR']:.2f}%  NE: {r['NE']:.2f}  SPL: {r['SPL']:.2f}%"
        )

    if output_file:
        outp = Path(output_file)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved three-mode results to: {outp}")

    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory validation results")
    parser.add_argument('--root_dir', type=str, default='Val/trajectory_results',
                       help="Results root directory (default: Val/trajectory_results)")
    parser.add_argument('--scene_id', type=str, default=None,
                       help="Scene id (e.g., City_1). If omitted, evaluate all scenes.")
    parser.add_argument('--output_file', type=str, default=None,
                       help="Output JSON path (optional)")
    parser.add_argument(
        "--three_modes",
        action="store_true",
        help="Evaluate full/simple/complex groups based on frame-0 distance (pooled across scenes)",
    )
    parser.add_argument(
        "--distance_threshold_m",
        type=float,
        default=250.0,
        help="Simple mode threshold (meters): distance < threshold => simple (default: 250)",
    )
    parser.add_argument(
        "--three_modes_output",
        type=str,
        default=None,
        help="Three-mode output JSON path (default: <root_dir>/evaluation_three_modes.json)",
    )

    args = parser.parse_args()

    if args.three_modes:
        tpath = args.three_modes_output
        if not tpath:
            tpath = str(Path(args.root_dir) / "evaluation_three_modes.json")
        evaluate_three_modes(
            args.root_dir,
            distance_threshold_m=args.distance_threshold_m,
            output_file=tpath,
        )
        return 0

    if len(sys.argv) == 1:
        print("="*60)
        print("Evaluating all scenes with default settings")
        print("="*60)
        print(f"Results root: {args.root_dir}")
        print("="*60)
        evaluate_all_scenes(args.root_dir, args.output_file)
    elif args.scene_id:
        evaluate_scene(args.root_dir, args.scene_id, args.output_file)
    else:
        evaluate_all_scenes(args.root_dir, args.output_file)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

