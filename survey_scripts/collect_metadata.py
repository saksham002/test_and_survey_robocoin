#!/usr/bin/env python3
"""
Collect metadata from all RoboCOIN datasets.
- Maps each repo to observation.state names, action names, camera names
- Creates embodiment (robot_type) to min/max/mean/std for state and action spaces

Outputs:
- survey_scripts/stats/info.json: per-repo metadata
- survey_scripts/stats/norm_stats.json: embodiment-wise normalization statistics
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
import pdb

PREFIX = "RoboCOIN/"
OUTPUT_DIR = Path(__file__).parent / "stats"
PLOTS_DIR = Path(__file__).parent / "plots"
EXAMPLES_DIR = Path(__file__).parent / "examples"

# Keys to track for normalization statistics
STAT_KEYS = ["observation.state", "action", "eef_sim_pose_state", "eef_sim_pose_action"]


@dataclass
class RunningStats:
    """Holds running statistics for a single feature using numpy arrays."""
    count: int = 0
    min_vals: Optional[np.ndarray] = None
    max_vals: Optional[np.ndarray] = None
    mean_vals: Optional[np.ndarray] = None
    m2_vals: Optional[np.ndarray] = None  # For Welford's algorithm
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict (converts numpy arrays to lists)."""
        result = {"count": self.count}
        if self.min_vals is not None:
            result["min"] = self.min_vals.tolist()
        if self.max_vals is not None:
            result["max"] = self.max_vals.tolist()
        if self.mean_vals is not None:
            result["mean"] = self.mean_vals.tolist()
        if self.m2_vals is not None and self.count > 0:
            result["std"] = np.sqrt(self.m2_vals / self.count).tolist()
        return result


def update_running_stats(running: RunningStats, episode_stats: Dict) -> None:
    """Update running statistics with a single episode's statistics using numpy."""
    
    ep_count = episode_stats["count"][0] if isinstance(episode_stats["count"], list) else episode_stats["count"]
    
    ep_min = np.array(episode_stats["min"], dtype=np.float64)
    ep_max = np.array(episode_stats["max"], dtype=np.float64)
    ep_mean = np.array(episode_stats["mean"], dtype=np.float64)
    ep_std = np.array(episode_stats["std"], dtype=np.float64)

    if np.isnan(ep_min).any() or np.isnan(ep_max).any() or np.isnan(ep_mean).any() or np.isnan(ep_std).any():
        return
    
    # Initialize if first update
    if running.count == 0:
        running.min_vals = ep_min.copy()
        running.max_vals = ep_max.copy()
        running.mean_vals = ep_mean.copy()
        running.m2_vals = ep_count * (ep_std ** 2)
        running.count = ep_count
        return
    
    # Update min/max element-wise
    np.minimum(running.min_vals, ep_min, out = running.min_vals)
    np.maximum(running.max_vals, ep_max, out = running.max_vals)
    
    # Parallel algorithm for combining mean and variance
    n_a = running.count
    n_b = ep_count
    n_ab = n_a + n_b
    
    delta = ep_mean - running.mean_vals
    new_mean = (n_a * running.mean_vals + n_b * ep_mean) / n_ab
    
    m2_b = n_b * (ep_std ** 2)
    new_m2 = running.m2_vals + m2_b + (delta ** 2) * n_a * n_b / n_ab
    
    running.mean_vals = new_mean
    running.m2_vals = new_m2
    running.count = n_ab


def extract_robot_type_from_repo_id(repo_id: str) -> str:
    """Extract robot type from repo_id by splitting on _ or - and taking first 2 elements.
    
    Args:
        repo_id: Repository ID, e.g., "RoboCOIN/robot-arm-dataset" or "RoboCOIN/robot_arm_dataset"
    
    Returns:
        Robot type as a string with first 2 parts joined by space, e.g., "robot arm"
    """
    # Remove the prefix (everything before and including the first /)
    if '/' in repo_id:
        repo_name = repo_id.split('/', 1)[1]
    else:
        repo_name = repo_id
    
    # Split on both _ and -
    parts = re.split(r'[_-]', repo_name)
    
    # Take first 2 elements and join with space
    if len(parts) >= 2:
        robot_type = ' '.join(parts[:2])
    else:
        # If there's only one part, use it as is
        robot_type = parts[0] if parts else repo_name
    
    return robot_type


def get_null_subtask_indices(repo_id: str) -> Optional[int]:
    """Download subtask_annotations.jsonl and find indices where subtask == 'null'.
    
    Returns:
        Set of null subtask indices, or None if file doesn't exist.
    """
    try:
        annotations_path = hf_hub_download(
            repo_id=repo_id,
            filename="annotations/subtask_annotations.jsonl",
            repo_type="dataset"
        )
        null_indices = set()
        with open(annotations_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data["subtask"] == "null":
                    null_indices.add(data["subtask_index"])
        assert len(null_indices) == 1, "Multiple null subtask indices found"
        return next(iter(null_indices))
    except Exception:
        return None


def extract_subtask_lengths(repo_id: str, fps: int) -> Tuple[List[int], List[float], bool]:
    """Extract subtask lengths from the first episode by downloading only episode_000000.parquet.
    
    Tracks multiple concurrent subtasks per row. When a subtask leaves the active set,
    its length (in steps) is recorded.
    
    Returns:
        Tuple of (subtask_steps, subtask_lengths_seconds, has_concurrent_transition)
        where has_concurrent_transition is True if condition on line 202 was met
    """
    # First get the null subtask index
    null_index = get_null_subtask_indices(repo_id)
    if null_index is None:
        return [], [], False
    
    try:
        # Download only the first episode's parquet file
        parquet_path = hf_hub_download(
            repo_id=repo_id,
            filename="data/chunk-000/episode_000000.parquet",
            repo_type="dataset"
        )
        
        # Read only the subtask_annotation column
        table = pq.read_table(parquet_path, columns = ["subtask_annotation"])
        subtask_annotations = table.column("subtask_annotation").to_pylist()
        
        subtask_steps = []
        current_subtasks = set()  # Set of currently active subtasks
        subtask_counts = {}  # Maps subtask -> count
        has_concurrent_transition = False  # Track if condition on line 202 is met
        
        for annotation in subtask_annotations:
            # annotation is an array of length 5 containing subtask indices
            # Filter out null index to get all valid subtasks
            valid_subtasks = set(s for s in annotation if s != null_index)
            
            # Find subtasks that ended (were in current but not in new)
            ended_subtasks = current_subtasks - valid_subtasks
            for subtask in ended_subtasks:
                subtask_steps.append(subtask_counts[subtask])
                del subtask_counts[subtask]
            
            # Find subtasks that started (are in new but not in current)
            new_subtasks = valid_subtasks - current_subtasks
            for subtask in new_subtasks:
                subtask_counts[subtask] = 1  # Start counting at 1 for this row
            
            # Increment count for continuing subtasks
            continuing_subtasks = valid_subtasks & current_subtasks
            for subtask in continuing_subtasks:
                subtask_counts[subtask] += 1

            if len(continuing_subtasks) > 0 and len(ended_subtasks) > 0:
                has_concurrent_transition = True
                
            current_subtasks = valid_subtasks
        
        # Don't forget subtasks still active at the end
        for subtask in current_subtasks:
            subtask_steps.append(subtask_counts[subtask])        
        # Convert to seconds
        subtask_lengths_sec = [s / fps for s in subtask_steps] if fps > 0 else []
        
        # Delete the downloaded parquet file and its parent directory if empty
        try:
            parquet_file = Path(parquet_path)
            if parquet_file.exists():
                parquet_file.unlink()
            # Try to remove parent directory if it's empty
            parent_dir = parquet_file.parent
            try:
                if parent_dir.exists() and not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
            except OSError:
                pass  # Directory not empty or other error, ignore
        except Exception:
            pass  # Ignore cleanup errors
        
        return subtask_steps, subtask_lengths_sec, has_concurrent_transition
        
    except Exception:
        return [], [], False


def select_camera_for_video(camera_names: List[str]) -> Optional[str]:
    """Select camera name for video based on priority rules.
    
    Priority:
    1. If exactly 1 camera, use that
    2. If num_cameras > 1:
       - Try cam_high_rgb
       - If not, try any with "head_rgb" in name
       - If not, try cam_front_rgb
       - If not, try any with "high" in name
       - Otherwise return None
    
    Returns:
        Selected camera name or None
    """
    if len(camera_names) == 1:
        return camera_names[0]
    
    if len(camera_names) == 0:
        return None
    
    # Try cam_high_rgb
    for cam in camera_names:
        if cam == "cam_high_rgb":
            return cam
    
    # Try any with "head_rgb" in it
    for cam in camera_names:
        if "head_rgb" in cam:
            return cam
    
    # Try cam_front_rgb
    for cam in camera_names:
        if cam == "cam_front_rgb":
            return cam
    
    # Try any with "high" in it
    for cam in camera_names:
        if "high" in cam:
            return cam
    
    return None


def save_example_video_and_annotations(repo_id: str, camera_name: str, examples_dir: Path) -> None:
    """Download and save video and annotation files for a repo.
    
    Args:
        repo_id: Repository ID
        camera_name: Selected camera name
        examples_dir: Directory to save files to
    """
    try:
        # Create repo-specific subdirectory (remove RoboCOIN_ prefix if present)
        repo_safe_name = repo_id.replace("/", "_")
        if repo_safe_name.startswith("RoboCOIN_"):
            repo_safe_name = repo_safe_name[len("RoboCOIN_"):]
        repo_dir = examples_dir / repo_safe_name
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        # Download video
        video_path = f"videos/chunk-000/observation.images.{camera_name}/episode_000000.mp4"
        try:
            video_download_path = hf_hub_download(
                repo_id=repo_id,
                filename=video_path,
                repo_type="dataset"
            )
            # Copy to examples directory with camera name
            video_dest = repo_dir / f"episode_000000_{camera_name}.mp4"
            shutil.copy2(video_download_path, video_dest)
            print(f"  Saved video to {video_dest}")
            
            # Delete downloaded video from cache
            try:
                Path(video_download_path).unlink()
            except Exception as del_e:
                print(f"  Warning: Could not delete video from cache: {del_e}")
        except Exception as e:
            print(f"  Warning: Could not download video: {e}")
        
        # Download subtask_annotations.jsonl
        try:
            annotations_path = hf_hub_download(
                repo_id=repo_id,
                filename="annotations/subtask_annotations.jsonl",
                repo_type="dataset"
            )
            annotations_dest = repo_dir / "subtask_annotations.jsonl"
            shutil.copy2(annotations_path, annotations_dest)
            print(f"  Saved subtask_annotations.jsonl to {annotations_dest}")
        except Exception as e:
            print(f"  Warning: Could not download subtask_annotations.jsonl: {e}")
        
        # Download scene_annotations.jsonl
        try:
            scene_annotations_path = hf_hub_download(
                repo_id=repo_id,
                filename="annotations/scene_annotations.jsonl",
                repo_type="dataset"
            )
            scene_annotations_dest = repo_dir / "scene_annotations.jsonl"
            shutil.copy2(scene_annotations_path, scene_annotations_dest)
            print(f"  Saved scene_annotations.jsonl to {scene_annotations_dest}")
        except Exception as e:
            print(f"  Warning: Could not download scene_annotations.jsonl: {e}")
        
        # Download meta/episodes.jsonl
        try:
            episodes_path = hf_hub_download(
                repo_id=repo_id,
                filename="meta/episodes.jsonl",
                repo_type="dataset"
            )
            episodes_dest = repo_dir / "episodes.jsonl"
            shutil.copy2(episodes_path, episodes_dest)
            print(f"  Saved episodes.jsonl to {episodes_dest}")
        except Exception as e:
            print(f"  Warning: Could not download episodes.jsonl: {e}")
        
        # Download meta/tasks.jsonl
        try:
            tasks_path = hf_hub_download(
                repo_id=repo_id,
                filename="meta/tasks.jsonl",
                repo_type="dataset"
            )
            tasks_dest = repo_dir / "tasks.jsonl"
            shutil.copy2(tasks_path, tasks_dest)
            print(f"  Saved tasks.jsonl to {tasks_dest}")
        except Exception as e:
            print(f"  Warning: Could not download tasks.jsonl: {e}")
            
    except Exception as e:
        print(f"  Error saving example files: {e}")


def extract_camera_info(features: Dict, global_fps: int, repo_id: str) -> Tuple[List[str], List[Tuple[int, ...]]]:
    """Extract camera names and shapes from features dictionary, verify FPS matches global FPS.
    
    Returns:
        Tuple of (camera_names, camera_shapes), both sorted by camera name.
    """
    cameras = []  # List of (name, shape) tuples
    for key, val in features.items():
        if isinstance(val, dict) and val["dtype"] == 'video':
            # Key format is "observation.images.<name>"
            parts = key.split('.')
            cam_name = ".".join(parts[2 : ])
            cam_shape = tuple(val["shape"])
            cameras.append((cam_name, cam_shape))
            
            # Assert camera FPS matches global FPS
            cam_fps = val["info"]["video.fps"]
            assert cam_fps == global_fps, \
                f"{repo_id}: Camera '{cam_name}' has fps={cam_fps}, expected global fps={global_fps}"
    
    # Sort by camera name
    cameras.sort(key = lambda x: x[0])
    camera_names = [c[0] for c in cameras]
    camera_shapes = [c[1] for c in cameras]
    return camera_names, camera_shapes


def main():
    api = HfApi()
    print(f"Listing datasets with prefix '{PREFIX}' from Hugging Face...")
    infos = api.list_datasets(search = PREFIX)
    repo_ids = sorted([d.id for d in infos if d.id.startswith(PREFIX)])
    
    print(f"Found {len(repo_ids)} datasets.")
    
    # Output dictionaries
    repo_info = {}  # repo_id -> metadata
    embodiment_stats = {}  # robot_type_key -> {stat_key -> RunningStats}
    robot_type_to_state_dim = {}  # robot_type -> state_dim (for tracking first occurrence)
    robot_type_has_variants = set()  # robot_types that have been detected to have multiple state_dims
    key_to_names = {}  # robot_type_key -> {"camera_names": [...], "state_names": [...], "action_names": [...]}
    episode_lengths_seconds = []  # List of individual episode lengths (in seconds)
    episode_steps = []  # List of individual episode lengths (in number of data points/frames)
    subtask_lengths_seconds = []  # List of individual subtask lengths (in seconds)
    subtask_steps = []  # List of individual subtask lengths (in number of data points/frames)
    examples_saved_count = 0  # Track how many example repos we've saved
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    for i, repo_id in enumerate(repo_ids, start=1):
        print(f"\n[{i}/{len(repo_ids)}] Processing {repo_id}...")
        
        try:
            # Download only meta/info.json
            info_path = hf_hub_download(
                repo_id = repo_id, 
                filename = "meta/info.json", 
                repo_type = "dataset"
            )
            
            with open(info_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
            
            # Extract metadata
            # Extract robot_type from repo_id instead of info.json
            robot_type = extract_robot_type_from_repo_id(repo_id)
            features = info_data["features"]
            
            # Get observation.state names
            state_names = features["observation.state"]["names"]
            
            # Get action names
            if "action" in features:
                action_names = features["action"]["names"]
            else:
                action_names = state_names
            
            # Get camera names and shapes (also verifies each camera's fps matches global fps)
            global_fps = info_data["fps"]
            camera_names, camera_shapes = extract_camera_info(features, global_fps, repo_id)
            
            # Get eef_sim_pose names if available
            eef_state_names = []
            if 'eef_sim_pose_state' in features:
                eef_state_names = features['eef_sim_pose_state']['names']
            
            eef_action_names = []
            if 'eef_sim_pose_action' in features:
                eef_action_names = features['eef_sim_pose_action']['names']
            
            # Calculate dimensions
            state_dim = features["observation.state"]["shape"][0]
            action_dim = features["action"]["shape"][0]
            
            total_episodes = info_data["total_episodes"]
            total_frames = info_data["total_frames"]
            
            # Store repo info
            repo_info[repo_id] = {
                "robot_type": robot_type,
                "fps": global_fps,
                "total_episodes": total_episodes,
                "total_frames": total_frames,
                "state_names": state_names,
                "action_names": action_names,
                "camera_names": camera_names,
                "camera_shapes": camera_shapes,
                "eef_state_names": eef_state_names,
                "eef_action_names": eef_action_names,
                "state_dim": state_dim,
                "action_dim": action_dim,
            }
            
            # Determine the key to use for embodiment_stats
            if robot_type in robot_type_has_variants:
                # This robot_type already has variants, always use formatted key
                robot_type_key = f"{robot_type} {state_dim}"
            elif robot_type in robot_type_to_state_dim:
                # We've seen this robot_type before, check if state_dim matches
                expected_state_dim = robot_type_to_state_dim[robot_type]
                if state_dim != expected_state_dim:
                    # Mismatch detected - need to rename existing key and create new one
                    old_key = robot_type
                    new_key = f"{robot_type} {state_dim}"
                    old_key_with_dim = f"{robot_type} {expected_state_dim}"
                    
                    # Rename the existing entry if it's still using the base robot_type key
                    embodiment_stats[old_key_with_dim] = embodiment_stats.pop(old_key)
                    
                    # Also copy the names to the new key
                    if old_key in key_to_names:
                        key_to_names[old_key_with_dim] = key_to_names.pop(old_key)
                    
                    print(f"  Renamed '{old_key}' -> '{old_key_with_dim}' (state_dim={expected_state_dim})")
                    
                    # Mark this robot_type as having variants
                    robot_type_has_variants.add(robot_type)
                    robot_type_key = new_key
                    print(f"  Using key '{robot_type_key}' for {robot_type} with state_dim={state_dim}")
                else:
                    # Same state_dim, use the base key (should still exist)
                    robot_type_key = robot_type
            else:
                # First time seeing this robot_type
                robot_type_to_state_dim[robot_type] = state_dim
                robot_type_key = robot_type
            
            # Maintain lists of name sets for this key
            if robot_type_key not in key_to_names:
                # First time seeing this key, initialize with lists
                key_to_names[robot_type_key] = {
                    "camera_names": [],
                    "state_names": [],
                    "action_names": []
                }
            
            # Check if this name set is already in the list, if not add it
            name_lists = key_to_names[robot_type_key]
            
            if camera_names not in name_lists["camera_names"]:
                name_lists["camera_names"].append(camera_names)
            
            if state_names not in name_lists["state_names"]:
                name_lists["state_names"].append(state_names)
            
            if action_names not in name_lists["action_names"]:
                name_lists["action_names"].append(action_names)
            
            # Download episodes_stats.jsonl for normalization stats
            stats_path = None
            first_episode_length_sec = None
            try:
                stats_path = hf_hub_download(
                    repo_id = repo_id,
                    filename = "meta/episodes_stats.jsonl",
                    repo_type = "dataset"
                )
                
                # Initialize embodiment stats if needed
                if robot_type_key not in embodiment_stats:
                    embodiment_stats[robot_type_key] = {key: RunningStats() for key in STAT_KEYS}
                
                # Read and aggregate episode stats
                with open(stats_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ep_data = json.loads(line)
                            stats_dict = ep_data["stats"]
                            
                            # Extract episode length from frame count
                            if "observation.state" in stats_dict:
                                ep_count = stats_dict["observation.state"]["count"]
                                if isinstance(ep_count, list):
                                    ep_count = ep_count[0]
                                episode_steps.append(ep_count)
                                if global_fps > 0:
                                    episode_length_sec = ep_count / global_fps
                                    episode_lengths_seconds.append(episode_length_sec)
                                    # Get first episode length for example saving check
                                    if line_num == 0:
                                        first_episode_length_sec = episode_length_sec
                            
                            for key in STAT_KEYS:
                                if key in stats_dict:
                                    if embodiment_stats[robot_type_key][key].mean_vals is not None and embodiment_stats[robot_type_key][key].mean_vals.shape[0] != len(stats_dict[key]["mean"]):
                                        pdb.set_trace()
                                    update_running_stats(
                                        embodiment_stats[robot_type_key][key],
                                        stats_dict[key]
                                    )
                        except json.JSONDecodeError:
                            continue
                
                print(f"  Collected metadata and stats for {robot_type_key}")
                
            except Exception as e:
                print(f"  Warning: Could not download episodes_stats.jsonl: {e}")
                print(f"  Collected metadata only")
            
            # Extract subtask lengths from first episode
            has_concurrent_transition = False
            try:
                repo_subtask_steps, repo_subtask_lengths, has_concurrent_transition = extract_subtask_lengths(repo_id, global_fps)
                if repo_subtask_steps:
                    subtask_steps.extend(repo_subtask_steps)
                    subtask_lengths_seconds.extend(repo_subtask_lengths)
                    print(f"  Collected {len(repo_subtask_steps)} subtask lengths from first episode")
            except Exception as e:
                print(f"  Warning: Could not extract subtask lengths: {e}")
            
            # Check if repo meets conditions for saving example
            should_save_example = False
            if has_concurrent_transition:
                should_save_example = True
                print(f"  Repo meets condition: concurrent subtask transition detected")
            elif first_episode_length_sec is not None and first_episode_length_sec > 60:
                should_save_example = True
                print(f"  Repo meets condition: first episode length ({first_episode_length_sec:.1f}s) > 1 minute")
            
            # Save example for first 5 repos that meet conditions
            if should_save_example and examples_saved_count < 5:
                selected_camera = select_camera_for_video(camera_names)
                if selected_camera:
                    print(f"  Saving example video and annotations (camera: {selected_camera})...")
                    save_example_video_and_annotations(repo_id, selected_camera, EXAMPLES_DIR)
                    examples_saved_count += 1
                else:
                    print(f"  Warning: Could not select camera for video, skipping example save")
            
        except Exception as e:
            print(f"  Error: {e}")
            repo_info[repo_id] = {"error": str(e)}
    
    # Convert embodiment stats to serializable format
    norm_stats = {}
    for robot_type_key, stats_by_key in embodiment_stats.items():
        norm_stats[robot_type_key] = {}
        
        # Add names if available (already in list format)
        if robot_type_key in key_to_names:
            norm_stats[robot_type_key]["camera_names"] = key_to_names[robot_type_key]["camera_names"]
            norm_stats[robot_type_key]["state_names"] = key_to_names[robot_type_key]["state_names"]
            norm_stats[robot_type_key]["action_names"] = key_to_names[robot_type_key]["action_names"]
        
        # Add statistics
        for key, running in stats_by_key.items():
            norm_stats[robot_type_key][key] = running.to_dict()
    
    # Calculate quantiles of episode lengths (seconds)
    episode_length_quantiles = {}
    if episode_lengths_seconds:
        lengths_array = np.array(episode_lengths_seconds)
        quantile_points = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        quantile_values = np.quantile(lengths_array, quantile_points).tolist()
        episode_length_quantiles = {
            "quantiles": {f"{int(q*100)}%": v for q, v in zip(quantile_points, quantile_values)},
            "mean": float(np.mean(lengths_array)),
            "std": float(np.std(lengths_array)),
            "count": len(lengths_array)
        }
    
    # Calculate quantiles of episode steps (number of data points)
    episode_steps_quantiles = {}
    if episode_steps:
        steps_array = np.array(episode_steps)
        quantile_points = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        quantile_values = np.quantile(steps_array, quantile_points).tolist()
        episode_steps_quantiles = {
            "quantiles": {f"{int(q*100)}%": v for q, v in zip(quantile_points, quantile_values)},
            "mean": float(np.mean(steps_array)),
            "std": float(np.std(steps_array)),
            "count": len(steps_array)
        }
    
    # Calculate quantiles of subtask lengths (seconds)
    subtask_length_quantiles = {}
    if subtask_lengths_seconds:
        lengths_array = np.array(subtask_lengths_seconds)
        quantile_points = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        quantile_values = np.quantile(lengths_array, quantile_points).tolist()
        subtask_length_quantiles = {
            "quantiles": {f"{int(q*100)}%": v for q, v in zip(quantile_points, quantile_values)},
            "mean": float(np.mean(lengths_array)),
            "std": float(np.std(lengths_array)),
            "count": len(lengths_array)
        }
    
    # Calculate quantiles of subtask steps (number of data points)
    subtask_steps_quantiles = {}
    if subtask_steps:
        steps_array = np.array(subtask_steps)
        quantile_points = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        quantile_values = np.quantile(steps_array, quantile_points).tolist()
        subtask_steps_quantiles = {
            "quantiles": {f"{int(q*100)}%": v for q, v in zip(quantile_points, quantile_values)},
            "mean": float(np.mean(steps_array)),
            "std": float(np.std(steps_array)),
            "count": len(steps_array)
        }
    
    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
    PLOTS_DIR.mkdir(parents = True, exist_ok = True)
    
    # Create histogram of episode lengths
    if episode_lengths_seconds:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(episode_lengths_seconds, bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
        ax.set_xlabel('Episode Length (seconds)', fontsize=12)
        ax.set_ylabel('Number of Episodes', fontsize=12)
        ax.set_title('Distribution of Episode Lengths', fontsize=14)
        
        # Add vertical lines for key quantiles
        if episode_length_quantiles:
            median = episode_length_quantiles['quantiles']['50%']
            mean = episode_length_quantiles['mean']
            ax.axvline(median, color='#E94F37', linestyle='--', linewidth=2, label=f'Median: {median:.1f}s')
            ax.axvline(mean, color='#F39C12', linestyle=':', linewidth=2, label=f'Mean: {mean:.1f}s')
            ax.legend(fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        hist_path = PLOTS_DIR / "episode_lengths_histogram.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Saved histogram to {hist_path}")
    
    # Create histogram of episode steps
    if episode_steps:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(episode_steps, bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
        ax.set_xlabel('Episode Length (steps)', fontsize=12)
        ax.set_ylabel('Number of Episodes', fontsize=12)
        ax.set_title('Distribution of Episode Steps', fontsize=14)
        
        # Add vertical lines for key quantiles
        if episode_steps_quantiles:
            median = episode_steps_quantiles['quantiles']['50%']
            mean = episode_steps_quantiles['mean']
            ax.axvline(median, color='#E94F37', linestyle='--', linewidth=2, label=f'Median: {median:.0f}')
            ax.axvline(mean, color='#F39C12', linestyle=':', linewidth=2, label=f'Mean: {mean:.0f}')
            ax.legend(fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        hist_path = PLOTS_DIR / "episode_steps_histogram.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Saved histogram to {hist_path}")
    
    # Create histogram of subtask lengths (seconds)
    if subtask_lengths_seconds:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(subtask_lengths_seconds, bins=50, edgecolor='black', alpha=0.7, color='#28A745')
        ax.set_xlabel('Subtask Length (seconds)', fontsize=12)
        ax.set_ylabel('Number of Subtasks', fontsize=12)
        ax.set_title('Distribution of Subtask Lengths', fontsize=14)
        
        # Add vertical lines for key quantiles
        if subtask_length_quantiles:
            median = subtask_length_quantiles['quantiles']['50%']
            mean = subtask_length_quantiles['mean']
            ax.axvline(median, color='#E94F37', linestyle='--', linewidth=2, label=f'Median: {median:.1f}s')
            ax.axvline(mean, color='#F39C12', linestyle=':', linewidth=2, label=f'Mean: {mean:.1f}s')
            ax.legend(fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        hist_path = PLOTS_DIR / "subtask_lengths_histogram.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Saved histogram to {hist_path}")
    
    # Create histogram of subtask steps
    if subtask_steps:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(subtask_steps, bins=50, edgecolor='black', alpha=0.7, color='#28A745')
        ax.set_xlabel('Subtask Length (steps)', fontsize=12)
        ax.set_ylabel('Number of Subtasks', fontsize=12)
        ax.set_title('Distribution of Subtask Steps', fontsize=14)
        
        # Add vertical lines for key quantiles
        if subtask_steps_quantiles:
            median = subtask_steps_quantiles['quantiles']['50%']
            mean = subtask_steps_quantiles['mean']
            ax.axvline(median, color='#E94F37', linestyle='--', linewidth=2, label=f'Median: {median:.0f}')
            ax.axvline(mean, color='#F39C12', linestyle=':', linewidth=2, label=f'Mean: {mean:.0f}')
            ax.legend(fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        hist_path = PLOTS_DIR / "subtask_steps_histogram.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Saved histogram to {hist_path}")
    
    # Write outputs
    info_path = OUTPUT_DIR / "info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(repo_info, f, indent = 2, ensure_ascii = False)
    print(f"\nWrote repo metadata to {info_path}")
    
    # Write episode length quantiles (seconds)
    quantiles_path = OUTPUT_DIR / "episode_length_quantiles.json"
    with open(quantiles_path, 'w', encoding='utf-8') as f:
        json.dump(episode_length_quantiles, f, indent = 2, ensure_ascii = False)
    print(f"Wrote episode length quantiles to {quantiles_path}")
    
    # Write episode steps quantiles (number of data points)
    steps_quantiles_path = OUTPUT_DIR / "episode_steps_quantiles.json"
    with open(steps_quantiles_path, 'w', encoding='utf-8') as f:
        json.dump(episode_steps_quantiles, f, indent = 2, ensure_ascii = False)
    print(f"Wrote episode steps quantiles to {steps_quantiles_path}")
    
    # Write subtask length quantiles (seconds)
    subtask_length_quantiles_path = OUTPUT_DIR / "subtask_length_quantiles.json"
    with open(subtask_length_quantiles_path, 'w', encoding='utf-8') as f:
        json.dump(subtask_length_quantiles, f, indent = 2, ensure_ascii = False)
    print(f"Wrote subtask length quantiles to {subtask_length_quantiles_path}")
    
    # Write subtask steps quantiles (number of data points)
    subtask_steps_quantiles_path = OUTPUT_DIR / "subtask_steps_quantiles.json"
    with open(subtask_steps_quantiles_path, 'w', encoding='utf-8') as f:
        json.dump(subtask_steps_quantiles, f, indent = 2, ensure_ascii = False)
    print(f"Wrote subtask steps quantiles to {subtask_steps_quantiles_path}")
    
    norm_path = OUTPUT_DIR / "norm_stats.json"
    with open(norm_path, 'w', encoding='utf-8') as f:
        json.dump(norm_stats, f, indent = 2, ensure_ascii = False)
    print(f"Wrote normalization stats to {norm_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total repos processed: {len(repo_info)}")
    print(f"Unique robot types: {len(norm_stats)}")
    for robot_type in norm_stats.keys():
        keys_with_stats = [k for k, v in norm_stats[robot_type].items()]
        print(f"  {robot_type}: {keys_with_stats}")
    
    if episode_length_quantiles:
        print("\nEpisode Length Quantiles (seconds):")
        print(f"  Count: {episode_length_quantiles['count']} episodes")
        print(f"  Mean: {episode_length_quantiles['mean']:.2f}s")
        print(f"  Std: {episode_length_quantiles['std']:.2f}s")
        for q_name, q_val in episode_length_quantiles['quantiles'].items():
            print(f"  {q_name}: {q_val:.2f}s")
    
    if episode_steps_quantiles:
        print("\nEpisode Steps Quantiles (data points):")
        print(f"  Count: {episode_steps_quantiles['count']} episodes")
        print(f"  Mean: {episode_steps_quantiles['mean']:.0f}")
        print(f"  Std: {episode_steps_quantiles['std']:.0f}")
        for q_name, q_val in episode_steps_quantiles['quantiles'].items():
            print(f"  {q_name}: {q_val:.0f}")
    
    if subtask_length_quantiles:
        print("\nSubtask Length Quantiles (seconds):")
        print(f"  Count: {subtask_length_quantiles['count']} subtasks")
        print(f"  Mean: {subtask_length_quantiles['mean']:.2f}s")
        print(f"  Std: {subtask_length_quantiles['std']:.2f}s")
        for q_name, q_val in subtask_length_quantiles['quantiles'].items():
            print(f"  {q_name}: {q_val:.2f}s")
    
    if subtask_steps_quantiles:
        print("\nSubtask Steps Quantiles (data points):")
        print(f"  Count: {subtask_steps_quantiles['count']} subtasks")
        print(f"  Mean: {subtask_steps_quantiles['mean']:.0f}")
        print(f"  Std: {subtask_steps_quantiles['std']:.0f}")
        for q_name, q_val in subtask_steps_quantiles['quantiles'].items():
            print(f"  {q_name}: {q_val:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

