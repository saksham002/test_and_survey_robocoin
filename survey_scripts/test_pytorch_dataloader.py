#!/usr/bin/env python3
"""
Test the iteration speed of the PyTorch DataLoader for the MultiLeRobotDataset.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional
import pdb

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from huggingface_hub import HfApi
import torch.nn.functional as F

# Ensure HF token is set
os.environ["HF_TOKEN"] = "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"

# ---------- Configuration ----------
DOWNLOAD_DIR = "/data/group_data/rl/saksham3/robocoin/"
CACHE_ROOT = Path(DOWNLOAD_DIR)
PREFIX = "RoboCOIN/"
DOWNLOAD_CMD = ["robocoin-download", "--hub", "huggingface", "--target-dir", DOWNLOAD_DIR, "--ds_lists"]

RATE_LIMIT_MARKERS = (
    "429 Too Many Requests",
)


def run_with_rate_limit_retry(
    cmd: List[str],
    sleep_seconds: int = 300,
    max_retries: Optional[int] = None,
) -> None:
    """
    Run a command, retrying on Hugging Face Hub rate limit errors.
    - sleep_seconds: wait time between retries (default 5 minutes).
    - max_retries: None = retry forever; else stop after N retries.
    """
    attempt = 0
    while True:
        attempt += 1
        p = subprocess.run(cmd, capture_output=True, text=True)

        out = (p.stdout or "") + "\n" + (p.stderr or "")

        # Check for rate limit
        if any(m in out for m in RATE_LIMIT_MARKERS):
            sys.stderr.write(
                f"[rate-limit] Command failed with rate limit markers. "
                f"Sleeping {sleep_seconds}s then retrying. Attempt={attempt}\n"
            )

            if max_retries is not None and attempt > max_retries:
                raise RuntimeError(f"Exceeded max_retries={max_retries} for command: {cmd}")

            time.sleep(sleep_seconds)
            continue

        if p.returncode == 0:
            # Print outputs to preserve logs
            if p.stdout:
                sys.stdout.write(p.stdout)
            if p.stderr:
                sys.stderr.write(p.stderr)
            return

        # Not a rate limit: surface error immediately
        sys.stderr.write(out)
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)


def get_robocoin_repo_ids() -> List[str]:
    """Get all RoboCOIN dataset repo_ids from Hugging Face Hub."""
    api = HfApi()
    print(f"Listing datasets with prefix '{PREFIX}' from Hugging Face...")
    infos = api.list_datasets(search=PREFIX)
    repo_ids = sorted({d.id for d in infos if d.id.startswith(PREFIX)})
    print(f"Found {len(repo_ids)} datasets.")
    return repo_ids


def download_datasets(repo_ids: List[str]) -> List[str]:
    """
    Download datasets using robocoin-download command.
    
    Returns list of successfully downloaded repo_ids.
    """
    successful = []
    failures = []

    for i, repo_id in enumerate(repo_ids, start=1):
        print(f"\n[{i}/{len(repo_ids)}] Downloading {repo_id}")

        try:
            # Remove prefix from repo_id for the command
            repo_id_no_prefix = repo_id[len(PREFIX):]
            cmd = DOWNLOAD_CMD + [repo_id_no_prefix]
            print(f"  running: {' '.join(cmd)}")
            run_with_rate_limit_retry(cmd, sleep_seconds=30, max_retries=None)
            successful.append(repo_id)
            print(f"  Successfully downloaded {repo_id}")
        except subprocess.CalledProcessError as e:
            msg = f"download command failed (exit {e.returncode})"
            failures.append((repo_id, msg))
            print(f"  ERROR: {msg}")
        except Exception as e:
            failures.append((repo_id, f"{type(e).__name__}: {e}"))
            print(f"  ERROR: {type(e).__name__}: {e}")

    if failures:
        print("\n==================== DOWNLOAD FAILURES ====================")
        for repo_id, msg in failures:
            print(f"  {repo_id}: {msg}")
        print("============================================================")

    print(f"\nSuccessfully downloaded {len(successful)}/{len(repo_ids)} datasets.")
    return successful

class PadToMaxDimension:
    def __init__(self, max_state_dim=118, max_action_dim=54):
        self.max_state = max_state_dim
        self.max_action = max_action_dim

    def __call__(self, item):
        # --- Pad State ---
        state = item['observation.state']
        pad_s = self.max_state - state.shape[-1]
        item['observation.state'] = F.pad(state, (0, pad_s), value=0)
        
        # Create State Mask
        mask_s = torch.zeros(self.max_state, dtype=torch.bool)
        mask_s[:state.shape[-1]] = True
        item['observation.state_mask'] = mask_s

        # --- Pad Action ---
        action = item['action']
        pad_a = self.max_action - action.shape[-1]
        item['action'] = F.pad(action, (0, pad_a), value=0)
        
        # Create Action Mask
        mask_a = torch.zeros(self.max_action, dtype=torch.bool)
        mask_a[:action.shape[-1]] = True
        item['action_mask'] = mask_a
        
        return item


class CropToSize:
    """Crop image to a specific size by taking the top-left corner."""
    def __init__(self, size=(224, 224)):
        self.size = size
    
    def __call__(self, img):
        # img is expected to be in CHW format (channels, height, width)
        # Crop to [:height, :width] for the last two dimensions
        return img[:, :self.size[0], :self.size[1]]


def build_multi_dataset(repo_ids: List[str], use_multi: bool = False, use_crop: bool = False):
    """
    Build a LeRobotDataset or MultiLeRobotDataset from the downloaded datasets.
    
    Args:
        repo_ids: List of repo_ids (with RoboCOIN/ prefix)
        use_multi: If True, use MultiLeRobotDataset; if False, use LeRobotDataset (only if len(repo_ids) == 1)
        use_crop: If True, use crop operation ([:224, :224]) instead of resize for images
    
    Returns:
        LeRobotDataset or MultiLeRobotDataset instance
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset

    # Use MultiLeRobotDataset if use_multi is True
    # Otherwise use LeRobotDataset for a single repo_id
    if use_multi:
        dataset_class = MultiLeRobotDataset
        dataset_name = "MultiLeRobotDataset"
        # MultiLeRobotDataset takes a list of repo_ids
        dataset_kwargs = {
            "repo_ids": repo_ids,
            "root": CACHE_ROOT,
            "download_videos": False,  # Already downloaded via robocoin-download
            "image_transforms": None,
            "transforms": PadToMaxDimension(),
        }
    else:
        dataset_class = LeRobotDataset
        dataset_name = "LeRobotDataset"
        # LeRobotDataset takes a single repo_id string, not a list
        dataset_kwargs = {
            "repo_id": repo_ids[0],
            "root": CACHE_ROOT / repo_ids[0],
            "download_videos": False,  # Already downloaded via robocoin-download
            "image_transforms": None,
            "transforms": PadToMaxDimension(),
        }

    print(f"\n==================== Building {dataset_name} ====================")
    print(f"Number of datasets: {len(repo_ids)}")
    print(f"Root directory: {CACHE_ROOT}")

    # Create image transform: resize/crop to 224x224 and normalize with ImageNet stats
    if use_crop:
        # Use inexpensive crop operation (just takes [:224, :224])
        image_transform = transforms.Compose([
            CropToSize((224, 224)),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])
    else:
        # Use resize operation (default)
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])

    # Apply image transform to the appropriate parameter
    dataset_kwargs["image_transforms"] = image_transform

    dataset = dataset_class(**dataset_kwargs)

    print("\n==================== Dataset Info ====================")
    print(dataset)
    print("=======================================================")

    return dataset


def iterate_dataset(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    max_batches: Optional[int] = None,
):
    """
    Iterate through the MultiLeRobotDataset using PyTorch DataLoader.

    Args:
        dataset: MultiLeRobotDataset instance
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        max_batches: Maximum number of batches to iterate (None for all)

    Returns:
        Dict with iteration statistics
    """
    print("\n==================== Iterating Through Dataset ====================")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Shuffle: {shuffle}")
    print(f"Total samples: {len(dataset)}")
    print(f"Expected batches: {(len(dataset) + batch_size - 1) // batch_size}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    total_samples = 0
    total_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            print(f"\nReached max_batches limit ({max_batches}). Stopping iteration.")
            break

        # Get batch size (may be smaller for last batch)
        current_batch_size = batch["index"].shape[0]
        total_samples += current_batch_size
        total_batches += 1

        # Log progress every 100 batches
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
            print(
                f"  Batch {batch_idx + 1}: "
                f"samples={total_samples}, "
                f"elapsed={elapsed:.1f}s, "
                f"throughput={samples_per_sec:.1f} samples/s"
            )

            # Print batch keys on first batch
            print(f"  Batch keys: {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")

    elapsed_time = time.time() - start_time
    samples_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0

    print("\n==================== Iteration Complete ====================")
    print(f"Total batches: {total_batches}")
    print(f"Total samples: {total_samples}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Throughput: {samples_per_sec:.2f} samples/s")
    print("=============================================================")

    return {
        "total_batches": total_batches,
        "total_samples": total_samples,
        "elapsed_time": elapsed_time,
        "samples_per_sec": samples_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Download RoboCOIN datasets and construct a LeRobotDataset or MultiLeRobotDataset")
    parser.add_argument(
        "--use_multi",
        action="store_true",
        help="Use MultiLeRobotDataset instead of LeRobotDataset (required if multiple datasets)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the DataLoader"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for the DataLoader"
    )
    parser.add_argument(
        "--use_crop",
        action="store_true",
        help="Use crop operation ([:224, :224]) instead of resize for images (cheaper but assumes images are >= 224x224)"
    )
    args = parser.parse_args()
    batch_size = args.batch_size
    num_workers = args.num_workers

    # 1. Get repo_ids from Hugging Face Hub
    repo_ids = get_robocoin_repo_ids()[0 : 101 : 100]
    # pdb.set_trace()

    if not repo_ids:
        print("No datasets found. Exiting.")
        return

    # 2. Download all datasets
    successful_repo_ids = download_datasets(repo_ids)

    if not successful_repo_ids:
        print("No datasets were successfully downloaded. Exiting.")
        return

    # Validate that use_multi is set when multiple datasets are present
    if len(successful_repo_ids) > 1 and not args.use_multi:
        print(f"Warning: Multiple datasets ({len(successful_repo_ids)}) require --use_multi. Using MultiLeRobotDataset anyway.")
        args.use_multi = True

    # 3. Build LeRobotDataset or MultiLeRobotDataset
    dataset = build_multi_dataset(successful_repo_ids, use_multi = args.use_multi, use_crop = args.use_crop)

    # Print summary
    print("\n==================== SUMMARY ====================")
    print(f"Total datasets: {len(successful_repo_ids)}")
    print(f"Dataset type: {type(dataset).__name__}")
    if len(successful_repo_ids) > 1:
        print(f"Robot types: {dataset.robot_types}")
        print(f"Stats per robot type: {list(dataset.stats_per_robot.keys())}")
    print(f"Total frames: {dataset.num_frames}")
    print(f"Total episodes: {dataset.num_episodes}")
    if len(successful_repo_ids) > 1:
        print(f"Stats per robot type: {list(dataset.stats_per_robot.keys())}")
    print("==================================================")

    # 4. Iterate through the dataset using DataLoader
    iteration_stats = iterate_dataset(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        max_batches=None,  # Set to a number to limit iterations, or None for full pass
    )

    return dataset, iteration_stats


if __name__ == "__main__":
    main()

