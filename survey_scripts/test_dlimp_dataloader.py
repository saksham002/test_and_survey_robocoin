#!/usr/bin/env python3
"""
Test the iteration speed of the DLIMP DataLoader for the RoboCOIN TFDS dataset.
"""

import argparse
import time
import threading
import queue
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

import pdb

# Disable GPU for TensorFlow (we only use it for data loading)
tf.config.experimental.set_visible_devices([], 'GPU')

import dlimp as dl
from dlimp.utils import vmap, parallel_vmap

# ---------- Configuration ----------
DATA_DIR = "/data/group_data/rl/saksham3/"
DATASET_NAME = "robocoin:1.0.0"
SPLIT = "train"
MAX_CAMERAS = 3
MAX_STATE_DIM = 118
MAX_ACTION_DIM = 54
IMAGE_SIZE = (224, 224)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype = np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype = np.float32)

# Directory for saving plots
PLOTS_DIR = Path(__file__).parent / "plots" / "images"
PLOTS_DIR.mkdir(parents = True, exist_ok = True)


def unnormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Reverse ImageNet normalization: (x * std) + mean, then clip to [0, 1] and scale to [0, 255].
    
    Args:
        image: Normalized float32 image of shape (H, W, C) with ImageNet normalization
        
    Returns:
        Unnormalized uint8 image of shape (H, W, C) in range [0, 255]
    """
    # Make a copy to avoid modifying the original
    image = image.copy()
    
    # Ensure image is in (H, W, C) format
    if image.ndim == 2:
        # Grayscale, add channel dimension
        image = np.expand_dims(image, axis=-1)
    elif image.ndim == 4:
        # Batch dimension, take first
        image = image[0]
    
    # Reverse normalization: (x * std) + mean
    # Handle both (H, W, C) and (H, W) cases
    if image.ndim == 3 and image.shape[2] == 3:
        # RGB image: apply per-channel normalization
        image = image * IMAGENET_STD.reshape(1, 1, 3) + IMAGENET_MEAN.reshape(1, 1, 3)
    else:
        # Grayscale or other: just reverse mean/std (use mean of means)
        mean_val = IMAGENET_MEAN.mean()
        std_val = IMAGENET_STD.mean()
        image = image * std_val + mean_val
    
    # Clip to [0, 1] range
    image = np.clip(image, 0.0, 1.0)
    
    # Convert to uint8 [0, 255]
    image = (image * 255.0).astype(np.uint8)
    
    return image


def plot_batch0_images(batch: Dict[str, Any], plots_dir: Path = PLOTS_DIR):
    """
    Plot and save images from the first example in batch 0.
    Only plots images that have non-zero values.
    
    Args:
        batch: Batch dictionary from the iterator
        plots_dir: Directory to save plots
    """
    # Extract first example from batch
    first_example = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            first_example[key] = value[0]
        else:
            first_example[key] = value
    
    # Find and plot camera images
    for cam_idx in range(MAX_CAMERAS):
        cam_key = f'observation/image/cam_{cam_idx}'
        
        if cam_key not in first_example:
            continue
        
        image = first_example[cam_key]
        
        # Check if image has non-zero values
        if isinstance(image, np.ndarray):
            if image.size == 0:
                continue
            
            # Check if image is non-zero (not padding)
            if np.all(image == 0):
                continue
            
            # Unnormalize image
            image = unnormalize_image(image)
            
            # Plot and save
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f'Camera {cam_idx}')
            
            save_path = plots_dir / f'cam_{cam_idx}.png'
            old_path = plots_dir / f'cam_{cam_idx}_old.png'
            
            # Rename existing file if it exists
            if save_path.exists():
                save_path.rename(old_path)
                print(f"  Renamed existing {save_path.name} to {old_path.name}")
            
            fig.savefig(save_path, bbox_inches = 'tight', dpi = 100)
            plt.close(fig)
            print(f"  Saved image to {save_path}")


class AsyncBatchPrefetcher:
    """
    Asynchronously prefetch batches in a background thread.
    Performs float32 conversion and ImageNet normalization on images.
    """
    
    def __init__(
        self, 
        iterator, 
        buffer_size: int = 2, 
        normalize_images: bool = True,
    ):
        """
        Initialize the prefetcher.
        
        Args:
            iterator: Iterator to fetch batches from
            buffer_size: Number of batches to prefetch
            normalize_images: Whether to normalize images to [0,1] and apply ImageNet stats
        """
        self.iterator = iterator
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.normalize_images = normalize_images
        self.thread = None
        self.stop_event = threading.Event()
        self.exception = None
        
    def _normalize_image_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Convert images to float32 and normalize with ImageNet statistics.
        
        Args:
            images: uint8 images of shape (..., H, W, C)
            
        Returns:
            Normalized float32 images
        """
        # Convert to float32 and scale to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization: (x - mean) / std
        images = (images - IMAGENET_MEAN) / IMAGENET_STD
        
        return images
        
    def _prefetch_worker(self):
        """Worker function that runs in a separate thread to prefetch batches."""
        try:
            for item in self.iterator:
                if self.stop_event.is_set():
                    break
                                
                # Normalize all camera images
                if self.normalize_images:
                    for cam_idx in range(MAX_CAMERAS):
                        cam_key = f'observation/image/cam_{cam_idx}'
                        if cam_key in item and item[cam_key] is not None:
                            # Check if images are non-empty (not padding)
                            # Images that are empty padding will be empty arrays
                            if item[cam_key].size > 0:
                                item[cam_key] = self._normalize_image_batch(item[cam_key])
                
                self.buffer.put(item)
        except Exception as e:
            print(f"Error in prefetch worker: {e}")
            self.exception = e
            self.buffer.put(None)  # Signal end
    
    def start(self):
        """Start the prefetching thread."""
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()
    
    def __iter__(self):
        """Make this object iterable."""
        return self
    
    def __next__(self):
        """Get the next prefetched batch."""
        if self.exception:
            raise self.exception
        
        item = self.buffer.get()
        if item is None:
            raise StopIteration
        return item
    
    def stop(self):
        """Stop the prefetching thread."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)


class PadToMaxDimension:
    """
    Pad observation.state and action to fixed maximum dimensions.
    Creates masks to indicate which dimensions are valid.
    Applied after flattening (per-frame transform).
    """
    
    def __init__(self, max_state_dim: int = MAX_STATE_DIM, max_action_dim: int = MAX_ACTION_DIM):
        self.max_state = max_state_dim
        self.max_action = max_action_dim
    
    def map(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Pad state and action to fixed dimensions."""
        # --- Pad State ---
        state = example['observation/state']
        state_dim = tf.shape(state)[-1]
        pad_s = self.max_state - state_dim
        
        # Pad state to max dimension
        example['observation/state'] = tf.pad(state, [[0, pad_s]], constant_values=0.0)
        
        # Create state mask (1 for valid, 0 for padding)
        state_mask = tf.concat([
            tf.ones(state_dim, dtype=tf.bool),
            tf.zeros(pad_s, dtype=tf.bool)
        ], axis=0)
        example['observation/state_mask'] = state_mask
        
        # --- Pad Action ---
        action = example['action']
        action_dim = tf.shape(action)[-1]
        pad_a = self.max_action - action_dim
        
        # Pad action to max dimension
        example['action'] = tf.pad(action, [[0, pad_a]], constant_values=0.0)
        
        # Create action mask
        action_mask = tf.concat([
            tf.ones(action_dim, dtype=tf.bool),
            tf.zeros(pad_a, dtype=tf.bool)
        ], axis=0)
        example['action_mask'] = action_mask
        
        return example


class ImageResizeTransform:
    """
    Decode JPEG images and resize to target size.
    Applied element-wise (per-frame) using Keras/TF functions.
    Does NOT normalize - normalization happens in the AsyncBatchPrefetcher.
    """
    
    def __init__(
        self, 
        target_size: tuple = IMAGE_SIZE,
        max_cameras: int = MAX_CAMERAS,
        ratio: int = 1,
        map_fn_type: str = "parallel_vmap",
    ):
        self.target_size = target_size
        self.max_cameras = max_cameras
        self.ratio = ratio
        self.map_fn_type = map_fn_type
    
    def _decode_and_resize(self, jpeg_bytes: tf.Tensor) -> tf.Tensor:
        """
        Decode a JPEG image and resize it.
        
        Args:
            jpeg_bytes: Encoded JPEG bytes
            
        Returns:
            Decoded and resized uint8 image of shape (H, W, C)
        """
        # Check if image is empty (padding camera)
        is_empty = tf.equal(tf.strings.length(jpeg_bytes), 0)
        
        def decode_resize():
            image = tf.io.decode_jpeg(jpeg_bytes, channels = 3, ratio = self.ratio)
            # Use Keras/TF resize function
            image = tf.image.resize(image, self.target_size, method = 'bilinear')
            # Keep as uint8 for now, conversion to float32 happens in prefetcher
            image = tf.cast(image, tf.uint8)
            return image
        
        def return_zeros():
            return tf.zeros((*self.target_size, 3), dtype = tf.uint8)
        
        return tf.cond(is_empty, return_zeros, decode_resize)
    
    def map(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode and resize all camera images in the example.
        Can process in parallel (vmap, parallel_vmap, tf_map_fn) or sequentially.
        """
        # Collect camera keys
        cam_keys = [f'observation/image/cam_{i}' for i in range(self.max_cameras)]
        
        # Process all cameras using the selected mapping function
        if self.map_fn_type == "sequential":
            # Sequential processing: process each camera one by one
            for cam_key in cam_keys:
                if cam_key in example:
                    example[cam_key] = self._decode_and_resize(example[cam_key])
        else:
            # Parallel processing: stack images and process together
            images = tf.stack([
                example.get(k, tf.constant(b'', dtype=tf.string)) 
                for k in cam_keys
            ])
            
            if self.map_fn_type == "vmap":
                resized = vmap(self._decode_and_resize)(images)
            elif self.map_fn_type == "parallel_vmap":
                resized = parallel_vmap(self._decode_and_resize)(images)
            elif self.map_fn_type == "tf_map_fn":
                resized = tf.map_fn(
                    self._decode_and_resize,
                    images,
                    parallel_iterations = self.max_cameras,
                    fn_output_signature = tf.TensorSpec(shape=(*self.target_size, 3), dtype=tf.uint8)
                )
            else:
                raise ValueError(f"Unknown map_fn_type: {self.map_fn_type}")
            
            # Unstack back to dict
            for i, cam_key in enumerate(cam_keys):
                if cam_key in example:
                    example[cam_key] = resized[i]
        
        return example


def create_robocoin_dataset_iterator(
    data_dir: str = DATA_DIR,
    split: str = SPLIT,
    batch_size: int = 64,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000,
    drop_remainder: bool = True,
    seed: int = 86,
    prefetch_buffer_size: int = 4,
    normalize_images: bool = True,
    drop_episode_metadata: bool = True,
    map_fn_type: str = "parallel_vmap",
    ratio: int = 1,
    map_fn_type: str = "parallel_vmap",
):
    """
    Create a DLIMP dataset iterator for the RoboCOIN dataset.
    
    Args:
        data_dir: Directory containing the TFDS dataset
        split: Dataset split to use ("train")
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        shuffle_buffer_size: Size of shuffle buffer
        drop_remainder: Whether to drop the last incomplete batch
        seed: Random seed for shuffling
        prefetch_buffer_size: Number of batches to prefetch asynchronously
        normalize_images: Whether to normalize images in the prefetcher
        
    Returns:
        AsyncBatchPrefetcher iterator or numpy iterator
    """
    print(f"\n==================== Building DLIMP Dataset ====================")
    print(f"Data directory: {data_dir}")
    print(f"Dataset name: {DATASET_NAME}")
    print(f"Split: {split}")
    print(f"Batch size: {batch_size}")
    print(f"Shuffle: {shuffle}")
    print(f"Shuffle buffer size: {shuffle_buffer_size}")
    print(f"Prefetch buffer size: {prefetch_buffer_size}")
    print(f"Drop episode metadata: {drop_episode_metadata}")
    print(f"Ratio: {ratio}")
    print(f"Map function type: {map_fn_type}")
    print("================================================================")
    
    # Build TFDS dataset
    builder = tfds.builder(DATASET_NAME, data_dir=data_dir)
    
    # Create DLIMP dataset from RLDS format
    dataset = dl.DLataset.from_rlds(builder, split = split, shuffle = shuffle, num_parallel_reads = -1)

    # Drop episode-level metadata that does not align with per-step length
    if drop_episode_metadata:
        def _drop_episode_metadata(episode: Any) -> Any:
            if isinstance(episode, dict):
                filtered = {
                    key: value
                    for key, value in episode.items()
                    if key not in ("traj_metadata", "episode_metadata")
                }
                if filtered:
                    return filtered
            return episode
        dataset = dataset.map(_drop_episode_metadata)
    
    # Repeat for continuous iteration
    dataset = dataset.repeat()
    
    # Flatten episodes to individual frames
    dataset = dataset.flatten()
    
    # Apply per-frame transforms
    # 1. Pad state and action to fixed dimensions
    # dataset = dataset.frame_map(PadToMaxDimension().map)
    
    # 2. Decode and resize images
    dataset = dataset.frame_map(ImageResizeTransform(ratio = ratio, map_fn_type = map_fn_type).map)
    
    # Shuffle if requested (after flattening for frame-level shuffling)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, seed = seed)
    
    # Batch the data
    dataset = dataset.batch(batch_size, drop_remainder = drop_remainder)
    
    # Set RAM budget
    dataset.with_ram_budget(1)
    
    # Create numpy iterator
    numpy_iterator = dataset.as_numpy_iterator()
    
    # Wrap with async prefetcher if requested
    if prefetch_buffer_size > 0:
        prefetcher = AsyncBatchPrefetcher(
            numpy_iterator,
            buffer_size = prefetch_buffer_size,
            normalize_images = normalize_images,
        )
        prefetcher.start()
        return prefetcher
    
    return numpy_iterator


def iterate_dataset(
    iterator,
    batch_size: int = 512,
    max_batches: Optional[int] = None,
    plot_images: bool = False,
):
    """
    Iterate through the dataset and measure throughput.
    
    Args:
        iterator: Dataset iterator
        batch_size: Number of samples per batch (for logging)
        max_batches: Maximum number of batches to iterate (None for indefinite)
        
    Returns:
        Dict with iteration statistics
    """
    print("\n==================== Iterating Through Dataset ====================")
    print(f"Batch size: {batch_size}")
    print(f"Max batches: {max_batches if max_batches else 'unlimited'}")
    
    total_samples = 0
    total_batches = 0
    batch_times = []
    start_time = time.time()
    
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            print(f"\nReached max_batches limit ({max_batches}). Stopping iteration.")
            break
                
        # Get batch size from the index field
        current_batch_size = batch["index"].shape[0]
        total_samples += current_batch_size
        total_batches += 1
        
        batch_end = time.time()
        batch_times.append(batch_end - batch_times[-1] if batch_times else 0)
        
        # Log progress every 100 batches or on first batch
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
            avg_batch_time = np.mean(batch_times) if batch_times else 0
            
            print(
                f"  Batch {batch_idx + 1}: "
                f"samples={total_samples}, "
                f"elapsed={elapsed:.1f}s, "
                f"throughput={samples_per_sec:.1f} samples/s, "
                f"avg_batch_time={avg_batch_time:.2f}s"
            )
            
            # Print batch keys and shapes on first batch
            if batch_idx == 0:
                print(f"  Batch keys: {list(batch.keys())}")
                for key, value in batch.items():
                    if isinstance(value, np.ndarray):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                
                # Plot and save images from first example in batch 0
                if plot_images:
                    print("\n  Plotting images from first example in batch 0...")
                    plot_batch0_images(batch, plots_dir = PLOTS_DIR)
    
    elapsed_time = time.time() - start_time
    samples_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0
    
    print("\n==================== Iteration Complete ====================")
    print(f"Total batches: {total_batches}")
    print(f"Total samples: {total_samples}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Throughput: {samples_per_sec:.2f} samples/s")
    if batch_times:
        print(f"Average batch time: {np.mean(batch_times):.2f}s")
        print(f"Batch time std: {np.std(batch_times):.2f}s")
        print(f"Min batch time: {np.min(batch_times):.2f}s")
        print(f"Max batch time: {np.max(batch_times):.2f}s")
    print("=============================================================")
    
    return {
        "total_batches": total_batches,
        "total_samples": total_samples,
        "elapsed_time": elapsed_time,
        "samples_per_sec": samples_per_sec,
        "batch_times": batch_times,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test DLIMP DataLoader for RoboCOIN TFDS dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for data loading (default: 64)"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=75,
        help="Maximum number of batches to iterate (default: None for unlimited)"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Whether to shuffle the data"
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10000,
        help="Size of shuffle buffer (default: 10000)"
    )
    parser.add_argument(
        "--prefetch_buffer_size",
        type=int,
        default=4,
        help="Number of batches to prefetch (default: 4)"
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable image normalization in prefetcher"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=86,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--keep_episode_metadata",
        action="store_true",
        help="Keep episode_metadata in the DLIMP pipeline (may cause shape mismatch)"
    )
    parser.add_argument(
        "--plot_images",
        action="store_true",
        help="Plot and save images from first example in batch 0"
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=1,
        help="Ratio for image decoding (default: 1)"
    )
    parser.add_argument(
        "--use_vmap",
        action="store_true",
        help="Use vmap for image processing (non-parallel, fused)"
    )
    parser.add_argument(
        "--use_parallel_vmap",
        action="store_true",
        help="Use parallel_vmap for image processing (parallel via tf.data)"
    )
    parser.add_argument(
        "--use_tf_map_fn",
        action="store_true",
        help="Use tf.map_fn for image processing"
    )
    args = parser.parse_args()
    
    # Determine map_fn_type from flags (default to sequential if none selected)
    map_fn_flags = [args.use_vmap, args.use_parallel_vmap, args.use_tf_map_fn]
    num_selected = sum(map_fn_flags)
    
    if num_selected > 1:
        raise ValueError(
            f"At most one of --use_vmap, --use_parallel_vmap, --use_tf_map_fn can be provided, "
            f"but {num_selected} were selected"
        )
    
    if args.use_vmap:
        map_fn_type = "vmap"
    elif args.use_parallel_vmap:
        map_fn_type = "parallel_vmap"
    elif args.use_tf_map_fn:
        map_fn_type = "tf_map_fn"
    else:
        # Default to sequential processing if no flags are provided
        map_fn_type = "sequential"
    
    print("=" * 60)
    print("DLIMP DataLoader Test for RoboCOIN Dataset")
    print("=" * 60)
    
    # Create dataset iterator
    iterator = create_robocoin_dataset_iterator(
        data_dir=DATA_DIR,
        split=SPLIT,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        shuffle_buffer_size=args.shuffle_buffer_size,
        drop_remainder=True,
        seed=args.seed,
        prefetch_buffer_size=args.prefetch_buffer_size,
        normalize_images=not args.no_normalize,
        drop_episode_metadata=not args.keep_episode_metadata,
        ratio=args.ratio,
        map_fn_type=map_fn_type,
    )
    
    # Iterate through dataset and measure performance
    iteration_stats = iterate_dataset(
        iterator,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        plot_images=args.plot_images,
    )
    
    # Stop prefetcher if using async prefetching
    if hasattr(iterator, 'stop'):
        iterator.stop()
    
    print("\n==================== FINAL SUMMARY ====================")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Total samples processed: {iteration_stats['total_samples']}")
    print(f"Total time: {iteration_stats['elapsed_time']:.2f}s")
    print(f"Final throughput: {iteration_stats['samples_per_sec']:.2f} samples/s")
    print("=======================================================")
    
    return iteration_stats


if __name__ == "__main__":
    main()

