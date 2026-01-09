#!/usr/bin/env python3
"""
Test the iteration speed of the DLIMP DataLoader for the RoboCOIN TFDS dataset.
"""

import argparse
import time
import threading
import queue
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Disable GPU for TensorFlow (we only use it for data loading)
tf.config.experimental.set_visible_devices([], 'GPU')

import dlimp as dl

# ---------- Configuration ----------
DATA_DIR = "/data/group_data/rl/saksham3/"
DATASET_NAME = "robocoin:1.0.0"
SPLIT = "train"
MAX_CAMERAS = 8
MAX_STATE_DIM = 118
MAX_ACTION_DIM = 54
IMAGE_SIZE = (224, 224)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype = np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype = np.float32)


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
    ):
        self.target_size = target_size
        self.max_cameras = max_cameras
    
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
            image = tf.io.decode_jpeg(jpeg_bytes, channels = 3)
            # Use Keras/TF resize function
            image = tf.image.resize(image, self.target_size, method = 'bilinear')
            # Keep as uint8 for now, conversion to float32 happens in prefetcher
            image = tf.cast(image, tf.uint8)
            return image
        
        def return_zeros():
            return tf.zeros((*self.target_size, 3), dtype=tf.uint8)
        
        return tf.cond(is_empty, return_zeros, decode_resize)
    
    def map(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode and resize all camera images in the example.
        """
        for cam_idx in range(self.max_cameras):
            cam_key = f'observation/image/cam_{cam_idx}'
            if cam_key in example:
                example[cam_key] = self._decode_and_resize(example[cam_key])
        
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
    print("================================================================")
    
    # Build TFDS dataset
    builder = tfds.builder(DATASET_NAME, data_dir=data_dir)
    
    # Create DLIMP dataset from RLDS format
    dataset = dl.DLataset.from_rlds(builder, split = split, shuffle = shuffle, num_parallel_reads = -1)
    
    # Repeat for continuous iteration
    dataset = dataset.repeat()
    
    # Flatten episodes to individual frames
    dataset = dataset.flatten()
    
    # Apply per-frame transforms
    # 1. Pad state and action to fixed dimensions
    dataset = dataset.frame_map(PadToMaxDimension().map)
    
    # 2. Decode and resize images
    dataset = dataset.frame_map(ImageResizeTransform().map)
    
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
    batch_size: int = 64,
    max_batches: Optional[int] = None,
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
        
        batch_start = time.time()
        
        # Get batch size from the index field
        current_batch_size = batch["index"].shape[0]
        total_samples += current_batch_size
        total_batches += 1
        
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        
        # Log progress every 100 batches or on first batch
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
            avg_batch_time = np.mean(batch_times) * 1000 if batch_times else 0
            
            print(
                f"  Batch {batch_idx + 1}: "
                f"samples={total_samples}, "
                f"elapsed={elapsed:.1f}s, "
                f"throughput={samples_per_sec:.1f} samples/s, "
                f"avg_batch_time={avg_batch_time:.2f}ms"
            )
            
            # Print batch keys and shapes on first batch
            if batch_idx == 0:
                print(f"  Batch keys: {list(batch.keys())}")
                for key, value in batch.items():
                    if isinstance(value, np.ndarray):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
    
    elapsed_time = time.time() - start_time
    samples_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0
    
    print("\n==================== Iteration Complete ====================")
    print(f"Total batches: {total_batches}")
    print(f"Total samples: {total_samples}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Throughput: {samples_per_sec:.2f} samples/s")
    if batch_times:
        print(f"Average batch time: {np.mean(batch_times)*1000:.2f}ms")
        print(f"Batch time std: {np.std(batch_times)*1000:.2f}ms")
        print(f"Min batch time: {np.min(batch_times)*1000:.2f}ms")
        print(f"Max batch time: {np.max(batch_times)*1000:.2f}ms")
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
        default=64,
        help="Batch size for data loading (default: 64)"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=604,
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
    args = parser.parse_args()

    print(f"Arguments: {args}")
    
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
    )
    
    # Iterate through dataset and measure performance
    iteration_stats = iterate_dataset(
        iterator,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
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

