# Survey RoboCOIN

Tools for surveying, analyzing, and working with the [RoboCOIN](https://huggingface.co/RoboCOIN) robotics dataset collection on Hugging Face.

## Overview

This repository provides:
- **Metadata collection** across all RoboCOIN datasets
- **Normalization statistics** computed per embodiment/robot type
- **Episode and subtask length analysis** with visualizations
- **Dataset building** tools for LeRobot/MultiLeRobotDataset
- **DataLoader benchmarking** for PyTorch and DLIMP

## Directory Structure

```
robocoin/
├── build_scripts/          # Dataset download and construction
│   └── concatenate_lerobot.py
├── survey_scripts/         # Metadata collection and analysis
│   ├── collect_metadata.py
│   ├── test_pytorch_dataloader.py
│   ├── test_dlimp_dataloader.py
│   └── web_scrape_memory.py
├── sbatch_scripts/         # SLURM job scripts
│   ├── test_pytorch_dataloader.sh
│   └── test_dlimp_dataloader.sh
└── logs/                   # Output logs
```

## Scripts

### Build Scripts

#### `build_scripts/concatenate_lerobot.py`

Downloads all RoboCOIN datasets and constructs a `LeRobotDataset` or `MultiLeRobotDataset`.

```bash
python build_scripts/concatenate_lerobot.py --use_multi --num_workers 4
```

### Survey Scripts

#### `survey_scripts/collect_metadata.py`

Main script that surveys all RoboCOIN datasets and produces:

**Outputs:**
- `survey_scripts/stats/info.json` - Per-repo metadata (robot type, fps, dimensions, camera info)
- `survey_scripts/stats/norm_stats.json` - Embodiment-wise normalization statistics (min/max/mean/std)
- `survey_scripts/stats/episode_length_quantiles.json` - Episode length distribution (seconds)
- `survey_scripts/stats/episode_steps_quantiles.json` - Episode length distribution (steps)
- `survey_scripts/stats/subtask_length_quantiles.json` - Subtask length distribution (seconds)
- `survey_scripts/stats/subtask_steps_quantiles.json` - Subtask length distribution (steps)
- `survey_scripts/plots/` - Histograms of episode and subtask length distributions
- `survey_scripts/examples/` - Example videos and annotation files for selected repos

**Features:**
- Extracts robot type from repo_id
- Computes running statistics using Welford's algorithm for numerical stability
- Handles multiple embodiment variants (same robot type with different state dimensions)
- Tracks subtask transitions with concurrent subtask support
- Saves example videos for repos meeting specific conditions

```bash
python survey_scripts/collect_metadata.py
```

#### `survey_scripts/test_pytorch_dataloader.py`

Benchmarks iteration speed and memory usage of the PyTorch DataLoader for `MultiLeRobotDataset`.

```bash
python survey_scripts/test_pytorch_dataloader.py --num_workers 4 --batch_size 32
```

#### `survey_scripts/test_dlimp_dataloader.py`

Tests the DLIMP (TensorFlow-based) dataloader for comparison.

#### `survey_scripts/web_scrape_memory.py`

Scrapes dataset statistics (total hours, size, episodes) from Hugging Face README files.

```bash
python survey_scripts/web_scrape_memory.py
```

### SLURM Scripts

- `sbatch_scripts/test_pytorch_dataloader.sh` - SLURM job for PyTorch dataloader benchmarks
- `sbatch_scripts/test_dlimp_dataloader.sh` - SLURM job for DLIMP dataloader benchmarks

```bash
sbatch sbatch_scripts/test_pytorch_dataloader.sh
```

## Requirements

```
numpy
matplotlib
pyarrow
huggingface_hub
torch
torchvision
lerobot
tensorflow
dlimp
```

## Installation

```bash
pip install numpy matplotlib pyarrow huggingface_hub torch torchvision
pip install lerobot tensorflow dlimp # For dataset loading
```

## Quick Start

1. **Collect metadata and statistics from all RoboCOIN datasets:**
   ```bash
   python survey_scripts/collect_metadata.py
   ```

2. **View generated statistics:**
   ```bash
   cat survey_scripts/stats/episode_length_quantiles.json
   ```

3. **Run dataloader benchmarks:**
   ```bash
   python survey_scripts/test_pytorch_dataloader.py
   ```

## Output Files

After running `collect_metadata.py`:

```
survey_scripts/
├── stats/
│   ├── info.json                      # Per-dataset metadata
│   ├── norm_stats.json                # Normalization statistics by embodiment
│   ├── episode_length_quantiles.json  # Episode length stats (seconds)
│   ├── episode_steps_quantiles.json   # Episode length stats (steps)
│   ├── subtask_length_quantiles.json  # Subtask length stats (seconds)
│   └── subtask_steps_quantiles.json   # Subtask length stats (steps)
├── plots/
│   ├── episode_lengths_histogram.png
│   ├── episode_steps_histogram.png
│   ├── subtask_lengths_histogram.png
│   └── subtask_steps_histogram.png
└── examples/                          # Example videos and annotations
    └── <dataset>/                     # Dataset name without RoboCOIN_ prefix
        ├── episode_000000_<camera>.mp4
        ├── subtask_annotations.jsonl
        ├── scene_annotations.jsonl
        ├── episodes.jsonl
        └── tasks.jsonl
```
