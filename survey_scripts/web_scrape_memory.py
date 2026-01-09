import re
import json
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

PREFIX = "RoboCOIN/"
OUTPUT_FILE = "survey_scripts/stats/all_stats.jsonl"

def extract_metric(content, metric_name):
    """Extracts a string value for a given metric from the Markdown table."""
    regex = rf"\|\s*(?:\*\*|__)?\s*{re.escape(metric_name)}\s*(?:\*\*|__)?\s*\|\s*([^|]+?)\s*\|"
    match = re.search(regex, content, re.IGNORECASE)
    return match.group(1).strip() if match else None

def parse_to_float(val_str):
    """Converts a string to a float, removing commas and non-numeric characters."""
    if not val_str: return 0.0
    try:
        clean_str = re.sub(r'[^\d.]', '', val_str)
        return float(clean_str)
    except ValueError:
        return 0.0

def parse_size_to_gb(size_str):
    """Parses a string like '92.8GB' or '500MB' into a float representing GB."""
    if not size_str:
        return 0.0
    s = size_str.strip().lower()
    match = re.match(r"^([\d\.]+)\s*([a-z]*)", s)
    if not match:
        return 0.0
    val_str, unit = match.groups()
    try:
        val = float(val_str)
        if 'tb' in unit: return val * 1024.0
        if 'gb' in unit: return val
        if 'mb' in unit: return val / 1024.0
        if 'kb' in unit: return val / (1024.0 * 1024.0)
        return val # Default to GB if unit is missing but number is present
    except ValueError:
        return 0.0

def get_quantiles(data_list):
    """Returns a dictionary of quantiles at 0, 25, 50, 75, and 100%."""
    if not data_list: return {}
    arr = np.array(data_list)
    return {
        "0%": float(np.min(arr)),
        "25%": float(np.quantile(arr, 0.25)),
        "50%": float(np.median(arr)),
        "75%": float(np.quantile(arr, 0.75)),
        "100%": float(np.max(arr))
    }

def main():
    api = HfApi()
    print(f"Listing datasets with prefix '{PREFIX}' from Hugging Face...")
    infos = api.list_datasets(search=PREFIX)
    repo_ids = sorted([d.id for d in infos if d.id.startswith(PREFIX)])
    
    results_map = {}
    metrics_data = {
        "tasks": [], 
        "episodes": [], 
        "frames": [], 
        "fps": [],
        "num_cameras": [],
        "sizes_gb": []
    }
    # Track size by num_cameras for summing
    size_by_cameras = {}  # {num_cameras: sum_of_sizes}
    
    for repo_id in repo_ids:
        try:
            readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="dataset")
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract raw strings
            v_str = extract_metric(content, "Total Videos")
            t_str = extract_metric(content, "Total Tasks")
            e_str = extract_metric(content, "Total Episodes")
            f_str = extract_metric(content, "Total Frames")
            fps_str = extract_metric(content, "FPS")
            sz_str = extract_metric(content, "Dataset Size")
            
            # Parse to numbers
            v_val = parse_to_float(v_str)
            t_val = parse_to_float(t_str)
            e_val = parse_to_float(e_str)
            f_val = parse_to_float(f_str)
            fps_val = parse_to_float(fps_str)
            sz_gb_val = parse_size_to_gb(sz_str)
            
            num_cameras = int(v_val // e_val) if e_val > 0 else 0

            # Line 1 entry: Add the parsed size in GB
            results_map[repo_id] = {
                "videos": v_str or "N/A", 
                "tasks": t_str or "N/A",
                "episodes": e_str or "N/A", 
                "frames": f_str or "N/A",
                "fps": fps_str or "N/A",
                "size_gb": sz_gb_val,
                "num_cameras": num_cameras
            }
            
            # Collect numeric data
            if t_val > 0: metrics_data["tasks"].append(t_val)
            if e_val > 0: metrics_data["episodes"].append(e_val)
            if f_val > 0: metrics_data["frames"].append(f_val)
            if fps_val > 0: metrics_data["fps"].append(fps_val)
            metrics_data["num_cameras"].append(num_cameras)
            if sz_gb_val > 0: metrics_data["sizes_gb"].append(sz_gb_val)
            
            # Accumulate size by num_cameras
            if num_cameras > 0 and sz_gb_val > 0:
                if num_cameras not in size_by_cameras:
                    size_by_cameras[num_cameras] = 0.0
                size_by_cameras[num_cameras] += sz_gb_val

            print(f"Collected data for {repo_id}")
            
        except Exception as e:
            results_map[repo_id] = f"Error: {str(e)}"

            print(f"Error: {str(e)} for {repo_id}")

    # Line 2 Summary
    summary = {
        "total_episodes_sum": float(sum(metrics_data["episodes"])),
        "total_frames_sum": float(sum(metrics_data["frames"])),
        "total_size_gb_sum": float(sum(metrics_data["sizes_gb"])),
        "quantiles_fps": get_quantiles(metrics_data["fps"]),
        "quantiles_tasks": get_quantiles(metrics_data["tasks"]),
        "quantiles_num_cameras": get_quantiles(metrics_data["num_cameras"]),
        "size_gb_sum_by_num_cameras": {str(k): float(v) for k, v in sorted(size_by_cameras.items())}
    }

    print(f"Writing metrics to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results_map) + "\n")
        f.write(json.dumps(summary) + "\n")

    print("Done.")

if __name__ == "__main__":
    main()