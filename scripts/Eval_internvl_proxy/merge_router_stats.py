import json
import glob
import os
import sys

def merge_stats(result_dir):
    # Find all router_stats*.json files
    # This matches router_stats.json, router_stats_1.json, router_stats_pid123.json etc.
    pattern = os.path.join(result_dir, "router_stats*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No router stats files found in {result_dir}")
        return

    print(f"Found {len(files)} stats files to merge: {files}")

    merged_stats = {}
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
            for module_name, stats in data.items():
                if module_name not in merged_stats:
                    # Initialize with the first file's data
                    merged_stats[module_name] = {"counts": stats["counts"]}
                else:
                    # Sum the counts
                    current_counts = merged_stats[module_name]["counts"]
                    new_counts = stats["counts"]
                    # Ensure lengths match
                    if len(current_counts) != len(new_counts):
                        print(f"Warning: Count length mismatch for {module_name} in {fpath}")
                        continue
                    merged_stats[module_name]["counts"] = [x + y for x, y in zip(current_counts, new_counts)]
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    # Re-calculate distribution
    for module_name in merged_stats:
        counts = merged_stats[module_name]["counts"]
        total = sum(counts)
        distribution = [c / total for c in counts] if total > 0 else [0] * len(counts)
        merged_stats[module_name]["distribution"] = distribution

    # Save merged stats to router_stats.json
    output_path = os.path.join(result_dir, "router_stats.json")
    
    try:
        with open(output_path, 'w') as f:
            json.dump(merged_stats, f, indent=2)
        print(f"Successfully merged stats into {output_path}")
        
        # Clean up partial files (except the main one we just wrote)
        for fpath in files:
            if os.path.abspath(fpath) != os.path.abspath(output_path):
                try:
                    os.remove(fpath)
                    print(f"Removed partial file: {fpath}")
                except OSError as e:
                    print(f"Error removing {fpath}: {e}")
                    
    except Exception as e:
        print(f"Failed to save merged stats: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        merge_stats(sys.argv[1])
    else:
        print("Usage: python merge_router_stats.py <result_dir>")
