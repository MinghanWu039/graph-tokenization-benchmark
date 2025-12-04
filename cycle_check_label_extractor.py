#!/usr/bin/env python3
"""
Extract labels from all algorithm subdirectories within a task directory.
Usage: python batch_label_extractor.py /path/to/tasks/cycle_check output_dir --split train
Examples:
    /tasks/cycle_check/ba/train
    /tasks/cycle_check/ba/test
"""

import json
import argparse
import sys
from pathlib import Path
import re


def extract_cycle_label_from_json(file_path):
    """
    Extract cycle detection label from a JSON file.
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        tuple: (graph_id, label) or (None, None) if not found
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both single object and list of objects
        if isinstance(data, list):
            data = data[0]  # Take first item if it's a list
        
        graph_id = data.get('graph_id', file_path.stem)
        text = data.get('text', '')
        # Look for pattern: <q> has_cycle <p> yes/no
        pattern = r'<q>\s*has_cycle\s*<p>\s*(yes|no)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            label = match.group(1).lower()
            # Convert to binary: yes -> 1, no -> 0
            binary_label = 1 if label == 'yes' else 0
            return graph_id, binary_label
        
        return None, None
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


def process_algorithm_directory(alg_dir, alg_name):
    """
    Process all JSON files in an algorithm directory.
    
    Args:
        alg_dir (Path): Path to algorithm directory (e.g., /tasks/cycle_check/ba/train)
        alg_name (str): Algorithm name (e.g., 'ba')
        
    Returns:
        list: List of tuples (graph_id, label)
    """
    results = []
    failed_count = 0
    
    # Find all JSON files in the directory
    json_files = list(alg_dir.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {alg_dir}")
        return results
    
    print(f"Processing {len(json_files)} JSON files from {alg_name}")
    
    for file_path in sorted(json_files):
        graph_id, label = extract_cycle_label_from_json(file_path)
        if graph_id and label is not None:
            results.append((graph_id, label))
        else:
            failed_count += 1
            if failed_count <= 5:  # Only show first 5 failures
                print(f"  Failed to extract label from: {file_path.name}")
    
    if failed_count > 5:
        print(f"  ... and {failed_count - 5} more failures")
    
    print(f"  Successfully processed: {len(results)} files, Failed: {failed_count} files")
    return results


def process_task_directory(task_dir, output_dir, split='train'):
    """
    Process all algorithm subdirectories within a task directory.
    
    Args:
        task_dir (str): Path to task directory (e.g., /tasks/cycle_check)
        output_dir (str): Directory to save individual algorithm label files
        split (str): Which split to process under each algorithm (e.g., 'train' or 'test')
    """
    task_path = Path(task_dir)
    output_path = Path(output_dir)
    
    if not task_path.exists():
        print(f"Error: Task directory {task_dir} does not exist!")
        sys.exit(1)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find algorithm subdirectories (look for the requested split under each algorithm)
    alg_dirs = []
    for item in task_path.iterdir():
        if item.is_dir():
            # Look for specified split subdirectory (train or test)
            split_dir = item / split
            print(f"Looking for split directory: {split_dir}")
            if split_dir.exists() and split_dir.is_dir():
                alg_dirs.append((item.name, split_dir))
    
    if not alg_dirs:
        print(f"No algorithm directories with '{split}' subdirectories found in {task_dir}")
        sys.exit(1)
    
    print(f"Found {len(alg_dirs)} algorithm directories:")
    for alg_name, _ in alg_dirs:
        print(f"  {alg_name}")
    
    # Process each algorithm
    total_results = 0
    for alg_name, split_dir in sorted(alg_dirs):
        print(f"\nProcessing algorithm: {alg_name} (split: {split})")
        
        # Process this algorithm's files
        results = process_algorithm_directory(split_dir, alg_name)
        
        if results:
            # Save results for this algorithm (include split in filename)
            output_file = output_path / f"{alg_name}.txt"
            with open(output_file, 'w') as f:
                for graph_id, label in results:
                    f.write(f"{graph_id} {label}\n")
            
            print(f"  Saved {len(results)} labels to {output_file}")
            total_results += len(results)
            
            # Show label distribution for this algorithm
            yes_count = sum(1 for _, label in results if label == 1)
            no_count = sum(1 for _, label in results if label == 0)
            print(f"  Label distribution - '1' (has cycle): {yes_count}, '0' (no cycle): {no_count}")
        else:
            print(f"  No valid results for {alg_name}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total labels extracted: {total_results}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    """Extract cycle labels from all algorithm directories within a task."""
    parser = argparse.ArgumentParser(description='Extract labels from all algorithms in a task directory')
    parser.add_argument('task_dir', help='Task directory path (e.g., /path/to/tasks/cycle_check)')
    parser.add_argument('output_dir', help='Output directory for algorithm label files')
    parser.add_argument('--split', choices=['train', 'test'], default='train', help="Which split to process under each algorithm (default: train)")
    
    args = parser.parse_args()
    
    process_task_directory(args.task_dir, args.output_dir, args.split)


if __name__ == "__main__":
    main()