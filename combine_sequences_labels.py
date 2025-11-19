#!/usr/bin/env python3
"""
Combine tokenized sequences and labels into a single dataset file.
Format: {alg}_{idx} token1 token2 ... tokenN label
"""

import argparse
import sys
from pathlib import Path


def load_sequences(sequence_file):
    """
    Load tokenized sequences from file.
    
    Args:
        sequence_file (str): Path to file with format: identifier token1 token2 ...
        
    Returns:
        dict: Dictionary mapping graph_id to token sequence
    """
    sequences = {}
    
    try:
        with open(sequence_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    print(f"Warning: Invalid format in {sequence_file} line {line_num}: {line}")
                    continue
                
                identifier = parts[0]  # e.g., "751"
                tokens = ' '.join(parts[1:])  # Rest are tokens
                sequences[identifier] = tokens
        
        print(f"Loaded {len(sequences)} sequences from {sequence_file}")
        return sequences
        
    except Exception as e:
        print(f"Error loading sequences from {sequence_file}: {e}")
        return {}


def load_labels(label_file):
    """
    Load labels from file.
    
    Args:
        label_file (str): Path to file with format: graph_id label
        
    Returns:
        dict: Dictionary mapping graph_id to label
    """
    labels = {}
    
    try:
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 2:
                    print(f"Warning: Invalid format in {label_file} line {line_num}: {line}")
                    continue
                
                graph_id, label = parts
                labels[graph_id.strip()] = label.strip()
        
        print(f"Loaded {len(labels)} labels from {label_file}")
        return labels
        
    except Exception as e:
        print(f"Error loading labels from {label_file}: {e}")
        return {}


def extract_alg_and_idx(graph_id):
    """
    Extract algorithm name and index from graph_id.
    
    Args:
        graph_id (str): Graph identifier like 'ba_train_751'
        
    Returns:
        tuple: (alg, idx) or (None, None) if parsing fails
    """
    # Handle formats like: ba_train_751, er_test_123, etc.
    parts = graph_id.split('_')
    if len(parts) >= 3:
        alg = parts[0]  # e.g., 'ba', 'er'
        idx = parts[-1]  # e.g., '751', '123'
        return alg, idx
    
    # Fallback: try to split on last underscore
    if '_' in graph_id:
        *alg_parts, idx = graph_id.split('_')
        alg = '_'.join(alg_parts)
        return alg, idx
    
    return None, None


def combine_sequences_and_labels(sequence_file, label_file, output_file, alg_name=None):
    """
    Combine sequences and labels into the required format.
    
    Args:
        sequence_file (str): Path to tokenized sequences file
        label_file (str): Path to labels file
        output_file (str): Path to output combined file
        alg_name (str, optional): Override algorithm name
    """
    # Algorithm-specific offsets for labels
    alg_offsets = {
        'er': 0,
        'ba': 500,
        'sbm': 1000,
        'sfn': 1500,
        'complete': 2000,
        'star': 2500,
        'path': 3000
    }
    
    print(f"Combining data from:")
    print(f"  Sequences: {sequence_file}")
    print(f"  Labels: {label_file}")
    print(f"  Output: {output_file}")
    
    # Load data
    sequences = load_sequences(sequence_file)
    labels = load_labels(label_file)
    
    if not sequences:
        print("Error: No sequences loaded")
        return
    
    if not labels:
        print("Error: No labels loaded")
        return
    
    # Find common graph IDs by matching the numeric part with algorithm offset
    sequence_ids = set(sequences.keys())  # e.g., {"0", "1", "2", ...}
    label_ids = set(labels.keys())       # e.g., {"ba_train_500", "ba_train_501", ...}
    
    # Determine algorithm name for offset calculation
    if alg_name:
        current_alg = alg_name
    else:
        # Extract algorithm name from label IDs
        sample_label = next(iter(label_ids)) if label_ids else ""
        current_alg = sample_label.split('_')[0] if sample_label else ""
    
    offset = alg_offsets.get(current_alg, 0)
    print(f"Using algorithm '{current_alg}' with offset {offset}")
    
    # Create mapping from sequence ID to label ID (accounting for offset)
    sequence_to_label = {}
    for seq_id in sequence_ids:
        # Convert sequence ID to expected label ID with offset
        try:
            seq_num = int(seq_id)
            expected_label_num = seq_num + offset
            
            # Find matching label ID
            for label_id in label_ids:
                label_num = int(label_id.split('_')[-1])
                if label_num == expected_label_num:
                    sequence_to_label[seq_id] = label_id
                    break
        except ValueError:
            continue
    
    common_count = len(sequence_to_label)
    
    print(f"\nData matching:")
    print(f"  Sequences: {len(sequence_ids)} graphs")
    print(f"  Labels: {len(label_ids)} graphs")
    print(f"  Common (with offset): {common_count} graphs")
    
    if common_count == 0:
        print("Error: No matching graph IDs found!")
        print("Sample sequence IDs:", list(sequence_ids)[:5])
        print("Sample label IDs:", list(label_ids)[:5])
        print(f"Expected offset: {offset}")
        return
    
    # Combine data
    combined_data = []
    
    for seq_id in sorted(sequence_to_label.keys(), key=int):
        # Get the corresponding label ID
        label_id = sequence_to_label[seq_id]
        
        if alg_name:
            # Use provided algorithm name
            alg, idx = alg_name, seq_id
        else:
            # Extract from label_id
            alg, idx = extract_alg_and_idx(label_id)
        
        if alg is None or idx is None:
            print(f"Warning: Could not parse algorithm and index from {label_id}")
            continue
        
        identifier = f"{alg}_{seq_id}"  # Use original sequence ID as index
        sequence = sequences[seq_id]    # Use sequence ID for sequences
        label = labels[label_id]        # Use label ID for labels
        
        # Format: identifier token1 token2 ... tokenN label
        combined_line = f"{identifier} {sequence} {label}"
        combined_data.append(combined_line)
    
    # Save combined data
    try:
        with open(output_file, 'w') as f:
            for line in combined_data:
                f.write(line + '\n')
        
        print(f"\nSuccessfully combined {len(combined_data)} graphs")
        print(f"Results saved to: {output_file}")
        
        # Show sample
        if combined_data:
            print(f"\nSample output:")
            for i, line in enumerate(combined_data[:3]):
                # Truncate long sequences for display
                parts = line.split()
                if len(parts) > 10:
                    sample = ' '.join(parts[:5] + ['...'] + parts[-3:])
                else:
                    sample = line
                print(f"  {sample}")
                
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")


def combine_all_algorithms(sequences_dir, labels_dir, output_file="cycle_check.txt"):
    """
    Combine sequences and labels for all algorithms into a single file.
    
    Args:
        sequences_dir (str): Directory containing sequence files
        labels_dir (str): Directory containing label files
        output_file (str): Single output file for all combined data
    """
    sequences_path = Path(sequences_dir)
    labels_path = Path(labels_dir)
    
    # Algorithm-specific offsets for labels
    alg_offsets = {
        'er': 0,
        'ba': 500,
        'sbm': 1000,
        'sfn': 1500,
        'complete': 2000,
        'star': 2500,
        'path': 3000
    }
    
    # Find all sequence files
    sequence_files = list(sequences_path.glob("*.txt"))
    
    print(f"Found {len(sequence_files)} sequence files")
    print(f"Combining all algorithms into: {output_file}")
    
    all_combined_data = []
    
    for seq_file in sorted(sequence_files):
        alg_name = seq_file.stem  # e.g., 'ba' from 'ba.txt'
        label_file = labels_path / f"{alg_name}.txt"
        
        if not label_file.exists():
            print(f"Warning: No label file found for {alg_name} (expected: {label_file})")
            continue
        
        print(f"\nProcessing algorithm: {alg_name}")
        
        # Load data for this algorithm
        sequences = load_sequences(str(seq_file))
        labels = load_labels(str(label_file))
        
        if not sequences or not labels:
            print(f"Skipping {alg_name} due to loading errors")
            continue
        
        # Get offset for this algorithm
        offset = alg_offsets.get(alg_name, 0)
        print(f"Using offset {offset} for {alg_name}")
        
        # Match sequences and labels with offset
        sequence_ids = set(sequences.keys())
        label_ids = set(labels.keys())
        
        sequence_to_label = {}
        for seq_id in sequence_ids:
            try:
                seq_num = int(seq_id)
                expected_label_num = seq_num + offset
                
                # Find matching label ID
                for label_id in label_ids:
                    label_num = int(label_id.split('_')[-1])
                    if label_num == expected_label_num:
                        sequence_to_label[seq_id] = label_id
                        break
            except ValueError:
                continue
        
        print(f"  Matched {len(sequence_to_label)} graphs for {alg_name}")
        
        # Combine data for this algorithm
        for seq_id in sorted(sequence_to_label.keys(), key=int):
            label_id = sequence_to_label[seq_id]
            
            identifier = f"{alg_name}_{seq_id}"
            sequence = sequences[seq_id]
            label = labels[label_id]
            
            # Format: identifier token1 token2 ... tokenN label
            combined_line = f"{identifier} {sequence} {label}"
            all_combined_data.append(combined_line)
    
    # Save all combined data to single file
    try:
        with open(output_file, 'w') as f:
            for line in all_combined_data:
                f.write(line + '\n')
        
        print(f"\n{'='*60}")
        print(f"Successfully combined all algorithms!")
        print(f"Total graphs: {len(all_combined_data)}")
        
        # Count by algorithm
        alg_counts = {}
        for line in all_combined_data:
            alg = line.split('_')[0]
            alg_counts[alg] = alg_counts.get(alg, 0) + 1
        
        print(f"Breakdown by algorithm:")
        for alg, count in sorted(alg_counts.items()):
            print(f"  {alg}: {count} graphs")
        
        print(f"Results saved to: {output_file}")
        
        # Show sample
        if all_combined_data:
            print(f"\nSample output:")
            for i, line in enumerate(all_combined_data[:5]):
                parts = line.split()
                if len(parts) > 10:
                    sample = ' '.join(parts[:5] + ['...'] + parts[-3:])
                else:
                    sample = line
                print(f"  {sample}")
        
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Combine tokenized sequences and labels')
    parser.add_argument('--sequences', help='Tokenized sequences file')
    parser.add_argument('--labels', help='Labels file')
    parser.add_argument('--output', help='Output combined file')
    parser.add_argument('--alg-name', help='Algorithm name override')
    parser.add_argument('--batch', action='store_true', 
                       help='Batch process all files into a single output file')
    parser.add_argument('--sequences-dir', default='autograph_sequences',
                       help='Directory containing sequence files (for batch mode)')
    parser.add_argument('--labels-dir', default='cycle_check_labels',
                       help='Directory containing label files (for batch mode)')
    parser.add_argument('--output-file', default='cycle_check.txt',
                       help='Output file name (for batch mode)')
    
    args = parser.parse_args()
    
    if args.batch:
        combine_all_algorithms(args.sequences_dir, args.labels_dir, args.output_file)
    else:
        if not all([args.sequences, args.labels, args.output]):
            print("Error: --sequences, --labels, and --output are required for single file mode")
            sys.exit(1)
        
        combine_sequences_and_labels(args.sequences, args.labels, args.output, args.alg_name)


if __name__ == "__main__":
    main()