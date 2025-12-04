#!/usr/bin/env python3
"""
Batch GraphML Tokenizer
Processes all GraphML files in a directory and saves tokenized sequences to a text file.
Each line contains: filename,tokenized_sequence
"""

import os
import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
import argparse
from pathlib import Path
from tqdm import tqdm

# Import AutoGraph tokenizer
from AutoGraph.autograph.datamodules.data.tokenizer import Graph2TrailTokenizer


def read_graphml_file(file_path):
    """
    Read a GraphML file and convert it to PyTorch Geometric Data format.
    
    Args:
        file_path (str): Path to the GraphML file
        
    Returns:
        tuple: (data, is_directed) or (None, None) if error
    """
    try:
        # Read the GraphML file using NetworkX
        G = nx.read_graphml(file_path)
        
        # Convert string node IDs to integers if needed
        if not all(isinstance(node, int) for node in G.nodes()):
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
        
        # Convert to PyTorch Geometric format
        data = from_networkx(G)
        
        # Ensure we have node count
        if not hasattr(data, 'num_nodes') or data.num_nodes is None:
            data.num_nodes = G.number_of_nodes()
        
        return data, G.is_directed()
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None


def tokenize_graph(data, is_directed=False, max_length=1000):
    """
    Tokenize a graph using AutoGraph tokenizer.
    
    Args:
        data: PyTorch Geometric Data object
        is_directed (bool): Whether the graph is directed
        max_length (int): Maximum sequence length
        
    Returns:
        torch.Tensor or None: Tokenized sequence
    """
    try:
        # Initialize tokenizer for unlabeled graphs
        tokenizer = Graph2TrailTokenizer(
            labeled_graph=False,
            undirected=not is_directed,
            max_length=max_length,
            append_eos=True
        )
        
        # Set the maximum number of nodes
        tokenizer.set_num_nodes(data.num_nodes)
        
        # Tokenize the graph
        tokenized_sequence = tokenizer.tokenize(data)
        return tokenized_sequence
        
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return None


def process_directory(input_dir, output_file, max_length=1000):
    """
    Process all GraphML files in a directory and save tokenized sequences.
    
    Args:
        input_dir (str): Directory containing GraphML files
        output_file (str): Output text file path
        max_length (int): Maximum sequence length for tokenization
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Directory {input_dir} does not exist!")
        return
    
    # Find all GraphML files
    graphml_files = list(input_path.glob("*.graphml"))
    if not graphml_files:
        print(f"No GraphML files found in {input_dir}")
        return
    
    print(f"Found {len(graphml_files)} GraphML files")
    print(f"Output will be saved to: {output_file}")
    
    successful_count = 0
    failed_files = []
    
    # Process files and write to output
    with open(output_file, 'w') as f:
        for file_path in tqdm(graphml_files, desc="Processing files"):
            try:
                # Read GraphML file
                data, is_directed = read_graphml_file(file_path)
                if data is None:
                    failed_files.append(str(file_path))
                    continue
                
                # Tokenize
                tokenized_sequence = tokenize_graph(data, is_directed, max_length)
                if tokenized_sequence is None:
                    failed_files.append(str(file_path))
                    continue
                
                # Write to file: filename,token1 token2 token3 ...
                filename = file_path.name.replace('.graphml', '')
                tokens_str = ' '.join(map(str, tokenized_sequence.tolist()))
                f.write(f"{filename} {tokens_str}\n")
                
                successful_count += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                failed_files.append(str(file_path))
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count}/{len(graphml_files)} files")
    
    if failed_files:
        print(f"Failed files ({len(failed_files)}):")
        for failed_file in failed_files[:10]:  # Show first 10 failed files
            print(f"  {failed_file}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    print(f"Results saved to: {output_file}")


def process_subdirectories(input_dir, output_dir, max_length=1000):
    """
    Process GraphML files in subdirectories and create separate output files.
    
    Args:
        input_dir (str): Root directory containing subdirectories with GraphML files
        output_dir (str): Directory to save output files
        max_length (int): Maximum sequence length for tokenization
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Directory {input_dir} does not exist!")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find subdirectories containing GraphML files
    subdirs_with_graphml = []
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            graphml_files = list(subdir.glob("*.graphml"))
            if graphml_files:
                subdirs_with_graphml.append((subdir, len(graphml_files)))
    
    if not subdirs_with_graphml:
        print(f"No subdirectories with GraphML files found in {input_dir}")
        return
    
    print(f"Found {len(subdirs_with_graphml)} subdirectories with GraphML files:")
    for subdir, count in subdirs_with_graphml:
        print(f"  {subdir.name}: {count} files")
    
    # Process each subdirectory
    for subdir, file_count in subdirs_with_graphml:
        print(f"\nProcessing subdirectory: {subdir.name}")
        output_file = output_path / f"{subdir.name}_tokens.txt"
        process_directory(str(subdir), str(output_file), max_length)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Batch tokenize GraphML files using AutoGraph')
    parser.add_argument('input_dir', help='Directory containing GraphML files or subdirectories')
    parser.add_argument('output', help='Output file path (for single dir) or output directory (for subdirs)')
    parser.add_argument('--max-length', type=int, default=1000, 
                       help='Maximum sequence length (default: 1000)')
    parser.add_argument('--subdirs', action='store_true',
                       help='Process subdirectories separately (each subdir gets its own output file)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    if args.subdirs:
        process_subdirectories(args.input_dir, args.output, args.max_length)
    else:
        process_directory(args.input_dir, args.output, args.max_length)


if __name__ == "__main__":
    main()