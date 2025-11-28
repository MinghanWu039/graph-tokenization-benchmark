#!/usr/bin/env python3
"""
Build aggregated shortest-path datasets directly from cycle-check sequences.
Each output line has the form:
    <index> <graph_sequence_without_cycle_label> <start end> <distance>
Labels are read from a separate directory containing per-graph shortest-path JSON files.
"""

import argparse
import json
import re
import sys
from pathlib import Path

LABEL_PATTERN = re.compile(r'\s*<q>\s*has_cycle\s*<p>\s*(yes|no)\s*', re.IGNORECASE)


def strip_cycle_label(text):
    """Remove the '<q> has_cycle <p> ...' snippet and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    stripped, _ = LABEL_PATTERN.subn(' ', text)
    return ' '.join(stripped.split()).strip()


def load_sequence_entries(file_path):
    """Load the cycle-check JSON file as a list of dict entries."""
    try:
        with open(file_path, 'r') as handle:
            data = json.load(handle)
    except Exception as exc:
        print(f"Error reading {file_path}: {exc}")
        return []

    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    if isinstance(data, dict):
        return [data]

    print(f"Warning: unexpected JSON payload in {file_path}")
    return []


def parse_graph_index(graph_file):
    """Extract the numeric suffix from a filename such as ba_train_500.json."""
    for part in reversed(graph_file.stem.split('_')):
        if part.isdigit():
            return int(part)
    return None


def find_split_start_index(split_dir):
    """Find the smallest index for all graph files in a split directory."""
    min_index = None
    for graph_file in split_dir.glob("*.json"):
        idx = parse_graph_index(graph_file)
        if idx is None:
            continue
        if min_index is None or idx < min_index:
            min_index = idx
    return min_index


def load_label_payload(label_file):
    """Load shortest-path label information from JSON."""
    try:
        with open(label_file, 'r') as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"Error reading label file {label_file}: {exc}")
        return None

    if not isinstance(payload, dict):
        print(f"Warning: label file {label_file} does not contain a JSON object")
        return None

    target = payload.get('target_node')
    distances = payload.get('distances')
    if target is None or not isinstance(distances, dict):
        print(f"Warning: label file {label_file} missing 'target_node' or 'distances'")
        return None

    return target, distances


def extract_index_token(entry, fallback):
    """Derive the identifier prefix for a graph (numeric suffix preferred)."""
    graph_id = entry.get('graph_id')
    candidates = [graph_id, fallback.name, fallback.stem]
    for candidate in candidates:
        if not candidate:
            continue
        parts = str(candidate).split('_')
        for part in reversed(parts):
            if part.isdigit():
                return part
    return fallback.stem


def label_to_lines(index_token, cleaned_text, start_node, distances):
    """Generate dataset lines for every destination node."""
    lines = []
    start_token = str(start_node)
    def sort_key(item):
        node_id_str, _ = item
        try:
            return int(node_id_str)
        except ValueError:
            return node_id_str

    for node_id_str, distance in sorted(distances.items(), key=sort_key):
        try:
            end_node = int(node_id_str)
        except ValueError:
            continue
        if distance is None:
            label_value = -1
        else:
            try:
                label_value = int(distance)
            except (TypeError, ValueError):
                label_value = -1
        lines.append(f"{index_token} {cleaned_text} <{start_token} {end_node}> {label_value}")
    return lines


def process_graph_file(graph_file, label_dir, base_index):
    """Convert a single graph JSON file into dataset lines using its label file."""
    graph_index = parse_graph_index(graph_file)
    if graph_index is None or base_index is None:
        print(f"Warning: unable to determine index for {graph_file}")
        return [], 0

    label_idx = graph_index - base_index
    if label_idx < 0:
        print(f"Warning: negative label index for {graph_file}")
        return [], 0

    label_file = label_dir / f"{label_idx}.json"
    if not label_file.exists():
        print(f"Warning: missing label file {label_file}")
        return [], 0

    payload = load_label_payload(label_file)
    if not payload:
        return [], 0

    start_node, distances = payload
    entries = load_sequence_entries(graph_file)
    output_lines = []
    graphs_with_lines = 0

    for entry in entries:
        cleaned_text = strip_cycle_label(entry.get('text', ''))
        if not cleaned_text:
            print(f"Warning: empty cleaned text for {graph_file}")
            continue
        index_token = extract_index_token(entry, graph_file)
        lines = label_to_lines(index_token, cleaned_text, start_node, distances)
        if lines:
            graphs_with_lines += 1
            output_lines.extend(lines)

    return output_lines, graphs_with_lines


def process_split(sequence_root, label_root, split_name, output_path):
    """Aggregate every graph for a split into a single text output."""
    total_graphs = 0
    total_lines = 0

    with open(output_path, 'w') as outfile:
        for alg_dir in sorted(sequence_root.iterdir()):
            if not alg_dir.is_dir():
                continue

            split_dir = alg_dir / split_name
            if not split_dir.is_dir():
                continue

            label_alg_dir = label_root / alg_dir.name / split_name
            if not label_alg_dir.is_dir():
                print(f"Warning: missing label directory {label_alg_dir}, skipping algorithm {alg_dir.name}")
                continue

            base_index = find_split_start_index(split_dir)
            if base_index is None:
                print(f"Warning: unable to determine base index for {split_dir}")
                continue

            for graph_file in sorted(split_dir.glob("*.json")):
                lines, graph_count = process_graph_file(graph_file, label_alg_dir, base_index)
                for line in lines:
                    outfile.write(line + "\n")
                total_graphs += graph_count
                total_lines += len(lines)

    return total_graphs, total_lines


def create_output_files(sequence_dir, label_dir, output_dir):
    """Create train.txt and test.txt by combining cycle-check sequences with shortest-path labels."""
    seq_path = Path(sequence_dir)
    label_path = Path(label_dir)
    out_path = Path(output_dir)

    for path, desc in [(seq_path, "sequence"), (label_path, "label")]:
        if not path.exists():
            print(f"Error: {desc} directory {path} does not exist")
            sys.exit(1)

    out_path.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ('train', 'test'):
        output_file = out_path / f"{split}.txt"
        graphs, lines = process_split(seq_path, label_path, split, output_file)
        summary[split] = (graphs, lines)
        print(f"Wrote {lines} lines from {graphs} graphs to {output_file}")

    print("\n" + "=" * 60)
    print("Finished constructing shortest-path datasets from cycle-check sequences")
    for split, (graphs, lines) in summary.items():
        print(f"{split.capitalize()}: {lines} lines from {graphs} graphs")
    print(f"Outputs stored in: {out_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Combine cycle-check graph sequences with shortest-path labels into aggregated text datasets.",
    )
    parser.add_argument('cycle_check_sequence_dir', help='Cycle-check sequence root (e.g., graph_token_outputs/tasks/cycle_check)')
    parser.add_argument('label_dir', help='Shortest-path label root (e.g., graph_token_outputs/shortest_path_labels)')
    parser.add_argument('output_dir', help='Directory for aggregated train.txt/test.txt files')

    args = parser.parse_args()
    create_output_files(args.cycle_check_sequence_dir, args.label_dir, args.output_dir)


if __name__ == "__main__":
    main()
