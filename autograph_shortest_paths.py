#!/usr/bin/env python3
"""
Full Autograph shortest-path pipeline that:
1. Tokenizes GraphML files into Autograph sequences using Graph2TrailTokenizer.
2. Augments the sequences with shortest-path labels from JSON files.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import networkx as nx
from torch_geometric.utils import from_networkx

try:
    from AutoGraph.autograph.datamodules.data.tokenizer import Graph2TrailTokenizer
except ImportError as exc:
    raise ImportError("Failed to import Graph2TrailTokenizer from AutoGraph") from exc


def read_graphml(graphml_path: Path):
    """Load GraphML as a PyTorch Geometric Data object."""
    graph = nx.read_graphml(graphml_path)
    if not all(isinstance(node, int) for node in graph.nodes()):
        mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping)
    data = from_networkx(graph)
    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        data.num_nodes = graph.number_of_nodes()
    return data, graph.is_directed()


def tokenize_graph(data, is_directed: bool, max_length: int) -> List[int]:
    tokenizer = Graph2TrailTokenizer(
        labeled_graph=False,
        undirected=not is_directed,
        max_length=max_length,
        append_eos=True,
    )
    tokenizer.set_num_nodes(data.num_nodes)
    tokens = tokenizer.tokenize(data)
    return tokens.tolist()


def tokenize_split(graphml_split_dir: Path, seq_output_path: Path, max_length: int) -> None:
    """Tokenize all GraphML files in a split and write Autograph sequences."""
    seq_output_path.parent.mkdir(parents=True, exist_ok=True)
    graphml_files = sorted(graphml_split_dir.glob("*.graphml"))
    with open(seq_output_path, "w") as outfile:
        for graphml_file in graphml_files:
            try:
                data, is_directed = read_graphml(graphml_file)
                tokens = tokenize_graph(data, is_directed, max_length)
                graph_idx = int(graphml_file.stem.split("_")[-1])
                outfile.write(f"{graph_idx} {' '.join(map(str, tokens))}\n")
            except Exception as exc:
                print(f"[WARN] Failed to tokenize {graphml_file}: {exc}")


def build_lines(graph_idx: int, tokens: List[str], start_node: int, distances: dict) -> Iterable[str]:
    base = f"{graph_idx} {' '.join(tokens)}"
    for node_id_str, distance in sorted(distances.items(), key=lambda kv: int(kv[0])):
        try:
            end_node = int(node_id_str)
        except ValueError:
            continue
        if distance is None:
            label = -1
        else:
            try:
                label = int(distance)
            except (TypeError, ValueError):
                label = -1
        yield f"{base} <{start_node} {end_node}> {label}"


def parse_sequence_line(line: str) -> Tuple[int, List[str]]:
    parts = line.strip().split()
    graph_idx = int(parts[0])
    return graph_idx, parts[1:]


def load_label(label_file: Path):
    with open(label_file, "r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Label file {label_file} must contain a JSON object")
    start_node = payload.get("target_node")
    distances = payload.get("distances")
    if start_node is None or not isinstance(distances, dict):
        raise ValueError(f"Label file {label_file} missing target_node/distances")
    return start_node, distances


def augment_sequences(sequence_path: Path, label_split_dir: Path, output_path: Path) -> Tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_lines = 0
    processed_graphs = 0
    with open(sequence_path, "r") as infile, open(output_path, "w") as outfile:
        for line_no, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                graph_idx, tokens = parse_sequence_line(line)
            except ValueError as exc:
                print(f"[WARN] {sequence_path}:{line_no}: {exc}")
                continue
            label_file = label_split_dir / f"{graph_idx}.json"
            if not label_file.exists():
                print(f"[WARN] Missing label file {label_file}")
                continue
            try:
                start_node, distances = load_label(label_file)
            except ValueError as exc:
                print(f"[WARN] {exc}")
                continue
            augmented_lines = list(build_lines(graph_idx, tokens, start_node, distances))
            if augmented_lines:
                processed_graphs += 1
                for augmented_line in augmented_lines:
                    outfile.write(augmented_line + "\n")
                    written_lines += 1
    return processed_graphs, written_lines


def combine_split_files(split_files: List[Path], combined_path: Path) -> None:
    """Combine per-algorithm files into a single train/test file with prefixed IDs."""
    if not split_files:
        print(f"[WARN] No files to combine for {combined_path.name}")
        return
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as outfile:
        for file_path in sorted(split_files):
            prefix = file_path.stem
            with open(file_path, "r") as infile:
                for line in infile:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    outfile.write(f"{prefix}_{stripped}\n")
    print(f"Combined {len(split_files)} files into {combined_path}")


def run_pipeline(graphml_root: Path, label_root: Path, sequence_root: Path, output_root: Path, max_length: int) -> None:
    total_graphs = 0
    total_lines = 0
    per_split_outputs = {"train": [], "test": []}
    for alg_dir in sorted(graphml_root.glob("*")):
        if not alg_dir.is_dir():
            continue
        alg_name = alg_dir.name
        for split in ("train", "test"):
            graphml_split_dir = alg_dir / split
            if not graphml_split_dir.is_dir():
                continue
            seq_split_dir = sequence_root / split
            seq_split_dir.mkdir(parents=True, exist_ok=True)
            sequence_path = seq_split_dir / f"{alg_name}.txt"
            print(f"\nTokenizing {graphml_split_dir} -> {sequence_path}")
            tokenize_split(graphml_split_dir, sequence_path, max_length)

            label_split_dir = label_root / alg_name / split
            if not label_split_dir.is_dir():
                print(f"[WARN] Missing label dir {label_split_dir}, skipping augmentation")
                continue
            output_split_dir = output_root / split
            output_split_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_split_dir / f"{alg_name}.txt"
            print(f"Augmenting {sequence_path} -> {output_path}")
            graphs, lines = augment_sequences(sequence_path, label_split_dir, output_path)
            per_split_outputs[split].append(output_path)
            total_graphs += graphs
            total_lines += lines

    print("\n" + "=" * 60)
    print("Autograph shortest-path pipeline complete")
    print(f"Total graphs processed: {total_graphs}")
    print(f"Total labeled lines written: {total_lines}")
    print(f"Augmented datasets stored in: {output_root}")
    combine_split_files(per_split_outputs["train"], output_root / "train.txt")
    combine_split_files(per_split_outputs["test"], output_root / "test.txt")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Autograph pipeline from GraphML + labels to labeled sequences")
    parser.add_argument("graphml_root", help="Root directory with GraphML files organized as alg/split/*.graphml")
    parser.add_argument("label_root", help="Root directory with label JSONs (alg/split/*.json)")
    parser.add_argument("output_root", help="Destination for augmented sequences")
    parser.add_argument("--sequence-root", help="Optional directory to cache raw Autograph sequences", default=None)
    parser.add_argument("--max-length", type=int, default=1000, help="Maximum token sequence length")
    return parser.parse_args()


def main():
    args = parse_args()
    graphml_root = Path(args.graphml_root)
    label_root = Path(args.label_root)
    output_root = Path(args.output_root)
    sequence_root = Path(args.sequence_root) if args.sequence_root else (output_root / "autograph_sequences")

    for path, desc in [
        (graphml_root, "GraphML root"),
        (label_root, "Label root"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} {path} does not exist")

    sequence_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    run_pipeline(graphml_root, label_root, sequence_root, output_root, args.max_length)


if __name__ == "__main__":
    main()
