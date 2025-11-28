#!/usr/bin/env python3
"""
Inspect a shortest-path GraphML graph and its generated labels.

This script loads a randomly selected (or user-specified) graph from
`graph_token_outputs/shortest_path_graphs`, combines it with the matching
JSON labels, and prints a quick PyG-style summary showing node features,
start node, and target distances.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("graph_token_outputs/shortest_path_graphs"),
        help="Root directory that stores algorithm/split GraphML trees.",
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("graph_token_outputs/shortest_path_labels"),
        help="Root directory mirroring graph-root but with JSON labels.",
    )
    parser.add_argument(
        "--graph-file",
        type=Path,
        default=None,
        help="Optional explicit .graphml file to inspect. Overrides --algorithm/--split.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="path",
        help="Algorithm subdirectory to sample from (when --graph-file unset).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "test"),
        default="train",
        help="Split subdirectory to sample from (when --graph-file unset).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible random selection.",
    )
    return parser.parse_args()


def canonical_node_id(node_id: Any) -> str:
    return str(node_id)


def build_label_path(graph_path: Path, graph_root: Path, label_root: Path) -> Path:
    rel_path = graph_path.relative_to(graph_root)
    rel_root = rel_path.with_suffix("")
    return label_root / rel_root.with_suffix(".json")


def load_label_payload(label_path: Path) -> Dict[str, Any]:
    with open(label_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if "target_node" not in payload or "distances" not in payload:
        raise ValueError(f"Invalid payload (missing required keys): {label_path}")
    return payload


def build_data(graph_path: Path, label_payload: Dict[str, Any]) -> Data:
    graph = nx.read_graphml(graph_path)
    if graph.is_directed():
        graph = graph.to_undirected()

    node_order = list(graph.nodes())
    node_to_idx = {canonical_node_id(node): idx for idx, node in enumerate(node_order)}

    data = from_networkx(graph)
    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        data.num_nodes = graph.number_of_nodes()

    if getattr(data, "x", None) is None:
        data.x = torch.arange(data.num_nodes, dtype=torch.float).unsqueeze(1)
    else:
        data.x = data.x.to(torch.float)

    start_node_id = canonical_node_id(label_payload["target_node"])
    if start_node_id not in node_to_idx:
        raise ValueError(f"Start node {start_node_id} not present in graph {graph_path}")
    start_idx = node_to_idx[start_node_id]

    distances = label_payload.get("distances", {})
    targets = torch.full((data.num_nodes,), fill_value=-1, dtype=torch.long)
    for node_id, distance in distances.items():
        node_idx = node_to_idx.get(canonical_node_id(node_id))
        if node_idx is None:
            continue
        targets[node_idx] = -1 if distance is None else int(distance)

    start_feature = torch.full((data.num_nodes, 1), 1000.0, dtype=torch.float)
    start_feature[start_idx, 0] = 0.0
    data.x = torch.cat([data.x, start_feature], dim=1)

    data.y = targets
    data.start_node = torch.tensor([start_idx], dtype=torch.long)
    return data


def pick_graph_file(args: argparse.Namespace) -> Path:
    if args.graph_file is not None:
        return args.graph_file

    rng = random.Random(args.seed)
    candidate_dir = args.graph_root / args.algorithm / args.split
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Graph directory not found: {candidate_dir}")

    candidates = sorted(candidate_dir.glob("*.graphml"))
    if not candidates:
        raise FileNotFoundError(f"No .graphml files found in {candidate_dir}")
    return rng.choice(candidates)


def main() -> None:
    args = parse_args()
    graph_path = pick_graph_file(args)
    label_path = build_label_path(graph_path, args.graph_root, args.label_root)
    if not label_path.exists():
        raise FileNotFoundError(f"Matching label file not found: {label_path}")

    label_payload = load_label_payload(label_path)
    data = build_data(graph_path, label_payload)
    start_idx = data.start_node.item()

    print(f"Graph file : {graph_path}")
    print(f"Label file : {label_path}")
    print(f"Nodes/Edges: {data.num_nodes} / {data.edge_index.size(1)}")
    print(f"Feature dim: {data.x.shape}")
    print(f"Start node : idx={start_idx}, feature={data.x[start_idx].tolist()}")

    unreachable = int((data.y < 0).sum())
    max_dist = int(data.y[data.y >= 0].max().item()) if (data.y >= 0).any() else -1
    print(f"Targets    : dtype={data.y.dtype}, max_distance={max_dist}, unreachable={unreachable}")

    print("\nPer-node summary (idx: features -> target):")
    for idx in range(data.num_nodes):
        features = ", ".join(f"{val:.2f}" for val in data.x[idx].tolist())
        print(f"  {idx:03d}: [{features}] -> {int(data.y[idx].item())}")


if __name__ == "__main__":
    main()
