#!/usr/bin/env python3
"""
Utilities for labeling GraphML graphs with single-source shortest path distances.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional
import xml.etree.ElementTree as ET

GRAPHML_NS = "{http://graphml.graphdrawing.org/xmlns}"


def label_shortest_paths(
    graphml_path: str, json_output_path: str, seed: Optional[int] = None
) -> Dict[str, object]:
    """
    Load a GraphML file, pick a random node, and save the node plus distances to JSON.
    The JSON schema is:
        {
            "target_node": <int | str>,
            "distances": {<node_id>: <distance | null>}
        }

    Parameters
    ----------
    graphml_path
        Path to the GraphML file describing an undirected graph.
    json_output_path
        Destination for the JSON file with the sampled node and shortest paths.
    seed
        Optional random seed for reproducible node selection.

    Returns
    -------
    Dict[str, object]
        The JSON-serializable payload that was written to disk.
    """

    nodes, adjacency = _load_graph(graphml_path)
    if not nodes:
        raise ValueError(f"No nodes found in GraphML file: {graphml_path}")

    rng = random.Random(seed)
    chosen_node = rng.choice(nodes)
    distances = _bfs_shortest_paths(adjacency, chosen_node, nodes)

    payload = {
        "target_node": _coerce_node_id(chosen_node),
        "distances": {
            _coerce_node_id(node): distance for node, distance in distances.items()
        },
    }

    with open(json_output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)

    return payload


def label_shortest_paths_tree(
    graphml_root: str, output_dir: str, seed: Optional[int] = None
) -> None:
    """
    Traverse graphml_root, process every .graphml file, and mirror results under output_dir.
    """

    root_path = Path(graphml_root)
    output_root = Path(output_dir)
    if not root_path.is_dir():
        raise ValueError(f"Expected a directory at {graphml_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    graphml_files = sorted(path for path in root_path.rglob("*.graphml") if path.is_file())
    rng = random.Random(seed)

    for graphml_file in graphml_files:
        relative_path = graphml_file.relative_to(root_path)
        json_path = output_root / relative_path
        json_path = json_path.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)

        per_file_seed = rng.randint(0, 2**63 - 1) if seed is not None else None
        label_shortest_paths(
            graphml_path=str(graphml_file),
            json_output_path=str(json_path),
            seed=per_file_seed,
        )


def _load_graph(graphml_path: str) -> (List[str], Dict[str, List[str]]):
    """Parse the GraphML file and return the node order plus adjacency list."""
    tree = ET.parse(graphml_path)
    root = tree.getroot()
    graph = root.find(f"{GRAPHML_NS}graph")
    if graph is None:
        raise ValueError("GraphML file is missing a <graph> element.")

    node_order: List[str] = []
    adjacency: Dict[str, List[str]] = {}
    for node in graph.findall(f"{GRAPHML_NS}node"):
        node_id = node.attrib.get("id")
        if node_id is None:
            raise ValueError("Encountered a node without an 'id' attribute.")
        if node_id not in adjacency:
            adjacency[node_id] = []
            node_order.append(node_id)

    for edge in graph.findall(f"{GRAPHML_NS}edge"):
        source = edge.attrib.get("source")
        target = edge.attrib.get("target")
        if source is None or target is None:
            raise ValueError("Encountered an edge without 'source' or 'target'.")
        if source not in adjacency:
            adjacency[source] = []
            node_order.append(source)
        if target not in adjacency:
            adjacency[target] = []
            node_order.append(target)
        adjacency[source].append(target)
        adjacency[target].append(source)

    return node_order, adjacency


def _bfs_shortest_paths(
    adjacency: MutableMapping[str, List[str]],
    start_node: str,
    ordered_nodes: List[str],
) -> Dict[str, Optional[int]]:
    """Run an unweighted BFS to get shortest path distances from start_node."""
    distances: Dict[str, Optional[int]] = {node: None for node in adjacency}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if distances[neighbor] is None:
                distances[neighbor] = distances[node] + 1  # type: ignore
                queue.append(neighbor)

    return {node: distances.get(node) for node in ordered_nodes}


def _coerce_node_id(node_id: str):
    """Cast numeric node identifiers to ints when possible for cleaner output."""
    try:
        return int(node_id)
    except (TypeError, ValueError):
        return node_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a GraphML graph with single-source shortest paths."
    )
    parser.add_argument(
        "input_path", help="Path to an input .graphml file or a directory of them."
    )
    parser.add_argument(
        "output_path",
        help="Output JSON file (for a single input) or output directory (for a tree).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed to sample a reproducible node.",
    )
    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if input_path.is_dir():
        if output_path.exists() and not output_path.is_dir():
            raise ValueError("Output path must be a directory when processing directories.")
        label_shortest_paths_tree(str(input_path), str(output_path), seed=args.seed)
    else:
        label_shortest_paths(str(input_path), str(output_path), seed=args.seed)


if __name__ == "__main__":
    main()
