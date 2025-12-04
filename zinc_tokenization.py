#!/usr/bin/env python3
"""Tokenize ZINC graphs with GraphTask or AutoGraph tokenization."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
from torch_geometric.utils import to_networkx

try:
    from AutoGraph.autograph.datamodules.data.tokenizer import Graph2TrailTokenizer
except ImportError:
    Graph2TrailTokenizer = None

TokenList = List[Union[str, int]]

_SPLITS: Tuple[str, ...] = ("train", "val", "test")


def build_edge_lookup(data: Data) -> Dict[Tuple[int, int], Optional[List[float]]]:
    """Return mapping from directed edge (u, v) to its feature vector."""
    lookup: Dict[Tuple[int, int], Optional[List[float]]] = {}
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        return lookup
    edge_index = edge_index.detach().cpu()
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr.detach().cpu()
    num_edges = edge_index.size(1)
    for i in range(num_edges):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        if edge_attr is None:
            feat = None
        else:
            attr = edge_attr[i]
            if attr.dim() == 0:
                feat = [float(attr.item())]
            else:
                feat = [float(x) for x in attr.tolist()]
        lookup[(u, v)] = feat
        lookup.setdefault((v, u), feat)
    return lookup


def _try_parse_node(token: Union[str, int]) -> Optional[int]:
    if isinstance(token, int):
        return token
    try:
        return int(token)
    except (TypeError, ValueError):
        return None


def _format_edge_token(feat: Optional[List[float]]) -> str:
    if not feat:
        return "<edge:-1>"
    if len(feat) == 1:
        return f"<edge:{feat[0]:.0f}>"
    values = ",".join(f"{val:.0f}" for val in feat)
    return f"<edge:{values}>"


def insert_edge_tokens(tokens: TokenList, edge_lookup: Dict[Tuple[int, int], Optional[List[float]]]) -> Tuple[TokenList, int]:
    """Insert edge tokens between consecutive node tokens."""
    if not tokens:
        return tokens, 0
    new_tokens: TokenList = []
    missing = 0
    for idx, token in enumerate(tokens):
        new_tokens.append(token)
        node_a = _try_parse_node(token)
        if node_a is None or idx == len(tokens) - 1:
            continue
        node_b = _try_parse_node(tokens[idx + 1])
        if node_b is None:
            continue
        feat = edge_lookup.get((node_a, node_b))
        if feat is None:
            missing += 1
        new_tokens.append(_format_edge_token(feat))
    return new_tokens, missing


def load_base_tokenizer(repo_root: Path) -> Callable[[nx.Graph], List[str]]:
    """Import `_base_tokens` from graph-token/graph_task.py at runtime."""
    graph_task_path = repo_root / "graph-token" / "graph_task.py"
    spec = importlib.util.spec_from_file_location("graph_task_module", graph_task_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module._base_tokens  # type: ignore[attr-defined]


def load_zinc_splits(root: Path, subset_name: str) -> Dict[str, ZINC]:
    """Replicate master_loader preformat: load train/val/test separately."""
    subset_flag = subset_name == "subset"
    return {
        split: ZINC(root=str(root), subset=subset_flag, split=split)
        for split in _SPLITS
    }


def pyg_to_networkx(data: Data) -> nx.Graph:
    """Convert a PyG Data object to a simple NetworkX graph."""
    graph = to_networkx(data, to_undirected=True)
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        graph = nx.Graph(graph)
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def ensure_num_nodes(data: Data) -> int:
    """Ensure data.num_nodes is set and return its value."""
    n = getattr(data, "num_nodes", None)
    if n is None:
        if getattr(data, "x", None) is not None:
            n = data.x.size(0)
        elif getattr(data, "edge_index", None) is not None and data.edge_index.numel() > 0:
            n = int(data.edge_index.max().item()) + 1
        else:
            n = 0
        data.num_nodes = int(n)
    return int(data.num_nodes)


def write_token_records(
    datasets: Dict[str, ZINC],
    tokenize_fn: Callable[[Data], TokenList],
    tokenizer_name: str,
    output_path: Path,
    limit: Optional[int] = None,
) -> int:
    """Tokenize each split and dump JSONL records to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    total_missing_edges = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for split in _SPLITS:
            dataset = datasets[split]
            for idx, data in enumerate(dataset):
                if limit is not None and idx >= limit:
                    break
                tokens = tokenize_fn(data)
                edge_lookup = build_edge_lookup(data)
                tokens, missing_edges = insert_edge_tokens(tokens, edge_lookup)
                total_missing_edges += missing_edges
                node_features = []
                feature_dim = 0
                if getattr(data, "x", None) is not None:
                    feats = data.x.detach().cpu().float()
                    if feats.dim() == 1:
                        feats = feats.unsqueeze(-1)
                    node_features = feats.tolist()
                    feature_dim = feats.size(1)
                record = {
                    "split": split,
                    "index": idx,
                    "graph_id": f"{split}_{idx}",
                    "tokenizer": tokenizer_name,
                    "tokens": tokens,
                    "token_str": " ".join(map(str, tokens)),
                    "target": float(data.y.item()) if getattr(data, "y", None) is not None else None,
                    "node_features": node_features,
                    "feature_dim": int(feature_dim),
                }
                json.dump(record, sink)
                sink.write("\n")
                count += 1
    print(f"Inserted edge tokens; missing edges encountered: {total_missing_edges}")
    return count


def build_graph_task_tokenizer(repo_root: Path) -> Tuple[Callable[[Data], TokenList], str]:
    base_tokenizer = load_base_tokenizer(repo_root)

    def _tokenize(data: Data) -> TokenList:
        graph = pyg_to_networkx(data)
        return base_tokenizer(graph)

    return _tokenize, "graph_task"


def build_autograph_tokenizer(args) -> Tuple[Callable[[Data], TokenList], str]:
    if Graph2TrailTokenizer is None:
        raise ImportError(
            "Graph2TrailTokenizer is unavailable. Install AutoGraph or adjust PYTHONPATH."
        )
    tokenizer = Graph2TrailTokenizer(
        labeled_graph=False,
        undirected=not args.autograph_directed,
        max_length=args.autograph_max_length,
        append_eos=args.autograph_append_eos,
    )

    def _tokenize(data: Data) -> TokenList:
        num_nodes = ensure_num_nodes(data)
        tokenizer.set_num_nodes(num_nodes)
        tokens = tokenizer.tokenize(data)
        if hasattr(tokens, "tolist"):
            return tokens.tolist()
        return list(tokens)

    return _tokenize, "autograph"


def build_tokenizer(args, repo_root: Path) -> Tuple[Callable[[Data], TokenList], str]:
    if args.tokenizer == "graph_task":
        return build_graph_task_tokenizer(repo_root)
    if args.tokenizer == "autograph":
        return build_autograph_tokenizer(args)
    raise ValueError(f"Unsupported tokenizer choice: {args.tokenizer}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_root = repo_root / "GraphGPS" / "datasets" / "ZINC"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Directory where PyG should cache ZINC (default: GraphGPS/datasets/ZINC).",
    )
    parser.add_argument(
        "--subset",
        choices=["subset", "full"],
        default="subset",
        help="Which ZINC variant to load (matches master_loader preformat).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("graph_token_outputs/zinc_tokens.jsonl"),
        help="Path to the JSONL file that will store tokenized samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of graphs tokenized per split.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["graph_task", "autograph"],
        default="graph_task",
        help="Select GraphGPS base tokens or AutoGraph Graph2TrailTokenizer.",
    )
    parser.add_argument(
        "--autograph-max-length",
        type=int,
        default=1024,
        help="Maximum sequence length for Graph2TrailTokenizer (used with --tokenizer autograph).",
    )
    parser.add_argument(
        "--autograph-directed",
        action="store_true",
        help="Treat molecules as directed graphs for Graph2TrailTokenizer (default: undirected).",
    )
    parser.add_argument(
        "--no-autograph-append-eos",
        dest="autograph_append_eos",
        action="store_false",
        help="Disable EOS token when using Graph2TrailTokenizer.",
    )
    parser.set_defaults(autograph_append_eos=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    tokenize_fn, tokenizer_name = build_tokenizer(args, repo_root)
    datasets = load_zinc_splits(args.root.expanduser(), args.subset)
    total = write_token_records(datasets, tokenize_fn, tokenizer_name, args.output, args.limit)
    print(
        f"Wrote {total} tokenized graphs to {args.output} "
        f"(subset={args.subset}, tokenizer={tokenizer_name})."
    )


if __name__ == "__main__":
    main()
