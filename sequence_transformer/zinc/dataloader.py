#!/usr/bin/env python3
"""Utilities to load tokenized ZINC sequences into PyTorch Geometric."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

Token = Union[str, int]

__all__ = ["TokenVocabulary", "ZincTokenDataset", "create_zinc_dataloaders", "load_token_records"]


def load_token_records(jsonl_path: Union[str, Path]) -> List[Dict[str, Union[str, float, List[Token]]]]:
    """Read JSONL tokenization output into memory."""
    records: List[Dict[str, Union[str, float, List[Token]]]] = []
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(jsonl_path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse line {lineno} of {jsonl_path}: {exc}") from exc
            required = {"split", "graph_id", "tokens"}
            if not required.issubset(record):
                missing = ",".join(sorted(required - set(record)))
                raise ValueError(f"Line {lineno} missing keys: {missing}")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {jsonl_path}")
    return records


def infer_feature_dim(records: Iterable[Dict]) -> int:
    """Infer one-hot feature dimensionality from token records."""
    max_idx = -1
    for record in records:
        rows = record.get("node_features")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, list) and row:
                val = row[0]
            elif isinstance(row, (int, float)):
                val = row
            else:
                continue
            try:
                idx = int(val)
            except (TypeError, ValueError):
                continue
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0


def _token_to_index(token: Token) -> Optional[int]:
    if isinstance(token, int):
        return token
    try:
        return int(token)
    except (TypeError, ValueError):
        return None


def build_one_hot_matrix(record: Dict, feature_dim: int) -> List[List[float]]:
    rows = record.get("node_features")
    if not isinstance(rows, list) or feature_dim <= 0:
        return []
    matrix: List[List[float]] = []
    for row in rows:
        if isinstance(row, list) and row:
            val = row[0]
        elif isinstance(row, (int, float)):
            val = row
        else:
            val = 0
        try:
            idx = int(val)
        except (TypeError, ValueError):
            idx = 0
        vec = [0.0] * feature_dim
        if 0 <= idx < feature_dim:
            vec[idx] = 1.0
        matrix.append(vec)
    return matrix


def build_token_feature_sequence(
    tokens: Sequence[Token],
    node_matrix: Sequence[Sequence[float]],
    feature_dim: int,
) -> List[List[float]]:
    if feature_dim <= 0:
        return [[] for _ in tokens]
    zero = [0.0] * feature_dim
    sequence: List[List[float]] = []
    for token in tokens:
        idx = _token_to_index(token)
        if idx is not None and 0 <= idx < len(node_matrix):
            vec = node_matrix[idx]
            if len(vec) != feature_dim:
                vec = list(vec) + [0.0] * (feature_dim - len(vec))
                vec = vec[:feature_dim]
            sequence.append(list(vec))
        else:
            sequence.append(list(zero))
    return sequence


@dataclass
class TokenVocabulary:
    """Vocabulary that maps string tokens to integer ids."""

    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    def __post_init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        for tok in (self.pad_token, self.unk_token):
            if tok not in self.token_to_id:
                self._add_token(tok)

    def _add_token(self, token: str) -> int:
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def _normalize(self, token: Token) -> str:
        if isinstance(token, str):
            return token
        return str(token)

    def add_sequence(self, tokens: Sequence[Token]) -> None:
        for token in tokens:
            normalized = self._normalize(token)
            if normalized not in self.token_to_id:
                self._add_token(normalized)

    def encode(self, tokens: Sequence[Token]) -> List[int]:
        encoded: List[int] = []
        for token in tokens:
            normalized = self._normalize(token)
            encoded.append(self.token_to_id.get(normalized, self.unk_id))
        return encoded

    @classmethod
    def build_from_records(
        cls,
        records: Iterable[Dict[str, Union[str, float, List[Token]]]],
        split: str = "train",
    ) -> "TokenVocabulary":
        vocab = cls()
        for record in records:
            if record.get("split") != split:
                continue
            tokens = record.get("tokens")
            if isinstance(tokens, list):
                vocab.add_sequence(tokens)
        return vocab


class ZincTokenDataset(Dataset):
    """PyG Dataset backed by the JSONL output of zinc_tokenization.py."""

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        split: Optional[str] = None,
        *,
        records: Optional[List[Dict[str, Union[str, float, List[Token]]]]] = None,
        vocab: Optional[TokenVocabulary] = None,
    ) -> None:
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        all_records = records or load_token_records(self.jsonl_path)
        if split is not None:
            allowed = {"train", "val", "test"}
            if split not in allowed:
                raise ValueError(f"split must be one of {allowed}")
            self.records = [rec for rec in all_records if rec.get("split") == split]
        else:
            self.records = list(all_records)
        if not self.records:
            raise ValueError(f"No samples found for split '{split}' in {self.jsonl_path}")

        if vocab is None:
            vocab = TokenVocabulary.build_from_records(self.records)
        self.vocab = vocab
        self.feature_dim = infer_feature_dim(self.records)
        self._encoded_sequences: List[List[int]] = []
        self._feature_sequences: List[List[List[float]]] = []
        for record in self.records:
            tokens = record.get("tokens")
            if not isinstance(tokens, list):
                raise ValueError(f"Record {record.get('graph_id')} missing token list.")
            encoded_tokens = self.vocab.encode(tokens)
            self._encoded_sequences.append(encoded_tokens)
            node_matrix = build_one_hot_matrix(record, self.feature_dim)
            token_features = build_token_feature_sequence(tokens, node_matrix, self.feature_dim)
            self._feature_sequences.append(token_features)

    def len(self) -> int:
        return len(self.records)

    def get(self, idx: int) -> Data:
        record = self.records[idx]
        seq = self._encoded_sequences[idx]
        tokens = torch.tensor(seq, dtype=torch.long)
        feat_seq = self._feature_sequences[idx]
        if self.feature_dim > 0:
            feature_tensor = torch.tensor(feat_seq, dtype=torch.float)
        else:
            feature_tensor = torch.zeros((tokens.size(0), 0), dtype=torch.float)
        target = record.get("target")
        if target is None:
            y = torch.tensor([], dtype=torch.float)
        else:
            y = torch.tensor([float(target)], dtype=torch.float)
        data = Data(
            token_ids=tokens,
            token_features=feature_tensor,
            length=torch.tensor(tokens.size(0), dtype=torch.long),
            y=y,
        )
        data.graph_id = record.get("graph_id")
        data.split = record.get("split")
        return data


def create_zinc_dataloaders(
    jsonl_path: Union[str, Path],
    batch_size: int = 32,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> Tuple[Dict[str, DataLoader], TokenVocabulary]:
    """Build PyG DataLoaders for train/val/test splits."""
    records = load_token_records(jsonl_path)
    vocab = TokenVocabulary.build_from_records(records, split="train")
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = ZincTokenDataset(jsonl_path, split=split, records=records, vocab=vocab)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    return loaders, vocab
