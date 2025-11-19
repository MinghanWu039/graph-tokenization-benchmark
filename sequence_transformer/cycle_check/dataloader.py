"""Data loader for cycle-check tokenized graphs.

File format (per line):
	<graph_index> <tok1> <tok2> ... <tokN> <label>

Where <graph_index> is a unique id (integer), tokens are integers, and
<label> is 0 or 1 (at the end of the line).

This module provides a PyTorch Dataset and helpers to create DataLoaders
for train/validation/test. Validation is created by splitting the train file.
"""

from typing import List, Tuple, Optional, Dict, Any
import random
import os
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset


class AutographDataset(Dataset):
	"""Dataset that reads tokenized graphs from a text file.

	Each sample returned by __getitem__ is a dict with keys:
	  - 'graph_index': int
	  - 'input_ids': List[int]
	  - 'label': int
	  - 'length': int
	"""

	def __init__(self, file_path: str, max_len: Optional[int] = None):
		if not os.path.isfile(file_path):
			raise FileNotFoundError(file_path)

		self.file_path = file_path
		self.max_len = max_len
		self.samples: List[Tuple[int, List[int], int]] = []

		with open(file_path, "r") as fh:
			for lineno, line in enumerate(fh, start=1):
				line = line.strip()
				if not line:
					continue
				parts = line.split()
				if len(parts) < 3:
					# need at least graph_index, one token, and label
					raise ValueError(f"Bad line {lineno} in {file_path}: '{line}'")
				try:
					graph_index = parts[0].strip()
					label = int(parts[-1])
					seq = [int(x) for x in parts[1:-1]]
				except ValueError as e:
					raise ValueError(f"Failed to parse line {lineno} in {file_path}: {e}")
				self.samples.append((graph_index, seq, label))

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		graph_index, seq, label = self.samples[idx]
		if self.max_len is not None and len(seq) > self.max_len:
			seq = seq[: self.max_len]
		return {"graph_index": graph_index, "input_ids": seq, "label": label, "length": len(seq)}

	def collate_fn(self, batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
		"""Collate function to produce padded tensors.

		Returns a dict with keys:
		  - input_ids: LongTensor (batch, seq_len)
		  - attention_mask: LongTensor (batch, seq_len)
		  - labels: LongTensor (batch,)
		  - graph_index: LongTensor (batch,)
		  - lengths: LongTensor (batch,)
		"""

		# determine target length
		batch_lens = [sample["length"] for sample in batch]
		if self.max_len is not None:
			tgt_len = min(max(batch_lens), self.max_len)
		else:
			tgt_len = max(batch_lens)

		input_ids = []
		attention_mask = []
		labels = []
		graph_indices = []

		for sample in batch:
			seq = sample["input_ids"]
			# truncate if needed
			if len(seq) > tgt_len:
				seq = seq[:tgt_len]
			# pad on the right
			pad_len = tgt_len - len(seq)
			input_ids.append(seq + [pad_token_id] * pad_len)
			attention_mask.append([1] * len(seq) + [0] * pad_len)
			labels.append(sample["label"])  # 0/1
			graph_indices.append(sample["graph_index"])

		return {
			"input_ids": torch.tensor(input_ids, dtype=torch.long),
			"attention_mask": torch.tensor(attention_mask, dtype=torch.long),
			"labels": torch.tensor(labels, dtype=torch.long),
			"graph_index": torch.tensor(graph_indices, dtype=torch.long),
			"lengths": torch.tensor(batch_lens, dtype=torch.long),
		}


class GraphTokenDataset(Dataset):
	"""Dataset that aggregates JSON files from all algorithms for a split."""

	def __init__(self, root_dir: str, split: str = "train"):
		self.root_dir = Path(root_dir)
		if split not in {"train", "test"}:
			raise ValueError("split must be 'train' or 'test'")
		if not self.root_dir.is_dir():
			raise FileNotFoundError(root_dir)

		self.split = split
		self.samples: List[Dict[str, Any]] = []
		self._load_all_json()

	def _load_all_json(self) -> None:
		for algo_dir in sorted(self.root_dir.iterdir()):
			if not algo_dir.is_dir():
				continue
			split_dir = algo_dir / self.split
			if not split_dir.is_dir():
				continue
			for json_file in sorted(split_dir.glob("*.json")):
				with open(json_file, "r") as fh:
					try:
						records = json.load(fh)
					except json.JSONDecodeError as exc:
						raise ValueError(f"Failed to parse {json_file}: {exc}") from exc
				if not isinstance(records, list):
					raise ValueError(f"Expected list in {json_file}")
				for entry in records:
					if not isinstance(entry, dict):
						continue
					graph_id = entry.get("graph_id")
					text = entry.get("text")
					if graph_id is None or text is None:
						continue
					tokens = text.strip().split()
					if len(tokens) < 2 or tokens[-1] != "<eos>":
						continue
					if len(tokens) < 2:
						continue
					label_token = tokens[-2]
					label_map = {"no": 0, "yes": 1}
					if label_token not in label_map:
						continue
					label = label_map[label_token]
					tokens.pop(-2)
					text_wo_label = " ".join(tokens)
					if not text_wo_label:
						continue
					self.samples.append(
						{
							"graph_id": graph_id,
							"text": text_wo_label,
							"length": len(tokens),
							"label": label,
						}
					)
		if not self.samples:
			raise ValueError(f"No samples found in {self.root_dir} for split '{self.split}'")

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		return self.samples[idx]

	def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
		return {
			"graph_id": [sample["graph_id"] for sample in batch],
			"text": [sample["text"] for sample in batch],
			"lengths": [sample["length"] for sample in batch],
			"labels": torch.tensor([sample["label"] for sample in batch], dtype=torch.long),
		}


def split_dataset(dataset: Dataset, val_fraction: float, seed: int = 42) -> Tuple[Subset, Subset]:
	"""Split a dataset into train and validation Subsets."""
	if not 0.0 < val_fraction < 1.0:
		raise ValueError("val_fraction must be between 0 and 1")

	n = len(dataset)
	indices = list(range(n))
	random.Random(seed).shuffle(indices)
	split = int(n * (1.0 - val_fraction))
	train_idx = indices[:split]
	val_idx = indices[split:]
	return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_dataloaders(
	train_file: str,
	test_file: Optional[str] = None,
	batch_size: int = 32,
	val_split: float = 0.0,
	shuffle: bool = True,
	num_workers: int = 0,
	max_len: Optional[int] = None,
	pad_token_id: int = 0,
	seed: int = 42,
) -> Dict[str, DataLoader]:
	"""Create dataloaders for train/(val)/test.

	Args:
		train_file: path to the train file (will be used to create train and optional val)
		test_file: optional path to a test file (keeps separate)
		batch_size: batch size
		val_split: fraction of train to reserve for validation (0.0 disables val)
		shuffle: whether to shuffle the training data
		num_workers: DataLoader num_workers
		max_len: maximum sequence length (truncate longer sequences)
		pad_token_id: token id used for padding
		seed: random seed for splitting

	Returns:
		dict with keys 'train', optionally 'val', optionally 'test' mapping to DataLoaders
	"""

	train_ds = AutographDataset(train_file, max_len=max_len)

	if val_split and val_split > 0.0:
		train_subset, val_subset = split_dataset(train_ds, val_split, seed=seed)
	else:
		train_subset = train_ds
		val_subset = None

	# Use the dataset instance's collate_fn when possible; if train_subset is a Subset,
	# we need to access the underlying dataset to get collate_fn. We'll use train_ds.collate_fn.
	collate = lambda batch: train_ds.collate_fn(batch, pad_token_id=pad_token_id)

	train_loader = DataLoader(
		train_subset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate,
	)

	loaders = {"train": train_loader}

	if val_subset is not None:
		val_loader = DataLoader(
			val_subset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			collate_fn=collate,
		)
		loaders["val"] = val_loader

	if test_file is not None:
		test_ds = AutographDataset(test_file, max_len=max_len)
		test_loader = DataLoader(
			test_ds,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			collate_fn=lambda batch: test_ds.collate_fn(batch, pad_token_id=pad_token_id),
		)
		loaders["test"] = test_loader

	return loaders


__all__ = ["AutographDataset", "GraphTokenDataset", "get_dataloaders", "split_dataset"]
