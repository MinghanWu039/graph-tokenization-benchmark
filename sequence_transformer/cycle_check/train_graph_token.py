#!/usr/bin/env python3
"""Train a transformer classifier on GraphTokenDataset cycle-check sequences."""

import argparse
import os
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
	from .dataloader import GraphTokenDataset, split_dataset
	from .model import CycleCheckTransformer
except ImportError:
	from dataloader import GraphTokenDataset, split_dataset
	from model import CycleCheckTransformer


# -----------------------------
# Utilities
# -----------------------------

class Vocab:
	"""Simple whitespace tokenizer vocabulary."""

	def __init__(self, tokens: Sequence[str]):
		self.pad_token = "<pad>"
		self.unk_token = "<unk>"
		self.stoi = {self.pad_token: 0, self.unk_token: 1}
		for tok in tokens:
			if tok not in self.stoi:
				self.stoi[tok] = len(self.stoi)
		self.itos = {idx: tok for tok, idx in self.stoi.items()}

	@property
	def pad_id(self) -> int:
		return self.stoi[self.pad_token]

	@property
	def unk_id(self) -> int:
		return self.stoi[self.unk_token]

	def encode(self, text: str) -> List[int]:
		return [self.stoi.get(tok, self.unk_id) for tok in text.split()]

	def __len__(self) -> int:
		return len(self.stoi)


def build_vocab(dataset: Dataset) -> Vocab:
	tokens = set()
	for sample in dataset:
		tokens.update(sample["text"].split())
	return Vocab(sorted(tokens))


def make_collate_fn(vocab: Vocab, max_len: int):
	def collate(batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
		sequences = []
		for sample in batch:
			ids = vocab.encode(sample["text"])
			if max_len > 0:
				ids = ids[:max_len]
			sequences.append(ids)
		target_len = max(len(seq) for seq in sequences)
		if max_len > 0:
			target_len = min(target_len, max_len)

		input_ids = []
		attention_mask = []
		for seq in sequences:
			if len(seq) < target_len:
				pad = [vocab.pad_id] * (target_len - len(seq))
				seq = seq + pad
			else:
				seq = seq[:target_len]
			input_ids.append(seq)
			attention_mask.append([1] * len(seq))
		input_ids = torch.tensor(input_ids, dtype=torch.long)
		attention_mask = torch.tensor(attention_mask, dtype=torch.long)
		labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
		return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

	return collate


# -----------------------------
# Training / Evaluation
# -----------------------------

def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def run_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	optimizer: torch.optim.Optimizer = None,
	collect_preds: bool = False,
) -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray]]:
	training = optimizer is not None
	model.train(training)
	total_loss = 0.0
	total_correct = 0
	total_examples = 0
	preds_buffer = [] if collect_preds else None
	labels_buffer = [] if collect_preds else None
	for batch in loader:
		input_ids = batch["input_ids"].to(device)
		attention_mask = batch["attention_mask"].to(device)
		labels = batch["labels"].to(device)

		logits = model(input_ids, attention_mask)
		loss = criterion(logits, labels)

		if training:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			preds = logits.argmax(dim=-1)
			total_correct += (preds == labels).sum().item()
			if collect_preds:
				preds_buffer.append(preds.detach().cpu())
				labels_buffer.append(labels.detach().cpu())
		total_loss += loss.item() * input_ids.size(0)
		total_examples += input_ids.size(0)

	metrics = {
		"loss": total_loss / total_examples if total_examples else 0.0,
		"accuracy": total_correct / total_examples if total_examples else 0.0,
	}
	if collect_preds:
		if preds_buffer:
			all_preds = torch.cat(preds_buffer).numpy()
			all_labels = torch.cat(labels_buffer).numpy()
		else:
			all_preds = np.array([])
			all_labels = np.array([])
		return metrics, (all_preds, all_labels)
	return metrics, (np.array([]), np.array([]))


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train transformer on GraphTokenDataset.")
	parser.add_argument("--data-root", type=str, required=True, help="Root directory of the dataset.")
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--num-heads", type=int, default=4)
	parser.add_argument("--hidden-dim", type=int, default=32)
	parser.add_argument("--num-layers", type=int, default=6)
	parser.add_argument("--mlp-dim", type=int, default=32, help="Feedforward dimension inside Transformer blocks.")
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--max-len", type=int, default=1024, help="Maximum input length (tokens >= max_len are truncated).")
	parser.add_argument(
		"--pos-emb",
		type=str,
		default="sinusoidal",
		choices=["sinusoidal", "simple-index", "learned"],
		help="Type of positional embedding to use.",
	)
	parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of train data for validation.")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
	parser.add_argument("--wandb-project", type=str, default="graph-token-cycle-check")
	parser.add_argument("--wandb-entity", type=str, default=None)
	parser.add_argument("--wandb-name", type=str, default=None)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
	set_seed(args.seed)
	wandb_run = None
	wandb_module = None
	if args.wandb:
		try:
			import wandb as wandb_module
		except ImportError as exc:
			raise ImportError("wandb is not installed but --use-wandb was set") from exc

	train_dataset = GraphTokenDataset(args.data_root, split="train")
	try:
		test_dataset = GraphTokenDataset(args.data_root, split="test")
	except Exception:
		test_dataset = None

	vocab = build_vocab(train_dataset)
	print(f"Vocabulary size: {len(vocab)}")

	if args.max_len <= 0:
		raise ValueError("max_len must be positive")

	if args.val_frac > 0.0:
		train_subset, val_subset = split_dataset(train_dataset, args.val_frac, seed=args.seed)
	else:
		train_subset, val_subset = train_dataset, None

	train_loader = DataLoader(
		train_subset,
		batch_size=args.batch_size,
		shuffle=True,
		collate_fn=make_collate_fn(vocab, args.max_len),
	)
	val_loader = (
		DataLoader(
			val_subset,
			batch_size=args.batch_size,
			shuffle=False,
			collate_fn=make_collate_fn(vocab, args.max_len),
		)
		if val_subset is not None
		else None
	)
	test_loader = (
		DataLoader(
			test_dataset,
			batch_size=args.batch_size,
			shuffle=False,
			collate_fn=make_collate_fn(vocab, args.max_len),
		)
		if test_dataset is not None
		else None
	)

	model = CycleCheckTransformer(
		vocab_size=len(vocab),
		hidden_dim=args.hidden_dim,
		num_heads=args.num_heads,
		num_layers=args.num_layers,
		mlp_dim=args.mlp_dim,
		max_len=args.max_len,
		dropout=args.dropout,
		pos_emb_type=args.pos_emb,
	)
	model.to(device)

	num_params = sum(p.numel() for p in model.parameters())
	print(f"Model parameters: {num_params:,}")
	if args.wandb:
		wandb_run = wandb_module.init(
			project=args.wandb_project,
			entity=args.wandb_entity,
			name=args.wandb_name,
			config={**vars(args), "vocab_size": len(vocab)},
		)
		wandb_run.summary["num_params"] = int(num_params)
		wandb_run.log({"num_params": int(num_params)}, step=0)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

	prev_val_acc = None
	best_val_acc = float("-inf")
	best_epoch = -1
	for epoch in range(1, args.epochs + 1):
		epoch_start = time.perf_counter()
		train_metrics, _ = run_epoch(model, train_loader, criterion, device, optimizer)
		log = [f"Epoch {epoch}/{args.epochs}", f"train_loss={train_metrics['loss']:.4f}", f"train_acc={train_metrics['accuracy']:.4f}"]
		val_metrics = None
		val_outputs = (np.array([]), np.array([]))
		if val_loader is not None:
			with torch.no_grad():
				val_metrics, val_outputs = run_epoch(model, val_loader, criterion, device, collect_preds=True)
			log.extend([f"val_loss={val_metrics['loss']:.4f}", f"val_acc={val_metrics['accuracy']:.4f}"])
			if val_metrics["accuracy"] > best_val_acc:
				best_val_acc = val_metrics["accuracy"]
				best_epoch = epoch
				if wandb_run is not None:
					wandb_run.summary["best_val_acc"] = float(best_val_acc)
					wandb_run.summary["best_epoch"] = int(best_epoch)
		print(" | ".join(log))

		epoch_time = time.perf_counter() - epoch_start
		if wandb_run is not None:
			log_data = {
				"epoch": epoch,
				"train_loss": train_metrics["loss"],
				"train_acc": train_metrics["accuracy"],
				"time_per_epoch": epoch_time,
			}
			if val_metrics is not None:
				log_data["val_loss"] = val_metrics["loss"]
				log_data["val_acc"] = val_metrics["accuracy"]
				if prev_val_acc is not None:
					delta = val_metrics["accuracy"] - prev_val_acc
					if delta > 0:
						log_data["time_per_1pct_sec"] = epoch_time / (delta * 100.0)
				prev_val_acc = val_metrics["accuracy"]
			wandb_run.log(log_data, step=epoch)
			val_preds, val_labels = val_outputs
			if val_preds.size and val_labels.size:
				try:
					class_names = [str(c) for c in sorted(set(val_labels.tolist()) | set(val_preds.tolist()))]
					cm = wandb_module.plot.confusion_matrix(
						y_true=val_labels.tolist(),
						preds=val_preds.tolist(),
						class_names=class_names,
					)
					wandb_run.log({"confusion_matrix/val": cm}, step=epoch)
				except Exception:
					try:
						uniq = np.unique(np.concatenate([val_labels, val_preds]))
						label_to_idx = {label: idx for idx, label in enumerate(uniq)}
						matrix = np.zeros((len(uniq), len(uniq)), dtype=int)
						for t, p in zip(val_labels, val_preds):
							matrix[label_to_idx[t], label_to_idx[p]] += 1
						wandb_run.log({"confusion_matrix/val": matrix.tolist()}, step=epoch)
					except Exception:
						pass

	if test_loader is not None:
		with torch.no_grad():
			test_metrics, (test_preds, test_labels) = run_epoch(model, test_loader, criterion, device, collect_preds=True)
		print(f"Test loss={test_metrics['loss']:.4f} | Test acc={test_metrics['accuracy']:.4f}")
		if wandb_run is not None:
			wandb_run.summary["test_loss"] = float(test_metrics["loss"])
			wandb_run.summary["test_acc"] = float(test_metrics["accuracy"])
			wandb_run.log({"test_loss": test_metrics["loss"], "test_acc": test_metrics["accuracy"]}, step=args.epochs + 1)
			if test_preds.size and test_labels.size:
				try:
					class_names = [str(c) for c in sorted(set(test_labels.tolist()) | set(test_preds.tolist()))]
					cm = wandb_module.plot.confusion_matrix(
						y_true=test_labels.tolist(),
						preds=test_preds.tolist(),
						class_names=class_names,
					)
					wandb_run.log({"confusion_matrix/test": cm}, step=args.epochs + 1)
				except Exception:
					try:
						uniq = np.unique(np.concatenate([test_labels, test_preds]))
						label_to_idx = {label: idx for idx, label in enumerate(uniq)}
						matrix = np.zeros((len(uniq), len(uniq)), dtype=int)
						for t, p in zip(test_labels, test_preds):
							matrix[label_to_idx[t], label_to_idx[p]] += 1
						wandb_run.log({"confusion_matrix/test": matrix.tolist()}, step=args.epochs + 1)
					except Exception:
						pass

	if wandb_run is not None:
		wandb_run.finish()


if __name__ == "__main__":
	main()
