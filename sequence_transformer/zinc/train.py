#!/usr/bin/env python3
"""Train a transformer regressor on ZINC token sequences."""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from .dataloader import (
        TokenVocabulary,
        load_token_records,
        infer_feature_dim,
        build_one_hot_matrix,
        build_token_feature_sequence,
    )
except ImportError:
    from dataloader import (  # type: ignore
        TokenVocabulary,
        load_token_records,
        infer_feature_dim,
        build_one_hot_matrix,
        build_token_feature_sequence,
    )


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EncodedSample:
    input_ids: List[int]
    target: float
    graph_id: str
    split: str
    token_features: List[List[float]]


class ZincSequenceDataset(Dataset):
    def __init__(self, samples: Sequence[EncodedSample], feature_dim: int) -> None:
        self.samples = list(samples)
        self.feature_dim = feature_dim
        if not self.samples:
            raise ValueError("Empty dataset.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> EncodedSample:
        return self.samples[idx]


def encode_records(
    records: Iterable[Dict],
    split: str,
    vocab: TokenVocabulary,
    feature_dim: int,
    min_length: int = 1,
) -> List[EncodedSample]:
    encoded: List[EncodedSample] = []
    for record in records:
        if record.get("split") != split:
            continue
        tokens = record.get("tokens")
        target = record.get("target")
        graph_id = record.get("graph_id", "unknown")
        if not isinstance(tokens, list) or target is None:
            continue
        ids = vocab.encode(tokens)
        if len(ids) < min_length:
            continue
        node_matrix = build_one_hot_matrix(record, feature_dim)
        token_features = build_token_feature_sequence(tokens, node_matrix, feature_dim)
        encoded.append(
            EncodedSample(
                input_ids=ids,
                target=float(target),
                graph_id=str(graph_id),
                split=split,
                token_features=token_features,
            )
        )
    if not encoded:
        raise ValueError(f"No samples found for split '{split}'.")
    return encoded


def make_pad_collate(pad_id: int, max_len: int, feature_dim: int):
    if max_len <= 0:
        raise ValueError("max_len must be positive.")

    def _collate(batch: Sequence[EncodedSample]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        lengths = [min(len(sample.input_ids), max_len) for sample in batch]
        target_len = max(lengths)
        input_ids = torch.full((batch_size, target_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, target_len), dtype=torch.bool)
        if feature_dim > 0:
            feature_tensor = torch.zeros((batch_size, target_len, feature_dim), dtype=torch.float)
        else:
            feature_tensor = torch.zeros((batch_size, target_len, 0), dtype=torch.float)
        targets = torch.zeros(batch_size, dtype=torch.float)
        for i, sample in enumerate(batch):
            seq = sample.input_ids[:max_len]
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :seq_len] = True
            if feature_dim > 0 and sample.token_features:
                feat_seq = sample.token_features[:max_len]
                if feat_seq:
                    feats = torch.tensor(feat_seq, dtype=torch.float)
                    feature_tensor[i, :seq_len] = feats
            targets[i] = sample.target
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_features": feature_tensor,
            "targets": targets,
        }

    return _collate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 16384, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds positional encoding limit {self.pe.size(0)}.")
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 16384, dropout: float = 0.0) -> None:
        super().__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds learned positional embedding limit {self.max_len}.")
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.embedding(positions)
        return self.dropout(x + pos_emb)


class ZincTransformerRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        feature_dim: int,
        pos_enc: str = "sinusoidal",
        max_position_embeddings: int = 16384,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        if feature_dim > 0:
            per_half = d_model // 2
            self.token_dim = per_half
            self.feature_dim_projected = d_model - per_half
        else:
            self.token_dim = d_model
            self.feature_dim_projected = 0
        self.embedding = nn.Embedding(vocab_size, self.token_dim, padding_idx=0)
        self.feature_proj = (
            nn.Linear(feature_dim, self.feature_dim_projected) if feature_dim > 0 else None
        )
        if pos_enc == "learned":
            self.positional = LearnedPositionalEncoding(d_model, max_position_embeddings, dropout)
        else:
            self.positional = SinusoidalPositionalEncoding(d_model, max_position_embeddings, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_tok = self.embedding(input_ids)
        if self.feature_proj is not None:
            if token_features is None:
                raise ValueError("token_features must be provided when feature_dim > 0.")
            feature_embed = self.feature_proj(token_features)
            x = torch.cat([x_tok, feature_embed], dim=-1)
        else:
            x = x_tok
        x = self.positional(x)
        key_padding_mask = ~attention_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(summed / denom)
        return self.reg_head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    preds_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_features = batch["token_features"].to(device)
        targets = batch["targets"].to(device)

        feature_arg = token_features if getattr(model, "feature_proj", None) is not None else None
        preds = model(input_ids, attention_mask, feature_arg)
        loss = criterion(preds, targets)

        if training:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        preds_list.append(preds.detach().cpu())
        targets_list.append(targets.detach().cpu())

    preds_cat = torch.cat(preds_list)
    targets_cat = torch.cat(targets_list)
    mae = torch.mean(torch.abs(preds_cat - targets_cat)).item()
    rmse = torch.sqrt(torch.mean((preds_cat - targets_cat) ** 2)).item()
    avg_loss = total_loss / total_examples if total_examples else 0.0
    return {"loss": avg_loss, "mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True, help="Path to zinc_tokens.jsonl generated by zinc_tokenization.py.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=512, help="Maximum sequence length (tokens beyond are truncated).")
    parser.add_argument("--max-position-emb", type=int, default=4096)
    parser.add_argument("--pos-encoding", choices=["sinusoidal", "learned"], default="sinusoidal")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-best", type=Path, default=None, help="Optional path to store the best checkpoint (based on val MAE).")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ZINC")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is not installed but --use-wandb was provided.") from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )

    records = load_token_records(args.tokens)
    splits = {split: [rec for rec in records if rec.get("split") == split] for split in ("train", "val", "test")}
    if not splits["train"]:
        raise ValueError("No train split found in token file.")
    if not splits["val"]:
        raise ValueError("No val split found in token file.")

    vocab = TokenVocabulary.build_from_records(records, split="train")
    feature_dim = infer_feature_dim(records)
    print(f"Vocabulary size (train split): {len(vocab)}")
    print(f"Feature dimension inferred: {feature_dim}")

    encoded_splits = {
        split: encode_records(records, split, vocab, feature_dim)
        for split in splits
        if splits[split]
    }

    collate_fn = make_pad_collate(vocab.pad_id, args.max_len, feature_dim)
    loaders: Dict[str, Optional[DataLoader]] = {}
    for split in ("train", "val", "test"):
        if split not in encoded_splits:
            loaders[split] = None
            continue
        dataset = ZincSequenceDataset(encoded_splits[split], feature_dim)
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn,
        )

    model = ZincTransformerRegressor(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        feature_dim=feature_dim,
        pos_enc=args.pos_encoding,
        max_position_embeddings=args.max_position_emb,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    if wandb_run is not None:
        wandb_run.summary["num_params"] = int(num_params)
        wandb_run.log({'num_params': int(num_params)})

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mae = float("inf")
    best_state = None
    improvement_time = 0.0
    improvement_delta = 0.0
    train_history: List[Dict[str, float]] = []
    val_history: List[Optional[Dict[str, float]]] = []
    full_epoch_times: List[float] = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer, grad_clip=args.grad_clip)
        train_history.append(train_metrics)
        log_parts = [
            f"Epoch {epoch}/{args.epochs}",
            f"train_loss={train_metrics['loss']:.4f}",
            f"train_mae={train_metrics['mae']:.4f}",
        ]
        val_metrics = None
        improvement_delta_epoch = 0.0
        if loaders["val"] is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, loaders["val"], criterion, device)
            log_parts.extend(
                [
                    f"val_loss={val_metrics['loss']:.4f}",
                    f"val_mae={val_metrics['mae']:.4f}",
                ]
            )
            if val_metrics["mae"] < best_val_mae:
                if best_val_mae != float("inf"):
                    improvement_delta_epoch = best_val_mae - val_metrics["mae"]
                best_val_mae = val_metrics["mae"]
                best_state = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "vocab_size": len(vocab),
                }
            val_history.append(
                {"loss": val_metrics["loss"], "mae": val_metrics["mae"], "rmse": val_metrics["rmse"]}
            )
        else:
            val_history.append(None)

        duration = time.time() - start
        full_epoch_times.append(duration)
        try:
            num_iters = len(loaders["train"]) if len(loaders["train"]) > 0 else 1
        except Exception:
            num_iters = 1
        time_per_iter = duration / float(num_iters)
        if improvement_delta_epoch > 0:
            improvement_time += duration
            improvement_delta += improvement_delta_epoch
        log_parts.append(f"time={duration:.1f}s")
        print(" | ".join(log_parts))

        if wandb_run is not None:
            avg_time_per_pct = None
            if improvement_delta > 0:
                avg_time_per_pct = improvement_time / (improvement_delta * 100.0)
            log_payload = {
                "train/loss": train_metrics["loss"],
                "train/mae": train_metrics["mae"],
                "train/rmse": train_metrics["rmse"],
                "train/time_epoch": duration,
                "train/time_iter": time_per_iter,
                "val/time_epoch": duration if val_metrics is not None else None,
                "val/time_iter": time_per_iter if val_metrics is not None else None,
            }
            if val_metrics is not None:
                log_payload.update(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/mae": val_metrics["mae"],
                        "val/rmse": val_metrics["rmse"],
                    }
                )
            if avg_time_per_pct is not None:
                log_payload["time_per_1pct_sec"] = avg_time_per_pct
            log_payload = {k: v for k, v in log_payload.items() if v is not None}
            wandb_run.log(log_payload, step=epoch)
        scheduler.step()

    if best_state is not None and args.save_best is not None:
        args.save_best.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, args.save_best)
        print(f"Saved checkpoint to {args.save_best} (val MAE={best_val_mae:.4f}).")

    if loaders["test"] is not None:
        if best_state is not None:
            model.load_state_dict(best_state["model_state"])
        with torch.no_grad():
            test_metrics = run_epoch(model, loaders["test"], criterion, device)
        print(f"Test loss={test_metrics['loss']:.4f} | Test MAE={test_metrics['mae']:.4f} | Test RMSE={test_metrics['rmse']:.4f}")
        if wandb_run is not None:
            wandb_run.summary["test_loss"] = float(test_metrics["loss"])
            wandb_run.summary["test_mae"] = float(test_metrics["mae"])
            wandb_run.summary["test_rmse"] = float(test_metrics["rmse"])
            wandb_run.log(
                {
                    "test/loss": test_metrics["loss"],
                    "test/mae": test_metrics["mae"],
                    "test/rmse": test_metrics["rmse"],
                },
                step=args.epochs + 1,
            )

    if wandb_run is not None:
        valid_records = [(idx, record) for idx, record in enumerate(val_history) if record is not None]
        if valid_records:
            best_epoch_idx, best_record = min(valid_records, key=lambda item: item[1]["mae"])
            summary_payload = {
                "best/epoch": best_epoch_idx + 1,
                "best/train_loss": train_history[best_epoch_idx]["loss"],
                "best/train_mae": train_history[best_epoch_idx]["mae"],
                "best/train_rmse": train_history[best_epoch_idx]["rmse"],
                "best/val_loss": best_record["loss"],
                "best/val_mae": best_record["mae"],
                "best/val_rmse": best_record["rmse"],
                "full_epoch_time_sum": float(np.sum(full_epoch_times)),
            }
            wandb_run.log(summary_payload, step=args.epochs + 1)
            wandb_run.summary.update(summary_payload)
        wandb_run.finish()


if __name__ == "__main__":
    main()
