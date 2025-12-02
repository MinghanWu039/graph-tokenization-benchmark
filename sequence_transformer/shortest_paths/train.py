#!/usr/bin/env python3
"""Train a transformer classifier on shortest-path aggregated sequences."""

import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .dataloader import create_dataloaders
    from .model import ShortestPathTransformer
except ImportError:
    from dataloader import create_dataloaders
    from model import ShortestPathTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    collect_preds: bool = False,
    missing_class_idx: Optional[int] = None,
) -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    preds_buffer = [] if collect_preds else None
    raw_labels_buffer = [] if collect_preds else None

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        raw_labels = labels.clone()
        if missing_class_idx is not None:
            labels = labels.clone()
            labels[labels < 0] = missing_class_idx

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
                raw_labels_buffer.append(raw_labels.detach().cpu())

        total_loss += loss.item() * input_ids.size(0)
        total_examples += input_ids.size(0)

    metrics = {
        "loss": total_loss / total_examples if total_examples else 0.0,
        "accuracy": total_correct / total_examples if total_examples else 0.0,
    }
    if collect_preds:
        if preds_buffer:
            all_preds = torch.cat(preds_buffer).numpy()
            all_labels = torch.cat(raw_labels_buffer).numpy()
        else:
            all_preds = np.array([])
            all_labels = np.array([])
        return metrics, (all_preds, all_labels)
    return metrics, (np.array([]), np.array([]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer on shortest-path sequences.")
    parser.add_argument("data_root", type=str, help="Directory containing train.txt and test.txt.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--tokenization", type=str, default="graph-token")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--mlp-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--pos-emb", type=str, default="sinusoidal", choices=["sinusoidal", "simple-index"])
    parser.add_argument("--num-classes", type=int, default=32, help="Number of discrete distance classes.")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-test", action="store_true", help="Skip evaluation on the test set.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="Shortest_Paths")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser.parse_args()


def compute_mae(preds: np.ndarray, labels: np.ndarray, missing_class_idx: int, missing_distance: int = 20) -> Optional[float]:
    if preds.size == 0 or labels.size == 0:
        return None
    pred_vals = preds.copy()
    label_vals = labels.copy()
    pred_vals[pred_vals == missing_class_idx] = missing_distance
    label_vals[label_vals < 0] = missing_distance
    return float(np.abs(pred_vals - label_vals).mean())


def log_confusion_matrix(
    run,
    preds: np.ndarray,
    labels: np.ndarray,
    split: str,
    step: int,
    wandb_module,
) -> None:
    if preds.size == 0 or labels.size == 0:
        return
    try:
        class_names = list(map(str, np.unique(np.concatenate([labels, preds]))))
        cm_plot = wandb_module.plot.confusion_matrix(
            preds=preds.tolist(), y_true=labels.tolist(), class_names=class_names
        )
        run.log({f"confusion_matrix/{split}": cm_plot}, step=step)
    except Exception:
        uniq = np.unique(np.concatenate([labels, preds]))
        index = {val: idx for idx, val in enumerate(uniq)}
        matrix = np.zeros((len(uniq), len(uniq)), dtype=int)
        for t, p in zip(labels, preds):
            matrix[index[t], index[p]] += 1
        run.log({f"confusion_matrix/{split}": matrix.tolist()}, step=step)


def main() -> None:
    args = parse_args()
    os.makedirs("checkpoints", exist_ok=True)

    wandb_run = None
    wandb_module = None
    if args.wandb:
        try:
            import wandb as wandb_module  # type: ignore
        except ImportError as exc:
            raise ImportError("wandb is not installed but --wandb was set") from exc

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    set_seed(args.seed)

    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        max_length=args.max_len,
        val_fraction=args.val_frac,
        type=args.tokenization,
        seed=args.seed,
    )

    model = ShortestPathTransformer(
        vocab_size=len(vocab),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_dim=args.mlp_dim,
        max_len=args.max_len,
        dropout=args.dropout,
        pos_emb_type=args.pos_emb,
        num_classes=args.num_classes,
    ).to(device)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = float("-inf")
    best_state = None
    improvement_time = 0.0
    improvement_delta = 0.0
    missing_idx = args.num_classes - 1
    train_history: List[Dict[str, float]] = []
    val_history: List[Optional[Dict[str, float]]] = []
    full_epoch_times: List[float] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        # Collect predictions for the training set so we can compute train MAE
        train_metrics, (train_preds, train_labels) = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            collect_preds=True,
            missing_class_idx=missing_idx,
        )
        train_history.append(train_metrics)
        # compute train MAE (returns None if preds/labels empty)
        train_mae = compute_mae(train_preds, train_labels, missing_idx)
        log = [
            f"Epoch {epoch}/{args.epochs}",
            f"train_loss={train_metrics['loss']:.4f}",
            f"train_acc={train_metrics['accuracy']:.4f}",
            f"train_mae={train_mae:.4f}" if train_mae is not None else "train_mae=NA",
        ]

        val_metrics = None
        val_mae = None
        improvement_delta_epoch = 0.0
        val_preds = val_labels = np.array([])
        if val_loader is not None:
            with torch.no_grad():
                val_metrics, (val_preds, val_labels) = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    collect_preds=True,
                    missing_class_idx=missing_idx,
                )
            val_mae = compute_mae(val_preds, val_labels, missing_idx)
            record = {"loss": val_metrics["loss"], "accuracy": val_metrics["accuracy"]}
            if val_mae is not None:
                record["mae"] = val_mae
            val_history.append(record)
            log.extend(
                [
                    f"val_loss={val_metrics['loss']:.4f}",
                    f"val_acc={val_metrics['accuracy']:.4f}",
                    f"val_mae={val_mae:.4f}" if val_mae is not None else "val_mae=NA",
                ]
            )
            if val_metrics["accuracy"] > best_val_acc:
                if best_val_acc != float("-inf"):
                    improvement_delta_epoch = val_metrics["accuracy"] - best_val_acc
                best_val_acc = val_metrics["accuracy"]
                best_state = model.state_dict()
        else:
            val_history.append(None)

        scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        full_epoch_times.append(elapsed)
        # compute per-iteration time (fallback to epoch time if loader length unknown)
        try:
            num_iters = len(train_loader) if len(train_loader) > 0 else 1
        except Exception:
            num_iters = 1
        time_per_iter = elapsed / float(num_iters)
        if improvement_delta_epoch > 0:
            improvement_time += elapsed
            improvement_delta += improvement_delta_epoch
        log.append(f"time={elapsed:.2f}s")
        print(" | ".join(log))

        if args.wandb and wandb_run is not None:
            avg_time_per_pct = None
            if improvement_delta > 0:
                avg_time_per_pct = improvement_time / (improvement_delta * 100.0)
            log_data = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/mae": train_mae,
                "train/accuracy": train_metrics["accuracy"],
                # requested timing metrics
                "train/time_epoch": elapsed,
                "train/time_iter": time_per_iter,
            }
            if val_metrics is not None:
                log_data["val/loss"] = val_metrics["loss"]
                log_data["val/accuracy"] = val_metrics["accuracy"]
            if val_mae is not None:
                log_data["val/mae"] = val_mae
            if avg_time_per_pct is not None:
                log_data["time_per_1pct_sec"] = avg_time_per_pct
            wandb_run.log(log_data, step=epoch)
            if val_metrics is not None:
                log_confusion_matrix(
                    wandb_run,
                    val_preds,
                    val_labels,
                    split="val",
                    step=epoch,
                    wandb_module=wandb_module,
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    if not args.no_test and test_loader is not None:
        with torch.no_grad():
            test_metrics, (test_preds, test_labels) = run_epoch(
                model,
                test_loader,
                criterion,
                device,
                collect_preds=True,
                missing_class_idx=missing_idx,
            )
        test_mae = compute_mae(test_preds, test_labels, missing_idx)
        test_mae_str = f"{test_mae:.4f}" if test_mae is not None else "NA"
        print(
            f"Test loss={test_metrics['loss']:.4f} | "
            f"Test acc={test_metrics['accuracy']:.4f} | "
            f"Test mae={test_mae_str}"
        )
        if args.wandb and wandb_run is not None:
            wandb_run.summary["test/loss"] = float(test_metrics["loss"])
            wandb_run.summary["test/accuracy"] = float(test_metrics["accuracy"])
            if test_mae is not None:
                wandb_run.summary["test/mae"] = float(test_mae)
            wandb_run.log(
                {"test/loss": test_metrics["loss"], "test/accuracy": test_metrics["accuracy"], "test/mae": test_mae},
                step=args.epochs + 1,
            )
            log_confusion_matrix(
                wandb_run,
                test_preds,
                test_labels,
                split="test",
                step=args.epochs + 1,
                wandb_module=wandb_module,
            )

    if args.wandb and wandb_run is not None and val_loader is not None:
        valid_records = [(idx, record) for idx, record in enumerate(val_history) if record is not None]
        if valid_records:
            best_epoch_idx, best_record = max(valid_records, key=lambda item: item[1]["accuracy"])
            summary_payload = {
                "best/epoch": best_epoch_idx + 1,
                "best/train_loss": train_history[best_epoch_idx]["loss"],
                "best/train_accuracy": train_history[best_epoch_idx]["accuracy"],
                "best/val_loss": best_record["loss"],
                "best/val_accuracy": best_record["accuracy"],
                "full_epoch_time_avg": float(np.mean(full_epoch_times)),
                "full_epoch_time_sum": float(np.sum(full_epoch_times)),
            }
            if "mae" in best_record:
                summary_payload["best/val_mae"] = best_record["mae"]
            wandb_run.log(summary_payload, step=args.epochs + 1)
            wandb_run.summary.update(summary_payload)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
