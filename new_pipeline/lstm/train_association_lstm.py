"""Train the association score head using the existing LSTMPredictor encoder.

The trained checkpoint contains only the lightweight MLP association head.
At inference time ByteTrackerLSTM reuses the hidden state from the already-run
motion LSTM, so no second LSTM pass is introduced.

Example:
    python new_pipeline/lstm/train_association_lstm.py \
        --ann_files datasets/mot/annotations/train_half.json \
        --val_ann_files datasets/mot/annotations/val_half.json \
        --lstm_ckpt checkpoints/lstm/best.pth \
        --save_dir checkpoints/lstm_assoc
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from new_pipeline.lstm.association_lstm import AssociationScoreHead  # noqa: E402
from new_pipeline.lstm.dataset_association_lstm import MOTAssociationDataset  # noqa: E402
from new_pipeline.lstm.lstm_predictor import LSTMPredictor  # noqa: E402


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train ByteTrack-LSTM association score head")
    parser.add_argument("--ann_files", nargs="+", required=True)
    parser.add_argument("--val_ann_files", nargs="+", default=None)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--min_history", type=int, default=2)
    parser.add_argument("--negatives_per_positive", type=int, default=3)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)

    parser.add_argument("--lstm_ckpt", default="checkpoints/lstm/best.pth")
    parser.add_argument("--train_lstm_encoder", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lstm_dropout", type=float, default=0.1)

    parser.add_argument("--assoc_dropout", type=float, default=0.1)
    parser.add_argument("--assoc_mlp_hidden", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_dir", default="checkpoints/lstm_assoc")
    parser.add_argument("--resume", default=None)
    return parser


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=shuffle,
    )


def label_counts(dataset) -> tuple[int, int]:
    if hasattr(dataset, "num_positive") and hasattr(dataset, "num_negative"):
        return int(dataset.num_positive), int(dataset.num_negative)
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        positives = sum(
            1 for idx in dataset.indices if dataset.dataset.samples[idx]["label"] > 0.5
        )
        return positives, len(dataset.indices) - positives
    positives = sum(1 for i in range(len(dataset)) if float(dataset[i]["label"]) > 0.5)
    return positives, len(dataset) - positives


def load_lstm_checkpoint(model: LSTMPredictor, path: str, device: torch.device):
    if not path or not os.path.isfile(path):
        print(f"[train_assoc] LSTM checkpoint not found: {path}. Using current weights.")
        return
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"[train_assoc] loaded LSTM encoder: {path}")


def run_epoch(
    encoder: LSTMPredictor,
    head: AssociationScoreHead,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    grad_clip: float,
    training: bool,
    train_encoder: bool,
    desc: str,
) -> dict:
    encoder.train(training and train_encoder)
    head.train(training)

    totals = {
        "loss": 0.0,
        "count": 0,
        "correct": 0.0,
        "pos_score": 0.0,
        "pos_count": 0.0,
        "neg_score": 0.0,
        "neg_count": 0.0,
    }

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        history = batch["history"].to(device).float()
        det = batch["det"].to(device).float()
        pair = batch["pair"].to(device).float()
        label = batch["label"].to(device).float()

        with torch.set_grad_enabled(training):
            if training and train_encoder:
                embedding = encoder.encode_history(history)
            else:
                with torch.no_grad():
                    embedding = encoder.encode_history(history)
            logits = head(embedding, det, pair)
            loss = criterion(logits, label)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
                    if train_encoder:
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                optimizer.step()

        probs = torch.sigmoid(logits.detach())
        pred = probs >= 0.5
        label_bool = label >= 0.5
        batch_size = label.numel()
        pos_mask = label_bool
        neg_mask = ~label_bool

        totals["loss"] += loss.item() * batch_size
        totals["count"] += batch_size
        totals["correct"] += (pred == label_bool).float().sum().item()
        totals["pos_score"] += probs[pos_mask].sum().item()
        totals["pos_count"] += pos_mask.float().sum().item()
        totals["neg_score"] += probs[neg_mask].sum().item()
        totals["neg_count"] += neg_mask.float().sum().item()
        pbar.set_postfix(
            loss=totals["loss"] / max(totals["count"], 1),
            acc=totals["correct"] / max(totals["count"], 1),
        )

    return {
        "loss": totals["loss"] / max(totals["count"], 1),
        "acc": totals["correct"] / max(totals["count"], 1),
        "pos_score": totals["pos_score"] / max(totals["pos_count"], 1),
        "neg_score": totals["neg_score"] / max(totals["neg_count"], 1),
    }


def save_checkpoint(path: str, head, optimizer, epoch: int, metrics: dict, args):
    torch.save(
        {
            "epoch": epoch,
            "model": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "head_type": "AssociationScoreHead",
        },
        path,
    )


def main():
    args = make_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = MOTAssociationDataset(
        args.ann_files,
        seq_len=args.seq_len,
        min_history=args.min_history,
        negatives_per_positive=args.negatives_per_positive,
        max_samples=args.max_train_samples,
    )
    if args.val_ann_files:
        val_ds = MOTAssociationDataset(
            args.val_ann_files,
            seq_len=args.seq_len,
            min_history=args.min_history,
            negatives_per_positive=args.negatives_per_positive,
            max_samples=args.max_val_samples,
        )
    else:
        n_val = max(1, int(0.1 * len(train_ds)))
        train_ds, val_ds = random_split(train_ds, [len(train_ds) - n_val, n_val])

    print(f"[train_assoc] train={len(train_ds)} val={len(val_ds)} device={device}")
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, device)

    encoder = LSTMPredictor(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.lstm_dropout,
    ).to(device)
    load_lstm_checkpoint(encoder, args.lstm_ckpt, device)

    head = AssociationScoreHead(
        hidden_size=args.hidden_size,
        dropout=args.assoc_dropout,
        mlp_hidden=args.assoc_mlp_hidden,
    ).to(device)

    if args.train_lstm_encoder:
        params = [
            {"params": encoder.parameters(), "lr": args.encoder_lr},
            {"params": head.parameters(), "lr": args.lr},
        ]
    else:
        for param in encoder.parameters():
            param.requires_grad_(False)
        params = head.parameters()

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    num_pos, num_neg = label_counts(train_ds)
    pos_weight = torch.tensor([max(num_neg / max(num_pos, 1), 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(
        f"[train_assoc] head_params={sum(p.numel() for p in head.parameters()):,} "
        f"pos_weight={pos_weight.item():.3f} train_encoder={args.train_lstm_encoder}"
    )

    start_epoch = 0
    best_val = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        head.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val = ckpt.get("metrics", {}).get("loss", best_val)
        print(f"[train_assoc] resumed {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            encoder,
            head,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip,
            True,
            args.train_lstm_encoder,
            f"train {epoch + 1}/{args.epochs}",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                encoder,
                head,
                val_loader,
                criterion,
                optimizer,
                device,
                args.grad_clip,
                False,
                False,
                f"val {epoch + 1}/{args.epochs}",
            )

        print(
            f"[epoch {epoch + 1:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
            f"val_pos={val_metrics['pos_score']:.3f} val_neg={val_metrics['neg_score']:.3f}"
        )

        save_checkpoint(os.path.join(args.save_dir, "last.pth"), head, optimizer, epoch, val_metrics, args)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(os.path.join(args.save_dir, "best.pth"), head, optimizer, epoch, val_metrics, args)
            print(f"[train_assoc] saved best: {best_val:.4f}")


if __name__ == "__main__":
    main()
