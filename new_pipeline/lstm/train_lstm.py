"""
Train the LSTMPredictor on MOT-format datasets.

Usage examples
──────────────
# Train on MOT17 with dedicated val split (recommended)
python new_pipeline/train.py \
    --ann_files datasets/mot/annotations/train.json \
    --val_ann_files datasets/mot/annotations/val_half.json \
    --save_dir checkpoints/lstm

# Resume training
python new_pipeline/train.py \
    --ann_files datasets/mot/annotations/train.json \
    --val_ann_files datasets/mot/annotations/val_half.json \
    --resume checkpoints/lstm/best.pth
"""

import argparse
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from new_pipeline.lstm_predictor import LSTMPredictor
from new_pipeline.dataset_lstm import MOTTrackletDataset


# ── Argument parsing ──────────────────────────────────────────────────────────

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train LSTMPredictor for ByteTrack")

    # Data
    p.add_argument("--ann_files", nargs="+", required=True,
                   help="COCO-format annotation JSON files for training")
    p.add_argument("--val_ann_files", nargs="+", default=None,
                   help="COCO-format annotation JSON files for validation "
                        "(recommended: val_half.json). Falls back to random "
                        "split of train data if not provided.")
    p.add_argument("--seq_len", type=int, default=32,
                   help="Sequence window length (frames)")
    p.add_argument("--miss_prob", type=float, default=0.15,
                   help="Probability of simulating a missing detection per frame")
    p.add_argument("--val_ratio", type=float, default=0.05,
                   help="Fraction of train data reserved for validation "
                        "(only used when --val_ann_files is not set)")

    # Model
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--save_every", type=int, default=5,
                   help="Save a numbered checkpoint every N epochs (best.pth always saved)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=8,
                   help="Early stopping patience (epochs without val improvement)")

    # Misc
    p.add_argument("--save_dir", type=str, default="checkpoints/lstm")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--device", type=str, default=None,
                   help="'cuda' / 'cpu' (auto-detected if not set)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_lambda(epoch: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0


def save_checkpoint(model: LSTMPredictor, optimizer, scheduler, epoch: int,
                    val_loss: float, path: str):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss": val_loss,
    }, path)


# ── Train / val loops ─────────────────────────────────────────────────────────

def run_epoch(
    model: LSTMPredictor,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    grad_clip: float,
    training: bool,
    desc: str = "",
) -> float:
    model.train(training)
    total_loss, total_samples = 0.0, 0

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        x_seq = batch["x_seq"].to(device)                # [B, T, 10]
        target = batch["target_residual"].to(device)      # [B, T, 4]
        mask = batch["mask"].to(device)                   # [B, T]

        B = x_seq.size(0)
        h0 = torch.zeros(model.num_layers, B, model.hidden_size, device=device)
        c0 = torch.zeros(model.num_layers, B, model.hidden_size, device=device)

        with torch.set_grad_enabled(training):
            residuals, _, _ = model(x_seq, h0, c0)
            loss = LSTMPredictor.huber_loss(residuals, target, mask)

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * B
        total_samples += B
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = make_parser().parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[train] device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    train_ds = MOTTrackletDataset(
        ann_files=args.ann_files,
        seq_len=args.seq_len,
        miss_prob=args.miss_prob,
    )
    if args.val_ann_files:
        val_ds = MOTTrackletDataset(
            ann_files=args.val_ann_files,
            seq_len=args.seq_len,
            miss_prob=0.0,   # no random drops during validation
            augment=False,
        )
    else:
        n_val = max(1, int(len(train_ds) * args.val_ratio))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(train_ds, [n_train, n_val])
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = LSTMPredictor(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"[train] params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    warmup = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: warmup_lambda(e, args.warmup_epochs)
    )

    start_epoch = 0
    best_val = float("inf")
    patience_counter = 0

    # ── Resume ───────────────────────────────────────────────────────────
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("val_loss", float("inf"))
        print(f"[train] resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    # ── Training loop ────────────────────────────────────────────────────
    epoch_bar = tqdm(range(start_epoch, args.epochs), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        lr = optimizer.param_groups[0]["lr"]
        train_loss = run_epoch(
            model, train_loader, optimizer, device, args.grad_clip,
            training=True, desc=f"train e{epoch+1}",
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, device, args.grad_clip,
            training=False, desc=f"val   e{epoch+1}",
        )

        if epoch < args.warmup_epochs:
            warmup.step()
        else:
            scheduler.step()

        improved = val_loss < best_val
        flag = " ← best" if improved else ""
        epoch_bar.write(
            f"[epoch {epoch+1:03d}/{args.epochs}]"
            f"  train={train_loss:.6f}  val={val_loss:.6f}"
            f"  lr={lr:.2e}{flag}"
        )

        # Save numbered checkpoint every save_every epochs
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pth"),
            )

        if improved:
            best_val = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(args.save_dir, "best.pth"),
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                epoch_bar.write(f"[train] early stopping at epoch {epoch + 1}")
                break

    print(f"[train] done. best val loss: {best_val:.6f}")
    print(f"[train] checkpoint: {os.path.join(args.save_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
