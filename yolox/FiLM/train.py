"""
Training script cho FiLM ReID model.

Sử dụng:
    python -m yolox.FiLM.train \
        --data-root   /path/to/tracklets_data \
        --output-dir  output/film_reid \
        --pretrained-reid  /path/to/fastreid_resnest50.pth \
        --fastreid-config  /path/to/fastreid_config.yml \
        --epochs 60 --batch-size 64

Chiến lược train:
    1. Warmup (5 epoch): chỉ train Motion Encoder + FiLM + Head, backbone frozen
    2. Full train (epoch 6+): unfreeze backbone với LR nhỏ hơn 10×
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

try:
    from .dataset import TrackletReIDDataset, RandomIdentitySampler
    from .film_resnest_reid import FiLMReIDModel
except ImportError:
    # Chạy trực tiếp: python3 yolox/FiLM/train.py
    # Thêm thư mục FiLM/ vào sys.path → import trực tiếp, tránh trigger yolox/__init__.py
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))
    for _p in [_here, os.path.join(_root, "fast-reid")]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from dataset import TrackletReIDDataset, RandomIdentitySampler
    from film_resnest_reid import FiLMReIDModel


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def batch_hard_triplet_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Batch Hard Triplet Loss.
    features: (B, D) L2-normalized
    labels:   (B,)   integer identity labels
    """
    dist = torch.cdist(features, features, p=2)   # (B, B)
    labels_col = labels.unsqueeze(1)              # (B, 1)

    mask_pos = (labels_col == labels_col.T).float()   # same identity
    mask_neg = (labels_col != labels_col.T).float()   # diff identity

    # Hardest positive (same identity, max distance)
    ap_dist = (dist * mask_pos).max(dim=1)[0]

    # Hardest negative (diff identity, min distance)
    inf_mask = mask_pos * 1e9
    an_dist  = (dist + inf_mask).min(dim=1)[0]

    loss = F.relu(ap_dist - an_dist + margin).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validate
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: FiLMReIDModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    lambda_motion: float = 0.1,
    triplet_margin: float = 0.3,
    ce_label_smooth: float = 0.1,
) -> dict:
    model.train()
    ce_fn = nn.CrossEntropyLoss(label_smoothing=ce_label_smooth)

    total = {"loss": 0.0, "tri": 0.0, "ce": 0.0, "mot": 0.0}
    n = 0

    for batch in loader:
        images     = batch["image"].to(device)
        traj_seq   = batch["traj_seq"].to(device)
        gate_feats = batch["gate_feats"].to(device)
        pids       = batch["pid"].to(device)
        next_box   = batch["next_box"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            out = model(images, traj_seq, gate_feats)

            # ReID losses
            feat_norm = F.normalize(out["features"], dim=1)
            loss_tri  = batch_hard_triplet_loss(feat_norm, pids, margin=triplet_margin)
            loss_ce   = ce_fn(out["cls_outputs"], pids)

            # Motion prediction loss (MSE on next bbox)
            loss_mot = F.mse_loss(out["motion_pred"], next_box)

            loss = loss_tri + loss_ce + lambda_motion * loss_mot

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        B = images.size(0)
        total["loss"] += loss.item() * B
        total["tri"]  += loss_tri.item() * B
        total["ce"]   += loss_ce.item() * B
        total["mot"]  += loss_mot.item() * B
        n += B

    return {k: v / n for k, v in total.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main train loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}")

    # ── Dataset ──────────────────────────────────────────────────────────
    train_dataset = TrackletReIDDataset(
        data_root=args.data_root,
        seq_len=args.seq_len,
        image_size=(args.img_h, args.img_w),
        augment=True,
    )
    # RandomIdentitySampler: P identities × K images/identity mỗi batch
    # → đảm bảo luôn có positive pairs trong batch cho triplet loss
    sampler = RandomIdentitySampler(
        train_dataset,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,    # sampler đã đảm bảo batch đầy đủ
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = FiLMReIDModel(
        num_classes=train_dataset.num_pids,
        motion_hidden_dim=args.motion_hidden_dim,
        fastreid_config=args.fastreid_config,
        pretrained_reid_path=args.pretrained_reid,
    ).to(device)

    # ── Optimizer: 2 param groups ─────────────────────────────────────────
    # Backbone dùng LR nhỏ hơn (fine-tuning), các module mới dùng LR lớn
    backbone_params = list(model.backbone.parameters())
    new_params = (
        list(model.motion_encoder.parameters())
        + list(model.film.parameters())
        + list(model.bnneck.parameters())
        + list(model.classifier.parameters())
    )

    # Warmup: freeze backbone hoàn toàn
    for p in backbone_params:
        p.requires_grad_(False)

    optimizer = optim.Adam(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": new_params,      "lr": args.lr},
        ],
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler("cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss  = float("inf")
    best_epoch = -1
    best_path  = os.path.join(args.output_dir, "film_reid_best.pth")
    final_path = os.path.join(args.output_dir, "film_reid_final.pth")

    for epoch in range(args.epochs):

        # ── Unfreeze backbone sau warmup ──────────────────────────────────
        if epoch == args.warmup_epochs:
            for p in backbone_params:
                p.requires_grad_(True)
            print(f"[Epoch {epoch}] Backbone unfrozen — full joint training bắt đầu")

        metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            lambda_motion=args.lambda_motion,
        )
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"[Epoch {epoch + 1:03d}/{args.epochs}] "
            f"loss={metrics['loss']:.4f}  tri={metrics['tri']:.4f}  "
            f"ce={metrics['ce']:.4f}  mot={metrics['mot']:.4f}  "
            f"lr={lr_now:.2e}"
        )

        # ── Checkpoint payload ────────────────────────────────────────────
        ckpt = {
            "epoch":     epoch + 1,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss":      metrics["loss"],
            "args":      vars(args),
        }

        # Lưu best checkpoint (ghi đè — luôn là model tốt nhất)
        is_best = metrics["loss"] < best_loss
        if is_best:
            best_loss  = metrics["loss"]
            best_epoch = epoch + 1
            torch.save(ckpt, best_path)
            print(f"  → Best  : {best_path}  (epoch {best_epoch}, loss={best_loss:.4f})")

        # Lưu periodic checkpoint mỗi save_freq epoch
        if (epoch + 1) % args.save_freq == 0:
            periodic = os.path.join(args.output_dir, f"film_reid_epoch{epoch + 1}.pth")
            torch.save(ckpt, periodic)
            print(f"  → Saved : {periodic}")

    # ── Luôn lưu final checkpoint sau epoch cuối ──────────────────────────
    torch.save(ckpt, final_path)
    print(f"\n[Done] Final checkpoint: {final_path}")
    print(f"[Done] Best  checkpoint: {best_path}  (epoch {best_epoch}, loss={best_loss:.4f})")

    # ── Zip nếu bật --zip ─────────────────────────────────────────────────
    if args.zip:
        _zip_checkpoints(args.output_dir, best_path, final_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _zip_checkpoints(output_dir: str, best_path: str, final_path: str) -> None:
    """Zip best + final checkpoint vào output_dir/film_reid_checkpoints.zip."""
    zip_path = os.path.join(output_dir, "film_reid_checkpoints.zip")
    files_to_zip = [p for p in [best_path, final_path] if os.path.isfile(p)]

    if not files_to_zip:
        print("[Zip] Không tìm thấy checkpoint nào để zip.")
        return

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files_to_zip:
            arcname = os.path.basename(fpath)
            zf.write(fpath, arcname)
            print(f"[Zip] Added: {arcname}")

    zip_size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"[Zip] Saved → {zip_path}  ({zip_size_mb:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser("FiLM ReID Training")

    # Data
    p.add_argument("--data-root",    required=True,  help="folder chứa tracklets.pkl")
    p.add_argument("--output-dir",   default="output/film_reid")
    p.add_argument("--img-h",        type=int, default=256)
    p.add_argument("--img-w",        type=int, default=128)
    p.add_argument("--seq-len",      type=int, default=20,
                   help="số frame history cho Motion Branch")

    # Model
    p.add_argument("--fastreid-config",  default=None,
                   help="path đến config .yml của fast-reid (optional)")
    p.add_argument("--pretrained-reid",  default=None,
                   help="path đến checkpoint fast-reid để init backbone")
    p.add_argument("--motion-hidden-dim", type=int, default=128)

    # Training
    p.add_argument("--epochs",         type=int,   default=60)
    p.add_argument("--warmup-epochs",  type=int,   default=5)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=3.5e-4)
    p.add_argument("--lambda-motion",  type=float, default=0.1,
                   help="weight cho motion prediction loss")
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--num-instances",  type=int,   default=4,
                   help="K: số ảnh mỗi identity mỗi batch (batch_size phải chia hết cho K)")
    p.add_argument("--save-freq",      type=int,   default=10)
    p.add_argument("--zip",            action="store_true",
                   help="sau khi train xong, zip best + final checkpoint lại")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
