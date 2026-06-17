"""Evaluate a trained LSTMPredictor checkpoint on MOTTrackletDataset.

Example:
python new_pipeline/eval_lstm.py \
    --ckpt checkpoints/lstm/best.pth \
    --ann_files datasets/mot/annotations/val_half.json
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from new_pipeline.dataset_lstm import MOTTrackletDataset
from new_pipeline.lstm_predictor import LSTMPredictor


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate LSTMPredictor checkpoint")
    parser.add_argument("--ckpt", default="checkpoints/lstm/best.pth")
    parser.add_argument(
        "--ann_files",
        nargs="+",
        default=["datasets/mot/annotations/val_half.json"],
        help="COCO-format annotation JSON files used for evaluation",
    )
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument(
        "--miss_prob",
        type=float,
        default=0.0,
        help="Use 0.0 for deterministic eval, or 0.15 to mimic training",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_samples", type=int, default=3)
    parser.add_argument("--no_progress", action="store_true")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_state_dict(ckpt):
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def infer_model_config(state_dict: dict) -> tuple[int, int]:
    hidden_size = int(state_dict["input_proj.weight"].shape[0])
    layer_ids = []
    for key in state_dict:
        if key.startswith("lstm.weight_ih_l"):
            layer_ids.append(int(key.rsplit("l", 1)[1]))
    num_layers = max(layer_ids) + 1 if layer_ids else 2
    return hidden_size, num_layers


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = unwrap_state_dict(ckpt)
    hidden_size, num_layers = infer_model_config(state_dict)

    model = LSTMPredictor(hidden_size=hidden_size, num_layers=num_layers).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    ds = MOTTrackletDataset(
        ann_files=args.ann_files,
        seq_len=args.seq_len,
        miss_prob=args.miss_prob,
        augment=False,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    total_loss = 0.0
    total_baseline_loss = 0.0
    total_abs = 0.0
    total_baseline_abs = 0.0
    total_valid_dims = 0.0
    total_samples = 0
    first_batch = None

    for batch in tqdm(
        loader,
        desc="eval",
        dynamic_ncols=True,
        disable=args.no_progress,
    ):
        x_seq = batch["x_seq"].to(device)
        target = batch["target_residual"].to(device)
        mask = batch["mask"].to(device)
        batch_size = x_seq.size(0)

        h0, c0 = model.init_hidden(batch_size=batch_size, device=device)
        pred, _, _ = model(x_seq, h0, c0)

        loss = LSTMPredictor.huber_loss(pred, target, mask)
        baseline = torch.zeros_like(pred)
        baseline_loss = LSTMPredictor.huber_loss(baseline, target, mask)

        valid = mask.unsqueeze(-1)
        valid_dims = valid.sum().item() * target.size(-1)
        total_loss += loss.item() * batch_size
        total_baseline_loss += baseline_loss.item() * batch_size
        total_abs += (torch.abs(pred - target) * valid).sum().item()
        total_baseline_abs += (torch.abs(target) * valid).sum().item()
        total_valid_dims += valid_dims
        total_samples += batch_size

        if first_batch is None:
            first_batch = (
                pred[: args.show_samples].detach().cpu(),
                target[: args.show_samples].detach().cpu(),
                mask[: args.show_samples].detach().cpu(),
            )

    mean_loss = total_loss / max(total_samples, 1)
    mean_baseline_loss = total_baseline_loss / max(total_samples, 1)
    mae = total_abs / max(total_valid_dims, 1)
    baseline_mae = total_baseline_abs / max(total_valid_dims, 1)
    improvement = 100.0 * (baseline_mae - mae) / max(baseline_mae, 1e-12)

    print("\nEvaluation")
    print(f"  checkpoint      : {args.ckpt}")
    print(f"  ann_files       : {args.ann_files}")
    print(f"  samples         : {len(ds)}")
    print(f"  device          : {device}")
    print(f"  hidden/layers   : {hidden_size}/{num_layers}")
    print(f"  seq_len         : {args.seq_len}")
    print(f"  miss_prob       : {args.miss_prob}")
    print(f"  huber_loss      : {mean_loss:.6f}")
    print(f"  zero_resid_loss : {mean_baseline_loss:.6f}")
    print(f"  residual_mae_px : {mae:.3f}")
    print(f"  baseline_mae_px : {baseline_mae:.3f}")
    print(f"  mae_improvement : {improvement:.2f}%")

    if first_batch is not None and args.show_samples > 0:
        pred, target, mask = first_batch
        print("\nFirst valid timesteps: pred_residual -> target_residual")
        for sample_idx in range(pred.size(0)):
            valid_idx = torch.nonzero(mask[sample_idx] > 0, as_tuple=False).flatten()
            valid_idx = valid_idx[:5]
            print(f"  sample {sample_idx}:")
            for t in valid_idx.tolist():
                p = pred[sample_idx, t].numpy()
                y = target[sample_idx, t].numpy()
                print(f"    t={t:02d}: {np.round(p, 2)} -> {np.round(y, 2)}")


if __name__ == "__main__":
    main()
