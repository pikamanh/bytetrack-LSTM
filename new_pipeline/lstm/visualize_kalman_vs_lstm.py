"""Export side-by-side style videos for one fixed MOT track.

The script writes two videos:
1. Kalman-only prediction.
2. Kalman prediction corrected by the trained LSTM residual model.

Both videos use exactly one (video_id, track_id). The green box is the
annotation box, and the red box is the prediction being evaluated.

Example:
python new_pipeline/visualize_kalman_vs_lstm.py \
    --ckpt checkpoints/lstm/best.pth \
    --ann_file datasets/mot/annotations/val_half.json \
    --img_root datasets/mot/train \
    --max_frames 150
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

from new_pipeline.dataset_lstm import _tlwh_to_xyah, _xyah_to_cxcywh
from new_pipeline.lstm_predictor import LSTMPredictor
from yolox.tracker.kalman_filter import KalmanFilter


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Visualize Kalman vs Kalman+LSTM")
    parser.add_argument("--ckpt", default="checkpoints/lstm/best.pth")
    parser.add_argument("--ann_file", default="datasets/mot/annotations/val_half.json")
    parser.add_argument("--img_root", default="datasets/mot/train")
    parser.add_argument("--output_dir", default="outputs/lstm_compare")
    parser.add_argument("--video_id", type=int, default=None)
    parser.add_argument("--track_id", type=int, default=None)
    parser.add_argument(
        "--start_frame",
        type=int,
        default=None,
        help="Start from this MOT frame_id after selecting the track.",
    )
    parser.add_argument(
        "--select",
        choices=["longest", "moving"],
        default="moving",
        help="Auto-select track when video_id/track_id are not provided.",
    )
    parser.add_argument(
        "--min_track_len",
        type=int,
        default=80,
        help="Minimum track length when auto-selecting a moving track.",
    )
    parser.add_argument("--max_frames", type=int, default=150)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--alpha0", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--missing_every",
        type=int,
        default=0,
        help="If >0, simulate missing detections every N frames.",
    )
    parser.add_argument(
        "--missing_span",
        type=int,
        default=1,
        help="Number of consecutive missing frames when missing_every is used.",
    )
    return parser


def tlwh_to_cxcywh(tlwh: np.ndarray) -> np.ndarray:
    x, y, w, h = tlwh
    return np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=np.float32)


def cxcywh_to_tlwh(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float32)


def tlwh_to_tlbr(tlwh: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = tlwh
    return (
        int(round(x)),
        int(round(y)),
        int(round(x + w)),
        int(round(y + h)),
    )


def draw_box(
    img: np.ndarray,
    tlwh: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = tlwh_to_tlbr(tlwh)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    y_text = max(18, y1 - 7)
    cv2.putText(
        img,
        label,
        (x1, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def center_error(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ca = tlwh_to_cxcywh(box_a)[:2]
    cb = tlwh_to_cxcywh(box_b)[:2]
    return float(np.linalg.norm(ca - cb))


def iou_tlwh(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = tlwh_to_tlbr(box_a)
    bx1, by1, bx2, by2 = tlwh_to_tlbr(box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def load_checkpoint(path: str, device: torch.device) -> LSTMPredictor:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    hidden_size = int(state["input_proj.weight"].shape[0])
    layer_ids = [
        int(k.rsplit("l", 1)[1])
        for k in state
        if k.startswith("lstm.weight_ih_l")
    ]
    num_layers = max(layer_ids) + 1 if layer_ids else 2
    model = LSTMPredictor(hidden_size=hidden_size, num_layers=num_layers).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_track(args):
    with open(args.ann_file, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    tracks = defaultdict(list)

    for ann in data["annotations"]:
        img = images[ann["image_id"]]
        key = (int(img["video_id"]), int(ann["track_id"]))
        tracks[key].append(
            {
                "frame_id": int(img["frame_id"]),
                "file_name": img["file_name"],
                "width": int(img["width"]),
                "height": int(img["height"]),
                "bbox": np.array(ann["bbox"], dtype=np.float32),
                "conf": float(ann.get("conf", 1.0)),
            }
        )

    if args.video_id is not None and args.track_id is not None:
        key = (args.video_id, args.track_id)
        if key not in tracks:
            raise ValueError(f"Track not found: video_id={args.video_id}, track_id={args.track_id}")
    else:
        if args.select == "longest":
            key = max(tracks, key=lambda k: len(tracks[k]))
        else:
            candidates = {
                k: sorted(v, key=lambda x: x["frame_id"])
                for k, v in tracks.items()
                if len(v) >= args.min_track_len
            }
            if not candidates:
                raise ValueError(
                    f"No track has at least {args.min_track_len} frames. "
                    "Lower --min_track_len or choose --video_id/--track_id."
                )

            def displacement(items):
                first = tlwh_to_cxcywh(items[0]["bbox"])[:2]
                last = tlwh_to_cxcywh(items[-1]["bbox"])[:2]
                return float(np.linalg.norm(last - first))

            key = max(candidates, key=lambda k: displacement(candidates[k]))

    frames = sorted(tracks[key], key=lambda x: x["frame_id"])
    if args.start_frame is not None:
        frames = [f for f in frames if f["frame_id"] >= args.start_frame]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames for track {key}, got {len(frames)}")

    return key, frames


def is_simulated_missing(frame_index: int, args) -> bool:
    if args.missing_every <= 0:
        return False
    if frame_index == 0:
        return False
    return (frame_index % args.missing_every) < max(1, args.missing_span)


def put_header(
    img: np.ndarray,
    title: str,
    frame_id: int,
    track_id: int,
    pred_box: np.ndarray,
    gt_box: np.ndarray,
    missing: bool,
) -> None:
    err = center_error(pred_box, gt_box)
    iou = iou_tlwh(pred_box, gt_box)
    lines = [
        title,
        f"frame={frame_id} track_id={track_id} IoU={iou:.3f} center_err={err:.1f}px",
    ]
    if missing:
        lines.append("simulated missing detection: no KF update")

    x, y = 18, 32
    for line in lines:
        cv2.putText(
            img,
            line,
            (x + 1, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_checkpoint(args.ckpt, device)
    h_lstm, c_lstm = model.init_hidden(batch_size=1, device=device)

    (video_id, track_id), frames = load_track(args)
    img_h, img_w = frames[0]["height"], frames[0]["width"]

    base_name = f"video{video_id:02d}_track{track_id}"
    if args.missing_every > 0:
        base_name += f"_missing{args.missing_every}x{args.missing_span}"
    kalman_path = os.path.join(args.output_dir, f"{base_name}_kalman.mp4")
    lstm_path = os.path.join(args.output_dir, f"{base_name}_kalman_lstm.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_kf = cv2.VideoWriter(kalman_path, fourcc, args.fps, (img_w, img_h))
    writer_lstm = cv2.VideoWriter(lstm_path, fourcc, args.fps, (img_w, img_h))
    if not writer_kf.isOpened() or not writer_lstm.isOpened():
        raise RuntimeError("Could not open video writer. Check OpenCV codec support.")

    kf_only = KalmanFilter()
    kf_lstm = KalmanFilter()
    mean_kf = cov_kf = None
    mean_lstm = cov_lstm = None

    last_bbox = tlwh_to_cxcywh(frames[0]["bbox"])
    velocity = np.zeros(2, dtype=np.float32)
    missing_count = 0
    is_missing = 0
    last_conf = frames[0]["conf"]

    kf_errors, lstm_errors = [], []
    kf_ious, lstm_ious = [], []

    for i, frame in enumerate(frames):
        img_path = os.path.join(args.img_root, frame["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        gt_tlwh = frame["bbox"]
        gt_cxcywh = tlwh_to_cxcywh(gt_tlwh)
        xyah = _tlwh_to_xyah(gt_tlwh)
        missing = is_simulated_missing(i, args)

        if i == 0:
            mean_kf, cov_kf = kf_only.initiate(xyah)
            mean_lstm, cov_lstm = kf_lstm.initiate(xyah)
            pred_kf_cxcywh = gt_cxcywh.copy()
            pred_lstm_cxcywh = gt_cxcywh.copy()

            init_feat = LSTMPredictor.build_input(
                bbox_cxcywh=gt_cxcywh,
                velocity=np.zeros(2, dtype=np.float32),
                delta_t=1.0,
                is_missing=0,
                missing_count=0,
                confidence=last_conf,
                img_w=img_w,
                img_h=img_h,
            )
            x_t = torch.tensor(init_feat[None], dtype=torch.float32, device=device)
            _, h_lstm, c_lstm = model.step(x_t, h_lstm, c_lstm)
        else:
            mean_kf, cov_kf = kf_only.predict(mean_kf, cov_kf)
            pred_kf_cxcywh = _xyah_to_cxcywh(mean_kf[:4])

            mean_lstm, cov_lstm = kf_lstm.predict(mean_lstm, cov_lstm)
            kf_for_lstm = _xyah_to_cxcywh(mean_lstm[:4])
            feat = LSTMPredictor.build_input(
                bbox_cxcywh=last_bbox,
                velocity=velocity,
                delta_t=1.0,
                is_missing=is_missing,
                missing_count=missing_count,
                confidence=last_conf,
                img_w=img_w,
                img_h=img_h,
            )
            x_t = torch.tensor(feat[None], dtype=torch.float32, device=device)
            residual, h_new, c_new = model.step(x_t, h_lstm, c_lstm)
            residual_np = residual[0].detach().cpu().numpy()
            alpha = args.alpha0 * np.exp(-args.beta * missing_count)
            pred_lstm_cxcywh = kf_for_lstm + alpha * residual_np
            pred_lstm_cxcywh = np.clip(
                pred_lstm_cxcywh,
                [0, 0, 1, 1],
                [img_w, img_h, img_w, img_h],
            ).astype(np.float32)

            if not missing:
                mean_kf, cov_kf = kf_only.update(mean_kf, cov_kf, xyah)
                mean_lstm, cov_lstm = kf_lstm.update(mean_lstm, cov_lstm, xyah)
                velocity = (gt_cxcywh[:2] - last_bbox[:2]).astype(np.float32)
                last_bbox = gt_cxcywh.astype(np.float32)
                last_conf = frame["conf"]
                missing_count = 0
                is_missing = 0
            else:
                last_bbox = pred_lstm_cxcywh.astype(np.float32)
                last_conf = 0.0
                missing_count += 1
                is_missing = 1
            h_lstm, c_lstm = h_new, c_new

        pred_kf_tlwh = cxcywh_to_tlwh(pred_kf_cxcywh)
        pred_lstm_tlwh = cxcywh_to_tlwh(pred_lstm_cxcywh)

        kf_errors.append(center_error(pred_kf_tlwh, gt_tlwh))
        lstm_errors.append(center_error(pred_lstm_tlwh, gt_tlwh))
        kf_ious.append(iou_tlwh(pred_kf_tlwh, gt_tlwh))
        lstm_ious.append(iou_tlwh(pred_lstm_tlwh, gt_tlwh))

        img_kf = img.copy()
        img_lstm = img.copy()
        draw_box(img_kf, gt_tlwh, (0, 220, 0), f"GT id={track_id}", 2)
        draw_box(img_kf, pred_kf_tlwh, (0, 0, 255), "Kalman pred", 2)
        draw_box(img_lstm, gt_tlwh, (0, 220, 0), f"GT id={track_id}", 2)
        draw_box(img_lstm, pred_lstm_tlwh, (0, 0, 255), "Kalman+LSTM pred", 2)
        put_header(
            img_kf,
            "Kalman only",
            frame["frame_id"],
            track_id,
            pred_kf_tlwh,
            gt_tlwh,
            missing,
        )
        put_header(
            img_lstm,
            "Kalman + LSTM",
            frame["frame_id"],
            track_id,
            pred_lstm_tlwh,
            gt_tlwh,
            missing,
        )
        writer_kf.write(img_kf)
        writer_lstm.write(img_lstm)

    writer_kf.release()
    writer_lstm.release()

    print("Exported videos")
    print(f"  Kalman      : {kalman_path}")
    print(f"  Kalman+LSTM : {lstm_path}")
    print(f"  video_id={video_id} track_id={track_id} frames={len(frames)}")
    print(f"  Kalman      mean_center_err={np.mean(kf_errors):.3f}px mean_iou={np.mean(kf_ious):.4f}")
    print(f"  Kalman+LSTM mean_center_err={np.mean(lstm_errors):.3f}px mean_iou={np.mean(lstm_ious):.4f}")


if __name__ == "__main__":
    main()
