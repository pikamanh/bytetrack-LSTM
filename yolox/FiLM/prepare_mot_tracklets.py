"""
Chuyển đổi MOT-format dataset sang tracklets.pkl cho FiLM training.

MOT format:
    <data_root>/
        <sequence>/
            img1/          ← frame images (000001.jpg, ...)
            gt/gt.txt      ← ground truth annotations

gt.txt format (comma-separated):
    frame, id, x1, y1, w, h, conf, class, visibility

Kết quả:
    <output_dir>/tracklets.pkl  — List[dict] mỗi dict là một tracklet

Usage:
    python -m yolox.FiLM.prepare_mot_tracklets \
        --data-root /path/to/MOT17/train \
        --output-dir /path/to/tracklets_data \
        --min-tracklet-len 10
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_gt_file(gt_path: str) -> dict:
    """
    Đọc gt.txt và trả về dict:
        {frame_id: [{id, x1, y1, w, h, conf, class}]}
    """
    annotations = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            frame_id  = int(parts[0])
            track_id  = int(parts[1])
            x1        = float(parts[2])
            y1        = float(parts[3])
            w         = float(parts[4])
            h         = float(parts[5])
            conf      = float(parts[6]) if len(parts) > 6 else 1.0
            obj_class = int(parts[7])   if len(parts) > 7 else 1

            # Chỉ lấy người (class=1) và visible (conf > 0)
            if obj_class != 1 or conf <= 0:
                continue

            annotations[frame_id].append({
                "id": track_id, "x1": x1, "y1": y1, "w": w, "h": h, "conf": conf
            })
    return annotations


def _detect_frame_format(img_dir: str, frame_id: int) -> str | None:
    """
    Tự động phát hiện format tên file frame.
    Thử các zero-padding phổ biến: 8 (DanceTrack), 6 (MOT17), 5, 4 digits.
    Trả về format string như '{:08d}' hoặc None nếu không tìm thấy.
    """
    for n_digits in [8, 6, 5, 4]:
        for ext in [".jpg", ".png"]:
            candidate = os.path.join(img_dir, f"{frame_id:0{n_digits}d}{ext}")
            if os.path.isfile(candidate):
                return f"{{:0{n_digits}d}}{ext}"
    # Fallback: list thư mục để đoán format
    try:
        files = sorted(os.listdir(img_dir))
        if files:
            name = files[0]
            stem, ext = os.path.splitext(name)
            n_digits = len(stem)
            return f"{{:0{n_digits}d}}{ext}"
    except OSError:
        pass
    return None


def build_tracklets_from_sequence(
    seq_dir: str,
    crop_output_dir: str,
    min_tracklet_len: int = 10,
    pid_offset: int = 0,
) -> list:
    """
    Xây dựng tracklets từ 1 sequence MOT.
    Crop và lưu từng frame của mỗi tracklet.

    Returns: List[dict] — danh sách tracklets
    """
    gt_path  = os.path.join(seq_dir, "gt", "gt.txt")
    img_dir  = os.path.join(seq_dir, "img1")
    seq_name = os.path.basename(seq_dir)

    if not os.path.isfile(gt_path):
        print(f"  [Skip] gt.txt not found: {gt_path}")
        return []

    annotations = load_gt_file(gt_path)
    frame_ids   = sorted(annotations.keys())

    # Nhóm theo track_id
    tracklets_raw = defaultdict(lambda: {"frames": [], "boxes_pixel": [], "confs": []})

    # Auto-detect frame naming format (MOT17=6 digits, DanceTrack=8 digits, ...)
    frame_fmt = _detect_frame_format(img_dir, frame_ids[0])
    if frame_fmt is None:
        print(f"  [Skip] Cannot detect frame format in: {img_dir}")
        return []

    # Lấy kích thước ảnh từ frame đầu tiên
    first_img_path = os.path.join(img_dir, frame_fmt.format(frame_ids[0]))
    sample_img = cv2.imread(first_img_path)
    if sample_img is None:
        print(f"  [Skip] Cannot read frame: {first_img_path}")
        return []
    img_h, img_w = sample_img.shape[:2]
    print(f"  Frame format: '{frame_fmt}'  size: {img_w}×{img_h}")

    for frame_id in frame_ids:
        for ann in annotations[frame_id]:
            tid = ann["id"]
            x1, y1, w, h = ann["x1"], ann["y1"], ann["w"], ann["h"]
            tracklets_raw[tid]["frames"].append(frame_id)
            tracklets_raw[tid]["boxes_pixel"].append([x1, y1, x1 + w, y1 + h])
            tracklets_raw[tid]["confs"].append(ann["conf"])

    # Cắt crop và build final tracklet dicts
    tracklets = []
    os.makedirs(crop_output_dir, exist_ok=True)

    for tid, raw in tqdm(tracklets_raw.items(), desc=f"  {seq_name}", leave=False):
        if len(raw["frames"]) < min_tracklet_len:
            continue

        frames       = raw["frames"]
        boxes_pixel  = np.array(raw["boxes_pixel"], dtype=np.float32)   # (T, 4) [x1,y1,x2,y2]
        confs        = np.array(raw["confs"],        dtype=np.float32)   # (T,)

        # Normalize boxes → [cx, cy, w, h] in [0,1]
        cx  = (boxes_pixel[:, 0] + boxes_pixel[:, 2]) / 2 / img_w
        cy  = (boxes_pixel[:, 1] + boxes_pixel[:, 3]) / 2 / img_h
        bw  = (boxes_pixel[:, 2] - boxes_pixel[:, 0]) / img_w
        bh  = (boxes_pixel[:, 3] - boxes_pixel[:, 1]) / img_h
        boxes_norm = np.stack([cx, cy, bw, bh], axis=-1)

        # Crop và lưu ảnh
        crop_paths = []
        for i, frame_id in enumerate(frames):
            img_path = os.path.join(img_dir, frame_fmt.format(frame_id))
            if not os.path.isfile(img_path):
                crop_paths.append(None)
                continue

            frame = cv2.imread(img_path)
            if frame is None:
                crop_paths.append(None)
                continue

            x1, y1, x2, y2 = boxes_pixel[i]
            x1 = max(0, int(x1));  y1 = max(0, int(y1))
            x2 = min(img_w, int(x2));  y2 = min(img_h, int(y2))
            if x2 <= x1 or y2 <= y1:
                crop_paths.append(None)
                continue

            crop = frame[y1:y2, x1:x2]
            crop_fname = f"{seq_name}_t{tid:04d}_f{frame_id:06d}.jpg"
            crop_path  = os.path.join(crop_output_dir, crop_fname)
            cv2.imwrite(crop_path, crop)
            crop_paths.append(crop_path)

        # Filter frames có crop hợp lệ
        valid_mask = [p is not None for p in crop_paths]
        if sum(valid_mask) < min_tracklet_len:
            continue

        valid_indices  = [i for i, v in enumerate(valid_mask) if v]
        frames_valid   = [frames[i]      for i in valid_indices]
        boxes_valid    = boxes_norm[valid_indices]
        confs_valid    = confs[valid_indices]
        crops_valid    = [crop_paths[i]  for i in valid_indices]

        # pid = seq_offset * 10000 + track_id (đảm bảo unique across sequences)
        pid = pid_offset + tid

        tracklets.append({
            "pid":    pid,
            "frames": frames_valid,
            "boxes":  boxes_valid,    # (T, 4) normalized [cx,cy,w,h]
            "confs":  confs_valid,    # (T,)
            "crops":  crops_valid,    # (T,) paths to cropped images
        })

    return tracklets


def build_tracklets_from_mot_root(
    data_root: str,
    output_dir: str,
    min_tracklet_len: int = 10,
):
    """
    Duyệt qua tất cả sequences trong data_root và build tracklets.pkl
    """
    os.makedirs(output_dir, exist_ok=True)
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    sequences = sorted([
        d for d in Path(data_root).iterdir()
        if d.is_dir() and (d / "gt" / "gt.txt").exists()
    ])

    if not sequences:
        raise FileNotFoundError(
            f"Không tìm thấy sequence nào với gt/gt.txt trong: {data_root}"
        )

    print(f"[Prepare] Tìm thấy {len(sequences)} sequences: {[s.name for s in sequences]}")

    all_tracklets = []
    for seq_idx, seq_path in enumerate(sequences):
        pid_offset = seq_idx * 100000   # đảm bảo pid unique across sequences
        print(f"\nProcessing: {seq_path.name}")
        tracklets = build_tracklets_from_sequence(
            str(seq_path),
            crop_output_dir=os.path.join(crops_dir, seq_path.name),
            min_tracklet_len=min_tracklet_len,
            pid_offset=pid_offset,
        )
        print(f"  → {len(tracklets)} tracklets (min_len={min_tracklet_len})")
        all_tracklets.extend(tracklets)

    output_path = os.path.join(output_dir, "tracklets.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_tracklets, f)

    unique_pids = len({t["pid"] for t in all_tracklets})
    total_frames = sum(len(t["frames"]) for t in all_tracklets)
    print(
        f"\n[Done] {len(all_tracklets)} tracklets, "
        f"{unique_pids} unique IDs, "
        f"{total_frames} total frames"
    )
    print(f"       Saved → {output_path}")


def parse_args():
    p = argparse.ArgumentParser("Prepare MOT tracklets for FiLM ReID training")
    p.add_argument("--data-root",       required=True,
                   help="path đến folder chứa các sequence MOT (có gt/gt.txt)")
    p.add_argument("--output-dir",      required=True,
                   help="folder lưu tracklets.pkl và crops")
    p.add_argument("--min-tracklet-len", type=int, default=10,
                   help="bỏ qua tracklets ngắn hơn N frames")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_tracklets_from_mot_root(
        data_root=args.data_root,
        output_dir=args.output_dir,
        min_tracklet_len=args.min_tracklet_len,
    )
