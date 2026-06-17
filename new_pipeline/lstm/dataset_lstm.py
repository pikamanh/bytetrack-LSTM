"""
MOTTrackletDataset — builds LSTM training sequences from COCO-format
annotations (the same format ByteTrack already uses).

Data flow
─────────
1. Load COCO JSON (mot/annotations/train.json or mix_det/annotations/train.json).
2. Group annotations by (video_id, track_id) → raw tracklets.
3. For each tracklet:
   a. Simulate Kalman filter frame-by-frame on the GT bboxes.
   b. Compute target residual = GT_cxcywh − KF_predicted_cxcywh.
   c. Randomly drop frames (simulate occlusion); set is_missing=1 for those.
   d. Slide a window of length `seq_len` over the tracklet.
4. Each sample: (x_seq [T, 10], target_residual [T, 4], mask [T]).

The mask is 1 for non-missing frames (where we compute the loss),
0 for artificially dropped frames.
"""

import json
import os
import random
import numpy as np
from torch.utils.data import Dataset

from yolox.tracker.kalman_filter import KalmanFilter


# ── Kalman helpers ────────────────────────────────────────────────────────────

def _tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
    """[x1, y1, w, h] → [cx, cy, aspect, h]"""
    ret = tlwh.copy().astype(np.float64)
    ret[0] += ret[2] / 2
    ret[1] += ret[3] / 2
    ret[2] = ret[2] / ret[3] if ret[3] > 0 else 1.0
    return ret


def _xyah_to_cxcywh(xyah: np.ndarray) -> np.ndarray:
    """[cx, cy, aspect, h] → [cx, cy, w, h]"""
    cx, cy, a, h = xyah
    return np.array([cx, cy, a * h, h], dtype=np.float32)


def simulate_kalman(
    tlwh_seq: np.ndarray,        # [T, 4]  GT bboxes (tlwh)
    frame_ids: np.ndarray,       # [T]     frame indices (may have gaps)
    miss_flags: np.ndarray,      # [T]     1 = artificially dropped
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a Kalman filter over a tracklet, skipping dropped frames.

    Returns
    -------
    kf_preds  : [T, 4]  KF-predicted bbox in [cx, cy, w, h] (before update)
    residuals : [T, 4]  GT_cxcywh − KF_pred (0 for missing frames)
    """
    kf = KalmanFilter()
    T = len(tlwh_seq)
    kf_preds = np.zeros((T, 4), dtype=np.float32)
    residuals = np.zeros((T, 4), dtype=np.float32)

    mean, cov = None, None

    for i, (tlwh, fid, is_miss) in enumerate(zip(tlwh_seq, frame_ids, miss_flags)):
        gt_cxcywh = np.array(
            [tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2, tlwh[2], tlwh[3]],
            dtype=np.float32,
        )
        xyah = _tlwh_to_xyah(tlwh)

        if mean is None:
            # First frame: initialise KF
            mean, cov = kf.initiate(xyah)
            kf_preds[i] = gt_cxcywh   # no prediction yet → no residual
            residuals[i] = 0.0
        else:
            # Predict step
            mean, cov = kf.predict(mean, cov)
            pred_cxcywh = _xyah_to_cxcywh(mean[:4])
            kf_preds[i] = pred_cxcywh

            if is_miss == 0 and tlwh[2] > 0 and tlwh[3] > 0:
                residuals[i] = gt_cxcywh - pred_cxcywh
                mean, cov = kf.update(mean, cov, xyah)
            # else: miss or degenerate bbox → skip KF update, residual stays 0

    return kf_preds, residuals


# ── Dataset ───────────────────────────────────────────────────────────────────

class MOTTrackletDataset(Dataset):
    """
    Parameters
    ----------
    ann_files    : list of COCO-format JSON paths (can mix datasets).
    seq_len      : sliding-window length (frames).
    miss_prob    : probability of randomly dropping a detected frame.
    min_track_len: discard tracklets shorter than this.
    max_missing  : normalisation denominator for missing_count feature.
    augment      : if True, apply random horizontal flip to bboxes.
    """

    def __init__(
        self,
        ann_files: str | os.PathLike | list[str | os.PathLike],
        seq_len: int = 32,
        miss_prob: float = 0.15,
        min_track_len: int = 8,
        max_missing: float = 30.0,
        augment: bool = True,
    ):
        self.seq_len = seq_len
        self.miss_prob = miss_prob
        self.max_missing = max_missing
        self.augment = augment

        self.samples: list[dict] = []   # each sample = one sliding window
        if isinstance(ann_files, (str, os.PathLike)):
            ann_files = [ann_files]
        self._build(ann_files, min_track_len)

    # ── Build ─────────────────────────────────────────────────────────────

    def _build(self, ann_files: list[str | os.PathLike], min_track_len: int):
        for ann_file in ann_files:
            print(f"[Dataset] loading {ann_file}")
            with open(ann_file) as f:
                data = json.load(f)

            # image_id → metadata
            img_meta: dict[int, dict] = {
                img["id"]: img for img in data["images"]
            }

            # group annotations by (video_id, track_id)
            tracklets: dict[tuple, list] = {}
            for ann in data["annotations"]:
                img = img_meta[ann["image_id"]]
                key = (img["video_id"], ann["track_id"])
                tracklets.setdefault(key, []).append({
                    "frame_id": img["frame_id"],
                    "bbox": ann["bbox"],          # [x, y, w, h]
                    "conf": ann.get("conf", 1.0),
                    "img_w": img["width"],
                    "img_h": img["height"],
                })

            for key, frames in tracklets.items():
                frames.sort(key=lambda x: x["frame_id"])
                if len(frames) < min_track_len:
                    continue
                self._add_tracklet(frames)

        print(f"[Dataset] total samples: {len(self.samples)}")

    def _add_tracklet(self, frames: list[dict]):
        """Slide a window over one tracklet and store samples."""
        # Drop frames with zero/negative width or height (degenerate annotations)
        frames = [f for f in frames if f["bbox"][2] > 0 and f["bbox"][3] > 0]
        if len(frames) < 2:
            return

        T_full = len(frames)
        img_w = float(frames[0]["img_w"])
        img_h = float(frames[0]["img_h"])

        tlwh_arr = np.array([f["bbox"] for f in frames], dtype=np.float32)
        fid_arr = np.array([f["frame_id"] for f in frames], dtype=np.int32)
        conf_arr = np.array([f["conf"] for f in frames], dtype=np.float32)

        stride = max(1, self.seq_len // 4)
        starts = list(range(0, T_full - self.seq_len + 1, stride))
        if not starts:
            starts = [0]

        for start in starts:
            end = min(start + self.seq_len, T_full)
            self.samples.append({
                "tlwh": tlwh_arr[start:end],
                "fid": fid_arr[start:end],
                "conf": conf_arr[start:end],
                "img_w": img_w,
                "img_h": img_h,
            })

    # ── Item ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        tlwh = s["tlwh"].copy()         # [T, 4]
        fid = s["fid"].copy()
        conf = s["conf"].copy()
        img_w, img_h = s["img_w"], s["img_h"]
        T = len(tlwh)

        # Pad if shorter than seq_len
        if T < self.seq_len:
            pad = self.seq_len - T
            tlwh = np.pad(tlwh, ((0, pad), (0, 0)))
            fid = np.pad(fid, (0, pad), constant_values=fid[-1] + 1)
            conf = np.pad(conf, (0, pad))

        # Optional horizontal flip augmentation
        if self.augment and random.random() < 0.5:
            tlwh[:, 0] = img_w - tlwh[:, 0] - tlwh[:, 2]

        # Randomly drop some frames (occlusion simulation)
        miss_flags = np.zeros(self.seq_len, dtype=np.float32)
        for i in range(1, self.seq_len):    # never drop the first frame
            if random.random() < self.miss_prob:
                miss_flags[i] = 1.0
        # Padded frames must be treated as missing so KF never receives
        # zero-bbox measurements (which would break positive-definiteness)
        miss_flags[T:] = 1.0

        # Valid mask: 1 on observed, non-padded frames
        valid_mask = np.ones(self.seq_len, dtype=np.float32)
        valid_mask[T:] = 0.0
        valid_mask[miss_flags == 1] = 0.0  # don't supervise missing frames

        # Simulate Kalman → residuals
        kf_preds, residuals = simulate_kalman(
            tlwh[:self.seq_len], fid[:self.seq_len], miss_flags
        )

        # Build LSTM input features
        cxcywh = np.stack([
            tlwh[:, 0] + tlwh[:, 2] / 2,
            tlwh[:, 1] + tlwh[:, 3] / 2,
            tlwh[:, 2],
            tlwh[:, 3],
        ], axis=-1).astype(np.float32)   # [T, 4]

        x_seq = self._build_feature_seq(
            cxcywh, miss_flags, conf, fid[:self.seq_len], img_w, img_h
        )  # [T, 10]

        # Residuals kept in pixel space — LSTM outputs pixels, applied directly
        # to KF bbox at inference without any denorm step.

        return {
            "x_seq": x_seq.astype(np.float32),           # [T, 10]
            "target_residual": residuals.astype(np.float32),  # [T, 4]
            "mask": valid_mask.astype(np.float32),        # [T]
        }

    # ── Feature construction ──────────────────────────────────────────────

    @staticmethod
    def _build_feature_seq(
        cxcywh: np.ndarray,      # [T, 4]
        miss_flags: np.ndarray,  # [T]
        conf: np.ndarray,        # [T]
        fid: np.ndarray,         # [T]  actual frame indices
        img_w: float,
        img_h: float,
        max_missing: float = 30.0,
    ) -> np.ndarray:             # [T, 10]
        T = len(cxcywh)
        x_seq = np.zeros((T, 10), dtype=np.float32)
        missing_count = 0

        for i in range(T):
            cx, cy, w, h = cxcywh[i]
            if i == 0:
                vx, vy = 0.0, 0.0
                delta_t = 1.0
            else:
                dt = float(fid[i] - fid[i - 1])
                delta_t = max(dt, 1.0)   # guard against padding artifacts
                vx = (cx - cxcywh[i - 1, 0]) / delta_t
                vy = (cy - cxcywh[i - 1, 1]) / delta_t

            is_miss = float(miss_flags[i])
            missing_count = missing_count + 1 if is_miss else 0
            score = 0.0 if is_miss else float(conf[i])

            x_seq[i] = [
                cx / img_w,
                cy / img_h,
                w / img_w,
                h / img_h,
                vx / img_w,
                vy / img_h,
                delta_t,
                is_miss,
                min(missing_count / max_missing, 1.0),
                score,
            ]
        return x_seq
