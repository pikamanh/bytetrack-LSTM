"""COCO/MOT dataset for LSTM association-score training."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset

from .association_lstm import build_detection_feature, build_pair_features, bbox_iou_cxcywh, tlwh_to_cxcywh
from .lstm_predictor import LSTMPredictor


class MOTAssociationDataset(Dataset):
    """Build positive/negative track-history and candidate-detection pairs.

    Each sample contains:
      history: [seq_len, 10] LSTM motion features ending before candidate frame
      det:     [5] normalized candidate bbox + confidence
      pair:    [8] relative motion/IoU features between track and candidate
      label:   1 if same GT track_id, else 0
    """

    def __init__(
        self,
        ann_files: str | os.PathLike | list[str | os.PathLike],
        seq_len: int = 16,
        min_history: int = 2,
        negatives_per_positive: int = 3,
        max_samples: int | None = None,
    ):
        if isinstance(ann_files, (str, os.PathLike)):
            ann_files = [ann_files]

        self.seq_len = seq_len
        self.min_history = min_history
        self.negatives_per_positive = negatives_per_positive
        self.tracklets: dict[tuple[int, int, int], list[dict]] = {}
        self.samples: list[dict] = []
        self.num_positive = 0
        self.num_negative = 0

        self._build(list(ann_files), max_samples=max_samples)

    def _build(self, ann_files: list[str | os.PathLike], max_samples: int | None):
        for file_idx, ann_path in enumerate(ann_files):
            with open(ann_path, "r") as f:
                data = json.load(f)

            images = {img["id"]: img for img in data["images"]}
            tracklets = defaultdict(list)
            frame_dets = defaultdict(list)

            for ann in data["annotations"]:
                bbox = ann.get("bbox", None)
                if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                img = images[ann["image_id"]]
                video_id = int(img["video_id"])
                frame_id = int(img["frame_id"])
                track_id = int(ann["track_id"])
                item = {
                    "file_idx": file_idx,
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "track_id": track_id,
                    "bbox": np.asarray(bbox, dtype=np.float32),
                    "conf": float(ann.get("conf", 1.0)),
                    "img_w": float(img["width"]),
                    "img_h": float(img["height"]),
                }
                key = (file_idx, video_id, track_id)
                tracklets[key].append(item)
                frame_dets[(file_idx, video_id, frame_id)].append(item)

            for key, frames in tracklets.items():
                frames.sort(key=lambda x: x["frame_id"])
                if len(frames) <= self.min_history:
                    continue
                self.tracklets[key] = frames

            for key, frames in self.tracklets.items():
                if key[0] != file_idx:
                    continue
                for det_idx in range(self.min_history, len(frames)):
                    pos = frames[det_idx]
                    self._append_sample(key, det_idx, pos, 1.0)

                    candidates = [
                        det
                        for det in frame_dets[(pos["file_idx"], pos["video_id"], pos["frame_id"])]
                        if det["track_id"] != pos["track_id"]
                    ]
                    for neg in self._select_hard_negatives(frames, det_idx, candidates):
                        self._append_sample(key, det_idx, neg, 0.0)

                    if max_samples is not None and len(self.samples) >= max_samples:
                        self.samples = self.samples[:max_samples]
                        self._recount_labels()
                        print(
                            f"[AssociationDataset] samples={len(self.samples)} "
                            f"pos={self.num_positive} neg={self.num_negative}"
                        )
                        return

        print(
            f"[AssociationDataset] samples={len(self.samples)} "
            f"pos={self.num_positive} neg={self.num_negative}"
        )

    def _append_sample(self, track_key: tuple[int, int, int], det_idx: int, candidate: dict, label: float):
        self.samples.append(
            {
                "track_key": track_key,
                "det_idx": det_idx,
                "candidate": candidate,
                "label": float(label),
            }
        )
        if label > 0.5:
            self.num_positive += 1
        else:
            self.num_negative += 1

    def _recount_labels(self):
        self.num_positive = sum(1 for sample in self.samples if sample["label"] > 0.5)
        self.num_negative = len(self.samples) - self.num_positive

    def _select_hard_negatives(self, frames: list[dict], det_idx: int, candidates: list[dict]) -> list[dict]:
        if not candidates or self.negatives_per_positive <= 0:
            return []

        last = frames[det_idx - 1]
        last_bbox = tlwh_to_cxcywh(last["bbox"])
        if det_idx >= 2:
            prev = tlwh_to_cxcywh(frames[det_idx - 2]["bbox"])
            dt = max(float(last["frame_id"] - frames[det_idx - 2]["frame_id"]), 1.0)
            velocity = (last_bbox[:2] - prev[:2]) / dt
        else:
            velocity = np.zeros(2, dtype=np.float32)

        delta_t = max(float(frames[det_idx]["frame_id"] - last["frame_id"]), 1.0)
        pred_bbox = last_bbox.copy()
        pred_bbox[:2] += velocity * delta_t

        def hardness(det: dict):
            det_bbox = tlwh_to_cxcywh(det["bbox"])
            iou = bbox_iou_cxcywh(pred_bbox, det_bbox)
            dist = np.linalg.norm(det_bbox[:2] - pred_bbox[:2])
            return (-iou, dist)

        ranked = sorted(candidates, key=hardness)
        return ranked[: self.negatives_per_positive]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        frames = self.tracklets[sample["track_key"]]
        det_idx = sample["det_idx"]
        candidate = sample["candidate"]

        history = self._build_history(frames, det_idx)
        det_bbox = tlwh_to_cxcywh(candidate["bbox"])
        det_feat = build_detection_feature(det_bbox, candidate["conf"], candidate["img_w"], candidate["img_h"])

        last = frames[det_idx - 1]
        last_bbox = tlwh_to_cxcywh(last["bbox"])
        velocity = self._last_velocity(frames, det_idx)
        delta_t = max(float(candidate["frame_id"] - last["frame_id"]), 1.0)
        pair_feat = build_pair_features(
            last_bbox,
            velocity,
            det_bbox,
            delta_t,
            candidate["img_w"],
            candidate["img_h"],
        )

        return {
            "history": history.astype(np.float32),
            "det": det_feat.astype(np.float32),
            "pair": pair_feat.astype(np.float32),
            "label": np.asarray(sample["label"], dtype=np.float32),
        }

    def _build_history(self, frames: list[dict], det_idx: int) -> np.ndarray:
        start = max(0, det_idx - self.seq_len)
        hist = frames[start:det_idx]
        features = np.zeros((self.seq_len, LSTMPredictor.INPUT_DIM), dtype=np.float32)
        offset = self.seq_len - len(hist)

        prev_bbox = None
        prev_frame = None
        for i, obs in enumerate(hist):
            bbox = tlwh_to_cxcywh(obs["bbox"])
            if prev_bbox is None:
                velocity = np.zeros(2, dtype=np.float32)
                delta_t = 1.0
            else:
                delta_t = max(float(obs["frame_id"] - prev_frame), 1.0)
                velocity = (bbox[:2] - prev_bbox[:2]) / delta_t

            features[offset + i] = LSTMPredictor.build_input(
                bbox_cxcywh=bbox,
                velocity=velocity,
                delta_t=delta_t,
                is_missing=0,
                missing_count=0,
                confidence=obs["conf"],
                img_w=obs["img_w"],
                img_h=obs["img_h"],
            )
            prev_bbox = bbox
            prev_frame = obs["frame_id"]

        return features

    def _last_velocity(self, frames: list[dict], det_idx: int) -> np.ndarray:
        if det_idx < 2:
            return np.zeros(2, dtype=np.float32)
        last = frames[det_idx - 1]
        prev = frames[det_idx - 2]
        last_bbox = tlwh_to_cxcywh(last["bbox"])
        prev_bbox = tlwh_to_cxcywh(prev["bbox"])
        delta_t = max(float(last["frame_id"] - prev["frame_id"]), 1.0)
        return ((last_bbox[:2] - prev_bbox[:2]) / delta_t).astype(np.float32)
