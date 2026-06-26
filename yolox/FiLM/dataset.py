"""
TrackletReIDDataset — Dataset cho FiLM ReID training.

Mỗi sample = (image_crop, trajectory_history, gate_feats, person_id, next_box)

Format dữ liệu đầu vào (tracklets.pkl):
    List[dict] mỗi dict là 1 tracklet:
    {
        'pid'    : int                  — person identity label
        'frames' : List[int]            — frame indices
        'boxes'  : np.ndarray (T, 4)    — [cx, cy, w, h] normalized [0,1]
        'confs'  : np.ndarray (T,)      — detection confidence tại mỗi frame
        'crops'  : List[str]            — path đến từng cropped image
    }

Dùng prepare_mot_tracklets.py (đính kèm) để tạo file tracklets.pkl từ MOT format.
"""

from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T


class RandomIdentitySampler(Sampler):
    """
    Sample P identities, K images per identity mỗi batch.
    Đảm bảo mỗi batch luôn có positive pairs → triplet loss hoạt động hiệu quả.

    Batch size thực tế = P * K (phải khớp với batch_size trong DataLoader).

    Args:
        data_source : TrackletReIDDataset
        batch_size  : total samples per batch (= P * K)
        num_instances: K — số samples per identity per batch
    """

    def __init__(self, data_source: TrackletReIDDataset, batch_size: int, num_instances: int = 4):
        super().__init__()
        self.data_source   = data_source
        self.batch_size    = batch_size
        self.num_instances = num_instances                  # K
        self.num_pids_per_batch = batch_size // num_instances  # P

        # pid_label → list of sample indices
        self.index_by_pid: dict[int, list[int]] = defaultdict(list)
        for idx, s in enumerate(data_source.samples):
            pid_label = data_source.pid2label[s["pid"]]
            self.index_by_pid[pid_label].append(idx)

        self.pids = list(self.index_by_pid.keys())

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        # Shuffle identities
        pids = self.pids.copy()
        random.shuffle(pids)

        batch_indices = []
        final_indices = []

        for pid in pids:
            idxs = self.index_by_pid[pid].copy()
            random.shuffle(idxs)

            # Nếu ít hơn K samples, sample lại có replacement
            if len(idxs) < self.num_instances:
                idxs = random.choices(idxs, k=self.num_instances)
            else:
                idxs = idxs[:self.num_instances]

            batch_indices.extend(idxs)

            if len(batch_indices) == self.batch_size:
                final_indices.extend(batch_indices)
                batch_indices = []

        # Bỏ phần dư (không đủ 1 batch)
        return iter(final_indices)


class TrackletReIDDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        seq_len: int = 20,
        image_size: tuple = (256, 128),
        min_tracklet_len: int = 6,    # cần ít nhất min_seq + 1 frame
        augment: bool = True,
        cold_start_prob: float = 0.3,
    ):
        self.seq_len = seq_len
        self.min_seq = min_tracklet_len - 1   # số frame history tối thiểu trước anchor
        self.augment = augment
        self.cold_start_prob = cold_start_prob

        # Transforms
        if augment:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Load tracklets
        pkl_path = os.path.join(data_root, "tracklets.pkl")
        with open(pkl_path, "rb") as f:
            tracklets = pickle.load(f)

        # Remap crop paths: replace whatever prefix was baked into the pkl
        # with the current data_root (handles path mismatch across machines).
        crops_root = os.path.join(data_root, "crops")
        for t in tracklets:
            remapped = []
            for p in t.get("crops", []):
                # Find 'crops/' segment and keep everything after it
                marker = "crops" + os.sep
                idx = p.find(marker)
                if idx != -1:
                    rel = p[idx + len(marker):]
                else:
                    rel = os.path.basename(p)
                remapped.append(os.path.join(crops_root, rel))
            t["crops"] = remapped

        self.samples, pid_set = self._build_samples(tracklets)

        # Remap pids to contiguous [0, num_pids)
        pid_list = sorted(pid_set)
        self.pid2label = {pid: i for i, pid in enumerate(pid_list)}
        self.num_pids = len(pid_list)

        print(
            f"[TrackletReIDDataset] {len(self.samples)} samples, "
            f"{self.num_pids} identities loaded from {pkl_path}"
        )

    def _build_samples(self, tracklets):
        samples = []
        pid_set = set()
        for t in tracklets:
            T_len = len(t["frames"])
            if T_len < self.min_seq + 1:
                continue
            pid_set.add(t["pid"])
            # Mỗi valid anchor: từ min_seq đến T-1
            # anchor_idx = i nghĩa là dùng frame i làm anchor,
            # history = frames [max(0, i-seq_len) .. i-1]
            # next_box = boxes[i+1] nếu có, không thì boxes[i]
            for anchor_idx in range(self.min_seq, T_len):
                samples.append({
                    "tracklet":    t,
                    "anchor_idx":  anchor_idx,
                    "pid":         t["pid"],
                })
        return samples, pid_set

    # ─────────────────────────────────────────────────────────────────────

    def _build_traj_seq(self, tracklet: dict, anchor_idx: int) -> np.ndarray:
        """
        Build trajectory sequence (N, 9) kết thúc ngay trước anchor_idx.

        Mỗi bước: [cx, cy, w, h, vx, vy, vw, vh, conf]
        Zero-pad về seq_len nếu history ngắn hơn (padding ở đầu).
        """
        boxes  = tracklet["boxes"]   # (T, 4)
        confs  = tracklet.get("confs", np.ones(len(tracklet["frames"]), dtype=np.float32))

        start = max(0, anchor_idx - self.seq_len)
        hist_boxes = boxes[start:anchor_idx].copy()    # (k, 4), k ≤ seq_len
        hist_confs = confs[start:anchor_idx].copy()    # (k,)

        # Velocities: finite difference
        if len(hist_boxes) > 1:
            vels = np.diff(hist_boxes, axis=0)         # (k-1, 4)
            vels = np.vstack([vels[:1], vels])          # (k, 4)  — pad first with same vel
        else:
            vels = np.zeros_like(hist_boxes)

        seq = np.concatenate(
            [hist_boxes, vels, hist_confs[:, None]], axis=-1
        ).astype(np.float32)                           # (k, 9)

        # Pad trái (history cũ nhất = zeros)
        if len(seq) < self.seq_len:
            pad = np.zeros((self.seq_len - len(seq), 9), dtype=np.float32)
            seq = np.vstack([pad, seq])

        return seq   # (seq_len, 9)

    def _build_gate_feats(self, tracklet: dict, anchor_idx: int) -> np.ndarray:
        """
        Tạo 3 features cho Reliability Gate:
            [track_age_norm, occ_ratio, delta_t_norm]
        """
        frames = tracklet["frames"]
        confs  = tracklet.get("confs", np.ones(len(frames), dtype=np.float32))

        track_age = anchor_idx
        start     = max(0, anchor_idx - self.seq_len)
        hist_confs = confs[start:anchor_idx]
        occ_ratio  = float(np.mean(hist_confs < 0.5))   # tỉ lệ frame bị occlusion

        delta_t = int(frames[anchor_idx]) - int(frames[anchor_idx - 1]) if anchor_idx > 0 else 1
        # delta_t=1 bình thường, >1 = gap
        delta_t_norm = min((delta_t - 1) / 30.0, 1.0)

        return np.array(
            [min(track_age / 100.0, 1.0), occ_ratio, delta_t_norm],
            dtype=np.float32,
        )

    def _load_crop(self, tracklet: dict, idx: int) -> Image.Image:
        path = tracklet["crops"][idx]
        return Image.open(path).convert("RGB")

    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s          = self.samples[idx]
        tracklet   = s["tracklet"]
        anchor_idx = s["anchor_idx"]
        pid_label  = self.pid2label[s["pid"]]

        # Crop ảnh tại anchor frame
        image = self.transform(self._load_crop(tracklet, anchor_idx))

        # Trajectory history (N, 9) kết thúc ngay trước anchor
        traj_seq   = torch.from_numpy(self._build_traj_seq(tracklet, anchor_idx))
        gate_feats = torch.from_numpy(self._build_gate_feats(tracklet, anchor_idx))

        # Cold-start augmentation: giả lập detection mới chưa có trajectory history
        # Để model học embedding nhất quán giữa cold-start và conditioned
        if self.augment and random.random() < self.cold_start_prob:
            traj_seq   = torch.zeros_like(traj_seq)
            gate_feats = torch.zeros_like(gate_feats)

        # Supervision cho motion prediction: next bounding box
        boxes    = tracklet["boxes"]
        next_idx = min(anchor_idx + 1, len(boxes) - 1)
        next_box = torch.from_numpy(boxes[next_idx].astype(np.float32))

        return {
            "image":      image,                          # (3, H, W)
            "traj_seq":   traj_seq,                       # (seq_len, 9)
            "gate_feats": gate_feats,                     # (3,)
            "pid":        torch.tensor(pid_label, dtype=torch.long),
            "next_box":   next_box,                       # (4,)
        }
