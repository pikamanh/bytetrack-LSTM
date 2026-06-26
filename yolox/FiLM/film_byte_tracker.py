"""
FiLMByteTracker — BYTETracker with trajectory-conditioned FiLM ReID.

Architecture:
    STrackFiLM  : STrack + trajectory history buffer (traj, frame_ids)
    FiLMExtractor: wraps FiLMReIDModel, handles crop/preprocess/batch
    FiLMByteTracker: BYTETracker subclass that uses FiLM features

Flow per frame:
    1. Cold-start FiLM features for all high-conf detections (traj_seq=None)
    2. First association: IoU + cold-start appearance (same as ByteTrack+ReID)
    3. Post-match refinement: recompute matched track features with trajectory context
    4. Second association: IoU only on low-conf dets (unchanged)
    5. Track state management (unchanged from BYTETracker)
"""

from __future__ import annotations

import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from yolox.tracker.byte_tracker import (
    STrack,
    BYTETracker,
    joint_stracks,
    sub_stracks,
    remove_duplicate_stracks,
)
from yolox.tracker.basetrack import TrackState
from yolox.tracker import matching

try:
    from .film_resnest_reid import FiLMReIDModel
except ImportError:
    _film_dir = os.path.dirname(os.path.abspath(__file__))
    if _film_dir not in sys.path:
        sys.path.insert(0, _film_dir)
    from film_resnest_reid import FiLMReIDModel


_SEQ_LEN = 20
_CROP_SIZE = (256, 128)   # H × W, matches training


# ─────────────────────────────────────────────────────────────────────────────
# STrackFiLM
# ─────────────────────────────────────────────────────────────────────────────

class STrackFiLM(STrack):
    """STrack with a trajectory history buffer for FiLM conditioning."""

    def __init__(self, tlwh, score, feat=None, alpha=0.9):
        super().__init__(tlwh, score, feat=feat, alpha=alpha)
        self._traj: list[np.ndarray] = []   # list of (9,): cx,cy,w,h,vx,vy,vw,vh,conf
        self._frame_ids: list[int] = []
        self._prev_cxcywh: np.ndarray | None = None

    # ── trajectory helpers ────────────────────────────────────────────────

    def _push_traj(self, tlwh: np.ndarray, score: float, frame_id: int) -> None:
        x1, y1, w, h = tlwh
        cx = x1 + w * 0.5
        cy = y1 + h * 0.5
        cur = np.array([cx, cy, w, h], dtype=np.float32)

        vel = cur - self._prev_cxcywh if self._prev_cxcywh is not None else np.zeros(4, np.float32)
        self._prev_cxcywh = cur.copy()

        step = np.concatenate([cur, vel, [score]], dtype=np.float32)
        self._traj.append(step)
        self._frame_ids.append(frame_id)
        if len(self._traj) > _SEQ_LEN:
            self._traj.pop(0)
            self._frame_ids.pop(0)

    def get_traj_seq(self) -> np.ndarray | None:
        """Returns (N, 9) array or None when history is empty."""
        return np.stack(self._traj, axis=0) if self._traj else None

    def get_gate_feats(self, frame_id: int) -> np.ndarray:
        """Returns [age_norm, occ_ratio, delta_t_norm] as (3,) float32."""
        age = self.tracklet_len
        occ_ratio = 0.0
        if len(self._traj) >= 2:
            confs = np.array([s[8] for s in self._traj], dtype=np.float32)
            occ_ratio = float(np.mean(confs < 0.5))
        delta_t = 1
        if len(self._frame_ids) >= 2:
            delta_t = int(self._frame_ids[-1]) - int(self._frame_ids[-2])
        return np.array([
            min(age / 100.0, 1.0),
            occ_ratio,
            min((delta_t - 1) / 30.0, 1.0),
        ], dtype=np.float32)

    # ── override lifecycle to append trajectory ───────────────────────────

    def activate(self, kalman_filter, frame_id: int) -> None:
        super().activate(kalman_filter, frame_id)
        self._push_traj(self.tlwh, self.score, frame_id)

    def re_activate(self, new_track, frame_id: int, new_id: bool = False) -> None:
        super().re_activate(new_track, frame_id, new_id=new_id)
        self._push_traj(self.tlwh, new_track.score, frame_id)

    def update(self, new_track, frame_id: int) -> None:
        super().update(new_track, frame_id)
        self._push_traj(self.tlwh, new_track.score, frame_id)


# ─────────────────────────────────────────────────────────────────────────────
# FiLMExtractor
# ─────────────────────────────────────────────────────────────────────────────

class FiLMExtractor:
    """Wraps FiLMReIDModel for per-frame crop extraction."""

    def __init__(
        self,
        ckpt_path: str,
        num_classes: int | None = None,
        device: str = "cuda",
        seq_len: int = _SEQ_LEN,
        motion_hidden_dim: int | None = None,
        fastreid_config: str | None = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        train_args = ckpt.get("args", {})

        # Auto-detect num_classes and motion_hidden_dim from checkpoint
        if num_classes is None:
            w = state_dict.get("classifier.weight")
            if w is not None:
                num_classes = int(w.shape[0])
            else:
                raise ValueError(
                    "Cannot detect num_classes from checkpoint. Pass --film-num-classes."
                )
        if motion_hidden_dim is None:
            motion_hidden_dim = int(train_args.get("motion_hidden_dim", 128))

        fastreid_root = os.path.join(_root, "fast-reid")
        if os.path.isdir(fastreid_root) and fastreid_root not in sys.path:
            sys.path.insert(0, fastreid_root)

        self.model = FiLMReIDModel(
            num_classes=num_classes,
            motion_hidden_dim=motion_hidden_dim,
            fastreid_config=fastreid_config,
        )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[FiLMExtractor] {len(missing)} missing keys (expected if backbone partial).")
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = T.Compose([
            T.Resize(_CROP_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(
            f"[FiLMExtractor] Loaded: {ckpt_path} "
            f"(num_classes={num_classes}, motion_hidden={motion_hidden_dim})"
        )

    def _read_frame(self, frame) -> np.ndarray | None:
        if isinstance(frame, str):
            frame = cv2.imread(frame)
        return frame

    def _crop_tlbrs(self, frame: np.ndarray, tlbrs) -> list:
        H, W = frame.shape[:2]
        crops = []
        for tlbr in tlbrs:
            x1, y1, x2, y2 = tlbr[:4]
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(W, int(np.ceil(x2)))
            y2 = min(H, int(np.ceil(y2)))
            if x2 <= x1 or y2 <= y1:
                patch = np.zeros((32, 16, 3), dtype=np.uint8)
            else:
                patch = frame[y1:y2, x1:x2]
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            crops.append(self.preprocess(Image.fromarray(rgb)))
        return crops

    @torch.no_grad()
    def extract(
        self,
        frame,
        tlbrs: list,
        traj_seqs: list | None = None,
        gate_feats_list: list | None = None,
    ) -> list[np.ndarray | None]:
        """
        frame        : ndarray (BGR) or file path, or None
        tlbrs        : list of [x1, y1, x2, y2]
        traj_seqs    : list of np.ndarray (N_i, 9) or None, one per detection
        gate_feats_list: list of np.ndarray (3,) or None, one per detection

        Returns list of np.ndarray (2048,) or None on failure.
        """
        if frame is None or len(tlbrs) == 0:
            return [None] * len(tlbrs)

        frame = self._read_frame(frame)
        if frame is None:
            return [None] * len(tlbrs)

        H, W = frame.shape[:2]
        crops = self._crop_tlbrs(frame, tlbrs)
        images = torch.stack(crops).to(self.device)   # (B, 3, H, W)

        # Build trajectory tensor
        traj_tensor = None
        if traj_seqs is not None and any(t is not None for t in traj_seqs):
            padded = []
            max_len = max(
                (len(t) for t in traj_seqs if t is not None and len(t) > 0), default=1
            )
            for traj in traj_seqs:
                if traj is None or len(traj) == 0:
                    padded.append(torch.zeros(max_len, 9, device=self.device))
                    continue
                t = traj.astype(np.float32).copy()
                # Normalize position/velocity by image size (match training convention)
                t[:, [0, 2, 4, 6]] /= W   # cx, w, vx, vw
                t[:, [1, 3, 5, 7]] /= H   # cy, h, vy, vh
                t = t[-self.seq_len:]
                n = t.shape[0]
                if n < max_len:
                    pad = np.zeros((max_len - n, 9), dtype=np.float32)
                    t = np.concatenate([pad, t], axis=0)
                padded.append(torch.from_numpy(t).to(self.device))
            traj_tensor = torch.stack(padded)   # (B, N, 9)

        # Build gate tensor
        gate_tensor = None
        if gate_feats_list is not None:
            gates = [
                np.zeros(3, dtype=np.float32) if g is None else g.astype(np.float32)
                for g in gate_feats_list
            ]
            gate_tensor = torch.from_numpy(np.stack(gates)).to(self.device)

        feats = self.model.extract_feat(images, traj_tensor, gate_tensor)  # (B, 2048)
        return [feats[i].cpu().numpy() for i in range(feats.shape[0])]


# ─────────────────────────────────────────────────────────────────────────────
# FiLMByteTracker
# ─────────────────────────────────────────────────────────────────────────────

class FiLMByteTracker(BYTETracker):
    """BYTETracker with FiLM trajectory-conditioned ReID."""

    def __init__(self, args, frame_rate: int = 30):
        # Prevent base class from building a separate ReID extractor
        _with_reid = getattr(args, "with_reid", False)
        args.with_reid = False
        super().__init__(args, frame_rate)
        args.with_reid = _with_reid

        self.film_extractor = FiLMExtractor(
            ckpt_path=args.film_ckpt,
            num_classes=getattr(args, "film_num_classes", None),
            device=getattr(args, "film_device", "cuda"),
            seq_len=getattr(args, "film_seq_len", _SEQ_LEN),
            motion_hidden_dim=getattr(args, "film_motion_hidden", None),
            fastreid_config=getattr(args, "film_fastreid_config", None),
        )
        self.reid_weight = getattr(args, "reid_weight", 0.35)
        self.reid_thresh = getattr(args, "reid_thresh", 0.7)
        self.reid_alpha = getattr(args, "reid_alpha", 0.9)

    # ── override to produce STrackFiLM ───────────────────────────────────

    def _make_detections(self, tlbrs, scores, features):
        return [
            STrackFiLM(STrackFiLM.tlbr_to_tlwh(tlbr), score, feat, alpha=self.reid_alpha)
            for tlbr, score, feat in zip(tlbrs, scores, features)
        ]

    # ── override ReID methods ─────────────────────────────────────────────

    def _extract_reid_features(self, frame, tlbrs):
        """Cold-start extraction — no trajectory context."""
        if frame is None or len(tlbrs) == 0:
            return [None] * len(tlbrs)
        return self.film_extractor.extract(frame, tlbrs)

    def _extract_conditioned_features(self, frame, tlbrs, tracks: list[STrackFiLM]):
        """Trajectory-conditioned extraction using each track's history."""
        traj_seqs = [t.get_traj_seq() for t in tracks]
        gate_feats = [t.get_gate_feats(self.frame_id) for t in tracks]
        return self.film_extractor.extract(frame, tlbrs, traj_seqs, gate_feats)

    def _fuse_reid(self, iou_dists, tracks, detections):
        """Always fuse appearance (overrides base class with_reid check)."""
        if iou_dists.size == 0:
            return iou_dists
        if not all(t.smooth_feat is not None for t in tracks):
            return iou_dists
        if not all(d.curr_feat is not None for d in detections):
            return iou_dists
        emb_dists = matching.embedding_distance(tracks, detections)
        emb_dists[emb_dists > self.reid_thresh] = 1.0
        return (1 - self.reid_weight) * iou_dists + self.reid_weight * emb_dists

    # ── main update (mirrors BYTETracker.update + post-match FiLM step) ──

    def update(self, output_results, img_info, img_size, frame=None):
        self.frame_id += 1
        activated_starcks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_second = np.logical_and(scores > 0.1, scores < self.args.track_thresh)

        dets          = bboxes[remain_inds]
        scores_keep   = scores[remain_inds]
        dets_second   = bboxes[inds_second]
        scores_second = scores[inds_second]

        features_keep = self._extract_reid_features(frame, dets)

        detections = self._make_detections(dets, scores_keep, features_keep) if len(dets) > 0 else []

        unconfirmed, tracked_stracks = [], []
        for track in self.tracked_stracks:
            (unconfirmed if not track.is_activated else tracked_stracks).append(track)

        # Step 2: First association — high-conf dets
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        dists = self._fuse_reid(dists, strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det   = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Post-match: refine matched tracks with trajectory-conditioned features
        # if frame is not None and len(matches) > 0:
        #     matched_tracks = [strack_pool[m[0]] for m in matches]
        #     matched_tlbrs  = [t.tlbr for t in matched_tracks]
        #     refined_feats  = self._extract_conditioned_features(frame, matched_tlbrs, matched_tracks)
        #     for track, feat in zip(matched_tracks, refined_feats):
        #         if feat is not None:
        #             track.update_features(feat)

        # Step 3: Second association — low-conf dets (IoU only)
        detections_second = self._make_detections(dets_second, scores_second, [None] * len(dets_second)) if len(dets_second) > 0 else []
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked[itracked]
            det   = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for it in u_track:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 4: Init new tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # Step 5: Expire lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks    = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks    = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        return [t for t in self.tracked_stracks if t.is_activated]
