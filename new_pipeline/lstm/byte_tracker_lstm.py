"""
BYTETrackerLSTM — ByteTrack with an LSTM residual corrector.

Pipeline per frame
──────────────────
1. Detector produces detections D_t.
2. All active tracks: Kalman predict → LSTM batch-predict residual → final bbox.
3. Cost matrix: IoU on LSTM-corrected positions + Mahalanobis gating.
4. Two-stage Hungarian matching (high-conf then low-conf) — same as ByteTrack.
5. Matched:   KF update + LSTM state commit with real detection.
6. Unmatched: KF stays, LSTM state commit with pseudo (predicted) bbox.
7. New detections: spawn STrackLSTM + init LSTM state.

The original BYTETracker code is NOT modified; this is a standalone class.
"""

import os
import numpy as np
import torch

from yolox.tracker.kalman_filter import KalmanFilter
from yolox.tracker.basetrack import TrackState

from .strack_lstm import STrackLSTM
from .lstm_predictor import LSTMPredictor
from .association_lstm import (
    LSTMAssociationHead,
    build_detection_feature,
    build_pair_features,
    tlwh_to_cxcywh,
)
from .matching_lstm import (
    linear_assignment,
    iou_distance_lstm,
    gate_cost_matrix,
    fuse_score,
)


def _joint(a: list, b: list) -> list:
    seen = {}
    out = []
    for t in a + b:
        if t.track_id not in seen:
            seen[t.track_id] = 1
            out.append(t)
    return out


def _sub(a: list, b: list) -> list:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(
    a: list[STrackLSTM], b: list[STrackLSTM]
) -> tuple[list, list]:
    if not a or not b:
        return a, b
    atlbrs = np.array([t.pred_tlbr for t in a])
    btlbrs = np.array([t.pred_tlbr for t in b])
    from cython_bbox import bbox_overlaps as bbox_ious
    iou = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64),
    )
    pairs = np.where(iou > 0.85)
    dup_a, dup_b = set(), set()
    for p, q in zip(*pairs):
        ta = a[p].frame_id - a[p].start_frame
        tb = b[q].frame_id - b[q].start_frame
        if ta > tb:
            dup_b.add(q)
        else:
            dup_a.add(p)
    return [t for i, t in enumerate(a) if i not in dup_a], \
           [t for i, t in enumerate(b) if i not in dup_b]


class BYTETrackerLSTM:
    """
    Parameters
    ----------
    args            : namespace with track_thresh, track_buffer, match_thresh,
                      mot20 (bool).
    frame_rate      : video frame rate (used to compute max_time_lost).
    lstm_ckpt       : optional path to a pre-trained LSTMPredictor checkpoint.
    hidden_size     : LSTM hidden size (must match checkpoint if loading).
    num_layers      : number of stacked LSTM layers.
    alpha0          : max residual scale (1.0 = full LSTM correction).
    beta            : decay rate for missing-frame residual scaling.
    lambda_iou      : weight for IoU cost in combined cost.
    device          : torch device; defaults to CUDA if available.
    """

    def __init__(
        self,
        args,
        frame_rate: int = 30,
        lstm_ckpt: str | None = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        alpha0: float = 1.0,
        beta: float = 0.3,
        assoc_ckpt: str | None = None,
        assoc_weight: float = 0.35,
        assoc_seq_len: int = 16,
        assoc_hidden_size: int = 128,
        assoc_num_layers: int = 1,
        assoc_dropout: float = 0.1,
        assoc_mlp_hidden: int = 128,
        assoc_min_history: int = 2,
        device: str | None = None,
    ):
        self.tracked: list[STrackLSTM] = []
        self.lost: list[STrackLSTM] = []
        self.removed: list[STrackLSTM] = []

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = KalmanFilter()

        # LSTM predictor
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.lstm = LSTMPredictor(hidden_size=hidden_size, num_layers=num_layers)
        self.lstm.to(self.device).eval()
        if lstm_ckpt and os.path.isfile(lstm_ckpt):
            state = torch.load(lstm_ckpt, map_location=self.device)
            self.lstm.load_state_dict(state["model"] if "model" in state else state)
            print(f"[BYTETrackerLSTM] loaded checkpoint: {lstm_ckpt}")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.alpha0 = alpha0
        self.beta = beta

        self.assoc_seq_len = assoc_seq_len
        self.assoc_weight = float(assoc_weight)
        self.assoc_min_history = assoc_min_history
        self.assoc_max_history = max(assoc_seq_len * 4, assoc_seq_len)
        self.assoc = None
        if assoc_ckpt and os.path.isfile(assoc_ckpt):
            self.assoc = LSTMAssociationHead(
                hidden_size=assoc_hidden_size,
                num_layers=assoc_num_layers,
                dropout=assoc_dropout,
                mlp_hidden=assoc_mlp_hidden,
            )
            state = torch.load(assoc_ckpt, map_location=self.device)
            self.assoc.load_state_dict(state["model"] if "model" in state else state)
            self.assoc.to(self.device).eval()
            print(f"[BYTETrackerLSTM] loaded association checkpoint: {assoc_ckpt}")

        # Image dims updated each frame (used for normalisation)
        self._img_w = 1920.0
        self._img_h = 1080.0

    # ── Public API ────────────────────────────────────────────────────────────

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.lstm.load_state_dict(state["model"] if "model" in state else state)
        self.lstm.eval()

    def update(
        self,
        output_results: torch.Tensor | np.ndarray,
        img_info: tuple,
        img_size: tuple,
    ) -> list[STrackLSTM]:
        """
        Parameters
        ----------
        output_results : tensor [N, 5] (x1,y1,x2,y2,score) or [N, 6] with cls.
        img_info       : (orig_h, orig_w)
        img_size       : (model_h, model_w)
        """
        self.frame_id += 1
        activated, refound, lost_new, removed_new = [], [], [], []

        # ── Parse detections ─────────────────────────────────────────────
        if isinstance(output_results, torch.Tensor):
            output_results = output_results.cpu().numpy()
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]

        img_h, img_w = img_info[0], img_info[1]
        self._img_h, self._img_w = float(img_h), float(img_w)
        scale = min(img_size[0] / img_h, img_size[1] / img_w)
        bboxes /= scale

        high_mask = scores > self.args.track_thresh
        low_mask = (scores > 0.1) & ~high_mask

        dets_high = [
            STrackLSTM(STrackLSTM.tlbr_to_tlwh(b), s, self.num_layers, self.hidden_size)
            for b, s in zip(bboxes[high_mask], scores[high_mask])
        ]
        dets_low = [
            STrackLSTM(STrackLSTM.tlbr_to_tlwh(b), s, self.num_layers, self.hidden_size)
            for b, s in zip(bboxes[low_mask], scores[low_mask])
        ]

        # ── Partition active tracks ───────────────────────────────────────
        unconfirmed: list[STrackLSTM] = []
        tracked: list[STrackLSTM] = []
        for t in self.tracked:
            (tracked if t.is_activated else unconfirmed).append(t)

        # ── Kalman predict → LSTM batch predict ──────────────────────────
        pool = _joint(tracked, self.lost)
        STrackLSTM.multi_predict(pool)
        if pool:
            self._lstm_batch_predict(pool)

        # ── Stage 1: match pool ↔ high-conf detections ───────────────────
        cost = iou_distance_lstm(pool, dets_high)
        if not self.args.mot20:
            cost = fuse_score(cost, dets_high)
        cost = gate_cost_matrix(self.kalman_filter, cost, pool, dets_high)
        cost = self._combine_association_cost(cost, pool, dets_high)
        matches, u_track, u_det = linear_assignment(cost, thresh=self.args.match_thresh)

        for ti, di in matches:
            track, det = pool[ti], dets_high[di]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refound.append(track)
            track.apply_lstm_matched(
                STrackLSTM._tlwh_to_cxcywh(det.tlwh), det.score
            )
            self._append_assoc_feature(track, is_missing=0)

        # Lost tracks that didn't match stage 1 still need LSTM state updated
        for i in u_track:
            t = pool[i]
            if t.state == TrackState.Lost:
                t.apply_lstm_missing()
                self._append_assoc_feature(t, is_missing=1)

        # ── Stage 2: remaining tracked ↔ low-conf detections ─────────────
        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        cost2 = iou_distance_lstm(r_tracked, dets_low)
        cost2 = self._combine_association_cost(cost2, r_tracked, dets_low)
        matches2, u_track2, _ = linear_assignment(cost2, thresh=0.5)

        for ti, di in matches2:
            track, det = r_tracked[ti], dets_low[di]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refound.append(track)
            track.apply_lstm_matched(
                STrackLSTM._tlwh_to_cxcywh(det.tlwh), det.score
            )
            self._append_assoc_feature(track, is_missing=0)

        for i in u_track2:
            t = r_tracked[i]
            if t.state != TrackState.Lost:
                t.mark_lost()
                lost_new.append(t)
            t.apply_lstm_missing()
            self._append_assoc_feature(t, is_missing=1)

        # ── Unconfirmed tracks ↔ remaining high-conf dets ─────────────────
        rem_dets = [dets_high[i] for i in u_det]
        cost3 = iou_distance_lstm(unconfirmed, rem_dets)
        if not self.args.mot20:
            cost3 = fuse_score(cost3, rem_dets)
        cost3 = self._combine_association_cost(cost3, unconfirmed, rem_dets)
        matches3, u_unc, u_det2 = linear_assignment(cost3, thresh=0.7)

        for ti, di in matches3:
            unconfirmed[ti].update(rem_dets[di], self.frame_id)
            unconfirmed[ti].apply_lstm_matched(
                STrackLSTM._tlwh_to_cxcywh(rem_dets[di].tlwh), rem_dets[di].score
            )
            self._append_assoc_feature(unconfirmed[ti], is_missing=0)
            activated.append(unconfirmed[ti])
        for i in u_unc:
            unconfirmed[i].mark_removed()
            removed_new.append(unconfirmed[i])

        # ── New tracks from unmatched high-conf dets ──────────────────────
        for i in u_det2:
            det = rem_dets[i]
            if det.score < self.det_thresh:
                continue
            det.activate(self.kalman_filter, self.frame_id)
            # Init LSTM with first observation
            self._lstm_init_track(det)
            self._append_assoc_feature(det, is_missing=0)
            activated.append(det)

        # ── Expire lost tracks ────────────────────────────────────────────
        for t in self.lost:
            if self.frame_id - t.end_frame > self.max_time_lost:
                t.mark_removed()
                removed_new.append(t)

        # ── State bookkeeping ─────────────────────────────────────────────
        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        self.tracked = _joint(self.tracked, activated)
        self.tracked = _joint(self.tracked, refound)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost_new)
        self.lost = _sub(self.lost, self.removed)
        self.removed.extend(removed_new)
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)

        return [t for t in self.tracked if t.is_activated]

    # ── LSTM batch operations ─────────────────────────────────────────────────

    @torch.no_grad()
    def _lstm_batch_predict(self, tracks: list[STrackLSTM]):
        """Run LSTM prediction for all tracks in one batched forward pass."""
        B = len(tracks)

        x_list, h_list, c_list = [], [], []
        for t in tracks:
            kf_bbox = t.get_kf_bbox_cxcywh()
            feat = LSTMPredictor.build_input(
                bbox_cxcywh=t.last_bbox_cxcywh,
                velocity=t.velocity,
                delta_t=1.0,
                is_missing=t.is_missing,
                missing_count=t.missing_count,
                confidence=t.last_confidence,
                img_w=self._img_w,
                img_h=self._img_h,
            )
            x_list.append(feat)
            h_list.append(t.h_lstm)          # [num_layers, H]
            c_list.append(t.c_lstm)

        x_t = torch.tensor(np.array(x_list), dtype=torch.float32, device=self.device)
        # Stack: [B, num_layers, H] → transpose → [num_layers, B, H]
        h_t = torch.tensor(
            np.array(h_list).transpose(1, 0, 2), dtype=torch.float32, device=self.device
        )
        c_t = torch.tensor(
            np.array(c_list).transpose(1, 0, 2), dtype=torch.float32, device=self.device
        )
        h_t = h_t.contiguous()
        c_t = c_t.contiguous()
        dt_dummy = torch.ones(B, 1, device=self.device)  # informational only

        residuals, h_new, c_new = self.lstm.step(x_t, h_t, c_t)

        residuals = residuals.cpu().numpy()   # [B, 4]
        # [num_layers, B, H] → [B, num_layers, H]
        h_new_np = h_new.cpu().numpy().transpose(1, 0, 2)
        c_new_np = c_new.cpu().numpy().transpose(1, 0, 2)

        for i, track in enumerate(tracks):
            # Missing-aware scale: reduce LSTM influence the longer a track is lost
            alpha = self.alpha0 * np.exp(-self.beta * track.missing_count)

            kf_bbox = track.get_kf_bbox_cxcywh()
            corrected = kf_bbox + alpha * residuals[i]   # residuals in pixel space
            corrected = np.clip(
                corrected,
                [0, 0, 1, 1],
                [self._img_w, self._img_h, self._img_w, self._img_h],
            )

            track.pred_bbox_cxcywh = corrected.astype(np.float32)
            track._lstm_residual = residuals[i]
            track._lstm_h_new = h_new_np[i]
            track._lstm_c_new = c_new_np[i]
            track.assoc_embed = h_new_np[i, -1].astype(np.float32)

    def _append_assoc_feature(self, track: STrackLSTM, is_missing: int):
        feature = LSTMPredictor.build_input(
            bbox_cxcywh=track.last_bbox_cxcywh,
            velocity=track.velocity,
            delta_t=1.0,
            is_missing=is_missing,
            missing_count=track.missing_count,
            confidence=0.0 if is_missing else track.last_confidence,
            img_w=self._img_w,
            img_h=self._img_h,
        )
        track.append_assoc_feature(feature, max_history=self.assoc_max_history)

    @torch.no_grad()
    def _association_scores(
        self,
        tracks: list[STrackLSTM],
        detections: list[STrackLSTM],
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        valid = np.zeros((len(tracks), len(detections)), dtype=bool)
        if self.assoc is None or not tracks or not detections:
            return scores, valid

        track_embeddings, det_features, pair_features, indices = [], [], [], []
        for row, track in enumerate(tracks):
            if len(track.assoc_history) < self.assoc_min_history:
                continue
            delta_t = max(float(self.frame_id - track.frame_id), 1.0)
            for col, det in enumerate(detections):
                det_bbox = tlwh_to_cxcywh(det.tlwh)
                track_embeddings.append(track.assoc_embed)
                det_features.append(
                    build_detection_feature(det_bbox, det.score, self._img_w, self._img_h)
                )
                pair_features.append(
                    build_pair_features(
                        track.last_bbox_cxcywh,
                        track.velocity,
                        det_bbox,
                        delta_t,
                        self._img_w,
                        self._img_h,
                    )
                )
                indices.append((row, col))

        if not indices:
            return scores, valid

        embed_t = torch.tensor(np.asarray(track_embeddings), dtype=torch.float32, device=self.device)
        det_t = torch.tensor(np.asarray(det_features), dtype=torch.float32, device=self.device)
        pair_t = torch.tensor(np.asarray(pair_features), dtype=torch.float32, device=self.device)

        probs = []
        batch_size = 4096
        for start in range(0, embed_t.size(0), batch_size):
            end = start + batch_size
            probs.append(self.assoc.predict_proba(embed_t[start:end], det_t[start:end], pair_t[start:end]))
        probs_np = torch.cat(probs, dim=0).cpu().numpy()

        for (row, col), prob in zip(indices, probs_np):
            scores[row, col] = float(prob)
            valid[row, col] = True
        return scores, valid

    def _combine_association_cost(
        self,
        base_cost: np.ndarray,
        tracks: list[STrackLSTM],
        detections: list[STrackLSTM],
    ) -> np.ndarray:
        if (
            self.assoc is None
            or self.assoc_weight <= 0.0
            or base_cost.size == 0
            or not tracks
            or not detections
        ):
            return base_cost

        assoc_scores, valid = self._association_scores(tracks, detections)
        assoc_cost = 1.0 - assoc_scores
        finite = np.isfinite(base_cost)
        mask = finite & valid
        if not np.any(mask):
            return base_cost

        combined = base_cost.copy()
        weight = min(max(self.assoc_weight, 0.0), 1.0)
        combined[mask] = (1.0 - weight) * base_cost[mask] + weight * assoc_cost[mask]
        combined[~finite] = np.inf
        return combined

    @torch.no_grad()
    def _lstm_init_track(self, track: STrackLSTM):
        """Run one LSTM step with the first detection to initialise the state."""
        feat = LSTMPredictor.build_input(
            bbox_cxcywh=track.last_bbox_cxcywh,
            velocity=np.zeros(2, dtype=np.float32),
            delta_t=1.0,
            is_missing=0,
            missing_count=0,
            confidence=track.score,
            img_w=self._img_w,
            img_h=self._img_h,
        )
        x_t = torch.tensor(feat[None], dtype=torch.float32, device=self.device)
        h0, c0 = self.lstm.init_hidden(1, self.device)
        _, h_new, c_new = self.lstm.step(x_t, h0, c0)
        track.h_lstm = h_new.cpu().numpy().transpose(1, 0, 2)[0]
        track.c_lstm = c_new.cpu().numpy().transpose(1, 0, 2)[0]
        track.assoc_embed = track.h_lstm[-1].astype(np.float32)
