"""
STrackLSTM — STrack extended with per-track LSTM hidden/cell states.

Drop-in replacement for the original STrack when using BYTETrackerLSTM.
The original ByteTrack code is untouched; this class only *adds* fields
needed by the LSTM predictor.
"""

import numpy as np

from yolox.tracker.basetrack import BaseTrack, TrackState
from yolox.tracker.kalman_filter import KalmanFilter


class STrackLSTM(BaseTrack):
    """
    Fields added over the original STrack
    ──────────────────────────────────────
    h_lstm, c_lstm   : LSTM hidden/cell state  [num_layers, H]  (numpy, CPU)
    last_bbox_cxcywh : previous observed bbox [cx, cy, w, h] in pixels
    velocity         : [vx, vy] estimated from consecutive detections
    missing_count    : consecutive frames without a matched detection
    is_missing       : 1 if current frame has no match, else 0
    last_confidence  : detection score of the last matched detection
    pred_bbox_cxcywh : LNN/LSTM-corrected predicted bbox (set by tracker)
    """

    shared_kalman = KalmanFilter()

    def __init__(
        self,
        tlwh: np.ndarray,
        score: float,
        num_layers: int = 2,
        hidden_size: int = 128,
    ):
        # ── Kalman state ────────────────────────────────────────────────
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter: KalmanFilter | None = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.is_activated: bool = False

        self.score = score
        self.tracklet_len = 0

        # ── LSTM state (stored as numpy; converted to tensor per batch) ─
        self.h_lstm = np.zeros((num_layers, hidden_size), dtype=np.float32)
        self.c_lstm = np.zeros((num_layers, hidden_size), dtype=np.float32)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # ── Motion history ──────────────────────────────────────────────
        self.last_bbox_cxcywh = self._tlwh_to_cxcywh(tlwh)
        self.velocity = np.zeros(2, dtype=np.float32)

        # ── Missing-aware state ─────────────────────────────────────────
        self.missing_count: int = 0
        self.is_missing: int = 0
        self.last_confidence: float = float(score)
        self.assoc_history: list[np.ndarray] = []

        # ── Scratch space filled by BYTETrackerLSTM before matching ────
        self.pred_bbox_cxcywh: np.ndarray = self.last_bbox_cxcywh.copy()
        self._lstm_residual: np.ndarray = np.zeros(4, dtype=np.float32)
        self._lstm_h_new: np.ndarray = self.h_lstm.copy()
        self._lstm_c_new: np.ndarray = self.c_lstm.copy()
        self.assoc_embed: np.ndarray = np.zeros(hidden_size, dtype=np.float32)

    # ── Kalman interface (mirrors original STrack exactly) ──────────────

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks: list["STrackLSTM"]):
        if not stracks:
            return
        multi_mean = np.array([s.mean.copy() for s in stracks])
        multi_cov = np.array([s.covariance for s in stracks])
        for i, s in enumerate(stracks):
            if s.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_cov = STrackLSTM.shared_kalman.multi_predict(
            multi_mean, multi_cov
        )
        for i, (m, c) in enumerate(zip(multi_mean, multi_cov)):
            stracks[i].mean = m
            stracks[i].covariance = c

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.pred_bbox_cxcywh = self._tlwh_to_cxcywh(self._tlwh)

    def re_activate(self, new_track: "STrackLSTM", frame_id: int, new_id: bool = False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self._reset_missing()

    def update(self, new_track: "STrackLSTM", frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self._reset_missing()

    # ── LSTM state commit helpers ────────────────────────────────────────

    def apply_lstm_matched(self, det_bbox_cxcywh: np.ndarray, det_score: float):
        """Commit LSTM state after a real detection match."""
        self.velocity = (det_bbox_cxcywh[:2] - self.last_bbox_cxcywh[:2]).astype(np.float32)
        self.last_bbox_cxcywh = det_bbox_cxcywh.astype(np.float32)
        self.last_confidence = float(det_score)
        self.h_lstm = self._lstm_h_new.copy()
        self.c_lstm = self._lstm_c_new.copy()
        self._reset_missing()

    def apply_lstm_missing(self):
        """Commit LSTM state when no detection matched this frame."""
        self.last_bbox_cxcywh = self.pred_bbox_cxcywh.astype(np.float32)
        self.last_confidence = 0.0
        self.h_lstm = self._lstm_h_new.copy()
        self.c_lstm = self._lstm_c_new.copy()
        self.missing_count += 1
        self.is_missing = 1

    def _reset_missing(self):
        self.missing_count = 0
        self.is_missing = 0

    def append_assoc_feature(self, feature: np.ndarray, max_history: int = 64):
        feature = np.asarray(feature, dtype=np.float32)
        self.assoc_history.append(feature)
        if len(self.assoc_history) > max_history:
            self.assoc_history = self.assoc_history[-max_history:]

    def get_assoc_history(self, seq_len: int, feature_dim: int = 10) -> np.ndarray:
        history = np.zeros((seq_len, feature_dim), dtype=np.float32)
        if not self.assoc_history:
            return history
        recent = self.assoc_history[-seq_len:]
        history[-len(recent) :] = np.asarray(recent, dtype=np.float32)
        return history

    # ── Bbox helpers ─────────────────────────────────────────────────────

    def to_xyah(self) -> np.ndarray:
        """Current KF position in [cx, cy, aspect, h] format."""
        return self.tlwh_to_xyah(self.tlwh)

    def get_kf_bbox_cxcywh(self) -> np.ndarray:
        """KF-predicted position in [cx, cy, w, h] pixel format."""
        if self.mean is None:
            return self._tlwh_to_cxcywh(self._tlwh)
        cx, cy, a, h = self.mean[:4]
        return np.array([cx, cy, a * h, h], dtype=np.float32)

    @property
    def pred_tlwh(self) -> np.ndarray:
        """LSTM-corrected predicted bbox in [x1, y1, w, h] format."""
        cx, cy, w, h = self.pred_bbox_cxcywh
        return np.array([cx - w / 2, cy - h / 2, w, h], dtype=np.float64)

    @property
    def pred_tlbr(self) -> np.ndarray:
        """LSTM-corrected predicted bbox in [x1, y1, x2, y2] format."""
        t = self.pred_tlwh
        return np.array([t[0], t[1], t[0] + t[2], t[1] + t[3]], dtype=np.float64)

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def _tlwh_to_cxcywh(tlwh: np.ndarray) -> np.ndarray:
        t = np.asarray(tlwh, dtype=np.float32)
        return np.array([t[0] + t[2] / 2, t[1] + t[3] / 2, t[2], t[3]], dtype=np.float32)

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"
