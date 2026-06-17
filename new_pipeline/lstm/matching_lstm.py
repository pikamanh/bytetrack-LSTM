"""
Matching utilities for the LSTM-enhanced ByteTrack pipeline.

Changes vs. original matching.py
─────────────────────────────────
- iou_distance_lstm   uses each track's LSTM-corrected pred_tlbr for IoU,
                      giving a more accurate cost estimate.
- gate_cost_matrix    uses KF mean/covariance for Mahalanobis gating
                      (unchanged semantics; KF covariance is still valid).
- fuse_score          identical to the original.
"""

import numpy as np
import lap
from cython_bbox import bbox_overlaps as bbox_ious
from yolox.tracker import kalman_filter as kf_module


# ── Low-level helpers ────────────────────────────────────────────────────────

def _ious(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    out = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if out.size == 0:
        return out
    return bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64),
    )


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Hungarian matching via lapjv. Mirrors the original implementation."""
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = [[i, mx] for i, mx in enumerate(x) if mx >= 0]
    return np.asarray(matches), np.where(x < 0)[0], np.where(y < 0)[0]


# ── IoU cost using LSTM-corrected positions ──────────────────────────────────

def iou_distance_lstm(atracks, btracks) -> np.ndarray:
    """
    1 - IoU cost matrix.

    atracks: use pred_tlbr (LSTM-corrected) when available, else tlbr.
    btracks: always uses tlbr (raw detections).
    """
    if not atracks or not btracks:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float64)

    atlbrs = [
        t.pred_tlbr if (not isinstance(t, np.ndarray) and hasattr(t, "pred_tlbr"))
        else (t if isinstance(t, np.ndarray) else t.tlbr)
        for t in atracks
    ]
    btlbrs = [t if isinstance(t, np.ndarray) else t.tlbr for t in btracks]

    return 1.0 - _ious(np.array(atlbrs), np.array(btlbrs))


# ── Mahalanobis gating (KF-based, same logic as original) ───────────────────

def gate_cost_matrix(
    kalman_filter,
    cost_matrix: np.ndarray,
    tracks,
    detections,
    only_position: bool = False,
) -> np.ndarray:
    """
    Set cost to np.inf for pairs whose Mahalanobis distance (from KF
    mean/covariance) exceeds the chi-square 95% threshold.
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    thresh = kf_module.chi2inv95[gating_dim]
    measurements = np.array([det.to_xyah() for det in detections])

    for row, track in enumerate(tracks):
        if track.mean is None:
            continue
        dist = kalman_filter.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, dist > thresh] = np.inf

    return cost_matrix


# ── Score fusion (identical to original) ────────────────────────────────────

def fuse_score(cost_matrix: np.ndarray, detections) -> np.ndarray:
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.expand_dims(
        np.array([d.score for d in detections]), 0
    ).repeat(cost_matrix.shape[0], axis=0)
    return 1.0 - iou_sim * det_scores
