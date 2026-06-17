"""Association score head for ByteTrack-LSTM.

Runtime design:
    one LSTM pass per track is done by ``LSTMPredictor``. This module only
    consumes that track hidden state plus candidate detection features and
    predicts whether the pair should match.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


DET_DIM = 5
PAIR_DIM = 8


def tlwh_to_cxcywh(tlwh: np.ndarray) -> np.ndarray:
    tlwh = np.asarray(tlwh, dtype=np.float32)
    return np.array(
        [tlwh[0] + tlwh[2] / 2.0, tlwh[1] + tlwh[3] / 2.0, tlwh[2], tlwh[3]],
        dtype=np.float32,
    )


def cxcywh_to_tlbr(cxcywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = np.asarray(cxcywh, dtype=np.float32)
    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        dtype=np.float32,
    )


def bbox_iou_cxcywh(a: np.ndarray, b: np.ndarray) -> float:
    a_tlbr = cxcywh_to_tlbr(a)
    b_tlbr = cxcywh_to_tlbr(b)
    x1 = max(float(a_tlbr[0]), float(b_tlbr[0]))
    y1 = max(float(a_tlbr[1]), float(b_tlbr[1]))
    x2 = min(float(a_tlbr[2]), float(b_tlbr[2]))
    y2 = min(float(a_tlbr[3]), float(b_tlbr[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a_tlbr[2] - a_tlbr[0])) * max(
        0.0, float(a_tlbr[3] - a_tlbr[1])
    )
    area_b = max(0.0, float(b_tlbr[2] - b_tlbr[0])) * max(
        0.0, float(b_tlbr[3] - b_tlbr[1])
    )
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom


def build_detection_feature(
    det_bbox_cxcywh: np.ndarray,
    confidence: float,
    img_w: float,
    img_h: float,
) -> np.ndarray:
    cx, cy, w, h = np.asarray(det_bbox_cxcywh, dtype=np.float32)
    return np.array(
        [cx / img_w, cy / img_h, w / img_w, h / img_h, float(confidence)],
        dtype=np.float32,
    )


def build_pair_features(
    last_bbox_cxcywh: np.ndarray,
    velocity: np.ndarray,
    det_bbox_cxcywh: np.ndarray,
    delta_t: float,
    img_w: float,
    img_h: float,
) -> np.ndarray:
    last_bbox_cxcywh = np.asarray(last_bbox_cxcywh, dtype=np.float32)
    det_bbox_cxcywh = np.asarray(det_bbox_cxcywh, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    delta_t = max(float(delta_t), 1.0)

    pred_bbox = last_bbox_cxcywh.copy()
    pred_bbox[0] += velocity[0] * delta_t
    pred_bbox[1] += velocity[1] * delta_t

    dx = (det_bbox_cxcywh[0] - pred_bbox[0]) / img_w
    dy = (det_bbox_cxcywh[1] - pred_bbox[1]) / img_h
    dw = (det_bbox_cxcywh[2] - pred_bbox[2]) / img_w
    dh = (det_bbox_cxcywh[3] - pred_bbox[3]) / img_h

    diag = float(np.hypot(img_w, img_h))
    center_dist = (
        float(
            np.hypot(
                det_bbox_cxcywh[0] - pred_bbox[0],
                det_bbox_cxcywh[1] - pred_bbox[1],
            )
        )
        / max(diag, 1.0)
    )
    iou = bbox_iou_cxcywh(pred_bbox, det_bbox_cxcywh)

    det_motion = (det_bbox_cxcywh[:2] - last_bbox_cxcywh[:2]) / delta_t
    denom = float(np.linalg.norm(velocity) * np.linalg.norm(det_motion))
    vel_cos = 0.0 if denom < 1e-6 else float(np.dot(velocity, det_motion) / denom)

    return np.array(
        [dx, dy, dw, dh, center_dist, iou, vel_cos, min(delta_t / 30.0, 1.0)],
        dtype=np.float32,
    )


class AssociationScoreHead(nn.Module):
    """MLP pair classifier over a reused LSTM track embedding."""

    def __init__(
        self,
        hidden_size: int = 128,
        det_dim: int = DET_DIM,
        pair_dim: int = PAIR_DIM,
        dropout: float = 0.1,
        mlp_hidden: int = 128,
        num_layers: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.det_dim = det_dim
        self.pair_dim = pair_dim
        self.num_layers = num_layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + det_dim + pair_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, max(mlp_hidden // 2, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(mlp_hidden // 2, 1), 1),
        )

    def forward(
        self,
        track_embedding: torch.Tensor,
        det_features: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([track_embedding, det_features, pair_features], dim=-1)
        return self.classifier(fused).squeeze(-1)

    @torch.no_grad()
    def predict_proba(
        self,
        track_embedding: torch.Tensor,
        det_features: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(track_embedding, det_features, pair_features))


LSTMAssociationHead = AssociationScoreHead
