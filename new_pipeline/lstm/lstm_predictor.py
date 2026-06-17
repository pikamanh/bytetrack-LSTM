"""
LSTM-based residual predictor for multi-object tracking.

Architecture
------------
Input (per track, per frame) — 10-dim normalised feature:
    [cx/W, cy/H, w/W, h/H, vx/W, vy/H, delta_t, is_missing,
     missing_count/max_missing, confidence]

LSTM (stacked, batch_first) processes a *single timestep* per
forward call during inference so that each track maintains its
own (h, c) state between frames.

Output
------
    residual  [4]  — bbox correction [dcx, dcy, dw, dh] in pixel space,
                     added on top of Kalman prediction.
    h_new, c_new  — updated LSTM hidden and cell states.

Zero-initialisation on the residual head ensures the model begins
as a pure Kalman filter before any training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMPredictor(nn.Module):
    INPUT_DIM = 10

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection (normalises scale before LSTM)
        self.input_proj = nn.Linear(self.INPUT_DIM, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.residual_head = nn.Linear(hidden_size, 4)

        # Zero-init residual → pure-Kalman behaviour before training
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    # ------------------------------------------------------------------
    # Single-step inference  (used by the tracker)
    # ------------------------------------------------------------------

    def step(
        self,
        x: torch.Tensor,      # [B, INPUT_DIM]
        h: torch.Tensor,      # [num_layers, B, hidden_size]
        c: torch.Tensor,      # [num_layers, B, hidden_size]
    ):
        """One-step forward, returns residual, h_new, c_new."""
        h = h.contiguous()
        c = c.contiguous()
        feat = F.relu(self.input_proj(x))          # [B, H]
        out, (h_new, c_new) = self.lstm(
            feat.unsqueeze(1), (h, c)              # seq_len=1
        )
        out = out.squeeze(1)                        # [B, H]
        return self.residual_head(out), h_new, c_new

    # ------------------------------------------------------------------
    # Sequence forward  (used during training)
    # ------------------------------------------------------------------

    def forward(
        self,
        x_seq: torch.Tensor,   # [B, T, INPUT_DIM]
        h0: torch.Tensor,      # [num_layers, B, H]
        c0: torch.Tensor,      # [num_layers, B, H]
    ):
        """Full-sequence forward for training (BPTT)."""
        feat = F.relu(self.input_proj(x_seq))      # [B, T, H]
        out, (h_n, c_n) = self.lstm(feat, (h0, c0))
        residuals = self.residual_head(out)         # [B, T, 4]
        return residuals, h_n, c_n

    def encode_history(
        self,
        x_seq: torch.Tensor,  # [B, T, INPUT_DIM]
        h0: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return top-layer LSTM embedding for a track history."""
        batch_size = x_seq.size(0)
        if h0 is None or c0 is None:
            h0, c0 = self.init_hidden(batch_size=batch_size, device=x_seq.device)
        feat = F.relu(self.input_proj(x_seq))
        _, (h_n, _) = self.lstm(feat, (h0, c0))
        return h_n[-1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def init_hidden(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    @staticmethod
    def build_input(
        bbox_cxcywh: np.ndarray,   # [cx, cy, w, h] pixel space
        velocity: np.ndarray,       # [vx, vy]  pixels / frame
        delta_t: float,             # frames elapsed since last update
        is_missing: int,            # 1 if no detection this frame
        missing_count: int,         # consecutive missing frames so far
        confidence: float,          # detection score (0 when missing)
        img_w: float,
        img_h: float,
        max_missing: float = 30.0,
    ) -> np.ndarray:               # float32  [INPUT_DIM]
        cx, cy, w, h = bbox_cxcywh
        vx, vy = velocity
        return np.array([
            cx / img_w,
            cy / img_h,
            w / img_w,
            h / img_h,
            vx / img_w,
            vy / img_h,
            float(delta_t),
            float(is_missing),
            min(missing_count / max_missing, 1.0),
            float(confidence),
        ], dtype=np.float32)

    @staticmethod
    def huber_loss(
        residuals: torch.Tensor,  # [B, T, 4]  predicted
        targets: torch.Tensor,    # [B, T, 4]  GT residuals
        mask: torch.Tensor,       # [B, T]     1 = valid frame, 0 = padding
        delta: float = 1.0,
    ) -> torch.Tensor:
        """
        Huber loss masked over valid frames.
        More robust than MSE: linear for large errors, quadratic for small ones.
        """
        per_dim = F.huber_loss(residuals, targets, reduction="none", delta=delta)
        loss = per_dim.mean(dim=-1)                 # [B, T]
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss
