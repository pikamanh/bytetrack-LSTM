from __future__ import annotations

import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    """
    LSTM encoder cho tracklet trajectory history.

    Input:
        traj_seq   : (B, N, input_dim)  — sequence bbox states
                     mỗi bước: [cx, cy, w, h, vx, vy, vw, vh, conf]
        gate_feats : (B, 3) optional    — [track_age_norm, occ_ratio, delta_t_norm]

    Output:
        h_motion    : (B, hidden_dim)  — motion context vector → inject vào ReID
        motion_pred : (B, 4)           — predicted next [cx, cy, w, h]
        g           : (B, 1)           — reliability gate ∈ (0, 1)
    """

    GATE_INPUT_DIM = 3  # track_age_norm, occ_ratio, delta_t_norm
    def __init__(self, input_dim: int = 9, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # LSTM encoder — 1 layer đủ cho seq_len ≤ 20
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Dự đoán next bounding box [cx, cy, w, h] (auxiliary loss)
        self.motion_pred_head = nn.Linear(hidden_dim, 4)

        # Reliability gate: học từ cả h_motion + external gate features
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + self.GATE_INPUT_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        # Motion pred head: init nhỏ để không ảnh hưởng lớn lúc đầu
        nn.init.normal_(self.motion_pred_head.weight, 0, 0.01)
        nn.init.zeros_(self.motion_pred_head.bias)

    def forward(
        self,
        traj_seq: torch.Tensor,
        gate_feats: torch.Tensor | None = None,
    ):
        """
        traj_seq:   (B, N, input_dim)
        gate_feats: (B, 3) hoặc None → dùng zeros khi cold-start
        """
        B = traj_seq.size(0)

        # Encode trajectory sequence
        _, (h_n, _) = self.lstm(traj_seq)   # h_n: (num_layers, B, hidden)
        h_motion = h_n[-1]                  # (B, hidden)

        # Auxiliary: dự đoán next box
        motion_pred = self.motion_pred_head(h_motion)  # (B, 4)

        # Reliability gate
        if gate_feats is None:
            gate_feats = torch.zeros(B, self.GATE_INPUT_DIM, device=h_motion.device)
        gate_input = torch.cat([h_motion, gate_feats], dim=-1)   # (B, hidden+3)
        g = self.gate_net(gate_input)                             # (B, 1)

        return h_motion, motion_pred, g
