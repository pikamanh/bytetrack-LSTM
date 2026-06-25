from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Generates γ và β từ motion context vector h_motion, sau đó:
        out = γ * feature_map + β

    Inject vào sau layer2 của ResNeSt (512 channels) — đây là điểm cân bằng
    giữa semantic depth và spatial resolution.

    Initialization strategy: γ=1, β=0 (identity) để lúc đầu model học ReID
    thuần trước, rồi dần dần motion context bắt đầu có tác động.
    """

    def __init__(self, motion_dim: int, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # Một linear layer tạo cả γ và β cùng lúc
        self.film_gen = nn.Linear(motion_dim, feature_dim * 2)

        self._init_as_identity()

    def _init_as_identity(self):
        """Init sao cho lúc đầu output ≈ identity: γ≈1, β≈0."""
        nn.init.normal_(self.film_gen.weight, 0, 0.001)
        # Bias: [γ_part=1, β_part=0]
        bias = torch.ones(self.feature_dim * 2)
        bias[self.feature_dim:] = 0.0          # β phần bằng 0
        self.film_gen.bias.data.copy_(bias)

    def forward(
        self,
        x: torch.Tensor,
        h_motion: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:        (B, C, H, W)   — feature map từ layer2
        h_motion: (B, motion_dim) — motion context vector
        g:        (B, 1)          — reliability gate (None = không dùng gate)

        Nếu g gần 0 (trajectory không đáng tin):
            γ ≈ 1, β ≈ 0  →  gần như identity, không ảnh hưởng ReID
        Nếu g gần 1 (trajectory tin cậy):
            FiLM hoạt động đầy đủ
        """
        params = self.film_gen(h_motion)           # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)      # each (B, C)

        if g is not None:
            # Gate: blend về identity khi g thấp
            gamma = g * gamma + (1.0 - g) * torch.ones_like(gamma)
            beta  = g * beta
            # (khi g=0: γ=1, β=0 → x không thay đổi)

        # Broadcast spatial dims
        gamma = gamma[:, :, None, None]   # (B, C, 1, 1)
        beta  = beta[:, :, None, None]    # (B, C, 1, 1)

        return gamma * x + beta
