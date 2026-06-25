"""
FiLMReIDModel — Trajectory-conditioned ReID với FiLM modulation.

Architecture:
    Motion Branch : LSTM(N×9) → h_motion + motion_pred + g (reliability gate)
    ReID Branch   : ResNeSt50 backbone, FiLM injected sau layer2 (512ch)
    Head          : GAP → BNNeck → 2048-dim embedding

Điểm inject FiLM: giữa layer2 và layer3 của ResNeSt50.
    layer2 output: 512 channels  ← FiLM ở đây
    layer3 output: 1024 channels
    layer4 output: 2048 channels

Training objective:
    L = L_triplet + L_ce + λ_motion * L_motion_pred
"""

from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .motion_encoder import MotionEncoder
    from .film_layer import FiLMLayer
except ImportError:
    from motion_encoder import MotionEncoder
    from film_layer import FiLMLayer


# Channels của ResNeSt50 tại mỗi stage
_RESNEST50_CHANNELS = {
    "layer1": 256,
    "layer2": 512,   # ← điểm inject FiLM
    "layer3": 1024,
    "layer4": 2048,
}
_FEAT_DIM = 2048


def _add_fastreid_to_path():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fastreid_root = os.path.join(root, "fast-reid")
    if fastreid_root not in sys.path:
        sys.path.insert(0, fastreid_root)
    return fastreid_root


class FiLMReIDModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        motion_hidden_dim: int = 128,
        motion_input_dim: int = 9,
        fastreid_config: str | None = None,
        pretrained_reid_path: str | None = None,
    ):
        """
        Args:
            num_classes:          số lượng identity trong tập train (cho CE head)
            motion_hidden_dim:    hidden size của LSTM motion encoder
            motion_input_dim:     chiều mỗi bước trong trajectory sequence (mặc định 9)
            fastreid_config:      path đến config file của fast-reid (.yml)
            pretrained_reid_path: path đến checkpoint fast-reid (.pth)
        """
        super().__init__()

        film_channels = _RESNEST50_CHANNELS["layer2"]

        # ── Motion Branch ──────────────────────────────────────────────────
        self.motion_encoder = MotionEncoder(
            input_dim=motion_input_dim,
            hidden_dim=motion_hidden_dim,
            num_layers=1,
        )

        # ── FiLM Injection Layer ───────────────────────────────────────────
        self.film = FiLMLayer(
            motion_dim=motion_hidden_dim,
            feature_dim=film_channels,
        )

        # ── ReID Backbone (ResNeSt50 từ fast-reid) ─────────────────────────
        self.backbone = self._build_backbone(fastreid_config, pretrained_reid_path)

        # ── Head: GAP + BNNeck + Classifier ───────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bnneck = nn.BatchNorm1d(_FEAT_DIM)
        self.bnneck.bias.requires_grad_(False)     # freeze bias theo convention fast-reid
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)

        self.classifier = nn.Linear(_FEAT_DIM, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, 0, 0.01)

    # ─────────────────────────────────────────────────────────────────────
    # Backbone builder
    # ─────────────────────────────────────────────────────────────────────

    def _build_backbone(self, fastreid_config, pretrained_reid_path):
        _add_fastreid_to_path()
        from fastreid.modeling.backbones.resnest import ResNeSt, Bottleneck

        # ── Bước 1: Build kiến trúc backbone ─────────────────────────────
        if fastreid_config is not None:
            # Dùng fast-reid config → nó tự xử lý ImageNet pretrain nếu PRETRAIN=True
            from fastreid.config import get_cfg
            from fastreid.modeling.backbones import build_backbone
            cfg = get_cfg()
            cfg.merge_from_file(fastreid_config)
            backbone = build_backbone(cfg)
            print("[FiLMReIDModel] Backbone built from fast-reid config")
            return backbone   # config đã handle pretrain → return sớm

        # Build ResNeSt50 thủ công (không có config)
        backbone = ResNeSt(
            last_stride=1,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            radix=2, groups=1, bottleneck_width=64,
            deep_stem=True, stem_width=32, avg_down=True,
            avd=True, avd_first=False, norm_layer="BN",
        )

        # ── Bước 2: Load weights theo thứ tự ưu tiên ─────────────────────
        if pretrained_reid_path and os.path.isfile(pretrained_reid_path):
            # Ưu tiên 1: ReID pretrained checkpoint (fast-reid format)
            _load_reid_weights(backbone, pretrained_reid_path)
        else:
            # Ưu tiên 2: ImageNet pretrained từ official ResNeSt repo
            # (tự động download ~90MB lần đầu, cache tại ~/.cache/torch)
            _load_imagenet_resnest50(backbone)

        return backbone

    # ─────────────────────────────────────────────────────────────────────
    # Forward với FiLM injection giữa layer2 và layer3
    # ─────────────────────────────────────────────────────────────────────

    def _forward_backbone_film(
        self,
        x: torch.Tensor,
        h_motion: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """
        ResNeSt forward với FiLM inject sau layer2.
        """
        bb = self.backbone

        # Stem
        x = bb.conv1(x)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)

        # Stages 1 & 2
        x = bb.layer1(x)   # (B, 256, H/4, W/4)
        x = bb.layer2(x)   # (B, 512, H/8, W/8)

        # ── FiLM injection ────────────────────────────────────────────────
        x = self.film(x, h_motion, g)

        # Stages 3 & 4
        x = bb.layer3(x)   # (B, 1024, H/16, W/16) hoặc H/8 nếu dilated
        x = bb.layer4(x)   # (B, 2048, ...)

        return x

    # ─────────────────────────────────────────────────────────────────────
    # Forward chính
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        traj_seq: torch.Tensor | None = None,
        gate_feats: torch.Tensor | None = None,
    ):
        """
        images:     (B, 3, H, W)  — normalized crops
        traj_seq:   (B, N, 9)     — trajectory history; None khi cold-start
        gate_feats: (B, 3)        — [age_norm, occ_ratio, delta_t_norm]; None OK

        Returns (eval):   normalized embedding (B, 2048)
        Returns (train):  dict với các keys: features, feat_bn, cls_outputs,
                          motion_pred, gate
        """
        B = images.size(0)

        # ── Motion Branch ──────────────────────────────────────────────────
        if traj_seq is not None:
            h_motion, motion_pred, g = self.motion_encoder(traj_seq, gate_feats)
        else:
            # Cold-start: FiLM hoạt động như identity (g=0 → γ=1, β=0)
            h_motion    = torch.zeros(B, self.motion_encoder.hidden_dim, device=images.device)
            motion_pred = torch.zeros(B, 4, device=images.device)
            g           = torch.zeros(B, 1, device=images.device)

        # ── ReID Branch + FiLM ────────────────────────────────────────────
        feat_map = self._forward_backbone_film(images, h_motion, g)  # (B, 2048, h, w)
        feat     = self.gap(feat_map).flatten(1)                      # (B, 2048)
        feat_bn  = self.bnneck(feat)                                  # (B, 2048)

        if not self.training:
            return F.normalize(feat_bn, dim=1)

        # ── Classification head (chỉ dùng lúc train) ─────────────────────
        cls_logits = self.classifier(feat_bn)   # (B, num_classes)

        return {
            "features":    feat,          # before BNNeck → dùng cho triplet loss
            "feat_bn":     feat_bn,        # after  BNNeck → dùng cho CE loss
            "cls_outputs": cls_logits,
            "motion_pred": motion_pred,
            "gate":        g,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Inference helper cho tracker (extract embedding với motion context)
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_feat(
        self,
        images: torch.Tensor,
        traj_seq: torch.Tensor | None = None,
        gate_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Trả về normalized embedding (B, 2048). Gọi khi eval()."""
        was_training = self.training
        self.eval()
        out = self(images, traj_seq, gate_feats)
        if was_training:
            self.train()
        return out


# ─────────────────────────────────────────────────────────────────────────
# Weight loading helpers
# ─────────────────────────────────────────────────────────────────────────

def _load_reid_weights(backbone: nn.Module, checkpoint_path: str) -> None:
    """Load backbone weights từ fast-reid checkpoint (.pth)."""
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("model", state)

    # fast-reid lưu toàn bộ model — chỉ lấy phần backbone
    backbone_state = {
        k.replace("backbone.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }
    if not backbone_state:
        # Thử load trực tiếp (checkpoint chỉ chứa backbone)
        backbone_state = state_dict

    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    print(
        f"[FiLMReIDModel] ReID checkpoint loaded: {checkpoint_path}\n"
        f"  missing={len(missing)}, unexpected={len(unexpected)}"
    )


def _load_imagenet_resnest50(backbone: nn.Module) -> None:
    """
    Load ImageNet pretrained weights cho ResNeSt50.
    Download tự động từ official repo lần đầu (~90MB), cache tại ~/.cache/torch.

    Đây là minimum requirement để train hội tụ mà không cần ReID checkpoint.
    """
    _add_fastreid_to_path()
    from fastreid.modeling.backbones.resnest import model_urls, short_hash

    url = model_urls["resnest50"]
    print(f"[FiLMReIDModel] Loading ImageNet pretrained ResNeSt50 from:\n  {url}")

    try:
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=True, map_location="cpu", check_hash=False
        )
        # ImageNet checkpoint có key kiểu khác — chỉ lấy feature layers
        # Bỏ qua fc (classifier head) vì fast-reid dùng head riêng
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        print(
            f"[FiLMReIDModel] ImageNet weights loaded — "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print(f"  Missing (expected nếu last_stride=1): {missing[:5]}")
    except Exception as e:
        print(
            f"[FiLMReIDModel] WARNING: ImageNet download failed ({e})\n"
            f"  Backbone sẽ dùng random init — khó converge.\n"
            f"  Hãy tải thủ công và truyền vào --pretrained-reid."
        )


# ─────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────

def build_film_reid_model(
    num_classes: int,
    motion_hidden_dim: int = 128,
    fastreid_config: str | None = None,
    pretrained_reid_path: str | None = None,
) -> FiLMReIDModel:
    return FiLMReIDModel(
        num_classes=num_classes,
        motion_hidden_dim=motion_hidden_dim,
        fastreid_config=fastreid_config,
        pretrained_reid_path=pretrained_reid_path,
    )
