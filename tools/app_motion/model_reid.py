import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AppearanceBranch(nn.Module):
    def __init__(self, out_dim=128):
        super(AppearanceBranch, self).__init__()
        # Sử dụng MobileNetV2 làm backbone vì nó cực nhẹ và nhanh
        # Tải pre-trained weights trên ImageNet để hội tụ nhanh hơn
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Bỏ đi lớp Classifier cuối cùng, chỉ lấy Feature Extractor
        self.backbone = mobilenet.features
        
        # Global Average Pooling để đưa feature map về 1 vector 1D
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Nén vector đặc trưng từ 1280 chiều của MobileNetV2 xuống out_dim (VD: 128)
        self.fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.backbone(x)     # [Batch, 1280, H', W']
        x = self.pool(x)         # [Batch, 1280, 1, 1]
        x = torch.flatten(x, 1)  # [Batch, 1280]
        x = self.fc(x)           # [Batch, out_dim]
        return x

class MotionBranch(nn.Module):
    def __init__(self, k_frames, out_dim=64):
        super(MotionBranch, self).__init__()
        # K frames thì sẽ có K-1 bước chuyển động (delta)
        # Mỗi delta có 4 giá trị (dx, dy, dw, dh)
        in_features = (k_frames - 1) * 4
        
        # Một mạng MLP 2 lớp cực nhẹ
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        # Đầu vào x có shape: [Batch, K-1, 4]
        x = torch.flatten(x, 1)  # Flatten thành [Batch, (K-1)*4]
        x = self.mlp(x)          # [Batch, out_dim]
        return x

class TwoStreamReID(nn.Module):
    def __init__(self, k_frames=5, appearance_dim=128, motion_dim=64, final_dim=128, num_classes=None):
        super(TwoStreamReID, self).__init__()
        
        self.appearance_net = AppearanceBranch(out_dim=appearance_dim)
        self.motion_net = MotionBranch(k_frames=k_frames, out_dim=motion_dim)
        
        # Ghép 2 vector lại và nén ra embedding cuối cùng
        self.fusion = nn.Sequential(
            nn.Linear(appearance_dim + motion_dim, final_dim),
            nn.BatchNorm1d(final_dim)
            # Lưu ý: Không dùng ReLU ở layer cuối của Embedding
        )
        
        # Đầu ra dùng để phân loại ID trong lúc training (Cross Entropy)
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(final_dim, num_classes)

    def forward(self, img, motion_history):
        # 1. Rút trích RGB (Ngoại hình)
        feat_a = self.appearance_net(img)
        
        # 2. Rút trích Quỹ đạo (Động học)
        feat_m = self.motion_net(motion_history)

        # print(f"shape feat: {feat_a.shape}")
        # print(f"feat_motion: {feat_m.shape}")
        
        # 3. Kết hợp (Concatenate)
        feat_concat = torch.cat([feat_a, feat_m], dim=1)
        
        # 4. Final Embedding
        embedding = self.fusion(feat_concat)
        
        # Chuẩn hóa L2 (L2 Normalize) bắt buộc để tính khoảng cách Cosine khi Matching
        embedding = F.normalize(embedding, p=2, dim=1)
        
        if self.training and self.num_classes is not None:
            # Lúc train trả về cả embedding (để tính Triplet Loss) 
            # và logits (để tính Cross Entropy Loss)
            logits = self.classifier(embedding)
            return embedding, logits
        else:
            # Lúc inference (tracking thực tế) chỉ cần embedding
            return embedding