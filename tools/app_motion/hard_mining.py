import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        embeddings: [B, D]
        labels: [B]
        """

        B = embeddings.size(0)

        # ==========================
        # Pairwise Distance Matrix
        # ==========================
        dist_mat = torch.cdist(
            embeddings,
            embeddings,
            p=2
        )  # [B, B]

        # ==========================
        # Positive Mask
        # ==========================
        labels = labels.view(-1, 1)

        positive_mask = (
            labels == labels.t()
        )

        # loại bỏ chính nó
        positive_mask.fill_diagonal_(False)

        # ==========================
        # Negative Mask
        # ==========================
        negative_mask = (
            labels != labels.t()
        )

        # ==========================
        # Hardest Positive
        # ==========================
        hardest_positive = (
            dist_mat * positive_mask.float()
        ).max(dim=1)[0]

        # ==========================
        # Hardest Negative
        # ==========================
        max_dist = dist_mat.max().detach()

        dist_neg = dist_mat.clone()

        dist_neg[~negative_mask] = max_dist + 1

        hardest_negative = (
            dist_neg.min(dim=1)[0]
        )
        # print("hardest_positive:")
        # print(hardest_positive[:8])
        # print("hardest_negative:")
        # print(hardest_negative[:8])

        # ==========================
        # Triplet Loss
        # ==========================
        loss = F.relu(
            hardest_positive
            - hardest_negative
            + self.margin
        )

        nonzero_ratio = (
            (loss > 0).float().mean()
        )
        # print(f"Unique labels: {torch.unique(labels)}")
        # print(f"labels shape: {labels.shape}")

        return loss.mean(), nonzero_ratio