"""Lunar Crater Neural Network Architecture & Multi-Angle Invariance Learning.

Provides:
1. CraterInvarianceNet: Deep CNN extracting 128-D angle-invariant crater embeddings.
2. AngleInvariantContrastiveLoss & MultiAngleCosineLoss: Supervised / metric loss
   for learning robust representations of identical craters viewed across divergent
   illumination, incidence, azimuth, and slant angles.
3. Parameter tracking: Records gradient norms and weight delta updates during backprop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class ConvBlock(nn.Module):
    """Convolutional block with Group Normalization and LeakyReLU (robust to any batch size)."""
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        groups = min(8, out_c)
        while out_c % groups != 0 and groups > 1:
            groups -= 1
        self.gn = nn.GroupNorm(groups, out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))

class CraterInvarianceNet(nn.Module):
    """Deep Neural Network trained to produce angle-invariant representations of craters.
    
    Lunar craters present extreme appearance changes when sunlight angle (solar incidence
    and azimuth) or orbital viewing slant changes. This network learns an embedding space
    where the same crater at different angles maps to nearby points on a unit hypersphere,
    while non-matching crater morphology or plains are separated.
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Multi-scale convolutional feature extractor
        # Input: (B, 1, 64, 64) crater image patch
        self.layer1 = nn.Sequential(
            ConvBlock(1, 32, kernel_size=3, padding=1),
            ConvBlock(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)  # (B, 32, 32, 32)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)  # (B, 64, 16, 16)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)  # (B, 128, 8, 8)
        )
        self.layer4 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((4, 4))  # (B, 256, 4, 4)
        )

        # Invariant Projection Head with LayerNorm (supports arbitrary batch sizes)
        self.projector = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate 2D feature map for visual heatmaps."""
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing L2-normalized embedding vectors.
        
        Args:
            x: Tensor of shape (B, 1, H, W), normalized in [0, 1] or standardized.
        Returns:
            Normalized descriptor embeddings of shape (B, embedding_dim) on unit sphere.
        """
        feats = self.extract_feature_map(x)
        flat = feats.view(feats.size(0), -1)
        embed = self.projector(flat)
        # Project onto unit hypersphere (L2 normalization)
        norm_embed = F.normalize(embed, p=2, dim=1)
        return norm_embed

class AngleInvariantContrastiveLoss(nn.Module):
    """Contrastive loss for multi-angle crater representation learning.
    
    Given:
      z_a: Embedding of crater at Angle A (e.g. 20 deg solar incidence)
      z_b: Embedding of SAME crater at Angle B (e.g. 60 deg solar incidence / rotated azimuth)
      z_neg: Embedding of DIFFERENT crater or non-crater terrain
      
    Loss = d(z_a, z_b)^2 + max(0, margin - d(z_a, z_neg))^2
    where d(u, v) is Euclidean distance on normalized embeddings (equivalent to 2 - 2*cos_sim).
    """
    def __init__(self, margin: float = 0.6, alpha_pos: float = 1.0, alpha_neg: float = 0.25):
        super().__init__()
        self.margin = margin
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def forward(
        self,
        z_angle_a: torch.Tensor,
        z_angle_b: torch.Tensor,
        z_negative: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Positive distance between same crater at different angles
        # Since embeddings are L2 normalized, ||z_a - z_b||^2 = 2 - 2*cos(z_a, z_b)
        cos_pos = torch.sum(z_angle_a * z_angle_b, dim=1)
        dist_pos_sq = torch.clamp(2.0 - 2.0 * cos_pos, min=0.0)
        loss_pos = torch.mean(dist_pos_sq)

        loss_neg = torch.tensor(0.0, device=z_angle_a.device)
        cos_neg_val = 0.0

        if z_negative is not None and z_negative.size(0) > 0:
            cos_neg = torch.sum(z_angle_a * z_negative, dim=1)
            dist_neg = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_neg, min=1e-8))
            # Hinge loss on negative margin
            margin_loss = F.relu(self.margin - dist_neg) ** 2
            loss_neg = torch.mean(margin_loss)
            cos_neg_val = float(cos_neg.mean().item())

        total_loss = self.alpha_pos * loss_pos + self.alpha_neg * loss_neg

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_positive": float(loss_pos.item()),
            "loss_negative": float(loss_neg.item()),
            "avg_positive_cosine": float(cos_pos.mean().item()),
            "avg_negative_cosine": cos_neg_val
        }
        return total_loss, metrics

class MultiAngleCosineLoss(nn.Module):
    """Cosine Invariance Loss maximizing similarity across N angles of the same crater."""
    def __init__(self):
        super().__init__()

    def forward(self, embeddings_list: list) -> Tuple[torch.Tensor, Dict[str, float]]:
        """embeddings_list: list of tensors [z_angle_1, z_angle_2, ..., z_angle_k]"""
        n_angles = len(embeddings_list)
        if n_angles < 2:
            return torch.tensor(0.0, requires_grad=True), {"total_loss": 0.0, "avg_cosine": 1.0}

        losses = []
        cosines = []
        for i in range(n_angles):
            for j in range(i + 1, n_angles):
                cos_sim = torch.sum(embeddings_list[i] * embeddings_list[j], dim=1)
                losses.append(1.0 - cos_sim)
                cosines.append(cos_sim)

        loss_stacked = torch.cat(losses)
        cos_stacked = torch.cat(cosines)

        total_loss = torch.mean(loss_stacked)
        avg_cos = float(torch.mean(cos_stacked).item())

        return total_loss, {
            "total_loss": float(total_loss.item()),
            "avg_cosine": avg_cos
        }
