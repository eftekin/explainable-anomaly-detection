"""
Coordinate Attention — Hou et al., CVPR 2021.

Separates spatial pooling into two 1-D operations:
  • X-direction avg pool: (B, C, H, W) → (B, C, H, 1)
  • Y-direction avg pool: (B, C, H, W) → (B, C, 1, W)

Concatenate along the spatial dimension, shared conv + BN + h-swish,
then split and produce two attention maps via 1×1 convs + Sigmoid.

Input / Output: (B, C, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        mid = max(8, in_channels // reduction)

        self.conv_mid = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn_mid   = nn.BatchNorm2d(mid)
        self.conv_h   = nn.Conv2d(mid, in_channels, kernel_size=1)
        self.conv_w   = nn.Conv2d(mid, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, H, W = x.shape

        pool_h  = x.mean(dim=3, keepdim=True)           # (B, C, H, 1)
        pool_w  = x.mean(dim=2, keepdim=True)           # (B, C, 1, W)
        pool_wt = pool_w.permute(0, 1, 3, 2)            # (B, C, W, 1)

        y = torch.cat([pool_h, pool_wt], dim=2)         # (B, C, H+W, 1)
        y = F.hardswish(self.bn_mid(self.conv_mid(y)))  # (B, mid, H+W, 1)

        y_h, y_w = torch.split(y, [H, W], dim=2)        # (B,mid,H,1), (B,mid,W,1)

        a_h = torch.sigmoid(self.conv_h(y_h))                      # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(y_w.permute(0, 1, 3, 2)))  # (B, C, 1, W)

        return identity * a_h * a_w
