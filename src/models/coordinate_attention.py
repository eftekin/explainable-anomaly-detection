from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """Coordinate attention block (Hou et al., 2021)."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        mid = max(8, channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.GELU()

        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, ch, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h_attn, x_w_attn = torch.split(y, [h, w], dim=2)

        a_h = self.conv_h(x_h_attn).sigmoid()
        a_w = self.conv_w(x_w_attn.permute(0, 1, 3, 2)).sigmoid()
        return x * a_h * a_w
