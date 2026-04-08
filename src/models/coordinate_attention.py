"""
Coordinate Attention (CA) module.

Decomposes global average pooling into two 1-D pooling operations along H and W,
then generates channel-wise attention weights that encode precise location information.
This helps the decoder localize defects accurately.

Reference: Section 4.3.3 of the paper.
"""

import torch
import torch.nn as nn
class CoordinateAttention(nn.Module):
    """
    Args:
        in_channels (int): Number of input feature channels.
        reduction (int):   Reduction ratio for the bottleneck MLP.
    """

    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, W)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W) – attention-weighted features
        """
        _, _, H, W = x.shape

        # Horizontal and vertical pooling
        x_h = self.pool_h(x)           # (B, C, H, 1)
        x_w = self.pool_w(x)           # (B, C, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1) – concatenate along H dim

        # Concatenate and apply shared bottleneck
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        # Split back
        x_h_att, x_w_att = torch.split(y, [H, W], dim=2)
        x_w_att = x_w_att.permute(0, 1, 3, 2)  # (B, mid, 1, W)

        # Generate attention maps
        att_h = torch.sigmoid(self.conv_h(x_h_att))  # (B, C, H, 1)
        att_w = torch.sigmoid(self.conv_w(x_w_att))  # (B, C, 1, W)

        return x * att_h * att_w
