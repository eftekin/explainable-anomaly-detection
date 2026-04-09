"""
CNN Decoder: (B, 768, 24, 24) → (B, 3, 384, 384)

6 layers:
  Layer 1-4: ConvTranspose2d ×2 upsample + BN + ReLU
  Layer 5:   Conv2d refinement + BN + ReLU  (same spatial size)
  Layer 6:   Conv2d + Tanh  (final output)

Channel progression: 768 → 512 → 256 → 128 → 64 → 32 → 3
"""

import torch
import torch.nn as nn


def _up_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def _same_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class Decoder(nn.Module):
    def __init__(self, in_channels: int = 768, out_channels: int = 3):
        super().__init__()
        self.layer1 = _up_block(768, 512)    # 24  → 48
        self.layer2 = _up_block(512, 256)    # 48  → 96
        self.layer3 = _up_block(256, 128)    # 96  → 192
        self.layer4 = _up_block(128,  64)    # 192 → 384
        self.layer5 = _same_block(64,  32)   # 384 → 384
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.layer6(x)   # (B, 3, 384, 384)
