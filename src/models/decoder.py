from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Bilinear upsampling decoder (no transposed convolutions)."""

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()

        def up_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )

        self.net = nn.Sequential(
            up_block(embed_dim, 512),
            up_block(512, 256),
            up_block(256, 128),
            up_block(128, 64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, groups=32, bias=False),
            nn.Conv2d(32, 3, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
