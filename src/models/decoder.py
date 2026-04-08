"""
Convolutional decoder.

Takes the memory-refined, attention-weighted feature map (B, 768, 24, 24)
and progressively upsamples it back to the input resolution (B, 3, 384, 384).

Reference: Section 4.3.4 of the paper.
"""

import torch
import torch.nn as nn


def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


_N_UPSAMPLE = 4  # stride-2 stages: 24 → 48 → 96 → 192 → 384


class Decoder(nn.Module):
    """
    Progressive upsampling via TransposedConv layers.

    Input:  (B, embed_dim=768, 24, 24)
    Output: (B, 3, 384, 384)

    6 stages total:
        Stages 0-3 (stride=2): 24 → 48 → 96 → 192 → 384
        Stages 4-5 (stride=1): refinement at 384×384
    """

    def __init__(self, embed_dim: int = 768, channels: list[int] = None):
        super().__init__()
        if channels is None:
            channels = [512, 256, 128, 64, 32, 16]

        layers = []
        in_c = embed_dim
        for i, out_c in enumerate(channels):
            if i < _N_UPSAMPLE:
                # Upsampling: doubles spatial resolution
                layers += [
                    nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    _conv_block(out_c, out_c),
                ]
            else:
                # Refinement: keeps spatial resolution
                layers += [
                    nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    _conv_block(out_c, out_c),
                ]
            in_c = out_c

        self.up_path = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_c, 3, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, embed_dim, H_p, W_p)
        Returns:
            recon: (B, 3, H, W) in [-1, 1] range (tanh output)
        """
        x = self.up_path(z)
        return torch.tanh(self.head(x))
