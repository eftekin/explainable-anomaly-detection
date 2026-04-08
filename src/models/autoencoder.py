"""
Full ViT-Memory Autoencoder.

Pipeline (Section 4.3 of the paper):
    Image → ViTEncoder → MemoryModule → CoordinateAttention → Decoder → Reconstructed Image

Anomaly score is the pixel-wise L2 difference between input and reconstruction.
"""

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ViTEncoder
from .memory_module import MemoryModule
from .coordinate_attention import CoordinateAttention
from .decoder import Decoder


class AnomalyAutoencoder(nn.Module):
    def __init__(
        self,
        vit_model: str = "google/vit-base-patch16-384",
        freeze_encoder: bool = False,
        memory_size: int = 100,
        embed_dim: int = 768,
        decoder_channels: list[int] = None,
    ):
        super().__init__()
        self.encoder = ViTEncoder(vit_model, freeze=freeze_encoder)
        self.memory = MemoryModule(memory_size, embed_dim)
        self.coord_attn = CoordinateAttention(embed_dim)
        self.decoder = Decoder(embed_dim, decoder_channels)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 384, 384)  – ImageNet-normalized input
        Returns:
            recon:   (B, 3, 384, 384) – reconstructed image (tanh)
            attn_w:  (B, H*W, M)      – memory attention weights for entropy loss
        """
        # Encode
        z = self.encoder(x)            # (B, 768, 24, 24)

        # Memory read – replace anomaly features with closest normal prototypes
        z_mem, attn_w = self.memory(z)  # (B, 768, 24, 24), (B, 576, M)

        # Coordinate attention for defect localization
        z_att = self.coord_attn(z_mem)  # (B, 768, 24, 24)

        # Decode back to image space
        recon = self.decoder(z_att)     # (B, 3, 384, 384)

        return recon, attn_w

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce a pixel-level anomaly heatmap.

        Args:
            x: (B, 3, 384, 384)
        Returns:
            heatmap: (B, 1, 384, 384)  values in [0, 1]
        """
        self.eval()
        recon, _ = self(x)

        # Bring reconstruction to same value range as input for fair comparison.
        # Input is ImageNet-normalized; reconstruction is tanh.  We undo tanh
        # normalization by mapping both to [0,1] via min-max per sample.
        diff = (x - recon).pow(2).mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Normalize per sample
        B = diff.shape[0]
        flat = diff.view(B, -1)
        mins = flat.min(dim=1).values.view(B, 1, 1, 1)
        maxs = flat.max(dim=1).values.view(B, 1, 1, 1)
        heatmap = (diff - mins) / (maxs - mins + 1e-8)

        # Gaussian smoothing (sigma=4) as described in the paper
        heatmap_np = heatmap.squeeze(1).cpu().numpy()  # (B, H, W)
        smoothed = np.stack([
            ndimage.gaussian_filter(h, sigma=4.0) for h in heatmap_np
        ])
        heatmap = torch.from_numpy(smoothed).unsqueeze(1).to(diff.device)
        return heatmap
