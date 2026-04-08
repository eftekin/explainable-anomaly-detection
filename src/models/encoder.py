"""
ViT-based encoder.

Wraps a pretrained HuggingFace ViT and exposes the patch token sequence
(excluding the [CLS] token) as a 2-D spatial feature map for downstream modules.
"""

import math

import torch
import torch.nn as nn
from transformers import ViTModel


class ViTEncoder(nn.Module):
    """
    Pretrained ViT encoder.

    For a 384x384 image with patch_size=16:
        num_patches = (384/16)^2 = 576
        feature map spatial size = 24x24

    Output shape: (B, embed_dim, H_p, W_p)  where H_p = W_p = image_size // patch_size
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-384", freeze: bool = False):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.embed_dim = self.vit.config.hidden_size        # 768
        self.patch_size = self.vit.config.patch_size        # 16
        self.image_size = self.vit.config.image_size        # 384
        self.num_patches_side = self.image_size // self.patch_size  # 24

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: (B, embed_dim, H_p, W_p)
        """
        outputs = self.vit(pixel_values=x)
        # last_hidden_state: (B, 1 + num_patches, embed_dim)
        # index 0 is [CLS], 1: are patch tokens
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B, 576, 768)

        B, N, C = patch_tokens.shape
        H = W = int(math.sqrt(N))
        # reshape to spatial map
        features = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)  # (B, 768, 24, 24)
        return features
