"""
ViT-Memory Autoencoder.

Encoder : pretrained google/vit-base-patch16-384 (HuggingFace transformers).
          Patch tokens → reshaped to (B, 768, 24, 24).
          Frozen for the first `freeze_epochs` epochs, then unfrozen.

Pipeline: X → ViT Encoder → F
                → MemoryModule → F_hat, w_hat
                → CoordinateAttention → F_attended
                → Decoder → X_hat

Returns (X_hat, w_hat) during forward.
"""

import torch
import torch.nn as nn
from transformers import ViTModel

from .memory_module import MemoryModule
from .coordinate_attention import CoordinateAttention
from .decoder import Decoder


class ViTMemoryAutoencoder(nn.Module):
    def __init__(
        self,
        encoder_name: str = "google/vit-base-patch16-384",
        embed_dim: int = 768,
        feat_side: int = 24,          # 384 / 16
        num_memory: int = 100,
        memory_temperature: float = 0.07,
        ca_reduction: int = 32,
    ):
        super().__init__()
        self.feat_side = feat_side

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = ViTModel.from_pretrained(encoder_name)
        # Start with encoder frozen; call unfreeze_encoder() at epoch 30
        self._set_encoder_grad(requires_grad=False)

        # ── Memory + Attention + Decoder ──────────────────────────────────────
        self.memory     = MemoryModule(num_memory, embed_dim, memory_temperature)
        self.coord_attn = CoordinateAttention(embed_dim, ca_reduction)
        self.decoder    = Decoder(embed_dim, out_channels=3)

    # ── Encoder freeze / unfreeze ─────────────────────────────────────────────

    def _set_encoder_grad(self, requires_grad: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = requires_grad

    def freeze_encoder(self) -> None:
        self._set_encoder_grad(False)

    def unfreeze_encoder(self) -> None:
        self._set_encoder_grad(True)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 384, 384) — ImageNet-normalised
        Returns:
            x_hat : (B, 3, 384, 384)
            w_hat : (B, H_f*W_f, N)
        """
        # ViTModel returns last_hidden_state: (B, 1+N_patches, D)
        # where index 0 is the [CLS] token
        outputs = self.encoder(pixel_values=x)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]   # (B, 576, 768)

        B, N, D = patch_tokens.shape
        H = W = self.feat_side
        # reshape to spatial feature map (B, D, H, W)
        F_map = patch_tokens.transpose(1, 2).reshape(B, D, H, W)  # (B, 768, 24, 24)

        F_hat, w_hat   = self.memory(F_map)       # (B, 768, 24, 24), (B, 576, N)
        F_attended     = self.coord_attn(F_hat)   # (B, 768, 24, 24)
        x_hat          = self.decoder(F_attended) # (B, 3, 384, 384)

        return x_hat, w_hat

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-pixel anomaly map (B, 1, H, W) via L2 diff + Gaussian blur.
        """
        self.eval()
        x_hat, _ = self(x)
        diff = (x - x_hat) ** 2                          # (B, 3, H, W)
        score_map = diff.mean(dim=1, keepdim=True)       # (B, 1, H, W)
        return _gaussian_blur(score_map, kernel_size=21, sigma=4.0)


# ── Utility ───────────────────────────────────────────────────────────────────

def _gaussian_blur(x: torch.Tensor, kernel_size: int = 21,
                   sigma: float = 4.0) -> torch.Tensor:
    C = x.shape[1]
    k = kernel_size
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = torch.outer(g, g)
    g /= g.sum()
    kernel = g.view(1, 1, k, k).repeat(C, 1, 1, 1)
    import torch.nn.functional as F
    return F.conv2d(x, kernel, padding=k // 2, groups=C)
