from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

from config import Config
from .coordinate_attention import CoordinateAttention
from .decoder import Decoder
from .memory_module import MemoryModule


def denorm_image(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Reverse ImageNet normalization to [0,1]."""
    out = tensor.clone().float()
    for idx, (m, s) in enumerate(zip(mean, std)):
        out[:, idx] = out[:, idx] * s + m
    return out.clamp(0, 1)


class TimmViTEncoder(nn.Module):
    def __init__(
        self, model_name: str = "vit_base_patch16_384", pretrained: bool = True
    ) -> None:
        super().__init__()
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.embed_dim = self.vit.embed_dim
        patch_size = self.vit.patch_embed.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.forward_features(x)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.ndim != 3:
            raise RuntimeError(
                f"Unexpected token shape from timm ViT: {tuple(tokens.shape)}"
            )

        num_patches = (x.shape[-2] // self.patch_size) * (
            x.shape[-1] // self.patch_size
        )
        if tokens.shape[1] == num_patches + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != num_patches:
            raise RuntimeError(
                f"Token length mismatch: got {tokens.shape[1]}, expected {num_patches} or {num_patches + 1}"
            )

        bsz, n_tokens, dim = tokens.shape
        h = w = int(n_tokens**0.5)
        return tokens.reshape(bsz, h, w, dim).permute(0, 3, 1, 2)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def get_cls_attention(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Returns [CLS]->patch attention from the final transformer block when available."""
        attn_store: list[torch.Tensor] = []

        def _hook(
            _module: nn.Module, _inp: tuple[torch.Tensor], out: torch.Tensor
        ) -> None:
            if isinstance(out, torch.Tensor):
                attn_store.append(out.detach())

        handle = self.vit.blocks[-1].attn.attn_drop.register_forward_hook(_hook)
        _ = self.forward(x)
        handle.remove()

        if not attn_store:
            return None

        attn = attn_store[0]
        if attn.ndim != 4:
            return None

        attn = attn.mean(dim=1)
        if attn.shape[-1] <= 1:
            return None
        return attn[:, 0, 1:]


class ViTMemoryAutoencoder(nn.Module):
    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = TimmViTEncoder(cfg.ENCODER_MODEL, pretrained=pretrained)
        self.memory = MemoryModule(
            num_slots=cfg.NUM_SLOTS,
            embed_dim=cfg.EMBED_DIM,
            shrink_threshold=cfg.SHRINK_THRESHOLD,
            shrink_gamma=cfg.SHRINK_GAMMA,
            temperature=cfg.MEM_TEMPERATURE,
        )
        self.coord_attn = CoordinateAttention(channels=cfg.EMBED_DIM)
        self.decoder = Decoder(embed_dim=cfg.EMBED_DIM)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        retrieved, entropy, weights = self.memory(feat)
        attended = self.coord_attn(retrieved)
        recon = self.decoder(attended)
        return recon, entropy, weights

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon, _, _ = self(x)
        x_01 = denorm_image(x)
        diff = (x_01 - recon).pow(2).mean(dim=1, keepdim=True)
        return diff, recon
