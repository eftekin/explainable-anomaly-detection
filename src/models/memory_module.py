from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModule(nn.Module):
    """Content-addressable memory with Yang & Guo style sparsification."""

    def __init__(
        self,
        num_slots: int,
        embed_dim: int,
        shrink_threshold: float = 0.0025,
        shrink_gamma: float = 2.0,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.shrink_threshold = shrink_threshold
        self.shrink_gamma = shrink_gamma
        self.temperature = temperature

        self.memory = nn.Parameter(torch.empty(num_slots, embed_dim))
        nn.init.trunc_normal_(self.memory, std=0.02)

    def _address(self, query: torch.Tensor) -> torch.Tensor:
        # Sharp softmax addressing with tau=0.05.
        q = F.normalize(query, p=2, dim=1)
        m = F.normalize(self.memory, p=2, dim=1)
        sim = torch.mm(q, m.t()) / self.temperature
        return F.softmax(sim, dim=1)

    def _shrinkage(self, retrieved: torch.Tensor) -> torch.Tensor:
        lam = (
            self.shrink_gamma * self.shrink_threshold / (retrieved.abs() + 1e-8)
        ).clamp(0, 1)
        return retrieved.sign() * F.relu(retrieved.abs() * (1 - lam))

    def _entropy(self, weights: torch.Tensor) -> torch.Tensor:
        return -(weights * (weights + 1e-8).log()).sum(dim=1).mean()

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map (B,D,H,W) -> retrieved features, entropy loss term, image-level slot usage."""
        bsz, dim, h, w = features.shape
        query = features.permute(0, 2, 3, 1).reshape(bsz * h * w, dim)

        weights = self._address(query)
        retrieved = torch.mm(weights, self.memory)
        retrieved = self._shrinkage(retrieved)
        entropy = self._entropy(weights)

        retrieved = retrieved.reshape(bsz, h, w, dim).permute(0, 3, 1, 2)
        weights_image = weights.reshape(bsz, h * w, self.num_slots).mean(dim=1)

        return retrieved, entropy, weights_image
