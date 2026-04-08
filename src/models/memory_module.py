"""
Memory Module.

Stores M prototype feature vectors of normal samples.
During the forward pass every spatial position in the input feature map is
"read" from memory via softmax attention and replaced with the weighted sum
of memory slots.  This forces the reconstruction to be normal-looking even
when the input contains anomalies.

Reference: Section 4.3.2 of the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModule(nn.Module):
    """
    Args:
        memory_size (int): Number of memory slots M.
        feature_dim (int): Dimensionality of each feature vector (768).
    """

    def __init__(self, memory_size: int = 100, feature_dim: int = 768):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim

        # Learnable memory bank: (M, C)
        self.memory = nn.Parameter(
            torch.randn(memory_size, feature_dim)
        )

    def forward(self, z: torch.Tensor, epoch: int | None = None):
        """
        Args:
            z:     encoder feature map (B, C, H, W)
            epoch: current training epoch; when provided and divisible by 10,
                   logs per-slot weight statistics to stdout.
        Returns:
            z_hat:   memory-reconstructed features (B, C, H, W)
            attn_w:  sparsified & L1-normalised attention weights (B, H*W, M)
                     – used for entropy loss
        """
        B, C, H, W = z.shape

        # Flatten spatial dims: (B, H*W, C)
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Normalize for cosine-like similarity
        z_norm = F.normalize(z_flat, dim=-1)                   # (B, N, C)
        mem_norm = F.normalize(self.memory, dim=-1)            # (M, C)

        # Attention scores → softmax weights  (Eq. 7-8)
        scores = torch.einsum("bnc,mc->bnm", z_norm, mem_norm)
        w = F.softmax(scores, dim=-1)                          # (B, N, M)

        # Log per-slot weight statistics every 10 epochs to track slot differentiation
        if epoch is not None and epoch % 10 == 0:
            # Average over batch and spatial positions to get one value per slot
            slot_w = w.detach().mean(dim=(0, 1))               # (M,)
            print(
                f"[MemoryModule] epoch {epoch:4d} | "
                f"slot-w  mean={slot_w.mean().item():.4f}  "
                f"std={slot_w.std().item():.4f}"
            )

        # Sparsification  (Eq. 10): shrinkage threshold λ = 1/M, then L1 normalise
        lam = 1.0 / self.memory_size
        eps = 1e-8
        attn_w = F.relu(w - lam) * w / (torch.abs(w - lam) + eps)   # hard shrinkage
        attn_w = attn_w / (attn_w.sum(dim=-1, keepdim=True) + eps)   # L1 normalise

        # Read from memory: weighted sum of slots  (Eq. 9 with sparsified weights)
        z_hat_flat = torch.einsum("bnm,mc->bnc", attn_w, self.memory)  # (B, N, C)

        # Reshape back to spatial map
        z_hat = z_hat_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)    # (B, C, H, W)

        return z_hat, attn_w
