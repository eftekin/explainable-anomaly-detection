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

_DEBUG_DONE = False  # print first-batch sparsification stats once


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
        global _DEBUG_DONE

        B, C, H, W = z.shape
        print(f"[MM] z            : {tuple(z.shape)}")

        # Flatten spatial dims: (B, H*W, C)
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C)
        print(f"[MM] z_flat       : {tuple(z_flat.shape)}")

        # Normalize for cosine-like similarity
        z_norm = F.normalize(z_flat, dim=-1)                   # (B, N, C)
        mem_norm = F.normalize(self.memory, dim=-1)            # (M, C)
        print(f"[MM] z_norm       : {tuple(z_norm.shape)}")
        print(f"[MM] mem_norm     : {tuple(mem_norm.shape)}")

        # Attention scores → softmax weights  (Eq. 7-8)
        scores = torch.einsum("bnc,mc->bnm", z_norm, mem_norm)
        print(f"[MM] scores       : {tuple(scores.shape)}  min={scores.min():.4f}  max={scores.max():.4f}")

        w = F.softmax(scores, dim=-1)                          # (B, N, M)
        print(f"[MM] w (softmax)  : {tuple(w.shape)}  min={w.min():.6f}  max={w.max():.6f}  mean={w.mean():.6f}")

        # Log per-slot weight statistics every 10 epochs to track slot differentiation
        if epoch is not None and epoch % 10 == 0:
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
        print(f"[MM] attn_w (spar): {tuple(attn_w.shape)}  min={attn_w.min():.6f}  max={attn_w.max():.6f}  mean={attn_w.mean():.6f}")

        # First-batch deep inspection: show how many slots survive sparsification
        if not _DEBUG_DONE:
            _DEBUG_DONE = True
            w0 = w.detach()[0]          # (N, M) — first sample in batch
            s0 = attn_w.detach()[0]     # (N, M) — after sparsification
            zeros_pct = (s0 == 0).float().mean().item() * 100
            # Per-slot survival: fraction of spatial positions where slot is non-zero
            slot_survival = (s0 > 0).float().mean(dim=0)   # (M,)
            top5_slots = slot_survival.topk(5).indices.tolist()
            print(
                f"\n[MM] === FIRST-BATCH SPARSIFICATION REPORT ===\n"
                f"  lambda (threshold)  : {lam:.6f}\n"
                f"  w  before  — min={w0.min():.6f}  max={w0.max():.6f}  mean={w0.mean():.6f}  std={w0.std():.6f}\n"
                f"  w  after   — min={s0.min():.6f}  max={s0.max():.6f}  mean={s0.mean():.6f}  std={s0.std():.6f}\n"
                f"  zeroed-out entries  : {zeros_pct:.1f}% of (N×M)\n"
                f"  top-5 active slots  : {top5_slots}  "
                f"(survival rates: {slot_survival[top5_slots].tolist()})\n"
                f"  avg active slots/pos: {(s0 > 0).float().sum(dim=1).mean():.1f} / {self.memory_size}\n"
                f"[MM] =============================================\n"
            )

        # Read from memory: weighted sum of slots  (Eq. 9 with sparsified weights)
        z_hat_flat = torch.einsum("bnm,mc->bnc", attn_w, self.memory)  # (B, N, C)
        print(f"[MM] z_hat_flat   : {tuple(z_hat_flat.shape)}")

        # Reshape back to spatial map
        z_hat = z_hat_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)    # (B, C, H, W)
        print(f"[MM] z_hat        : {tuple(z_hat.shape)}")

        return z_hat, attn_w
