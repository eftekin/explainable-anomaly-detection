"""
Memory Module with temperature-scaled cosine similarity and sparsification.

Forward pass per query z:
  1. Cosine similarity:  d(z, m_i) = z·m_i^T / (‖z‖·‖m_i‖)          [Eq. 7]
  2. Temperature scale + softmax:  w = softmax(d / τ)                  [Eq. 8]
  3. Sparsification:  ŵ_i = max(w_i-λ,0)·w_i / (|w_i-λ|+ε)          [Eq. 10]
  4. L1 normalisation
  5. Weighted sum:  ẑ = Σ ŵ_i · m_i                                   [Eq. 9]

Input : F  (B, C, H_f, W_f)
Output: F_hat (B, C, H_f, W_f),  w_hat (B, H_f*W_f, N)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryModule(nn.Module):
    def __init__(self, num_memory: int = 100, feature_dim: int = 768,
                 temperature: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self.N = num_memory
        self.C = feature_dim
        self.temperature = temperature
        self.eps = eps
        self.lam = 1.0 / num_memory   # shrinkage threshold λ = 1/N

        # Memory initialised on the unit hypersphere
        self.memory = nn.Parameter(
            F.normalize(torch.randn(num_memory, feature_dim), dim=1)
        )

    def forward(self, F_map: torch.Tensor):
        B, C, H, W = F_map.shape
        P = H * W

        # Flatten spatial dims: (B, C, H, W) → (B*P, C)
        z = F_map.permute(0, 2, 3, 1).reshape(B * P, C)

        # ── Step 1 & 2: temperature-scaled cosine similarity + softmax ──
        z_norm = F.normalize(z, dim=1)           # (B*P, C)
        m_norm = F.normalize(self.memory, dim=1) # (N,   C)
        scores = torch.mm(z_norm, m_norm.t()) / self.temperature  # (B*P, N)
        w = F.softmax(scores, dim=-1)            # (B*P, N)

        # ── Step 3: Sparsification [Eq. 10] ──
        diff = w - self.lam
        w_hat = F.relu(diff) * w / (diff.abs() + self.eps)  # (B*P, N)

        # ── Step 4: L1 normalisation ──
        w_hat = w_hat / w_hat.sum(dim=1, keepdim=True).clamp(min=self.eps)

        # ── Step 5: Weighted sum [Eq. 9] ──
        z_hat = torch.mm(w_hat, self.memory)    # (B*P, C)

        F_hat = z_hat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        w_hat_spatial = w_hat.reshape(B, P, self.N)              # (B, P, N)

        return F_hat, w_hat_spatial
