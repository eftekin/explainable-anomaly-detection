"""
Loss functions as described in Section 4.4 of the paper.

L_recon = ||A - A'||^2 + lambda_ssim * (1 - SSIM(A, A'))
L_entropy = -sum(w * log(w + eps))   over memory attention weights
L_total = L_recon + lambda_entropy * L_entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g)
    return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        kernel = _gaussian_kernel(window_size)
        self.register_buffer("kernel", kernel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: (B, C, H, W)
        Returns:
            scalar loss = 1 - mean_SSIM
        """
        C = pred.shape[1]
        kernel = self.kernel.to(pred.device).expand(C, 1, -1, -1)  # (C, 1, k, k)

        pad = self.window_size // 2

        mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
        mu2 = F.conv2d(target, kernel, padding=pad, groups=C)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
        sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu1_mu2

        ssim_map = (
            (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        ) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

class ReconstructionLoss(nn.Module):
    """
    L_recon = MSE(A, A') + lambda_ssim * SSIM_loss(A, A')
    """

    def __init__(self, lambda_ssim: float = 1.0):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.ssim = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        ssim = self.ssim(pred, target)
        return mse + self.lambda_ssim * ssim


# ---------------------------------------------------------------------------
# Entropy loss (memory module regularization)
# ---------------------------------------------------------------------------

class EntropyLoss(nn.Module):
    """
    Encourages sparse (confident) memory addressing.
    L_entropy = mean( -sum_m( w_m * log(w_m + eps) ) )

    Args:
        attn_w: (B, N, M) softmax attention weights from MemoryModule
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, attn_w: torch.Tensor) -> torch.Tensor:
        entropy = -(attn_w * torch.log(attn_w + self.eps)).sum(dim=-1)  # (B, N)
        return entropy.mean()


# ---------------------------------------------------------------------------
# Total loss
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    def __init__(self, lambda_ssim: float = 1.0, lambda_entropy: float = 0.001):
        super().__init__()
        self.recon_loss = ReconstructionLoss(lambda_ssim)
        self.entropy_loss = EntropyLoss()
        self.lambda_entropy = lambda_entropy

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        attn_w: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        l_recon = self.recon_loss(pred, target)
        if attn_w is not None:
            l_entropy = self.entropy_loss(attn_w)
            total = l_recon + self.lambda_entropy * l_entropy
        else:
            l_entropy = torch.tensor(0.0, device=pred.device)
            total = l_recon
        return {"total": total, "recon": l_recon, "entropy": l_entropy}
