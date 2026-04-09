"""
Training script.

Usage:
    python train.py [--data_root ./data/mvtec/bottle] [--epochs 200] ...

Encoder is frozen for the first `freeze_epochs` epochs (default 30),
then unfrozen. CosineAnnealingLR over T_max=200 epochs. Early stopping
with patience=50.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from src.models.autoencoder import ViTMemoryAutoencoder
from src.data.dataset import get_train_loader


# ── Loss ──────────────────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    kernel: torch.Tensor  # registered buffer

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        kernel = self._make_kernel(window_size)
        self.register_buffer("kernel", kernel)
        self.ws = window_size

    @staticmethod
    def _make_kernel(size: int) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = torch.outer(g, g)
        g /= g.sum()
        return g.unsqueeze(0).unsqueeze(0)  # (1, 1, ws, ws)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        pad = self.ws // 2
        k = self.kernel.repeat(C, 1, 1, 1)

        mu_x  = F.conv2d(x, k, padding=pad, groups=C)
        mu_y  = F.conv2d(y, k, padding=pad, groups=C)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        sig_x2 = F.conv2d(x * x, k, padding=pad, groups=C) - mu_x2
        sig_y2 = F.conv2d(y * y, k, padding=pad, groups=C) - mu_y2
        sig_xy = F.conv2d(x * y, k, padding=pad, groups=C) - mu_xy

        num = (2 * mu_xy + self.C1) * (2 * sig_xy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sig_x2 + sig_y2 + self.C2)
        ssim_map = num / (den + 1e-8)
        return (1.0 - ssim_map).mean()


class AnomalyLoss(nn.Module):
    def __init__(self, lambda_ssim: float = 1.0, lambda_recon: float = 1.0,
                 lambda_ent: float = 0.001):
        super().__init__()
        self.lambda_ssim  = lambda_ssim
        self.lambda_recon = lambda_recon
        self.lambda_ent   = lambda_ent
        self.ssim = SSIMLoss()

    def forward(self, x, x_hat, w_hat):
        l2   = F.mse_loss(x_hat, x)
        ssim = self.ssim(x, x_hat)

        # Entropy loss [Eq. 14]: minimise negative entropy → push weights sparse
        w_c  = w_hat.clamp(min=1e-8)
        ent  = (w_c * torch.log(w_c)).sum(dim=-1).mean()

        l_recon = l2 + self.lambda_ssim * ssim
        total   = self.lambda_recon * l_recon + self.lambda_ent * ent
        return total, l2.detach(), ssim.detach(), ent.detach()


# ── Training utilities ────────────────────────────────────────────────────────

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total, n = 0.0, 0

    for batch_idx, batch in enumerate(loader):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad()
        x_hat, w_hat = model(x)
        loss, l2, ssim, ent = criterion(x, x_hat, w_hat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total += loss.item()
        n += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Ep {epoch:3d} | B {batch_idx+1:4d}/{len(loader)} "
                  f"| loss={loss.item():.4f} l2={l2.item():.4f} "
                  f"ssim={ssim.item():.4f} ent={ent.item():.4f}")

    return total / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: Config) -> None:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    device = _device()
    print(f"Device: {device}")

    loader = get_train_loader(
        cfg.data_root, cfg.img_size, cfg.batch_size, cfg.num_workers
    )

    model = ViTMemoryAutoencoder(
        encoder_name=cfg.encoder_name,
        embed_dim=cfg.embed_dim,
        feat_side=cfg.feat_side,
        num_memory=cfg.num_memory,
        memory_temperature=cfg.memory_temperature,
        ca_reduction=cfg.ca_reduction,
    ).to(device)

    print(f"Encoder frozen for first {cfg.freeze_epochs} epochs.")

    # Only optimise non-encoder params at the start
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01,
    )
    criterion = AnomalyLoss(cfg.lambda_ssim, cfg.lambda_recon, cfg.lambda_ent).to(device)

    best_loss   = float("inf")
    patience_ct = 0

    for epoch in range(1, cfg.epochs + 1):
        # ── Encoder freeze / unfreeze ─────────────────────────────────────────
        if epoch == cfg.freeze_epochs + 1:
            print(f"\n[Epoch {epoch}] Unfreezing encoder and re-initialising optimizer.")
            model.unfreeze_encoder()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.lr * 0.1, betas=(0.9, 0.999),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.epochs - cfg.freeze_epochs,
                eta_min=cfg.lr * 0.001,
            )

        t0 = time.time()
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device, epoch)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{cfg.epochs} | loss={avg_loss:.4f} "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")

        # ── Checkpoint ───────────────────────────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_ct = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, cfg.best_model_path)
            print(f"  ✓ Best model saved (loss={best_loss:.4f})")
        else:
            patience_ct += 1
            if patience_ct >= cfg.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs "
                      f"(no improvement for {cfg.patience} epochs).")
                break

    print(f"\nDone. Best loss: {best_loss:.4f}  → {cfg.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    default="./data/mvtec/bottle")
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=0.0002)
    parser.add_argument("--num_memory",   type=int,   default=100)
    parser.add_argument("--temperature",  type=float, default=0.07)
    parser.add_argument("--freeze_epochs",type=int,   default=30)
    parser.add_argument("--patience",     type=int,   default=50)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    args = parser.parse_args()

    cfg = Config(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_memory=args.num_memory,
        memory_temperature=args.temperature,
        freeze_epochs=args.freeze_epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        best_model_path=os.path.join(args.checkpoint_dir, "best_model.pth"),
    )
    main(cfg)
