from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.data.dataset import MVTecDataset
from src.models.autoencoder import ViTMemoryAutoencoder, denorm_image

try:
    from pytorch_msssim import ssim as ssim_fn

    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReconLoss(nn.Module):
    """L_recon = alpha * L1 + (1 - alpha) * (1 - SSIM)."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x01: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(x_hat, x01)
        if HAS_SSIM:
            ssim_loss = 1.0 - ssim_fn(x_hat, x01, data_range=1.0, size_average=True)
            return self.alpha * l1 + (1 - self.alpha) * ssim_loss
        return l1


class TotalLoss(nn.Module):
    """L_total = L_recon + lambda_ent * H(slot_weights)."""

    def __init__(self, lambda_ent: float = 0.1) -> None:
        super().__init__()
        self.lambda_ent = lambda_ent
        self.recon = ReconLoss(alpha=0.5)

    def forward(
        self, x01: torch.Tensor, x_hat: torch.Tensor, entropy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = self.recon(x01, x_hat)
        ent_loss = self.lambda_ent * entropy
        return recon_loss + ent_loss, recon_loss, ent_loss


class CollapseDetector:
    """Flags representation collapse when encoder feature variance gets too small."""

    def __init__(self, threshold: float = 1e-4) -> None:
        self.threshold = threshold
        self.variances: list[float] = []
        self.epochs: list[int] = []

    @torch.no_grad()
    def check(
        self,
        model: ViTMemoryAutoencoder,
        loader: DataLoader,
        device: torch.device,
        epoch: int,
    ) -> bool:
        model.eval()
        feats = []
        for idx, batch in enumerate(loader):
            if idx >= 5:
                break
            encoded = model.encoder(batch["image"].to(device)).cpu().flatten(1)
            feats.append(encoded)

        variance = torch.cat(feats, 0).var(0).mean().item()
        self.variances.append(variance)
        self.epochs.append(epoch)

        status = "OK" if variance > self.threshold else "COLLAPSE"
        print(f"  [Collapse] ep {epoch}: variance={variance:.2e} [{status}]")
        model.train()
        return variance > self.threshold


def build_train_loader(cfg: Config) -> DataLoader:
    train_ds = MVTecDataset(
        cfg.DATA_ROOT, cfg.CATEGORY, split="train", image_size=cfg.IMAGE_SIZE
    )
    return DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def train_epoch(
    model: ViTMemoryAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: TotalLoss,
    device: torch.device,
    grad_clip: float,
    epoch: int,
) -> tuple[float, float, float]:
    model.train()

    total = 0.0
    recon = 0.0
    ent = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        x01 = denorm_image(x)

        optimizer.zero_grad(set_to_none=True)
        x_hat, entropy, _ = model(x)
        loss, l_recon, l_ent = criterion(x01, x_hat, entropy)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total += float(loss.item())
        recon += float(l_recon.item())
        ent += float(l_ent.item())

    denom = max(1, len(loader))
    return total / denom, recon / denom, ent / denom


def run_training(
    model: ViTMemoryAutoencoder,
    train_loader: DataLoader,
    criterion: TotalLoss,
    cfg: Config,
    device: torch.device,
) -> tuple[dict[str, list[float]], CollapseDetector, Path]:
    freeze_until = cfg.FREEZE_UNTIL_EPOCH
    num_epochs = cfg.NUM_EPOCHS
    collapse_detector = CollapseDetector(threshold=cfg.VARIANCE_THRESHOLD)

    # Phase 1: freeze encoder.
    model.encoder.freeze()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, freeze_until),
        eta_min=1e-6,
    )

    history: dict[str, list[float]] = {"total": [], "recon": [], "ent": [], "phase": []}
    best_loss = float("inf")
    best_epoch = 0
    ckpt_path = Path(cfg.CHECKPOINT_PATH) / "best_model.pth"

    print(
        f"Training {num_epochs} epochs | tau={cfg.MEM_TEMPERATURE} | "
        f"lambda_ent={cfg.LAMBDA_ENT} | freeze_until={freeze_until}"
    )

    for epoch in range(1, num_epochs + 1):
        if epoch == freeze_until + 1:
            model.encoder.unfreeze()
            enc_params = list(model.encoder.parameters())
            enc_ids = {id(p) for p in enc_params}
            other_params = [p for p in model.parameters() if id(p) not in enc_ids]

            optimizer = torch.optim.Adam(
                [
                    {"params": other_params, "lr": cfg.LEARNING_RATE},
                    {
                        "params": enc_params,
                        "lr": cfg.LEARNING_RATE * cfg.ENCODER_LR_SCALE,
                    },
                ],
                betas=(0.9, 0.999),
                weight_decay=1e-5,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_epochs - freeze_until),
                eta_min=1e-6,
            )
            print(
                "\n[Phase 2] encoder unfrozen | "
                f"enc_lr={cfg.LEARNING_RATE * cfg.ENCODER_LR_SCALE:.2e}, "
                f"rest_lr={cfg.LEARNING_RATE:.2e}"
            )

        phase = 1 if epoch <= freeze_until else 2
        total, recon, ent = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=cfg.GRAD_CLIP,
            epoch=epoch,
        )
        scheduler.step()

        history["total"].append(total)
        history["recon"].append(recon)
        history["ent"].append(ent)
        history["phase"].append(float(phase))

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"ep {epoch:03d}/{num_epochs} | total={total:.5f} | "
                f"recon={recon:.5f} | ent={ent:.5f} | phase={phase}"
            )

        if epoch % 10 == 0:
            collapse_detector.check(model, train_loader, device, epoch)

        if total < best_loss:
            best_loss = total
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "loss": best_loss,
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                },
                ckpt_path,
            )

    print(f"Best loss {best_loss:.5f} at epoch {best_epoch}. Loading best checkpoint.")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    return history, collapse_detector, ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT-memory anomaly detector.")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.category:
        cfg.CATEGORY = args.category
    if args.data_root:
        cfg.DATA_ROOT = args.data_root
    if args.checkpoint_path:
        cfg.CHECKPOINT_PATH = args.checkpoint_path
    if args.output_path:
        cfg.OUTPUT_PATH = args.output_path
    if args.epochs is not None:
        cfg.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        cfg.NUM_WORKERS = args.num_workers
    if args.seed is not None:
        cfg.RANDOM_SEED = args.seed
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(Config(), args)
    cfg.ensure_dirs()

    set_seed(cfg.RANDOM_SEED)
    device = cfg.get_device()
    print(f"Device: {device}")
    print(
        f"Config | category={cfg.CATEGORY} | image={cfg.IMAGE_SIZE} | "
        f"tau={cfg.MEM_TEMPERATURE} | lambda_ent={cfg.LAMBDA_ENT} | freeze={cfg.FREEZE_UNTIL_EPOCH}"
    )

    train_loader = build_train_loader(cfg)
    model = ViTMemoryAutoencoder(cfg, pretrained=not args.no_pretrained).to(device)
    criterion = TotalLoss(lambda_ent=cfg.LAMBDA_ENT).to(device)

    history, collapse_detector, ckpt_path = run_training(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        cfg=cfg,
        device=device,
    )

    history_path = Path(cfg.OUTPUT_PATH) / "training_history.json"
    history_payload: dict[str, Any] = {
        "history": history,
        "collapse": {
            "epochs": collapse_detector.epochs,
            "variances": collapse_detector.variances,
        },
        "checkpoint": str(ckpt_path),
        "config": cfg.to_dict(),
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
