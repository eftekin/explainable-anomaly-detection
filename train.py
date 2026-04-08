"""
Training script.

Usage:
    python train.py --category bottle --epochs 200
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import model_cfg, train_cfg, data_cfg
from src.data import MVTecDataset
from src.models import AnomalyAutoencoder
from src.utils import TotalLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", default=data_cfg.category)
    p.add_argument("--data_root", default=str(data_cfg.data_root))
    p.add_argument("--epochs", type=int, default=train_cfg.epochs)
    p.add_argument("--batch_size", type=int, default=train_cfg.batch_size)
    p.add_argument("--lr", type=float, default=train_cfg.learning_rate)
    p.add_argument("--memory_size", type=int, default=model_cfg.memory_size)
    p.add_argument(
        "--freeze_encoder", action="store_true", default=model_cfg.freeze_encoder
    )
    p.add_argument("--checkpoint_dir", default=str(train_cfg.checkpoint_dir))
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir) / args.category
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_dataset = MVTecDataset(
        root=args.data_root,
        category=args.category,
        split="train",
        image_size=model_cfg.image_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
    )
    log.info(f"Training samples: {len(train_dataset)}")

    # Model
    model = AnomalyAutoencoder(
        vit_model=model_cfg.vit_model,
        freeze_encoder=args.freeze_encoder,
        memory_size=args.memory_size,
        embed_dim=model_cfg.embed_dim,
        decoder_channels=model_cfg.decoder_channels,
        use_memory=model_cfg.use_memory,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = TotalLoss(
        lambda_ssim=train_cfg.lambda_ssim,
        lambda_entropy=train_cfg.lambda_entropy,
    )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        log.info(f"Resumed from epoch {start_epoch}")

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = recon_loss = entropy_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)

            recon, attn_w = model(images, epoch=epoch)
            losses = criterion(recon, images, attn_w)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            total_loss += losses["total"].item()
            recon_loss += losses["recon"].item()
            entropy_loss += losses["entropy"].item()

        n = len(train_loader)
        scheduler.step()
        log.info(
            f"Epoch [{epoch+1}/{args.epochs}]  "
            f"loss={total_loss/n:.4f}  "
            f"recon={recon_loss/n:.4f}  "
            f"entropy={entropy_loss/n:.4f}"
        )

        # Checkpoint
        if (epoch + 1) % train_cfg.save_every == 0:
            path = checkpoint_dir / f"epoch_{epoch+1:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                path,
            )

        # Best model + early stopping
        avg = total_loss / n
        if avg < best_loss:
            best_loss = avg
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_dir / "best.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.patience:
                log.info("Early stopping triggered.")
                break

    log.info(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
