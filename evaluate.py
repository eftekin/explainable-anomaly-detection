"""
Evaluation script – computes image-level and pixel-level AUROC on MVTec test set.

Usage:
    python evaluate.py --category bottle --checkpoint results/checkpoints/bottle/best.pth
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import model_cfg, data_cfg
from src.data import MVTecDataset
from src.models import AnomalyAutoencoder
from src.utils import image_auroc, pixel_auroc, save_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", default=data_cfg.category)
    p.add_argument("--data_root", default=str(data_cfg.data_root))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--save_heatmaps", action="store_true")
    p.add_argument("--output_dir", default="results/heatmaps")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    test_dataset = MVTecDataset(
        root=args.data_root,
        category=args.category,
        split="test",
        image_size=model_cfg.image_size,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    log.info(f"Test samples: {len(test_dataset)}")

    # Model
    model = AnomalyAutoencoder(
        vit_model=model_cfg.vit_model,
        memory_size=model_cfg.memory_size,
        embed_dim=model_cfg.embed_dim,
        decoder_channels=model_cfg.decoder_channels,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"Loaded checkpoint from epoch {ckpt['epoch']+1}")

    all_scores, all_labels = [], []
    all_score_maps, all_masks = [], []
    output_dir = Path(args.output_dir) / args.category

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            masks = batch["mask"]
            labels = batch["label"]

            heatmap = model.anomaly_map(images)  # (1, 1, H, W)

            # Image-level score = max pixel score
            score = heatmap.squeeze().cpu().max().item()
            all_scores.append(score)
            all_labels.append(labels.item())

            all_score_maps.append(heatmap.squeeze().cpu().numpy())
            all_masks.append(masks.squeeze().numpy())

            if args.save_heatmaps:
                path = output_dir / f"{i:04d}_label{labels.item()}.png"
                save_result(images[0].cpu(), heatmap[0].cpu(), path)

    img_auc = image_auroc(all_scores, all_labels)
    log.info(f"Image-level AUROC: {img_auc:.4f}")

    # Pixel-level only where GT masks exist (anomalous samples)
    has_mask = [l == 1 for l in all_labels]
    if any(has_mask):
        pix_maps = np.stack([all_score_maps[i] for i, h in enumerate(has_mask) if h])
        pix_masks = np.stack([all_masks[i] for i, h in enumerate(has_mask) if h])
        pix_auc = pixel_auroc(pix_maps, pix_masks)
        log.info(f"Pixel-level AUROC: {pix_auc:.4f}")

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
