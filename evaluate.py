from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.data.dataset import MVTecDataset
from src.models.autoencoder import ViTMemoryAutoencoder


def build_test_loader(cfg: Config) -> DataLoader:
    test_ds = MVTecDataset(
        cfg.DATA_ROOT, cfg.CATEGORY, split="test", image_size=cfg.IMAGE_SIZE
    )
    return DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(
    model: ViTMemoryAutoencoder,
    loader: DataLoader,
    device: torch.device,
    top_k_ratio: float,
) -> tuple[float, float]:
    model.eval()

    image_scores: list[float] = []
    image_labels: list[int] = []
    pixel_scores: list[float] = []
    pixel_labels: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            x = batch["image"].to(device)
            label = int(batch["label"].item())
            mask = batch["mask"]

            diff, _ = model.anomaly_map(x)
            flat = diff[0, 0].cpu().flatten().numpy()

            k = max(1, int(top_k_ratio * len(flat)))
            image_scores.append(float(np.sort(flat)[-k:].mean()))
            image_labels.append(label)

            if label == 1 and mask is not None:
                gt = mask[0, 0].numpy().flatten()
                if gt.max() > 0:
                    pixel_scores.extend(flat.tolist())
                    pixel_labels.extend(gt.tolist())

    if len(set(image_labels)) < 2:
        raise RuntimeError(
            "Image-level AUROC requires both normal and anomaly samples."
        )

    img_auroc = roc_auc_score(image_labels, image_scores)

    pix_auroc = 0.0
    if pixel_scores:
        pix_bin = (np.array(pixel_labels) > 0.5).astype(int)
        if len(np.unique(pix_bin)) > 1:
            pix_auroc = float(roc_auc_score(pix_bin, pixel_scores))

    return float(img_auroc), float(pix_auroc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ViT-memory anomaly detector."
    )
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--top-k-ratio", type=float, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.category:
        cfg.CATEGORY = args.category
    if args.data_root:
        cfg.DATA_ROOT = args.data_root
    if args.output_path:
        cfg.OUTPUT_PATH = args.output_path
    if args.top_k_ratio is not None:
        cfg.TOP_K_RATIO = args.top_k_ratio
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(Config(), args)
    cfg.ensure_dirs()

    device = cfg.get_device()
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ViTMemoryAutoencoder(cfg, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    test_loader = build_test_loader(cfg)
    img_auroc, pix_auroc = evaluate(model, test_loader, device, cfg.TOP_K_RATIO)

    print("=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"Category:          {cfg.CATEGORY}")
    print(f"Image-level AUROC: {img_auroc:.4f} ({img_auroc * 100:.1f}%)")
    print(f"Pixel-level AUROC: {pix_auroc:.4f} ({pix_auroc * 100:.1f}%)")
    print(f"Top-k ratio:       {cfg.TOP_K_RATIO}")
    print("Scoring direction: error up -> score up")
    print("=" * 55)

    metrics_path = Path(cfg.OUTPUT_PATH) / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "category": cfg.CATEGORY,
                "image_auroc": img_auroc,
                "pixel_auroc": pix_auroc,
                "top_k_ratio": cfg.TOP_K_RATIO,
                "checkpoint": str(checkpoint_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
