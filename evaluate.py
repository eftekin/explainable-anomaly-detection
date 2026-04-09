"""
Evaluation script — image-level and pixel-level AUROC on MVTec bottle.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth \
                       --data_root  ./data/mvtec/bottle

Outputs:
    Image-level AUROC: <score>
    Pixel-level  AUROC: <score>
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import Config
from src.models.autoencoder import ViTMemoryAutoencoder
from src.data.dataset import get_test_loader


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: ViTMemoryAutoencoder, loader, device: torch.device):
    """
    Returns:
        img_auroc  : float  — image-level AUROC
        pixel_auroc: float  — pixel-level  AUROC
    """
    model.eval()

    all_img_scores: list = []   # one scalar per image
    all_img_labels: list = []   # 0=normal, 1=anomaly
    all_pix_scores: list = []   # flattened per-pixel scores
    all_pix_labels: list = []   # flattened per-pixel GT masks

    for imgs, masks, labels in tqdm(loader, desc="Evaluating"):
        imgs   = imgs.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        score_map = model.anomaly_score(imgs)  # (B, 1, H, W)

        # Image-level score = max of the anomaly map (common in literature)
        B = imgs.shape[0]
        img_scores = score_map.reshape(B, -1).max(dim=1).values  # (B,)

        all_img_scores.append(img_scores.cpu().numpy())
        all_img_labels.append(labels.cpu().numpy())

        # Pixel-level
        all_pix_scores.append(score_map.squeeze(1).cpu().numpy().ravel())
        all_pix_labels.append(masks.squeeze(1).cpu().numpy().ravel())

    img_scores = np.concatenate(all_img_scores)
    img_labels = np.concatenate(all_img_labels)
    pix_scores = np.concatenate(all_pix_scores)
    pix_labels = np.concatenate(all_pix_labels)

    img_auroc   = roc_auc_score(img_labels, img_scores)
    pixel_auroc = roc_auc_score(pix_labels, pix_scores)

    _print_score_stats(img_scores, img_labels)

    return img_auroc, pixel_auroc


def _print_score_stats(img_scores: np.ndarray, img_labels: np.ndarray) -> None:
    """Print mean/min/max/std of image-level anomaly scores per group."""
    normal  = img_scores[img_labels == 0]
    anomaly = img_scores[img_labels == 1]

    def stats(arr):
        return arr.mean(), arr.min(), arr.max(), arr.std()

    n_mean, n_min, n_max, n_std = stats(normal)
    a_mean, a_min, a_max, a_std = stats(anomaly)

    print("\n── Score statistics (image-level) ───────────────────────")
    print(f"  Normal   (n={len(normal):3d})  mean={n_mean:.4f}  min={n_min:.4f}  max={n_max:.4f}  std={n_std:.4f}")
    print(f"  Anomaly  (n={len(anomaly):3d})  mean={a_mean:.4f}  min={a_min:.4f}  max={a_max:.4f}  std={a_std:.4f}")

    if n_mean > a_mean:
        print("\n  [WARNING] Normal mean > Anomaly mean — scoring is INVERTED.")
        print("            Anomalous images are getting lower scores than normal ones.")
    else:
        print(f"\n  [OK] Anomaly mean is {a_mean - n_mean:.4f} higher than normal mean.")


def main(args) -> None:
    device = _device()
    print(f"Device: {device}")

    cfg = Config(
        data_root=args.data_root,
        encoder_name=args.encoder_name,
        num_memory=args.num_memory,
        memory_temperature=args.temperature,
    )

    model = ViTMemoryAutoencoder(
        encoder_name=cfg.encoder_name,
        embed_dim=cfg.embed_dim,
        feat_side=cfg.feat_side,
        num_memory=cfg.num_memory,
        memory_temperature=cfg.memory_temperature,
        ca_reduction=cfg.ca_reduction,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")

    loader = get_test_loader(
        cfg.data_root, cfg.img_size, batch_size=1, num_workers=cfg.num_workers
    )

    img_auroc, pixel_auroc = evaluate(model, loader, device)
    print(f"\nImage-level AUROC : {img_auroc:.4f}")
    print(f"Pixel-level  AUROC: {pixel_auroc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to saved checkpoint (.pth)")
    parser.add_argument("--data_root",    default="./data/mvtec/bottle")
    parser.add_argument("--encoder_name", default="google/vit-base-patch16-384")
    parser.add_argument("--num_memory",   type=int,   default=100)
    parser.add_argument("--temperature",  type=float, default=0.07)
    main(parser.parse_args())
