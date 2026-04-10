from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from src.data.dataset import MVTecDataset
from src.models.autoencoder import ViTMemoryAutoencoder, denorm_image


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


def image_score(anomaly_map: np.ndarray, top_k_ratio: float) -> float:
    flat = anomaly_map.reshape(-1)
    k = max(1, int(len(flat) * top_k_ratio))
    return float(np.sort(flat)[-k:].mean())


def collect_visualization_data(
    model: ViTMemoryAutoencoder,
    loader: DataLoader,
    device: torch.device,
    top_k_ratio: float,
    num_per_class: int,
) -> tuple[list[dict[str, np.ndarray | float | int]], dict[int, list[float]]]:
    selected: dict[int, list[dict[str, np.ndarray | float | int]]] = {0: [], 1: []}
    score_dist: dict[int, list[float]] = {0: [], 1: []}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            label = int(batch["label"].item())
            mask = batch["mask"][0, 0].cpu().numpy()

            diff, recon = model.anomaly_map(x)
            anomaly_map = diff[0, 0].cpu().numpy()
            score = image_score(anomaly_map, top_k_ratio)
            score_dist[label].append(score)

            if len(selected[label]) < num_per_class:
                original = denorm_image(x)[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
                reconstruction = recon[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
                selected[label].append(
                    {
                        "original": original,
                        "reconstruction": reconstruction,
                        "anomaly_map": anomaly_map,
                        "mask": mask,
                        "score": score,
                        "label": label,
                    }
                )

    examples = selected[0] + selected[1]
    return examples, score_dist


def save_example_panel(
    examples: list[dict[str, np.ndarray | float | int]], output_path: Path
) -> None:
    if not examples:
        raise RuntimeError("No visualization examples found in the test split.")

    fig, axes = plt.subplots(len(examples), 4, figsize=(16, 4 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, item in enumerate(examples):
        original = item["original"]
        reconstruction = item["reconstruction"]
        anomaly_map = item["anomaly_map"]
        mask = item["mask"]
        score = float(item["score"])
        label = int(item["label"])

        norm_map = anomaly_map / (float(np.max(anomaly_map)) + 1e-8)
        label_name = "NORMAL" if label == 0 else "DEFECT"

        axes[idx, 0].imshow(original)
        axes[idx, 0].set_title(f"{label_name} | score={score:.4f}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(reconstruction)
        axes[idx, 1].set_title("Reconstruction")
        axes[idx, 1].axis("off")

        im_err = axes[idx, 2].imshow(norm_map, cmap="hot", vmin=0, vmax=1)
        axes[idx, 2].set_title("Anomaly map")
        axes[idx, 2].axis("off")
        plt.colorbar(im_err, ax=axes[idx, 2], fraction=0.046)

        axes[idx, 3].imshow(original)
        axes[idx, 3].imshow(norm_map, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        if float(np.max(mask)) > 0:
            axes[idx, 3].contour(mask, levels=[0.5], colors=["lime"], linewidths=1.5)
        axes[idx, 3].set_title("Overlay + GT contour")
        axes[idx, 3].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution(
    score_dist: dict[int, list[float]], output_path: Path
) -> None:
    fig = plt.figure(figsize=(8, 5))
    if score_dist[0]:
        plt.hist(score_dist[0], bins=20, alpha=0.7, label="Normal", color="green")
    if score_dist[1]:
        plt.hist(score_dist[1], bins=20, alpha=0.7, label="Defect", color="red")
    plt.xlabel("Image-level anomaly score")
    plt.ylabel("Count")
    plt.title("Score distribution (top-k reconstruction error)")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate anomaly visualizations from a trained checkpoint."
    )
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--top-k-ratio", type=float, default=None)
    parser.add_argument("--num-per-class", type=int, default=3)
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
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ViTMemoryAutoencoder(cfg, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    test_loader = build_test_loader(cfg)
    examples, score_dist = collect_visualization_data(
        model=model,
        loader=test_loader,
        device=device,
        top_k_ratio=cfg.TOP_K_RATIO,
        num_per_class=max(1, args.num_per_class),
    )

    panel_path = Path(cfg.OUTPUT_PATH) / "visualization_examples.png"
    hist_path = Path(cfg.OUTPUT_PATH) / "visualization_score_distribution.png"

    save_example_panel(examples, panel_path)
    save_score_distribution(score_dist, hist_path)

    print(f"Saved example panel: {panel_path}")
    print(f"Saved score histogram: {hist_path}")
    print(
        "Scored samples | "
        f"normal={len(score_dist[0])}, defect={len(score_dist[1])}, "
        f"shown_per_class={max(1, args.num_per_class)}"
    )


if __name__ == "__main__":
    main()
