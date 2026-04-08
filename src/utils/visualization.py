"""
Visualization utilities – generate explainability heatmaps overlaid on images.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert ImageNet-normalized tensor (C,H,W) → uint8 HWC numpy array."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def heatmap_overlay(
    image: torch.Tensor,
    score_map: torch.Tensor,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a colorized anomaly score map on the original image.

    Args:
        image:     (3, H, W) ImageNet-normalized tensor
        score_map: (1, H, W) or (H, W) float tensor, values in [0, 1]
        alpha:     blend factor for the heatmap overlay
        colormap:  OpenCV colormap

    Returns:
        overlay: (H, W, 3) uint8 BGR image
    """
    img_np = denormalize(image)                              # HWC RGB uint8
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    sm = score_map.squeeze().cpu().numpy()                   # (H, W)
    sm = sm - sm.min()
    sm = sm / (sm.max() + 1e-8)
    sm_uint8 = np.uint8(255 * np.clip(sm, 0, 1))
    heatmap = cv2.applyColorMap(sm_uint8, colormap)          # HWC BGR

    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay


def save_result(
    image: torch.Tensor,
    score_map: torch.Tensor,
    save_path: str | Path,
    threshold: float | None = None,
):
    """
    Save a side-by-side panel: original | heatmap | (optional) binary mask.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    orig = cv2.cvtColor(denormalize(image), cv2.COLOR_RGB2BGR)
    overlay = heatmap_overlay(image, score_map)

    panels = [orig, overlay]

    if threshold is not None:
        sm = score_map.squeeze().cpu().numpy()
        binary = (sm >= threshold).astype(np.uint8) * 255
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        panels.append(binary_bgr)

    result = np.concatenate(panels, axis=1)
    cv2.imwrite(str(save_path), result)
    return result
