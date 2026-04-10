from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecDataset(Dataset):
    """MVTec AD loader with ImageNet normalization."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self, root: str, category: str, split: str = "train", image_size: int = 384
    ) -> None:
        self.image_size = image_size
        self.samples: List[Tuple[Path, Optional[Path], int]] = []

        data_dir = Path(root) / category / split
        gt_dir = Path(root) / category / "ground_truth"

        if not data_dir.exists():
            raise FileNotFoundError(f"MVTec split path not found: {data_dir}")

        for cls_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
            label = 0 if cls_dir.name == "good" else 1

            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in {".png", ".jpg", ".bmp", ".jpeg"}:
                    continue

                mask_path: Optional[Path] = None
                if label == 1:
                    candidate = gt_dir / cls_dir.name / f"{img_path.stem}_mask.png"
                    if candidate.exists():
                        mask_path = candidate
                self.samples.append((img_path, mask_path, label))

        self.img_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )
        self.mask_tf = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | str]:
        img_path, mask_path, label = self.samples[idx]
        image = self.img_tf(Image.open(img_path).convert("RGB"))

        if mask_path is not None:
            mask = self.mask_tf(Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "path": str(img_path),
        }
