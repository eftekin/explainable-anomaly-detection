"""
MVTec bottle dataset loader.

Directory layout expected under `root`:
  root/
    train/good/*.png          ← normal training images
    test/good/*.png           ← normal test images
    test/<defect_class>/*.png ← anomalous test images
    ground_truth/<defect_class>/*.png  ← pixel-level GT masks
                                          (absent for "good")

get_train_loader → DataLoader of normal images (label=0)
get_test_loader  → DataLoader of (image, mask, label) triples
                   label: 0=normal, 1=anomaly
                   mask : (1, H, W) float in [0,1]; all-zeros for normal samples
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _train_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _test_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _mask_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])


class MVTecBottleDataset(Dataset):
    """MVTec bottle dataset — train or test split."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 384,
    ):
        assert split in ("train", "test"), f"Invalid split: {split}"
        self.root     = Path(root)
        self.split    = split
        self.img_size = img_size

        if split == "train":
            self.img_tf  = _train_transforms(img_size)
            self.samples = self._collect_train()
        else:
            self.img_tf   = _test_transforms(img_size)
            self.mask_tf  = _mask_transforms(img_size)
            self.samples  = self._collect_test()

    # ── collection helpers ────────────────────────────────────────────────────

    def _collect_train(self) -> List[Path]:
        good_dir = self.root / "train" / "good"
        if not good_dir.exists():
            raise FileNotFoundError(f"Training images not found at: {good_dir}")
        return sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg"))

    def _collect_test(self) -> List[Tuple[Path, Optional[Path], int]]:
        """Returns list of (image_path, mask_path_or_None, label)."""
        test_dir = self.root / "test"
        gt_dir   = self.root / "ground_truth"
        samples  = []

        for class_dir in sorted(test_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            is_normal = class_dir.name == "good"
            label = 0 if is_normal else 1

            for img_path in sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg")):
                if is_normal:
                    mask_path = None
                else:
                    # GT mask: ground_truth/<class>/<stem>_mask.png
                    mask_path = gt_dir / class_dir.name / (img_path.stem + "_mask.png")
                    if not mask_path.exists():
                        # fallback: same name
                        mask_path = gt_dir / class_dir.name / img_path.name
                samples.append((img_path, mask_path, label))

        if not samples:
            raise FileNotFoundError(f"No test images found under: {test_dir}")
        return samples

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.split == "train":
            img_path = self.samples[idx]
            img = Image.open(img_path).convert("RGB")
            return self.img_tf(img)

        img_path, mask_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.img_tf(img)

        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_tf(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.img_size, self.img_size)

        return img, mask, torch.tensor(label, dtype=torch.long)


def get_train_loader(root: str, img_size: int = 384,
                     batch_size: int = 8, num_workers: int = 4) -> DataLoader:
    dataset = MVTecBottleDataset(root, split="train", img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_test_loader(root: str, img_size: int = 384,
                    batch_size: int = 1, num_workers: int = 4) -> DataLoader:
    dataset = MVTecBottleDataset(root, split="test", img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
