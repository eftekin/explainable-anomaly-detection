"""
MVTec AD dataset loader.

Directory structure expected:
    data/mvtec/<category>/train/good/*.png
    data/mvtec/<category>/test/good/*.png
    data/mvtec/<category>/test/<defect_type>/*.png
    data/mvtec/<category>/ground_truth/<defect_type>/*.png
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int, split: str = "train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_mask_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.Resampling.NEAREST),
        transforms.ToTensor(),
    ])


class MVTecDataset(Dataset):
    """
    Loads MVTec AD images for a single category.

    Training split: only 'good' images (unsupervised).
    Test split: good + all defect types, with binary masks.
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        image_size: int = 384,
    ):
        assert split in ("train", "test")
        self.root = Path(root) / category
        self.split = split
        self.image_size = image_size
        self.transform = get_transforms(image_size, split)
        self.mask_transform = get_mask_transform(image_size)

        self.image_paths: list[Path] = []
        self.mask_paths: list[Path | None] = []
        self.labels: list[int] = []  # 0=normal, 1=anomaly

        self._load_paths()

    def _load_paths(self):
        split_dir = self.root / self.split

        for defect_dir in sorted(split_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            is_good = defect_dir.name == "good"
            label = 0 if is_good else 1

            for img_path in sorted(defect_dir.glob("*.png")):
                self.image_paths.append(img_path)
                self.labels.append(label)

                if is_good or self.split == "train":
                    self.mask_paths.append(None)
                else:
                    mask_path = (
                        self.root / "ground_truth" / defect_dir.name /
                        (img_path.stem + "_mask.png")
                    )
                    self.mask_paths.append(mask_path if mask_path.exists() else None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = self.transform(img)

        label = self.labels[idx]

        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.mask_transform(mask)
        else:
            mask_tensor = __import__("torch").zeros(1, self.image_size, self.image_size)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "label": label,
            "path": str(self.image_paths[idx]),
        }
