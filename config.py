"""
Global configuration for Explainable Anomaly Detection project.
Hyperparameters taken directly from Table 1 of the paper.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    # ViT encoder settings
    image_size: int = 384
    patch_size: int = 16
    vit_model: str = "google/vit-base-patch16-384"  # HuggingFace pretrained ViT
    embed_dim: int = 768
    freeze_encoder: bool = False

    # Memory module
    memory_size: int = 100  # number of memory slots
    memory_dim: int = 768   # must match embed_dim

    # Decoder
    # 4 stages: 24→48→96→192→384 (24 * 2^4 = 384 ✓)
    decoder_channels: list = field(default_factory=lambda: [512, 256, 128, 64])


@dataclass
class TrainConfig:
    # Table 1 hyperparameters
    learning_rate: float = 0.0002
    batch_size: int = 8
    epochs: int = 200
    optimizer: str = "adam"

    # Loss weights (from paper notation)
    lambda_ssim: float = 1.0       # weight for SSIM term in reconstruction loss
    lambda_entropy: float = 0.001  # weight for entropy loss (memory)

    # Data
    num_workers: int = 2
    pin_memory: bool = True

    # Checkpointing
    save_every: int = 10
    checkpoint_dir: Path = Path("results/checkpoints")

    # Early stopping
    patience: int = 20


@dataclass
class DataConfig:
    data_root: Path = Path("data/mvtec")
    category: str = "bottle"  # one of the 15 MVTec categories
    image_size: int = 384

    # MVTec AD categories
    TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]
    OBJECT_CATEGORIES = [
        "bottle", "cable", "capsule", "hazelnut", "metal_nut",
        "pill", "screw", "toothbrush", "transistor", "zipper"
    ]
    ALL_CATEGORIES = TEXTURE_CATEGORIES + OBJECT_CATEGORIES


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    results_dir: Path = Path("results/heatmaps")


model_cfg = ModelConfig()
train_cfg = TrainConfig()
data_cfg = DataConfig()
app_cfg = AppConfig()
