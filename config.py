from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass
class Config:
    # Data
    CATEGORY: str = "bottle"
    IMAGE_SIZE: int = 384
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2

    # Encoder
    ENCODER_MODEL: str = "vit_base_patch16_384"
    EMBED_DIM: int = 768
    PATCH_SIZE: int = 16

    # Memory
    NUM_SLOTS: int = 100
    SHRINK_THRESHOLD: float = 0.0025
    SHRINK_GAMMA: float = 2.0
    MEM_TEMPERATURE: float = 0.05

    # Training
    LEARNING_RATE: float = 2e-4
    ENCODER_LR_SCALE: float = 0.1
    NUM_EPOCHS: int = 100
    LAMBDA_ENT: float = 0.1
    GRAD_CLIP: float = 1.0
    FREEZE_UNTIL_EPOCH: int = 50
    VARIANCE_THRESHOLD: float = 1e-4

    # Scoring
    TOP_K_RATIO: float = 0.1

    # Paths
    DATA_ROOT: str = "data/mvtec"
    CHECKPOINT_PATH: str = "checkpoints"
    OUTPUT_PATH: str = "outputs"

    # Runtime
    RANDOM_SEED: int = 42

    def ensure_dirs(self) -> None:
        Path(self.CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    def get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
