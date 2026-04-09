from dataclasses import dataclass


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_root: str = "./data/mvtec/bottle"
    img_size: int = 384
    batch_size: int = 8
    num_workers: int = 4

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder_name: str = "google/vit-base-patch16-384"
    embed_dim: int = 768          # ViT-Base hidden size
    feat_side: int = 24           # 384 / 16 = 24 patches per side
    num_memory: int = 100
    memory_temperature: float = 0.07
    ca_reduction: int = 32

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 200
    lr: float = 0.0002
    freeze_epochs: int = 30       # encoder frozen for epochs 0 … 29
    patience: int = 50            # early stopping

    # ── Loss weights (paper Table 1) ──────────────────────────────────────────
    lambda_recon: float = 1.0
    lambda_ssim: float = 1.0
    lambda_ent: float = 0.001

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    best_model_path: str = "./checkpoints/best_model.pth"
    log_dir: str = "./logs"
