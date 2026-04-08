"""
FastAPI backend for the Explainable Anomaly Detection web app.

Endpoints:
    POST /predict  – upload an image, returns anomaly score + heatmap (base64 PNG)
    GET  /health   – liveness check
"""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import model_cfg, app_cfg
from src.models import AnomalyAutoencoder
from src.utils.visualization import heatmap_overlay

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Explainable Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model (loaded once at startup)
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: AnomalyAutoencoder | None = None
CHECKPOINT_PATH = Path("results/checkpoints/best.pth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREPROCESS = transforms.Compose([
    transforms.Resize((model_cfg.image_size, model_cfg.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@app.on_event("startup")
def load_model():
    global MODEL
    MODEL = AnomalyAutoencoder(
        vit_model=model_cfg.vit_model,
        memory_size=model_cfg.memory_size,
        embed_dim=model_cfg.embed_dim,
        decoder_channels=model_cfg.decoder_channels,
    ).to(DEVICE)

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        MODEL.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"Warning: no checkpoint at {CHECKPOINT_PATH}, using random weights.")

    MODEL.eval()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    heatmap_b64: str  # base64-encoded PNG of the overlay


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read and preprocess image
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    # Inference
    with torch.no_grad():
        heatmap = MODEL.anomaly_map(tensor)  # (1, 1, H, W)

    score = float(heatmap.squeeze().max().item())
    is_anomaly = score >= threshold

    # Build overlay PNG
    overlay_bgr = heatmap_overlay(tensor[0].cpu(), heatmap[0].cpu())
    _, buffer = cv2.imencode(".png", overlay_bgr)
    heatmap_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return PredictResponse(
        anomaly_score=round(score, 4),
        is_anomaly=is_anomaly,
        heatmap_b64=heatmap_b64,
    )
