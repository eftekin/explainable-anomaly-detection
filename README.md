# Explainable Anomaly Detection in Industrial Images using Vision Transformers

Unsupervised anomaly detection on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset using a ViT-based autoencoder with a memory module and coordinate attention. The model is trained **only on normal images** and produces pixel-level anomaly heatmaps at inference time.

---

## Architecture

```
Input Image (384×384)
        │
        ▼
┌───────────────┐
│  ViT Encoder  │  pretrained google/vit-base-patch16-384
│  patch 16×16  │  → (B, 768, 24, 24) feature map
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Memory Module │  100 learnable normal prototypes
│ (softmax read)│  anomaly features → replaced with nearest normal
└───────┬───────┘
        │
        ▼
┌──────────────────────┐
│ Coordinate Attention │  H+W directional pooling for defect localization
└───────┬──────────────┘
        │
        ▼
┌─────────────┐
│   Decoder   │  4× TransposedConv: 24→48→96→192→384
└───────┬─────┘
        │
        ▼
Reconstructed Image  ──diff──►  Anomaly Heatmap
```

**Loss:** `L = MSE(A, A') + λ_ssim·(1 − SSIM(A, A')) + λ_entropy·H(attn)`

| Hyperparameter | Value |
|---|---|
| Input resolution | 384 × 384 |
| Patch size | 16 × 16 |
| Embedding dim | 768 |
| Memory slots | 100 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Batch size | 8 |
| Epochs | 200 |
| λ_ssim | 1.0 |
| λ_entropy | 0.001 |

---

## Project Structure

```
explainable-anomaly-detection/
├── config.py                   # All hyperparameters
├── train.py                    # Training script
├── evaluate.py                 # AUROC evaluation + heatmap export
├── src/
│   ├── data/
│   │   └── dataset.py          # MVTec AD dataset loader
│   ├── models/
│   │   ├── encoder.py          # ViT encoder
│   │   ├── memory_module.py    # Memory module
│   │   ├── coordinate_attention.py
│   │   ├── decoder.py          # Transposed-conv decoder
│   │   └── autoencoder.py      # Full pipeline + anomaly_map()
│   └── utils/
│       ├── losses.py           # MSE + SSIM + Entropy loss
│       ├── metrics.py          # Image/pixel AUROC
│       └── visualization.py    # Heatmap overlay
├── app/
│   └── backend/
│       └── main.py             # FastAPI inference API
├── notebooks/                  # Experiment notebooks
├── data/
│   └── mvtec/                  # Dataset (not tracked by git)
└── results/
    ├── checkpoints/            # Saved model weights (not tracked)
    └── heatmaps/               # Output visualizations (not tracked)
```

---

## Installation

```bash
git clone https://github.com/eftekin/explainable-anomaly-detection.git
cd explainable-anomaly-detection
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.0+ are required. GPU training is strongly recommended.

---

## Dataset

Download MVTec AD from the [official page](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it:

```bash
mkdir -p data/mvtec
tar -xf mvtec_anomaly_detection.tar.xz -C data/mvtec/
```

Expected structure:
```
data/mvtec/
├── bottle/
│   ├── train/good/
│   ├── test/good/
│   ├── test/broken_large/
│   └── ground_truth/broken_large/
├── cable/
└── ...  (15 categories total)
```

---

## Usage

### Train

```bash
# Train on a single category (default: bottle)
python train.py --category bottle

# Common options
python train.py --category carpet --epochs 200 --batch_size 8 --lr 0.0002
```

Checkpoints are saved to `results/checkpoints/<category>/`.

### Evaluate

```bash
python evaluate.py \
    --category bottle \
    --checkpoint results/checkpoints/bottle/best.pth \
    --save_heatmaps
```

Outputs image-level and pixel-level AUROC to the console.  
Heatmap panels (original | overlay | binary mask) are saved to `results/heatmaps/`.

### Web API

```bash
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Upload image → anomaly score + base64 heatmap PNG |

Example:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.png" \
  -F "threshold=0.5"
```

Response:
```json
{
  "anomaly_score": 0.843,
  "is_anomaly": true,
  "heatmap_b64": "iVBORw0KGgo..."
}
```

---

## Results

| Category | Image AUROC | Pixel AUROC |
|---|---|---|
| bottle | — | — |
| carpet | — | — |
| *...* | — | — |

> Results will be populated after full training runs.

---

## References

1. Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021
2. Yang & Guo, *Unsupervised Industrial Anomaly Detection with ViT-Based Autoencoder*, Sensors 2024
3. Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset*, CVPR 2019
4. Defard et al., *PaDiM: A Patch Distribution Modeling Framework*, ICPR 2021
5. Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022
