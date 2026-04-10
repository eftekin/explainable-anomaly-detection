# Explainable Anomaly Detection (ViT + Memory + Coordinate Attention)

This repository contains a module-based implementation of unsupervised anomaly detection for MVTec AD.

The pipeline is:

1. ViT encoder (timm vit_base_patch16_384)
2. Memory module with sharp slot addressing
3. Coordinate attention refinement
4. Bilinear upsampling decoder
5. Reconstruction-error based anomaly scoring

Training uses only normal images (train/good). Evaluation is done on mixed normal and defect test images.

## Key Settings

- Encoder: timm vit_base_patch16_384
- Input size: 384 x 384
- Embedding dim: 768
- Memory slots: 100
- Memory temperature: 0.05
- Entropy weight: 0.1
- Freeze epochs: 50 (encoder frozen first, then unfrozen)
- Decoder: bilinear upsample + Conv2d (no ConvTranspose2d)
- Image normalization: ImageNet mean/std
- Image-level score: mean of top 10 percent reconstruction-error pixels

## Repository Layout

```
explainable-anomaly-detection/
├── LICENSE
├── README.md
├── config.py
├── train.py
├── evaluate.py
├── requirements.txt
├── train_colab.ipynb
├── data/
│   └── mvtec/
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── dataset.py
    └── models/
        ├── __init__.py
        ├── autoencoder.py
        ├── memory_module.py
        ├── coordinate_attention.py
        └── decoder.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Place MVTec AD under data/mvtec so the category folders are directly inside data/mvtec.

Expected structure example:

```
data/mvtec/
├── bottle/
│   ├── train/good/
│   ├── test/good/
│   └── ground_truth/
├── cable/
└── ...
```

## Training

Default run:

```bash
python train.py
```

Common options:

```bash
python train.py \
  --category bottle \
  --data-root data/mvtec \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 2 \
  --checkpoint-path checkpoints \
  --output-path outputs
```

Useful flags:

- --seed INT
- --no-pretrained

Outputs:

- checkpoints/best_model.pth
- outputs/training_history.json

## Evaluation

```bash
python evaluate.py \
  --category bottle \
  --data-root data/mvtec \
  --checkpoint checkpoints/best_model.pth \
  --top-k-ratio 0.1 \
  --output-path outputs
```

Outputs:

- Console summary (image-level and pixel-level AUROC)
- outputs/evaluation_metrics.json

## Colab

The notebook train_colab.ipynb provides a clean Colab flow:

1. Install requirements
2. Set Kaggle credentials
3. Download/unpack MVTec AD
4. Run python train.py
5. Run python evaluate.py

## License

MIT License. See LICENSE.
