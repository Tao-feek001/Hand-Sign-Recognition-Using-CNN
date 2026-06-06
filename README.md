# Hand Sign Recognition using CNN — CSY3081 PJ1

A reproducible PyTorch implementation of CNN-based American Sign Language (ASL) hand sign recognition on a 24-class static-sign dataset (A–Y, excluding J and Z which require motion).

## Project Structure

```
FINAL_AS2/
├── notebook.ipynb              # Main deliverable (run end-to-end)
├── requirements.txt            # Pinned dependency versions
├── README.md                   # This file
├── data/
│   ├── Train/                  # 26,755 images across 24 class folders
│   └── Test/                   # 7,272 images across 24 class folders
└── outputs/
    ├── models/                 # Saved model weights (.pth)
    └── plots/                  # Saved figures (.png)
```

## Running the notebook

### Option A — Google Colab (recommended; uses free GPU)

1. Upload the entire `FINAL_AS2/` folder to your Google Drive (e.g., to `MyDrive/FINAL_AS2/`).
2. Open `notebook.ipynb` in Colab (right-click in Drive → Open with → Google Colaboratory).
3. Set runtime: `Runtime → Change runtime type → T4 GPU`.
4. Run all cells (`Runtime → Run all`).

The first cell will detect the Colab environment and mount Google Drive automatically.

### Option B — Local

1. Create a fresh Python 3.10+ virtual environment.
2. `pip install -r requirements.txt`.
3. Launch Jupyter: `jupyter notebook notebook.ipynb`.
4. Run all cells.

Local CPU training is feasible (images are 28×28 grayscale) but slower than Colab GPU. Expect ~3–5× slower per epoch on CPU.

## Reproducibility

- All random seeds are fixed (see `set_seed()` function).
- `torch.backends.cudnn.deterministic = True`, `cudnn.benchmark = False`.
- Dependency versions pinned in `requirements.txt`.
- Multi-seed final evaluation (3 seeds) reports mean ± std.
- All intermediate artifacts saved to `outputs/`.

## What the notebook does

1. **Setup & reproducibility** — environment detection, seeding, config.
2. **Exploratory Data Analysis** — class distribution, pixel statistics, mean images per class, visually-confusable pair inspection, entropy analysis, train/test distribution comparison.
3. **Preprocessing** — dataset-derived normalisation statistics, stratified train/val split, domain-aware augmentation (no horizontal flip).
4. **Baseline CNN** — minimal 2-conv-layer model for performance floor.
5. **Custom CNN** — 4 conv blocks with BatchNorm and Dropout.
6. **Hyperparameter sweep** — 3 optimisers × 3 learning rates on Custom CNN.
7. **Augmentation ablation** — empirical test of whether augmentation helps on this dataset.
8. **MobileNetV2 transfer learning** — adapted for 1-channel input + 24-class head.
9. **Multi-seed final evaluation** — mean ± std across 3 seeds.
10. **Grad-CAM interpretability** — visualisations of what the network attends to.
11. **Error analysis** — top-5 confused class pairs and misclassification examples.
12. **Inference benchmark** — latency per sample on CPU and GPU.

## Author

CSY3081 — AI Concepts and Applications (PJ1 Individual Course Project)
