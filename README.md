# Hand Sign Recognition using CNN (CSY3081)

PyTorch project for recognising 24 static ASL hand signs (A–Y, no J/Z) from 28×28 grayscale images using convolutional neural networks.

## What’s here

- `notebook.ipynb` – end‑to‑end pipeline (EDA → training → evaluation).
- `requirements.txt` – Python dependencies.
- `README.md` – project overview.
- `data/` – local Train/Test folders (ignored in git).
- `outputs/` – saved models and plots (ignored in git).

## How to run

**Colab (recommended)**  
Upload the `FINAL_AS2/` folder to Drive, open `notebook.ipynb` with Colab, select a GPU runtime, and “Run all”. The notebook sets paths and seeds automatically.

**Local**  
Create a Python 3.10+ env, then:

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Run the cells in order (CPU is slower but works).

## What the notebook does

- Exploratory data analysis and dataset integrity checks.  
- Baseline CNN vs deeper CustomCNN vs MobileNetV2 (grayscale‑adapted).  
- Hyperparameter and augmentation experiments.  
- Grad‑CAM visualisations and error analysis.  
- Simple latency benchmark (CPU vs GPU).

## Results (summary)

The CustomCNN achieves ~99.5% test accuracy and macro‑F1 with low variance across seeds, while remaining small and fast enough for real‑time use on modest hardware.

## Author

Taofeek Abimbolu — CSY3081 AI Concepts and Applications (PJ1)
