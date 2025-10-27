# NTMP / CSMPU – Reference Implementation

Reproducible code for our paper on **N‑Tuple with M Positives (NTMP)** / **Cost‑Sensitive Multi‑Positive‑Unlabeled (CSMPU)** learning.

> **Environment summary.** Python 3.12.9; PyTorch 2.8.0.dev20250323+cu128 / TorchVision 0.22.0.dev20250324+cu128 / Torchaudio 2.6.0.dev20250324+cu128; CUDA: CPU-only (or MPS on macOS).

---

## 1) Environment

This repo ships a conda environment file: `torch_backup.yaml`.

### Create & activate
```bash
conda env create -f torch_backup.yaml
conda activate torch
```

### Notes
- Channels:
```
  - defaults
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
  - https://repo.anaconda.com/pkgs/msys2
```
- If you are on **NVIDIA CUDA**, the environment declares: **CPU-only (or MPS on macOS)**.
- On **macOS** (Apple Silicon) or **CPU‑only**, ignore CUDA; PyTorch will fall back to MPS/CPU.
- To export the exact spec (for archival):
```bash
conda env export --no-builds > environment_exact.yml
```

---

## 2) Datasets

Only **TorchVision** datasets:
`MNIST`, `FashionMNIST`, `USPS`, `KMNIST`, `SVHN`.
They download automatically on first use.

---

## 3) Quick Start (single run)

The entry point is a CLI:

```bash
python experiment_.py   --dataset MNIST   --classes 4   --mode ABS   --kprior 0.5   --noise 1.0   --epochs 50   --batchsize 512   --seed 42   --save-dir runs   --save-margins
```

Outputs (per run):
```
runs/<dataset>_<mode>_<classes>_<kprior>_<seed>/
  ├─ metrics.csv
  ├─ pred.npy        # if --save-margins
  └─ y.npy           # if --save-margins
```

---

## 4) Reproducibility

- Set `--seed` for deterministic runs (Python/NumPy/Torch).
- Use `--save-margins` to export logits/labels for figure/table scripts.
- We recommend logging hardware, PyTorch/CUDA versions when reporting numbers.

---

## 5) Project Layout

```
repo/
├─ experiment_.py          # CLI entry point
├─ custommodel.py             # MLP / ResNet backbones
├─ dataset.py                 # TorchVision datasets only
├─ loss.py                    # mpan_loss and related
├─ torch_backup.yaml          # Conda environment (this file)
├─ runs/                      # outputs (git-ignored)
└─ results/                   # tables/figures (git-ignored)
```

---

## 6) Troubleshooting

- **CUDA mismatch**: ensure driver/toolkit matches `CPU-only (or MPS on macOS)`; otherwise consider CPU or MPS.
- **Torch import errors**: verify compatible triplet of `pytorch/torchvision/torchaudio` — current spec: PyTorch 2.8.0.dev20250323+cu128 / TorchVision 0.22.0.dev20250324+cu128 / Torchaudio 2.6.0.dev20250324+cu128.
- **Large test batch**: `experiment_.py` evaluates with full test set in one batch; lower the batch size if OOM.

