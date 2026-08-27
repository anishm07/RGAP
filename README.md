# RGAP: Redundancy-Guided Adaptive Pruning for Vision Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## Description

RGAP is a structured pruning framework for Vision Transformers (ViTs) that replaces the conventional fixed global pruning ratio with a **per-layer adaptive pruning budget** derived from the model's own learned importance distribution.

Existing search-based pruning methods learn per-component importance scores but apply a single fixed ratio uniformly across all transformer layers — ignoring substantial variation in layer-wise redundancy and causing unstable pruning outcomes across different search initialisations. RGAP addresses this by:

1. **Learning** per-component importance scores (zeta values) for attention heads and MLP neurons during a short search phase
2. **Extracting** three statistical redundancy descriptors per transformer layer — mean (μ), standard deviation (σ), and entropy (H) — from the learned importance distribution
3. **Combining** these descriptors through jointly learned weights (α, β, γ) into a composite redundancy score Sℓ per layer
4. **Mapping** each layer's redundancy score to a bounded per-layer pruning ratio ρℓ via a closed-form min-max rescaling

**Key results on ImageNet-100 / DeiT-Base:**
- RGAP achieves **92.90% Top-1 accuracy** at **56.18% FLOPs retained** (seed 0)
- Cross-seed accuracy range: **0.24 pp** across 3 independent seeds vs. **7.20 pp** for the fixed-ratio baseline
- Every RGAP seed exceeds the unpruned baseline (91.90%)

---

## Dataset Information

### Primary Dataset — ImageNet-100

ImageNet-100 is a 100-class subset of the full ImageNet-1K dataset (ILSVRC 2012).

- **Source:** ImageNet Large Scale Visual Recognition Challenge (ILSVRC 2012)
- **URL:** https://www.image-net.org/
- **Classes:** 100 classes selected from the original 1,000
- **Training images:** 126,697 training images (~1,300 images per class)
- **Validation images:** 5,000 (50 images per class)
- **Image format:** JPEG, resized to 224×224 for DeiT-Base input

The 100-class subset is constructed by selecting a fixed list of class IDs from the ImageNet-1K training and validation splits. The class list used in this work is defined in `datasets.py`.

To obtain ImageNet-1K, register and download from:
> https://www.image-net.org/download.php

Then extract the 100 classes specified in `datasets.py` to form ImageNet-100.

---

## Code Information

### Core pipeline files

| File | Description |
|------|-------------|
| `adaptive_search.py` | Phase 1 — Adaptive search: jointly learns zeta importance scores and redundancy weights α, β, γ |
| `adaptive_pruning.py` | Core RGAP module: implements RedundancyWeights class and get_adaptive_thresholds() for per-layer redundancy scoring and ratio mapping |
| `adaptive_prune.py` | Phase 2 — Loads search checkpoint, computes per-layer thresholds, generates pruning masks, writes the structurally pruned model |
| `finetune.py` | Phase 3 — Fine-tunes the structurally pruned model on ImageNet-100 |
| `search.py` | Standard (non-adaptive) search script — used for fixed-ratio baseline experiments |

### Model and Architecture files

| File | Description |
|------|-------------|
| `vision_transformer.py` | Modified DeiT/ViT architecture with zeta-based learnable importance masks on attention heads and MLP neurons |
| `models.py` | Model registration and creation utilities |

### Evaluation and Analysis files

| File | Description |
|------|-------------|
| `compute_flops.py` | Computes GFLOPs for the pruned model |
| `compute_flops_matched.py` | FLOPs computation for matched-budget comparison |
| `compute_flops_baseline.py` | FLOPs computation for fixed-ratio baseline |
| `compute_flops_ablation.py` | FLOPs for ablation variants (μ-only, σ-only, H-only) |
| `compute_flops_ablation2.py` | Extended ablation FLOPs computation |
| `head_deactivation_study.py` | Empirically deactivates each of 144 attention heads and measures accuracy impact |
| `plot_head_deactivation.py` | Plots head-deactivation results |

### Supporting files

| File | Description |
|------|-------------|
| `datasets.py` | Dataset loading for ImageNet-100; defines the 100-class ImageNet subset |
| `engine.py` | Training and evaluation engine (train_one_epoch, evaluate) |
| `losses.py` | Loss functions including knowledge distillation |
| `utils.py` | Utility functions: threshold computation, checkpoint saving, redundancy statistics logging |
| `samplers.py` | Distributed training samplers |
| `test_imports.py` | Import verification script |


---

## Requirements

### Hardware
- GPU: NVIDIA RTX 3090 (24 GB VRAM) or equivalent
- RAM: 32 GB minimum recommended
- Storage: ~150 GB for ImageNet-100 data and checkpoints

### Software Dependencies

```
Python >= 3.8
PyTorch >= 2.0
torchvision >= 0.15
timm >= 0.6.12
numpy >= 1.21
scipy >= 1.7
Pillow >= 9.0
einops >= 0.6.0
tensorboard >= 2.11
tqdm >= 4.64
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### Step 0 — Prepare the Dataset

The ImageNet-100 dataset consists of 100 classes selected from ImageNet-1K, comprising 126,697 training images and 5,000 validation images (50 per class).
The dataset should be organised as a standard image folder structure with integer class indices (0–99):

```
data/
└── imagenet100_images/
    ├── train/
    │   ├── 0/        (~1300 images per class)
    │   ├── 1/
    │   └── ...       (100 classes total)
    └── val/
        ├── 0/        (50 images per class)
        ├── 1/
        └── ...       (100 classes total)
```

### Step 1 — Phase 1: Adaptive Search

Jointly learn zeta importance scores and redundancy weights α, β, γ:

```bash
python adaptive_search.py \
    --model deit_base_patch16_224 \
    --data-path /path/to/imagenet100 \
    --data-set ImageNet100 \
    --pretrained-path ./deit_base_patch16_224-b5f2ef4d.pth \
    --output-dir ./output/search \
    --epochs 10 \
    --batch-size 128 \
    --lr 5e-4 \
    --lambda-head 1e-2 \
    --lambda-mlp 1e-4 \
    --lambda-reg 1e-3
```

### Step 2 — Phase 2: Adaptive Pruning

Compute per-layer redundancy scores and generate the structurally pruned model:

```bash
python adaptive_prune.py \
    --model deit_base_patch16_224 \
    --checkpoint ./output/search/checkpoint.pth \
    --output-dir ./output/pruned \
    --head-rho-min 0.10 \
    --head-rho-max 0.85 \
    --mlp-rho-min 0.10 \
    --mlp-rho-max 0.95 \
    --device cuda
```

### Step 3 — Phase 3: Fine-Tuning

Fine-tune the structurally pruned model:

```bash
python finetune.py \
    --model deit_base_patch16_224 \
    --data-path /path/to/imagenet100 \
    --data-set ImageNet100 \
    --retrain \
    --prune_head \
    --prune_mlp \
    --checkpoint_path ./output/pruned/pruned_checkpoint.pth \
    --search_checkpoint ./output/search/checkpoint.pth \
    --output_dir ./output/finetuned \
    --epochs 30 \
    --batch-size 128 \
    --lr 1e-4 \
    --warmup-epochs 5 \
    --sched cosine \
    --weight-decay 0.05 \
    --smoothing 0.1 \
    --cutmix 1.0 \
    --mixup 0.8
```

### Step 4 — Compute FLOPs

```bash
python compute_flops_matched.py \
    --checkpoint ./output/finetuned/best_checkpoint.pth \
    --search_checkpoint ./output/search/checkpoint.pth
```

### Step 5 — Head Deactivation Study (Optional)

```bash
python head_deactivation_study.py \
    --checkpoint ./output/search/checkpoint.pth \
    --data-path /path/to/imagenet100 \
    --output-dir ./head_deactivation_results_versionA
```

---

## Methodology

RGAP operates in three sequential phases:

### Phase 1 — Adaptive Search (10 epochs)

Per-component zeta importance scores are attached to every attention head and MLP neuron. These are jointly optimised with three redundancy weighting scalars (α, β, γ) under the adaptive search loss:

```
L_adaptive = L_CE + λ_head·Σ|z^H| + λ_mlp·Σ|z^M| + λ_reg·(α+β+γ−1)²
```

Hyperparameters: λ_head = 1×10⁻², λ_mlp = 1×10⁻⁴, λ_reg = 1×10⁻³

The redundancy weights α, β, γ are optimised with a separate learning rate of 1×10⁻³ and zero weight decay, decoupled from the backbone and zeta parameter group (learning rate 5×10⁻⁴).

### Phase 2 — Adaptive Pruning (one-shot, no additional training)

Three redundancy descriptors are computed per layer from the converged zeta distribution:
- μℓ = mean(zℓ) — average importance level of the layer
- σℓ = std(zℓ) — within-layer diversity of importance
- Hℓ = −Σ pᵢ log pᵢ, where p = softmax(zℓ) — distributional entropy

These are combined as:

```
Sℓ = α(1−μℓ) + βσℓ + γHℓ
```

And mapped to bounded per-layer pruning ratios via min-max rescaling:

```
ρℓ = ρ_min + (ρ_max − ρ_min) · (Sℓ − S_min) / (S_max − S_min + ε)
```

Parameters: ρ_min = 0.10 (safety floor), ρ_max tuned to target FLOPs budget.
The mapping is applied independently to the attention-head and MLP-neuron branches.

### Phase 3 — Fine-Tuning (30 epochs)

Standard AdamW fine-tuning with cosine LR schedule (LR = 1×10⁻⁴, warmup 5 epochs),
CutMix, MixUp, and label smoothing on the structurally pruned model.

---

## Data Preprocessing

All images are resized to 224×224 pixels to match the DeiT-Base patch size (16×16).
Standard ImageNet normalisation is applied:
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

Training augmentation includes:
- Random resized crop (scale 0.08–1.0)
- Random horizontal flip
- AutoAugment (rand-m9-mstd0.5-inc1)
- Random erasing (probability 0.25)
- CutMix (alpha 1.0) and MixUp (alpha 0.8)

Validation uses centre crop with ratio 0.875.

---

## Citations

If you use this code in your research, please cite the accompanying paper:

Anish M. George and Shajimon K. John. "RGAP: Redundancy-Guided Adaptive Pruning for Vision Transformers." Submitted to PeerJ Computer Science, 
2026.

The code is archived on Zenodo:
DOI: 10.5281/zenodo.22124637
URL: https://doi.org/10.5281/zenodo.22124637

The ImageNet-100 dataset is derived from ImageNet-1K:Russakovsky et al. (2015). ImageNet Large Scale Visual Recognition Challenge. International Journal 
of Computer Vision, 115(3), 211-252. DOI: 10.1007/s11263-015-0816-y

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Anish M. George

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contribution Guidelines

1. Open an issue describing the problem or suggestion
2. Fork the repository
3. Create a feature branch: `git checkout -b fix/your-fix`
4. Commit your changes: `git commit -m 'Fix: description'`
5. Push to the branch: `git push origin fix/your-fix`
6. Open a Pull Request

---

*For questions about the paper or code, contact: anish.mg@saintgits.org*
