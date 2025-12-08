# ISL for Implicit Generative Models

Code for training **implicit generative models** using the  
**Invariant Statistical Loss (ISL)** and its sliced / heavy–tailed variants.

This repository accompanies the paper

> **Robust training of implicit generative models for multivariate and heavy-tailed distributions with an invariant statistical loss**  
> J. M. de Frutos et al., JMLR (under review).

It also builds on the original AISTATS work where ISL was first introduced for 1D implicit models.

---

## Overview

This repo provides:

- A **PyTorch implementation of ISL** in 1D (`isl.loss_1d`) and its **sliced extension** to \(\mathbb{R}^d\) (`isl.sliced`).
- **Rank-based utilities** (`isl.ranks`) and metrics/helpers (`isl.metrics`, `isl.utils`).
- Simple **MLP generators** and other models (`isl.models`).
- Reproducible **experiments** for:
  - 1D toy targets (Gaussian, mixtures, heavy-tailed, etc.).
  - 2D toy targets with **sliced ISL** (random vs “smart” projections).
  - 1D heavy-tailed real data (**keystroke intervals**) with **Pareto-ISL**.
  - Time series experiments on the **ETT** datasets:
    - Univariate RNN + ISL (Gaussian latent).
    - Multivariate RNN + **sliced ISL** (random directions) with 1-step forecasts.

The code is intentionally simple and modular: you can reuse the core ISL pieces in your own generative setups.

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-user>/isl-implicit-generative-models.git
cd isl-implicit-generative-models
```

## Repository structure

Rough layout (main bits):

```text
isl-implicit-generative-models/
├── README.md
├── CITATION.cff
├── pyproject.toml / setup.py          # (if present)
├── src/
│   └── isl/
│       ├── __init__.py
│       ├── loss_1d.py                 # 1D ISL (hard & soft surrogate)
│       ├── ranks.py                   # rank utilities
│       ├── sliced.py                  # sliced ISL in R^d
│       ├── metrics.py                 # simple metrics (KS/KSD etc.) for experiments
│       ├── models.py                  # small MLP generators, etc.
│       └── utils.py                   # seeding, device helpers, filesystem
└── experiments/
    ├── 1d_univariate/
    │   ├── run_1d_isl_targets.py      # 1D toy targets + K ablations
    │   └── ...
    ├── 2d_toy/
    │   ├── train_2d_sliced_isl_random.py
    │   ├── train_2d_sliced_isl_smart.py
    │   └── ...
    ├── heavy_tails/
    │   └── keystrokes_pareto_isl.py   # keystroke intervals, Pareto-ISL
    └── time_series/
        ├── train_rnn_isl_ett_gaussian.py      # univariate ETT + 1D ISL
        └── train_rnn_sliced_isl_ett.py        # multivariate ETT + sliced ISL
```

