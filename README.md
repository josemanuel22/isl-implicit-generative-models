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
## Running the experiments

All commands below assume you are in the **repo root**:

```bash
cd isl-implicit-generative-models
```

---

### 1. 1D toy implicit models

Script: `experiments/1d_univariate/run_1d_isl_targets.py`

This script trains a 1D MLP generator \(G : \mathbb{R}^{d_z} \to \mathbb{R}\) on various 1D targets, such as:

- Gaussian / Laplace,
- mixtures of Gaussians,
- heavy-tailed targets (e.g. Cauchy mixtures, Pareto, etc.).

It uses `isl.loss_1d.isl_1d_soft` and typically produces:

- real vs generated **histograms** (with sensible clipping for heavy tails),
- (optionally) true vs learned densities when the analytic density is known,
- diagnostics for different values of **K** (number of bins).

Example:

```bash
python experiments/1d_univariate/run_1d_isl_targets.py   --target cauchy_mix   --K_list 8 16 32 64 128
```

---

### 2. 2D toy models with sliced ISL

Scripts under `experiments/2d_toy/` illustrate 2D use-cases:

- 2D targets (e.g. Gaussians, rings, moons, mixtures),
- 2D generator \(G : \mathbb{R}^{d_z} \to \mathbb{R}^2\),
- sliced ISL via:
  - random projections: `sliced_isl_random`,
  - “smart” / max-sliced projections: `sliced_isl_smart`.

Example:

```bash
python experiments/2d_toy/train_2d_sliced_isl_random.py   --target moons   --n_slices 32   --steps 10000
```

The scripts usually produce scatter plots of **real vs generated samples**, and optionally compare random vs smart slicing.

---

### 3. Heavy-tailed 1D real data (keystrokes + Pareto-ISL)

Script: `experiments/heavy_tails/keystrokes_pareto_isl.py`

This experiment:

- loads a 1D array of inter-keystroke intervals from
  `data/keystrokes_intervals.npy` or `.npz` (key: `intervals`),
- optionally applies a log-transform `log(x + eps)`,
- trains a 1D MLP generator with ISL using:
  - Gaussian latent, or
  - **GPD (Pareto-like) latent**,
- prints **tail diagnostics** (quantiles, Hill estimator),
- saves linear- and log-y **histograms** for real vs generated samples.

Example:

```bash
python experiments/heavy_tails/keystrokes_pareto_isl.py   --data_path data/keystrokes_intervals.npy   --latent_type gpd   --log_transform   --steps 5000
```

You must provide the keystroke intervals file under `data/`.

---

### 4. Time series: univariate ETT + ISL

Script: `experiments/time_series/train_rnn_isl_ett_gaussian.py`

This script:

- loads an ETT CSV (e.g. `data/ETTm1.csv`),
- selects a **single** column (default: `OT`),
- optionally applies `log(x + eps)`,
- normalises the series to zero mean and unit variance,
- builds a sliding-window dataset of length `seq_len`,
- trains a GRU-based RNN + MLP decoder with **Gaussian latent** and **1D ISL** on the one-step-ahead conditional distribution,
- produces:
  - an ISL training curve,
  - real vs generated histograms,
  - a **1-step-ahead forecast plot** using **teacher forcing** (always reuse the true past).

Example:

```bash
python experiments/time_series/train_rnn_isl_ett_gaussian.py   --data_path data/ETTm1.csv   --ett_column OT   --seq_len 64   --steps 10000   --forecast_horizon 500
```

Outputs are saved in `experiments/ett_rnn_isl/`.

---

### 5. Time series: multivariate ETT + sliced ISL

Script: `experiments/time_series/train_rnn_sliced_isl_ett.py`

This is the multivariate extension:

- uses several numeric columns as a multivariate series in \(\mathbb{R}^d\),
- applies per-channel normalisation,
- trains an RNN (GRU) + MLP decoder in \(\mathbb{R}^d\),
- minimises **sliced ISL** (`sliced_isl_random`) on the joint one-step-ahead distribution,
- reports 1D diagnostics and a forecast plot for a chosen **plot column** (e.g. `OT`).

Example:

```bash
python experiments/time_series/train_rnn_sliced_isl_ett.py   --data_path data/ETTm1.csv   --feature_cols HUFL HULL MUFL MULL LUFL LULL OT   --plot_column OT   --seq_len 64   --steps 10000   --n_slices 32   --forecast_horizon 500
```
