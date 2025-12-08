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

