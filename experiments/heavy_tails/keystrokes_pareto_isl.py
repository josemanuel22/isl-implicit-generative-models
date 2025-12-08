#!/usr/bin/env python
"""
Keystroke intervals with Pareto-ISL (1D heavy-tailed ISL experiment).

This script trains a 1D implicit generator on a real keystroke
inter-arrival-time dataset using the 1D Invariant Statistical Loss (ISL).

Assumptions
-----------
- You have a preprocessed 1D array of inter-keystroke intervals stored
  in a NumPy .npy file, e.g. shape (N,) with entries in seconds or ms.

- We treat the intervals as an i.i.d. heavy-tailed sample and learn a
  generator G: R^{noise_dim} -> R with either:

    * Gaussian latent:  z ~ N(0, I)
    * GPD latent:       z ~ GPD(ξ, σ)  ("Pareto-like" latent)

- Optionally, we can log-transform the data and model:

      x_log = log(x + eps)

  which often helps numerics while preserving tail structure.

What it does
------------
1. Loads keystroke intervals from a .npy file.
2. Optionally log-transforms the data.
3. Trains a 1D generator with ISL (isl.loss_1d.isl_1d_soft).
4. Saves:
   - training curve (ISL loss vs steps),
   - histograms (linear & log-y) real vs generated (clipped to [-200, 200]).
5. Prints tail diagnostics (quantiles + Hill-type tail index).

Usage
-----
Example:

    python keystrokes_pareto_isl.py \
        --data_path data/keystrokes_intervals.npy \
        --latent_type gpd \
        --log_transform

"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
#  Repo paths and imports
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # .../isl-implicit-generative-models
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isl.loss_1d import isl_1d_soft
from isl.models import MLPGenerator
from isl.utils import set_seed, get_device, ensure_dir


# ---------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------

def load_keystroke_intervals(
    path: str | Path,
    device: torch.device,
    log_transform: bool = False,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Load 1D keystroke inter-arrival times from a .npy or .npz file.

    Expected format:
      - .npy : a NumPy array of shape (N,) or (N,1)
      - .npz : expects key "intervals" with shape (N,) or (N,1)

    Parameters
    ----------
    path : str or Path
        Path to .npy or .npz file.
    device : torch.device
        Device for the resulting tensor.
    log_transform : bool, default=False
        If True, model log(x + eps) instead of x.
    eps : float, default=1e-3
        Small positive constant to avoid log(0) if log-transforming.

    Returns
    -------
    intervals : Tensor, shape (N,)
        Keystroke intervals (possibly log-transformed), on `device`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        if "intervals" not in data:
            raise KeyError(f".npz file must contain 'intervals' key, got keys: {list(data.keys())}")
        arr = data["intervals"]
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}. Use .npy or .npz")

    arr = np.asarray(arr).reshape(-1)  # shape (N,)

    if log_transform:
        arr = np.log(arr + eps)

    intervals = torch.tensor(arr, dtype=torch.float32, device=device)
    return intervals


# ---------------------------------------------------------------------
#  Heavy-tailed latent (GPD) for Pareto-ISL
# ---------------------------------------------------------------------

def sample_gpd(
    n: int,
    device: torch.device,
    xi: float = 0.5,
    sigma: float = 1.0,
    loc: float = 0.0,
) -> torch.Tensor:
    """
    Sample from a 1D Generalised Pareto (GPD) using inverse CDF:

        F(x) = 1 - (1 + ξ (x - loc) / σ)^(-1/ξ),   x >= loc (ξ > 0)
    """
    u = torch.rand(n, device=device)  # U ~ Uniform(0,1)
    xi_t = torch.tensor(xi, device=device, dtype=torch.float32)
    sigma_t = torch.tensor(sigma, device=device, dtype=torch.float32)
    loc_t = torch.tensor(loc, device=device, dtype=torch.float32)

    if abs(xi) < 1e-6:
        # ξ -> 0 limit => exponential
        x = -sigma_t * torch.log1p(-u)
    else:
        x = sigma_t / xi_t * ((1 - u) ** (-xi_t) - 1.0)

    return loc_t + x


def sample_latent(
    latent_type: str,
    n: int,
    noise_dim: int,
    device: torch.device,
    gpd_xi: float = 0.5,
    gpd_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Sample latent noise Z in R^{noise_dim}.

    latent_type:
      - "gaussian": iid N(0,1)
      - "gpd"     : iid GPD(ξ, σ)
    """
    if latent_type == "gaussian":
        return torch.randn(n, noise_dim, device=device)
    elif latent_type == "gpd":
        # Independent GPD along each dimension
        z_list = [
            sample_gpd(n, device=device, xi=gpd_xi, sigma=gpd_sigma, loc=0.0)
            for _ in range(noise_dim)
        ]
        return torch.stack(z_list, dim=1)  # (n, noise_dim)
    else:
        raise ValueError(f"Unknown latent_type: {latent_type!r}")


# ---------------------------------------------------------------------
#  Tail diagnostics (Hill estimator, quantiles)
# ---------------------------------------------------------------------

def hill_tail_index(x: np.ndarray, k: int = 500) -> float:
    """
    Hill estimator for tail index alpha based on top-k order stats.

    For Pareto-like tails:

        alpha_hat ~ 1 / mean( log(X_i / X_(k)) ),

    where X_(k) is the smallest of the top-k order statistics.
    """
    x = np.asarray(x)
    x = x[x > 0]  # need positivity for log
    x_sorted = np.sort(x)
    n = x_sorted.shape[0]
    if n < 10:
        return np.nan
    if k >= n:
        k = max(1, n // 2)

    tail = x_sorted[-k:]
    x_k = tail[0]
    logs = np.log(tail / x_k)
    mean_log = logs.mean()
    if mean_log <= 0:
        return np.nan
    return 1.0 / mean_log


def print_tail_stats(name: str, x: np.ndarray) -> None:
    """
    Print tail quantiles and Hill estimate for a sample.
    """
    qs = [0.9, 0.95, 0.99, 0.995, 0.999]
    print(f"\n{name} tail diagnostics:")
    for q in qs:
        val = np.quantile(x, q)
        print(f"  q={q:.3f}: {val:.4f}")
    alpha_hat = hill_tail_index(x, k=min(1000, max(100, x.shape[0] // 10)))
    print(f"  Hill alpha_hat ≈ {alpha_hat:.4f}")


# ---------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------

def train_keystrokes_isl(
    data_path: str | Path,
    latent_type: str = "gpd",       # "gaussian" or "gpd"
    log_transform: bool = False,
    steps: int = 5000,
    batch_size: int = 512,
    noise_dim: int = 4,
    hidden_dims: tuple[int, ...] = (64, 64),
    lr: float = 1e-3,
    K: int = 64,
    cdf_bandwidth: float = 0.15,
    hist_sigma: float = 0.05,
    gpd_xi: float = 0.5,
    gpd_sigma: float = 1.0,
    log_every: int = 200,
    device: torch.device | None = None,
    outdir: Path | None = None,
) -> None:
    """
    Train a 1D generator on keystroke intervals with ISL and a chosen latent.
    """
    if device is None:
        device = get_device(prefer_gpu=True)
    if outdir is None:
        outdir = ROOT / "experiments" / "keystrokes"
    ensure_dir(outdir)

    print("=" * 70)
    print("Keystrokes Pareto-ISL experiment")
    print(f"  data_path      : {data_path}")
    print(f"  latent_type    : {latent_type}  (gaussian | gpd)")
    print(f"  log_transform  : {log_transform}")
    print(f"  steps          : {steps}")
    print(f"  batch_size     : {batch_size}")
    print(f"  noise_dim      : {noise_dim}")
    print(f"  hidden_dims    : {hidden_dims}")
    print(f"  lr             : {lr}")
    print(f"  K              : {K}")
    print(f"  cdf_bandwidth  : {cdf_bandwidth}")
    print(f"  hist_sigma     : {hist_sigma}")
    print(f"  gpd_xi         : {gpd_xi}")
    print(f"  gpd_sigma      : {gpd_sigma}")
    print(f"  device         : {device}")
    print(f"  outdir         : {outdir}")
    print("=" * 70)

    torch.set_num_threads(1)

    # Load data
    x_all = load_keystroke_intervals(
        path=data_path,
        device=device,
        log_transform=log_transform,
        eps=1e-3,
    )
    N = x_all.shape[0]
    print(f"Loaded {N} keystroke intervals.")

    # Generator: R^{noise_dim} -> R
    gen = MLPGenerator(
        noise_dim=noise_dim,
        data_dim=1,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = optim.Adam(gen.parameters(), lr=lr)
    losses: list[float] = []

    # ------------------- training loop ------------------- #
    for step in range(1, steps + 1):
        # Sample a batch of real keystroke intervals
        idx = torch.randint(low=0, high=N, size=(batch_size,), device=device)
        x_real = x_all[idx]  # (batch_size,)

        # Sample latent
        z = sample_latent(
            latent_type,
            batch_size,
            noise_dim=noise_dim,
            device=device,
            gpd_xi=gpd_xi,
            gpd_sigma=gpd_sigma,
        )

        x_fake = gen(z).view(-1)  # shape (batch_size,)

        loss = isl_1d_soft(
            x_real.view(-1),
            x_fake,
            K=K,
            cdf_bandwidth=cdf_bandwidth,
            hist_sigma=hist_sigma,
            reduction="mean",
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

        if step % log_every == 0 or step == 1 or step == steps:
            print(f"[{step:5d}/{steps}] ISL loss = {loss.item():.6f}")

    # ------------------- Save checkpoint ------------------- #
    suffix = f"keystrokes_latent-{latent_type}_K{K}" + ("_logx" if log_transform else "")
    ckpt_path = outdir / f"generator_{suffix}.pt"
    torch.save(gen.state_dict(), ckpt_path)
    print(f"\nSaved generator checkpoint to {ckpt_path}")

    # ------------------- Training curve ------------------- #
    steps_arr = np.arange(1, len(losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(steps_arr, losses)
    plt.xlabel("Training step")
    plt.ylabel("ISL loss")
    plt.title(f"Keystrokes ISL training curve ({latent_type} latent)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    curve_path = outdir / f"keystrokes_isl_loss_curve_{suffix}.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved training curve to {curve_path}")

    # ------------------- Evaluation: real vs generated ------------------- #
    gen.eval()
    with torch.no_grad():
        # Use all data (or sample) for "real" evaluation
        x_real_eval = x_all.detach().cpu().numpy()

        N_eval = min(20000, N)
        z_eval = sample_latent(
            latent_type,
            N_eval,
            noise_dim=noise_dim,
            device=device,
            gpd_xi=gpd_xi,
            gpd_sigma=gpd_sigma,
        )
        x_fake_eval = gen(z_eval).view(-1).cpu().numpy()

    print_tail_stats("Real keystrokes", x_real_eval)
    print_tail_stats(f"Generated ({latent_type} latent)", x_fake_eval)

    # ------------------- Histograms (linear & log-y) ------------------- #
    xmin, xmax = -200.0, 200.0  # fixed window for visibility/comparability

    # Linear scale
    plt.figure(figsize=(6, 4))
    plt.hist(
        x_real_eval,
        bins=200,
        range=(xmin, xmax),
        density=True,
        alpha=0.5,
        label="real",
    )
    plt.hist(
        x_fake_eval,
        bins=200,
        range=(xmin, xmax),
        density=True,
        alpha=0.5,
        label="generated",
    )
    plt.xlim(xmin, xmax)
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.xlabel("x" if not log_transform else "log(x + eps)")
    plt.ylabel("density (hist)")
    plt.title(f"Keystrokes ({latent_type} latent) - linear scale")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    hist_lin_path = outdir / f"keystrokes_hist_linear_{suffix}.png"
    plt.savefig(hist_lin_path, dpi=200)
    plt.close()
    print(f"Saved linear-scale histogram to {hist_lin_path}")

    # Log-y scale
    plt.figure(figsize=(6, 4))
    plt.hist(
        x_real_eval,
        bins=200,
        range=(xmin, xmax),
        density=True,
        alpha=0.5,
        label="real",
    )
    plt.hist(
        x_fake_eval,
        bins=200,
        range=(xmin, xmax),
        density=True,
        alpha=0.5,
        label="generated",
    )
    plt.xlim(xmin, xmax)
    plt.yscale("log")
    plt.ylim(bottom=1e-6)
    plt.legend()
    plt.xlabel("x" if not log_transform else "log(x + eps)")
    plt.ylabel("density (log scale)")
    plt.title(f"Keystrokes ({latent_type} latent) - log-y scale")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    hist_log_path = outdir / f"keystrokes_hist_logy_{suffix}.png"
    plt.savefig(hist_log_path, dpi=200)
    plt.close()
    print(f"Saved log-y histogram to {hist_log_path}")


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keystroke inter-arrival times with Pareto-ISL (1D ISL)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to .npy or .npz file containing 1D keystroke intervals.",
    )
    parser.add_argument(
        "--latent_type",
        type=str,
        default="gpd",
        choices=["gaussian", "gpd"],
        help="Latent noise type: Gaussian or GPD (Pareto-like).",
    )
    parser.add_argument(
        "--log_transform",
        action="store_true",
        help="Model log(x + eps) instead of x.",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--noise_dim", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--cdf_bandwidth", type=float, default=0.15)
    parser.add_argument("--hist_sigma", type=float, default=0.05)
    parser.add_argument("--gpd_xi", type=float, default=0.5)
    parser.add_argument("--gpd_sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    device = get_device(prefer_gpu=True)

    train_keystrokes_isl(
        data_path=args.data_path,
        latent_type=args.latent_type,
        log_transform=args.log_transform,
        steps=args.steps,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        hidden_dims=(64, 64),
        lr=args.lr,
        K=args.K,
        cdf_bandwidth=args.cdf_bandwidth,
        hist_sigma=args.hist_sigma,
        gpd_xi=args.gpd_xi,
        gpd_sigma=args.gpd_sigma,
        log_every=200,
        device=device,
    )


if __name__ == "__main__":
    main()
