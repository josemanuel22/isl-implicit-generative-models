#!/usr/bin/env python
"""
Train a 1D implicit generator with ISL loss.

This script trains an MLP generator G : R^{noise_dim} -> R using the 1D soft
ISL divergence between real samples X_real (from a chosen 1D target) and
generated samples X_fake = G(Z), Z ~ N(0, I).

Supported targets:
  - "gaussian":
        X_real ~ N(0, 1)

  - "mixture":
        X_real ~ 0.5 * N(-delta, 1) + 0.5 * N(+delta, 1)

  - "laplace":
        X_real ~ Laplace(0, b)

  - "student":
        X_real ~ StudentT(df=nu)  (centered heavy-tailed)

  - "lognormal":
        X_real ~ LogNormal(0, sigma)  (positive, skewed)

  - "pareto":
        X_real ~ Pareto(xm, alpha)   (positive heavy tail)

Outputs:
  - A trained generator checkpoint.
  - Training curve: ISL loss vs optimisation steps.
  - Histogram overlay of real vs generated samples.
  - For the Gaussian case, KSD between generated samples and N(0,1).

Usage (from repo root), example:
    python experiments/1d_univariate/run_1d_isl.py \
        --target gaussian \
        --steps 5000 \
        --K 32

    python experiments/1d_univariate/run_1d_isl.py \
        --target student \
        --student_df 3.0 \
        --steps 5000
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Laplace, StudentT, LogNormal, Pareto

# ---------------------------------------------------------------------
#  Make sure 'isl' is importable from src/ layout
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]   # .../isl-implicit-generative-models
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isl.models import MLPGenerator
from isl.loss_1d import isl_1d_soft
from isl.utils import set_seed, get_device, ensure_dir
from isl.metrics import ksd_rbf


# ---------------------------------------------------------------------
#  Target distributions
# ---------------------------------------------------------------------

def sample_real_gaussian(n: int, device: torch.device) -> torch.Tensor:
    """
    Sample from the 1D standard normal N(0, 1).
    """
    return torch.randn(n, device=device)


def sample_real_mixture(n: int, device: torch.device, delta: float) -> torch.Tensor:
    """
    Sample from a symmetric 2-component Gaussian mixture:

        X ~ 0.5 * N(-delta, 1) + 0.5 * N(+delta, 1).
    """
    signs = torch.where(
        torch.rand(n, device=device) < 0.5,
        torch.full((n,), -1.0, device=device),
        torch.full((n,), +1.0, device=device),
    )
    eps = torch.randn(n, device=device)
    return signs * delta + eps


def sample_real_laplace(n: int, device: torch.device, scale: float) -> torch.Tensor:
    """
    Sample from a Laplace(0, scale) distribution (double exponential).
    """
    loc = torch.tensor(0.0, device=device)
    sc = torch.tensor(scale, device=device)
    dist = Laplace(loc=loc, scale=sc)
    return dist.sample((n,))


def sample_real_student(n: int, device: torch.device, df: float) -> torch.Tensor:
    """
    Sample from a centered Student-t distribution with df degrees of freedom.
    """
    df_t = torch.tensor(df, device=device)
    dist = StudentT(df_t)
    return dist.sample((n,))


def sample_real_lognormal(n: int, device: torch.device, sigma: float) -> torch.Tensor:
    """
    Sample from LogNormal(0, sigma) (positive, skewed).
    """
    mean = torch.tensor(0.0, device=device)
    std = torch.tensor(sigma, device=device)
    dist = LogNormal(mean, std)
    return dist.sample((n,))


def sample_real_pareto(n: int, device: torch.device, xm: float, alpha: float) -> torch.Tensor:
    """
    Sample from a Pareto(xm, alpha) distribution (positive heavy tail).

    Support: x >= xm > 0.
    """
    scale = torch.tensor(xm, device=device)
    a = torch.tensor(alpha, device=device)
    dist = Pareto(scale=scale, alpha=a)
    return dist.sample((n,))

def sample_real_mog3(
    n: int,
    device: torch.device,
    means: torch.Tensor | None = None,
    stds: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sample from a 1D 3-component Gaussian mixture:

        X ~ sum_j w_j * N(means_j, stds_j^2),

    with a default choice:
        means  = [-3, 0, 3]
        stds   = [1, 0.5, 1]
        weights = [0.3, 0.4, 0.3]

    Parameters
    ----------
    n : int
        Number of samples.
    device : torch.device
        Device where tensors are allocated.
    means, stds, weights : torch.Tensor, optional
        Custom mixture parameters. If None, defaults are used.

    Returns
    -------
    x : torch.Tensor of shape (n,)
        Mixture samples.
    """
    if means is None:
        means = torch.tensor([-3.0, 0.0, 3.0], device=device)
    if stds is None:
        stds = torch.tensor([1.0, 0.5, 1.0], device=device)
    if weights is None:
        weights = torch.tensor([0.3, 0.4, 0.3], device=device)

    # Normalise weights just in case
    weights = weights / weights.sum()

    k = means.numel()
    # Categorical sampling of component indices
    comp_idx = torch.multinomial(weights, num_samples=n, replacement=True)  # (n,)

    # Gather means/stds for each selected component
    comp_means = means[comp_idx]   # (n,)
    comp_stds = stds[comp_idx]     # (n,)

    eps = torch.randn(n, device=device)
    x = comp_means + comp_stds * eps
    return x



# ---------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------

def train_isl_1d(
    target: str = "gaussian",
    steps: int = 5000,
    batch_size: int = 512,
    K: int = 32,
    cdf_bandwidth: float = 0.15,
    hist_sigma: float = 0.05,
    noise_dim: int = 4,
    hidden_dims: tuple[int, ...] = (64, 64),
    lr: float = 1e-3,
    mixture_delta: float = 2.0,
    laplace_scale: float = 1.0,
    student_df: float = 3.0,
    lognorm_sigma: float = 0.5,
    pareto_xm: float = 1.0,
    pareto_alpha: float = 2.5,
    log_every: int = 200,
    device: torch.device | None = None,
    outdir: Path | None = None,
) -> None:
    """
    Train a 1D generator with ISL loss for a chosen 1D target distribution.

    Parameters
    ----------
    target : {"gaussian", "mixture", "laplace", "student", "lognormal", "pareto"}
        Target distribution type.
    steps : int
        Number of optimisation steps.
    batch_size : int
        Number of real/fake samples per step.
    K : int
        Number of bins in the 1D ISL estimator.
    cdf_bandwidth : float
        Bandwidth for the soft CDF smoothing in isl_1d_soft.
    hist_sigma : float
        Bandwidth for the soft histogram smoothing in isl_1d_soft.
    noise_dim : int
        Dimension of the latent noise z ~ N(0, I).
    hidden_dims : tuple of int
        Hidden sizes for the MLPGenerator.
    lr : float
        Learning rate for Adam.
    mixture_delta : float
        Half-separation of the modes for the "mixture" target.
    laplace_scale : float
        Scale (b) for Laplace(0, b).
    student_df : float
        Degrees of freedom for Student-t(df).
    lognorm_sigma : float
        Sigma for LogNormal(0, sigma).
    pareto_xm : float
        Scale (xm) for Pareto(xm, alpha).
    pareto_alpha : float
        Shape (alpha) for Pareto(xm, alpha).
    log_every : int
        Print training info every log_every steps.
    device : torch.device, optional
        Device to use; if None, uses get_device().
    outdir : Path, optional
        Output directory; if None, uses experiments/1d_univariate/runs.
    """
    if device is None:
        device = get_device(prefer_gpu=True)
    if outdir is None:
        outdir = ROOT / "experiments" / "1d_univariate" / "runs"
    ensure_dir(outdir)

    print("=" * 70)
    print("Train 1D generator with ISL")
    print(f"  target         : {target}")
    print(f"  steps          : {steps}")
    print(f"  batch_size     : {batch_size}")
    print(f"  K              : {K}")
    print(f"  cdf_bandwidth  : {cdf_bandwidth}")
    print(f"  hist_sigma     : {hist_sigma}")
    print(f"  noise_dim      : {noise_dim}")
    print(f"  hidden_dims    : {hidden_dims}")
    print(f"  lr             : {lr}")
    print(f"  mixture_delta  : {mixture_delta}")
    print(f"  laplace_scale  : {laplace_scale}")
    print(f"  student_df     : {student_df}")
    print(f"  lognorm_sigma  : {lognorm_sigma}")
    print(f"  pareto_xm      : {pareto_xm}")
    print(f"  pareto_alpha   : {pareto_alpha}")
    print(f"  device         : {device}")
    print(f"  outdir         : {outdir}")
    print("=" * 70)

    # -----------------------------------------------------------------
    #  Model and optimiser
    # -----------------------------------------------------------------
    torch.set_num_threads(1)

    gen = MLPGenerator(
        noise_dim=noise_dim,
        data_dim=1,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = optim.Adam(gen.parameters(), lr=lr)
    losses: list[float] = []

    # -----------------------------------------------------------------
    #  Training loop
    # -----------------------------------------------------------------
    for step in range(1, steps + 1):
        # Real samples
        if target == "gaussian":
            x_real = sample_real_gaussian(batch_size, device=device)
        elif target == "mixture":
            x_real = sample_real_mixture(batch_size, device=device, delta=mixture_delta)
        elif target == "laplace":
            x_real = sample_real_laplace(batch_size, device=device, scale=laplace_scale)
        elif target == "student":
            x_real = sample_real_student(batch_size, device=device, df=student_df)
        elif target == "lognormal":
            x_real = sample_real_lognormal(batch_size, device=device, sigma=lognorm_sigma)
        elif target == "pareto":
            x_real = sample_real_pareto(batch_size, device=device, xm=pareto_xm, alpha=pareto_alpha)
        elif target == "mog3":
            x_real = sample_real_mog3(batch_size, device=device)
        else:
            raise ValueError(f"Unknown target: {target!r}")

        # Generator samples
        z = torch.randn(batch_size, noise_dim, device=device)
        x_fake = gen(z).view(-1)  # (batch_size,)

        # ISL loss
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

        losses.append(loss.item())

        if step % log_every == 0 or step == 1 or step == steps:
            print(
                f"[{step:5d}/{steps}] ISL loss = {loss.item():.6f}"
            )

    # -----------------------------------------------------------------
    #  Save checkpoint
    # -----------------------------------------------------------------
    ckpt_path = outdir / f"generator_1d_{target}_K{K}.pt"
    torch.save(gen.state_dict(), ckpt_path)
    print(f"\nSaved generator checkpoint to {ckpt_path}")

    # -----------------------------------------------------------------
    #  Plot training curve
    # -----------------------------------------------------------------
    steps_arr = np.arange(1, len(losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(steps_arr, losses)
    plt.xlabel("Training step")
    plt.ylabel("ISL loss")
    plt.title(f"1D ISL training curve ({target}, K={K})")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    curve_path = outdir / f"isl_1d_loss_curve_{target}_K{K}.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved training curve to {curve_path}")

    # -----------------------------------------------------------------
    #  Evaluate: real vs generated histogram
    # -----------------------------------------------------------------
    gen.eval()
    with torch.no_grad():
        N_eval = 20000

        if target == "gaussian":
            x_real_eval = sample_real_gaussian(N_eval, device=device)
        elif target == "mixture":
            x_real_eval = sample_real_mixture(N_eval, device=device, delta=mixture_delta)
        elif target == "laplace":
            x_real_eval = sample_real_laplace(N_eval, device=device, scale=laplace_scale)
        elif target == "student":
            x_real_eval = sample_real_student(N_eval, device=device, df=student_df)
        elif target == "lognormal":
            x_real_eval = sample_real_lognormal(N_eval, device=device, sigma=lognorm_sigma)
        elif target == "pareto":
            x_real_eval = sample_real_pareto(N_eval, device=device, xm=pareto_xm, alpha=pareto_alpha)
        elif target == "mog3":
            x_real_eval = sample_real_mog3(N_eval, device=device)
        else:
            raise ValueError(f"Unknown target: {target!r}")

        z_eval = torch.randn(N_eval, noise_dim, device=device)
        x_fake_eval = gen(z_eval).view(-1)

    x_real_np = x_real_eval.cpu().numpy()
    x_fake_np = x_fake_eval.cpu().numpy()

    # Basic stats
    print("\nSample statistics (generated):")
    print(f"  mean ≈ {x_fake_np.mean():.4f}, std ≈ {x_fake_np.std():.4f}")
    print("  (real)    mean ≈ {:.4f}, std ≈ {:.4f}".format(
        x_real_np.mean(), x_real_np.std()
    ))

    # Histogram overlay
    plt.figure(figsize=(6, 4))
    # Choose a common range to make comparison fair
    xmin = min(x_real_np.min(), x_fake_np.min())
    xmax = max(x_real_np.max(), x_fake_np.max())
    plt.hist(
        x_real_np,
        bins=100,
        range=(xmin, xmax),
        density=True,
        alpha=0.5,
        label="real",
    )
    plt.hist(
        x_fake_np,
        bins=100,
        range=(xmin, xmax),
        density=True,
        alpha=0.5,
        label="generated",
    )
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("density (hist)")
    plt.title(f"Real vs generated histogram ({target}, K={K})")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    hist_path = outdir / f"isl_1d_hist_{target}_K{K}.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved histogram plot to {hist_path}")

    # -----------------------------------------------------------------
    #  Optional: KSD for Gaussian target
    # -----------------------------------------------------------------
    if target == "gaussian":
        print("\nComputing KSD against N(0,1)...")

        samples = torch.from_numpy(x_fake_np).float().unsqueeze(1).to(device)

        def score_fn(x: torch.Tensor) -> torch.Tensor:
            # Score of N(0,1): ∇_x log p(x) = -x
            return -x

        with torch.no_grad():
            ksd_val = ksd_rbf(samples, score_fn).item()
        print(f"KSD (generated vs N(0,1)) ≈ {ksd_val:.4e}")


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 1D generator with ISL loss."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="gaussian",
        choices=["gaussian", "mixture", "laplace", "student", "lognormal", "pareto", "mog3"],
        help="Target 1D distribution to learn.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of optimisation steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=32,
        help="Number of bins in the ISL estimator.",
    )
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=4,
        help="Dimensionality of latent noise z.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--mixture_delta",
        type=float,
        default=2.0,
        help="Half-separation delta for the 2-component mixture.",
    )
    parser.add_argument(
        "--laplace_scale",
        type=float,
        default=1.0,
        help="Scale parameter b for Laplace(0, b).",
    )
    parser.add_argument(
        "--student_df",
        type=float,
        default=3.0,
        help="Degrees of freedom for Student-t(df).",
    )
    parser.add_argument(
        "--lognorm_sigma",
        type=float,
        default=0.5,
        help="Sigma for LogNormal(0, sigma).",
    )
    parser.add_argument(
        "--pareto_xm",
        type=float,
        default=1.0,
        help="Scale xm for Pareto(xm, alpha).",
    )
    parser.add_argument(
        "--pareto_alpha",
        type=float,
        default=2.5,
        help="Shape alpha for Pareto(xm, alpha).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    device = get_device(prefer_gpu=True)

    train_isl_1d(
        target=args.target,
        steps=args.steps,
        batch_size=args.batch_size,
        K=args.K,
        cdf_bandwidth=0.15,
        hist_sigma=0.05,
        noise_dim=args.noise_dim,
        hidden_dims=(64, 64),
        lr=args.lr,
        mixture_delta=args.mixture_delta,
        laplace_scale=args.laplace_scale,
        student_df=args.student_df,
        lognorm_sigma=args.lognorm_sigma,
        pareto_xm=args.pareto_xm,
        pareto_alpha=args.pareto_alpha,
        log_every=200,
        device=device,
    )


if __name__ == "__main__":
    main()
