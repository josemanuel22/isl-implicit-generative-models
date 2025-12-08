#!/usr/bin/env python
"""
Train a 2D generator with sliced ISL (using isl.sliced.sliced_isl_random)
on various 2D toy distributions.

Targets implemented:

  - "ring"        : ring of Gaussians (K modes on a circle)
  - "gaussian"    : single correlated 2D Gaussian
  - "grid"        : grid of Gaussians on a lattice
  - "checkerboard": checkerboard-like pattern
  - "moons"       : classic two-moons dataset
  - "spiral"      : spiral distribution

We train an MLP generator G: R^{noise_dim} -> R^2 and use sliced_isl_random
(defined in isl/sliced.py) as the training loss.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

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

from isl.models import MLPGenerator
from isl.utils import set_seed, get_device, ensure_dir
from isl.sliced import sliced_isl_random


# ---------------------------------------------------------------------
#  2D toy target samplers
# ---------------------------------------------------------------------

def sample_ring_gaussians(
    n: int,
    device: torch.device,
    n_modes: int = 8,
    radius: float = 2.5,
    sigma: float = 0.15,
) -> torch.Tensor:
    """
    Ring of Gaussians:

        - Means equally spaced on a circle of radius `radius`
        - Each component has isotropic covariance sigma^2 I_2
    """
    comps = torch.randint(low=0, high=n_modes, size=(n,), device=device)
    angles = 2 * np.pi * comps.float() / float(n_modes)
    mu_x = radius * torch.cos(angles)
    mu_y = radius * torch.sin(angles)
    mu = torch.stack([mu_x, mu_y], dim=1)  # (n, 2)
    eps = sigma * torch.randn(n, 2, device=device)
    return mu + eps


def sample_gaussian_2d(
    n: int,
    device: torch.device,
    mean: tuple[float, float] = (0.0, 0.0),
    rho: float = 0.8,
    sx: float = 1.0,
    sy: float = 0.5,
) -> torch.Tensor:
    """
    Single correlated 2D Gaussian:

        X ~ N(mean, Σ), with

            Σ = [[sx^2, rho*sx*sy],
                 [rho*sx*sy, sy^2]]
    """
    mean_vec = torch.tensor(mean, device=device, dtype=torch.float32)
    cov = torch.tensor(
        [[sx**2, rho * sx * sy],
         [rho * sx * sy, sy**2]],
        device=device,
        dtype=torch.float32,
    )
    L = torch.linalg.cholesky(cov)  # (2, 2)
    z = torch.randn(n, 2, device=device)
    x = z @ L.T + mean_vec
    return x


def sample_grid_gaussians(
    n: int,
    device: torch.device,
    grid_size: int = 5,
    spacing: float = 2.0,
    sigma: float = 0.15,
) -> torch.Tensor:
    """
    Grid of Gaussians on a grid_size x grid_size lattice.

    Means at coordinates:
        {-(g-1)/2, ..., (g-1)/2} x {-(g-1)/2, ..., (g-1)/2} * spacing
    """
    g = grid_size
    coords_1d = torch.linspace(-(g - 1) / 2, (g - 1) / 2, steps=g, device=device)
    xs, ys = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    means = torch.stack([xs.flatten(), ys.flatten()], dim=1) * spacing  # (g^2, 2)
    n_comp = means.shape[0]

    comp_idx = torch.randint(low=0, high=n_comp, size=(n,), device=device)
    mu = means[comp_idx]
    eps = sigma * torch.randn(n, 2, device=device)
    return mu + eps


def sample_checkerboard(
    n: int,
    device: torch.device,
    scale: float = 2.0,
) -> torch.Tensor:
    """
    Simple checkerboard-like distribution.

    Idea:
        - Sample (u, v) in [0, 4) x [0, 4)
        - Shift v by +2 on every other vertical strip to create a checkerboard.

    This is not the canonical GAN benchmark formula, but gives a clear
    checkerboard structure.
    """
    u = 4 * torch.rand(n, device=device)
    v = 4 * torch.rand(n, device=device)

    # Shift every other vertical strip
    mask = (torch.floor(u) % 2 == 0)
    v = v + 2 * mask.float()

    x = u - 2  # center roughly at 0
    y = v - 4  # center roughly at 0
    return scale * torch.stack([x, y], dim=1)


def sample_two_moons(
    n: int,
    device: torch.device,
    noise: float = 0.08,
    radius: float = 1.0,
) -> torch.Tensor:
    """
    Two-moons dataset, constructed manually.
    """
    n1 = n // 2
    n2 = n - n1

    # First moon
    theta1 = torch.rand(n1, device=device) * np.pi  # [0, π]
    x1 = torch.stack(
        [
            radius * torch.cos(theta1),
            radius * torch.sin(theta1),
        ],
        dim=1,
    )

    # Second moon (shifted + flipped)
    theta2 = torch.rand(n2, device=device) * np.pi
    x2 = torch.stack(
        [
            radius * torch.cos(theta2) + radius,
            -radius * torch.sin(theta2) + 0.5,
        ],
        dim=1,
    )

    x = torch.cat([x1, x2], dim=0)
    x = x + noise * torch.randn_like(x)
    return x


def sample_spiral(
    n: int,
    device: torch.device,
    n_turns: float = 3.0,
    noise: float = 0.05,
) -> torch.Tensor:
    """
    Simple 2D spiral distribution.

    Sample r ~ Uniform(0, 1), angle θ = r * n_turns * 2π, then:

        x = r cos θ + noise
        y = r sin θ + noise
    """
    r = torch.rand(n, device=device)
    theta = r * n_turns * 2 * np.pi

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    pts = torch.stack([x, y], dim=1)
    pts = pts + noise * torch.randn_like(pts)
    return pts


def sample_target(
    target: str,
    n: int,
    device: torch.device,
    **kwargs,
) -> torch.Tensor:
    """
    Dispatch to the appropriate 2D toy sampler.
    """
    if target == "ring":
        return sample_ring_gaussians(
            n,
            device=device,
            n_modes=kwargs.get("n_modes", 8),
            radius=kwargs.get("radius", 2.5),
            sigma=kwargs.get("sigma", 0.15),
        )
    elif target == "gaussian":
        return sample_gaussian_2d(
            n,
            device=device,
            mean=kwargs.get("mean", (0.0, 0.0)),
            rho=kwargs.get("rho", 0.8),
            sx=kwargs.get("sx", 1.0),
            sy=kwargs.get("sy", 0.5),
        )
    elif target == "grid":
        return sample_grid_gaussians(
            n,
            device=device,
            grid_size=kwargs.get("grid_size", 5),
            spacing=kwargs.get("spacing", 2.0),
            sigma=kwargs.get("sigma", 0.15),
        )
    elif target == "checkerboard":
        return sample_checkerboard(
            n,
            device=device,
            scale=kwargs.get("scale", 1.0),
        )
    elif target == "moons":
        return sample_two_moons(
            n,
            device=device,
            noise=kwargs.get("noise", 0.08),
            radius=kwargs.get("radius", 1.0),
        )
    elif target == "spiral":
        return sample_spiral(
            n,
            device=device,
            n_turns=kwargs.get("n_turns", 3.0),
            noise=kwargs.get("noise", 0.05),
        )
    else:
        raise ValueError(f"Unknown 2D target: {target!r}")


# ---------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------

def train_sliced_isl_2d(
    target: str = "ring",
    steps: int = 5000,
    batch_size: int = 512,
    noise_dim: int = 4,
    hidden_dims: tuple[int, ...] = (128, 128),
    lr: float = 1e-3,
    n_projections: int = 128,
    K: int = 64,
    cdf_bandwidth: float = 0.15,
    hist_sigma: float = 0.05,
    target_kwargs: dict | None = None,
    log_every: int = 200,
    device: torch.device | None = None,
    outdir: Path | None = None,
) -> None:
    """
    Train a 2D generator with sliced_isl_random on a chosen 2D toy target.

    This uses `isl.sliced.sliced_isl_random` (random directions only, no smart
    selection) as defined in your sliced.py.
    """
    if device is None:
        device = get_device(prefer_gpu=True)
    if outdir is None:
        outdir = ROOT / "experiments" / "2d_toy" / "runs"
    ensure_dir(outdir)

    if target_kwargs is None:
        target_kwargs = {}

    print("=" * 70)
    print("Train 2D generator with sliced ISL (random directions)")
    print(f"  target         : {target}")
    print(f"  steps          : {steps}")
    print(f"  batch_size     : {batch_size}")
    print(f"  noise_dim      : {noise_dim}")
    print(f"  hidden_dims    : {hidden_dims}")
    print(f"  lr             : {lr}")
    print(f"  n_projections  : {n_projections}")
    print(f"  K              : {K}")
    print(f"  cdf_bandwidth  : {cdf_bandwidth}")
    print(f"  hist_sigma     : {hist_sigma}")
    print(f"  target_kwargs  : {target_kwargs}")
    print(f"  device         : {device}")
    print(f"  outdir         : {outdir}")
    print("=" * 70)

    torch.set_num_threads(1)

    # Generator: R^{noise_dim} -> R^2
    gen = MLPGenerator(
        noise_dim=noise_dim,
        data_dim=2,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = optim.Adam(gen.parameters(), lr=lr)
    losses = []

    # ------------------- training loop ------------------- #
    for step in range(1, steps + 1):
        x_real = sample_target(
            target,
            batch_size,
            device=device,
            **target_kwargs,
        )

        z = torch.randn(batch_size, noise_dim, device=device)
        x_fake = gen(z)

        loss = sliced_isl_random(
            x_real,
            x_fake,
            m=n_projections,
            K=K,
            cdf_bandwidth=cdf_bandwidth,
            hist_sigma=hist_sigma,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

        if step % log_every == 0 or step == 1 or step == steps:
            print(f"[{step:5d}/{steps}] sliced ISL loss = {loss.item():.6f}")

    # ------------------- save checkpoint ------------------- #
    ckpt_path = outdir / f"generator_2d_{target}_K{K}_m{n_projections}.pt"
    torch.save(gen.state_dict(), ckpt_path)
    print(f"\nSaved generator checkpoint to {ckpt_path}")

    # ------------------- training curve ------------------- #
    steps_arr = np.arange(1, len(losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(steps_arr, losses)
    plt.xlabel("Training step")
    plt.ylabel("sliced ISL loss")
    plt.title(f"2D sliced ISL training curve ({target})")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    curve_path = outdir / f"sliced_isl_2d_loss_curve_{target}_K{K}_m{n_projections}.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved training curve to {curve_path}")

    # ------------------- visualize learned generator ------------------- #
    gen.eval()
    with torch.no_grad():
        N_vis = 5000
        x_real_vis = sample_target(
            target,
            N_vis,
            device=device,
            **target_kwargs,
        ).cpu().numpy()

        z_vis = torch.randn(N_vis, noise_dim, device=device)
        x_fake_vis = gen(z_vis).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(
        x_real_vis[:, 0],
        x_real_vis[:, 1],
        s=5,
        alpha=0.4,
        label="real",
    )
    plt.scatter(
        x_fake_vis[:, 0],
        x_fake_vis[:, 1],
        s=5,
        alpha=0.4,
        label="generated",
    )
    plt.axis("equal")
    plt.legend()
    plt.title(f"2D toy ({target}) with sliced ISL (random)")
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    scatter_path = outdir / f"sliced_isl_2d_{target}_real_vs_gen_K{K}_m{n_projections}.png"
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"Saved scatter plot to {scatter_path}")


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 2D generator with sliced ISL (random directions) "
                    "on various 2D toy datasets."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="ring",
        choices=["ring", "gaussian", "grid", "checkerboard", "moons", "spiral"],
        help="2D toy distribution to learn.",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--noise_dim", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_projections", type=int, default=128)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--cdf_bandwidth", type=float, default=0.15)
    parser.add_argument("--hist_sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    device = get_device(prefer_gpu=True)

    # Default target kwargs (you can later make these CLI flags if you want)
    if args.target == "ring":
        target_kwargs = dict(n_modes=8, radius=2.5, sigma=0.15)
    elif args.target == "gaussian":
        target_kwargs = dict(mean=(0.0, 0.0), rho=0.8, sx=1.0, sy=0.5)
    elif args.target == "grid":
        target_kwargs = dict(grid_size=5, spacing=2.0, sigma=0.15)
    elif args.target == "checkerboard":
        target_kwargs = dict(scale=1.0)
    elif args.target == "moons":
        target_kwargs = dict(noise=0.08, radius=1.0)
    elif args.target == "spiral":
        target_kwargs = dict(n_turns=3.0, noise=0.05)
    else:
        target_kwargs = {}

    train_sliced_isl_2d(
        target=args.target,
        steps=args.steps,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        hidden_dims=(128, 128),
        lr=args.lr,
        n_projections=args.n_projections,
        K=args.K,
        cdf_bandwidth=args.cdf_bandwidth,
        hist_sigma=args.hist_sigma,
        target_kwargs=target_kwargs,
        log_every=200,
        device=device,
    )


if __name__ == "__main__":
    main()
