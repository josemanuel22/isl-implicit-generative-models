#!/usr/bin/env python
"""
MNIST implicit generator trained with sliced ISL (DCGAN generator, GPU + AMP).

This script trains a DCGAN-style implicit generator G(z) -> x on MNIST using the
Sliced Invariant Statistical Loss (ISL) in R^784:

    - Data: MNIST 28x28, flattened to R^784 and scaled to [-1, 1].
    - Latent: z ~ N(0, I) in R^{noise_dim}.
    - Generator: DCGAN-style convnet producing 1x28x28 images (tanh output).
    - Loss: sliced ISL with random directions (isl.sliced.sliced_isl_random).

It saves:
    - a training curve (sliced ISL vs steps),
    - a grid of generated samples after training.

This version:
    - explicitly uses CUDA (and raises if not available),
    - uses torch.cuda.amp (autocast + GradScaler) for mixed precision on GPU.
"""

from __future__ import annotations

#import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys
import argparse
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# ---------------------------------------------------------------------
#  Repo paths and imports
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # .../isl-implicit-generative-models
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isl.sliced import sliced_isl_random
from isl.utils import set_seed, ensure_dir


# ---------------------------------------------------------------------
#  DCGAN-style generator for MNIST
# ---------------------------------------------------------------------

class DCGANGeneratorMNIST(nn.Module):
    """
    DCGAN-style generator for 28x28 grayscale images.

    Input:  z of shape (B, z_dim)
    Output: x_fake of shape (B, 1, 28, 28), in [-1, 1] (tanh)
    """

    def __init__(self, z_dim: int = 64, ngf: int = 64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.ngf = ngf

        self.net = nn.Sequential(
            # Input: (B, z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, ngf * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),          # (B, 4*ngf, 7, 7)

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),          # (B, 2*ngf, 14, 14)

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),          # (B, ngf, 28, 28)

            nn.ConvTranspose2d(ngf, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),              # (B, 1, 28, 28) in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim)
        B, Dz = z.shape
        if Dz != self.z_dim:
            raise ValueError(f"Expected z_dim={self.z_dim}, got {Dz}")
        z = z.view(B, Dz, 1, 1)
        return self.net(z)


# ---------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------

def get_mnist_loader(
    data_root: str | Path,
    batch_size: int,
    train: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """
    MNIST DataLoader with images in [0,1] as float tensors.
    Flattening + rescaling to [-1,1] is done in the training loop.
    """
    transform = transforms.ToTensor()  # returns (1, 28, 28) in [0,1]

    ds = datasets.MNIST(
        root=str(data_root),
        train=train,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# ---------------------------------------------------------------------
#  Latent sampler
# ---------------------------------------------------------------------

def sample_latent(
    n: int,
    noise_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample latent noise Z in R^{noise_dim}, iid N(0,1).
    """
    return torch.randn(n, noise_dim, device=device)


# ---------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------

def train_mnist_sliced_isl_dcgan(
    data_root: str | Path,
    steps: int = 50000,
    batch_size: int = 128,
    noise_dim: int = 64,
    ngf: int = 64,
    lr: float = 2e-4,
    n_slices: int = 32,
    K: int = 64,
    cdf_bandwidth: float = 0.15,
    hist_sigma: float = 0.05,
    log_every: int = 500,
    eval_n_samples: int = 64,
    device: Optional[torch.device] = None,
    outdir: Optional[Path] = None,
) -> None:
    """
    Train a DCGAN-style implicit generator on MNIST with sliced ISL in R^784.

    Parameters
    ----------
    data_root : str or Path
        Directory for torchvision MNIST download.
    steps : int
        Number of training steps (mini-batches).
    batch_size : int
        Batch size.
    noise_dim : int
        Dimension of latent z.
    ngf : int
        Base number of feature maps in the generator (DCGAN convention).
    lr : float
        Learning rate (Adam).
    n_slices : int
        Number of random directions for sliced ISL.
    K : int
        Number of bins in 1D ISL surrogate.
    cdf_bandwidth : float
        Soft CDF bandwidth.
    hist_sigma : float
        Soft histogram bandwidth.
    log_every : int
        Print loss every this many steps.
    eval_n_samples : int
        Number of samples to generate for the final image grid.
    device : torch.device, optional
        Device where training is run (expects CUDA here).
    outdir : Path, optional
        Output directory for checkpoints / plots.
    """
    # Device handling: expect GPU
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but a GPU run was requested.")
        device = torch.device("cuda")
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA device, got {device}")

    use_amp = True  # we want AMP on GPU

    if outdir is None:
        outdir = ROOT / "experiments" / "mnist_sliced_isl"
    ensure_dir(outdir)

    print("=" * 70)
    print("MNIST implicit generator with sliced ISL (DCGAN generator, GPU + AMP)")
    print(f"  data_root   : {data_root}")
    print(f"  steps       : {steps}")
    print(f"  batch_size  : {batch_size}")
    print(f"  noise_dim   : {noise_dim}")
    print(f"  ngf         : {ngf}")
    print(f"  lr          : {lr}")
    print(f"  n_slices    : {n_slices}")
    print(f"  K           : {K}")
    print(f"  cdf_bw      : {cdf_bandwidth}")
    print(f"  hist_sigma  : {hist_sigma}")
    print(f"  device      : {device}")
    print(f"  use_amp     : {use_amp}")
    print(f"  outdir      : {outdir}")
    print("=" * 70)

    torch.set_num_threads(1)

    # Data
    loader = get_mnist_loader(
        data_root=data_root,
        batch_size=batch_size,
        train=True,
        num_workers=2,
    )

    # Model: G: R^{z_dim} -> 1x28x28
    gen = DCGANGeneratorMNIST(z_dim=noise_dim, ngf=ngf).to(device)

    optimizer = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

    scaler = GradScaler(enabled=use_amp)

    losses: List[float] = []
    step = 0
    epoch = 0

    # ------------------- Training loop ------------------- #
    while step < steps:
        epoch += 1
        for x, _ in loader:
            if step >= steps:
                break

            # move to GPU with non_blocking for perf
            x = x.to(device, non_blocking=True)  # (B, 1, 28, 28), in [0,1]
            B = x.size(0)

            # Flatten to (B,784) and rescale to [-1,1]
            x_real_img = x * 2.0 - 1.0              # (B, 1, 28, 28) in [-1,1]
            x_real = x_real_img.view(B, -1)         # (B, 784)

            z = sample_latent(
                n=B,
                noise_dim=noise_dim,
                device=device,
            )  # (B, noise_dim)

            with autocast(enabled=use_amp):
                x_fake_img = gen(z)                 # (B, 1, 28, 28) in [-1,1]
                x_fake = x_fake_img.view(B, -1)     # (B, 784)

                loss = sliced_isl_random(
                    x_real,
                    x_fake,
                    m=n_slices,
                    K=K,
                    cdf_bandwidth=cdf_bandwidth,
                    hist_sigma=hist_sigma,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            losses.append(float(loss.item()))

            if step % log_every == 0 or step == 1 or step == steps:
                print(f"[step {step:6d}/{steps}] sliced ISL = {loss.item():.6f}, epoch={epoch}")

    # ------------------- Save checkpoint ------------------- #
    suffix = f"MNIST_sliced_ISL_dcgan_slices{n_slices}_K{K}_z{noise_dim}_ngf{ngf}"
    ckpt_path = outdir / f"generator_{suffix}.pt"
    torch.save(gen.state_dict(), ckpt_path)
    print(f"\nSaved generator checkpoint to {ckpt_path}")

    # ------------------- Training curve ------------------- #
    steps_arr = np.arange(1, len(losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(steps_arr, losses)
    plt.xlabel("Training step")
    plt.ylabel("Sliced ISL loss")
    plt.title(f"MNIST sliced ISL (DCGAN) training curve (m={n_slices}, K={K})")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    curve_path = outdir / f"mnist_sliced_isl_loss_curve_{suffix}.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved training curve to {curve_path}")

    # ------------------- Generate and save sample grid ------------------- #
    gen.eval()
    with torch.no_grad(), autocast(enabled=use_amp):
        n_samples = eval_n_samples
        z = sample_latent(
            n=n_samples,
            noise_dim=noise_dim,
            device=device,
        )
        x_fake_img = gen(z)  # (N, 1, 28, 28) in [-1,1]

        # Back to [0,1] for visualisation, ensure float32 for matplotlib
        imgs = (x_fake_img + 1.0) / 2.0          # still possibly float16
        imgs = torch.clamp(imgs, 0.0, 1.0)
        imgs = imgs.float()                      # <-- cast to float32

    grid = make_grid(imgs, nrow=int(np.sqrt(n_samples)) or 8, padding=2)
    grid = grid.float()                          # just to be safe
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(grid_np.squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("MNIST samples – sliced ISL (DCGAN generator)")
    plt.tight_layout()
    img_path = outdir / f"mnist_sliced_isl_samples_{suffix}.png"
    plt.savefig(img_path, dpi=200)
    plt.close()
    print(f"Saved generated samples grid to {img_path}")



# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MNIST implicit generator trained with sliced ISL (DCGAN generator, GPU + AMP)."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Directory where MNIST will be downloaded / looked for.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Number of training steps (mini-batches).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size.",
    )
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=64,
        help="Latent dimension z_dim.",
    )
    parser.add_argument(
        "--ngf",
        type=int,
        default=64,
        help="Base number of feature maps in the generator (DCGAN convention).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--n_slices",
        type=int,
        default=32,
        help="Number of random directions for sliced ISL.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=64,
        help="Number of bins in 1D ISL surrogate.",
    )
    parser.add_argument(
        "--cdf_bandwidth",
        type=float,
        default=0.15,
        help="Bandwidth for soft CDF.",
    )
    parser.add_argument(
        "--hist_sigma",
        type=float,
        default=0.05,
        help="Bandwidth for soft histogram.",
    )
    parser.add_argument(
        "--eval_n_samples",
        type=int,
        default=64,
        help="Number of samples for final grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    set_seed(args.seed, deterministic=False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but this script is configured for GPU training.")
    device = torch.device("cuda")
    print("Using device:", device)

    train_mnist_sliced_isl_dcgan(
        data_root=args.data_root,
        steps=args.steps,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        ngf=args.ngf,
        lr=args.lr,
        n_slices=args.n_slices,
        K=args.K,
        cdf_bandwidth=args.cdf_bandwidth,
        hist_sigma=args.hist_sigma,
        log_every=500,
        eval_n_samples=args.eval_n_samples,
        device=device,
    )


if __name__ == "__main__":
    main()
