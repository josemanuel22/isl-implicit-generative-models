#!/usr/bin/env python
"""
MNIST – DCGAN pretrained with sliced ISL, then fine-tuned with GAN loss.

Stage 1 (pretrain):
    - Data: MNIST 28x28, flattened to R^784 and scaled to [-1, 1].
    - Latent: z ~ N(0, I) in R^{noise_dim}.
    - Generator: DCGAN-style convnet producing 1x28x28 images (tanh output).
    - Loss: sliced ISL on flattened images via isl.sliced.sliced_isl_random.
    - No discriminator used in this phase.

Stage 2 (GAN):
    - Initialise the same DCGAN generator with the pretrained weights.
    - Add a DCGAN-style discriminator.
    - Train a standard non-saturating GAN:
          L_D = BCE(D(x_real), 1) + BCE(D(x_fake), 0)
          L_G = BCE(D(x_fake), 1)
      (NO ISL term in this phase.)

Outputs:
    - Checkpoints for G after pretrain and after GAN fine-tuning.
    - Checkpoint for D after GAN.
    - Training curves (ISL pretrain + GAN losses).
    - Sample grids after pretrain and after GAN.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys
import argparse
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
from isl.utils import set_seed, get_device, ensure_dir


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
        B, Dz = z.shape
        if Dz != self.z_dim:
            raise ValueError(f"Expected z_dim={self.z_dim}, got {Dz}")
        z = z.view(B, Dz, 1, 1)
        return self.net(z)


# ---------------------------------------------------------------------
#  DCGAN-style discriminator for MNIST
# ---------------------------------------------------------------------

class DCGANDiscriminatorMNIST(nn.Module):
    """
    DCGAN-style discriminator for 28x28 grayscale images.

    Input:  x of shape (B, 1, 28, 28) in [-1, 1]
    Output: logits of shape (B, 1)
    """

    def __init__(self, ndf: int = 64) -> None:
        super().__init__()
        self.ndf = ndf

        self.net = nn.Sequential(
            # (B, 1, 28, 28) -> (B, ndf, 14, 14)
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, ndf, 14, 14) -> (B, 2*ndf, 7, 7)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 2*ndf, 7, 7) -> (B, 4*ndf, 4, 4)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 4*ndf, 4, 4) -> (B, 1, 1, 1)
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)          # (B, 1, 1, 1)
        return out.view(-1, 1)     # (B, 1) logits


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
    Rescaling to [-1,1] and flattening is done in the training loop.
    """
    transform = transforms.ToTensor()  # (1, 28, 28) in [0,1]

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

def train_mnist_isl_pretrain_then_gan(
    data_root: str | Path,
    pretrain_steps: int = 20000,
    gan_steps: int = 30000,
    batch_size: int = 128,
    noise_dim: int = 64,
    ngf: int = 64,
    ndf: int = 64,
    lr_pretrain: float = 2e-4,
    lr_gan_g: float = 2e-4,
    lr_gan_d: float = 2e-4,
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
    Two-stage training:
      1) Pretrain DCGAN generator with sliced ISL only.
      2) Fine-tune the same generator with standard DCGAN GAN loss.
    """
    if device is None:
        device = get_device(prefer_gpu=True)
    if outdir is None:
        outdir = ROOT / "experiments" / "mnist_isl_plus_gan"
    ensure_dir(outdir)

    print("=" * 70)
    print("MNIST DCGAN – Stage 1: sliced ISL pretraining, Stage 2: GAN fine-tuning")
    print(f"  data_root      : {data_root}")
    print(f"  pretrain_steps : {pretrain_steps}")
    print(f"  gan_steps      : {gan_steps}")
    print(f"  batch_size     : {batch_size}")
    print(f"  noise_dim      : {noise_dim}")
    print(f"  ngf            : {ngf}")
    print(f"  ndf            : {ndf}")
    print(f"  lr_pretrain    : {lr_pretrain}")
    print(f"  lr_gan_g       : {lr_gan_g}")
    print(f"  lr_gan_d       : {lr_gan_d}")
    print(f"  n_slices       : {n_slices}")
    print(f"  K              : {K}")
    print(f"  cdf_bw         : {cdf_bandwidth}")
    print(f"  hist_sigma     : {hist_sigma}")
    print(f"  device         : {device}")
    print(f"  outdir         : {outdir}")
    print("=" * 70)

    torch.set_num_threads(1)

    # Data
    loader = get_mnist_loader(
        data_root=data_root,
        batch_size=batch_size,
        train=True,
        num_workers=2,
    )

    # ------------------------------------------------------------------
    # Stage 1: Pretrain generator with sliced ISL
    # ------------------------------------------------------------------
    print("\n=== Stage 1: ISL pretraining of the generator ===")

    G = DCGANGeneratorMNIST(z_dim=noise_dim, ngf=ngf).to(device)
    opt_G_pre = optim.Adam(G.parameters(), lr=lr_pretrain, betas=(0.5, 0.999))

    isl_losses: List[float] = []
    step = 0
    epoch = 0

    while step < pretrain_steps:
        epoch += 1
        for x, _ in loader:
            if step >= pretrain_steps:
                break

            x = x.to(device, non_blocking=True)    # (B, 1, 28, 28) in [0,1]
            B = x.size(0)

            # Rescale to [-1,1] and flatten
            x_real_img = x * 2.0 - 1.0             # (B, 1, 28, 28)
            x_real = x_real_img.view(B, -1)        # (B, 784)

            z = sample_latent(B, noise_dim, device=device)
            x_fake_img = G(z)                      # (B, 1, 28, 28)
            x_fake = x_fake_img.view(B, -1)        # (B, 784)

            loss_isl = sliced_isl_random(
                x_real,
                x_fake,
                m=n_slices,
                K=K,
                cdf_bandwidth=cdf_bandwidth,
                hist_sigma=hist_sigma,
            )

            opt_G_pre.zero_grad()
            loss_isl.backward()
            opt_G_pre.step()

            step += 1
            isl_losses.append(float(loss_isl.item()))

            if step % log_every == 0 or step == 1 or step == pretrain_steps:
                print(
                    f"[Stage1 step {step:6d}/{pretrain_steps}] "
                    f"sliced ISL = {loss_isl.item():.6f}, epoch={epoch}"
                )

    # Save pretrained generator
    pre_suffix = f"MNIST_pretrain_ISL_dcgan_slices{n_slices}_K{K}_z{noise_dim}_ngf{ngf}"
    ckpt_G_pre = outdir / f"generator_pretrained_{pre_suffix}.pt"
    torch.save(G.state_dict(), ckpt_G_pre)
    print(f"\nSaved ISL-pretrained generator checkpoint to {ckpt_G_pre}")

    # Plot ISL training curve
    steps_arr_pre = np.arange(1, len(isl_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(steps_arr_pre, isl_losses)
    plt.xlabel("Pretrain step")
    plt.ylabel("Sliced ISL loss")
    plt.title(f"MNIST DCGAN – ISL pretraining (m={n_slices}, K={K})")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    curve_pre_path = outdir / f"mnist_isl_pretraining_curve_{pre_suffix}.png"
    plt.savefig(curve_pre_path, dpi=200)
    plt.close()
    print(f"Saved ISL pretraining curve to {curve_pre_path}")

    # Sample grid after ISL pretraining
    G.eval()
    with torch.no_grad():
        n_samples = eval_n_samples
        z = sample_latent(n_samples, noise_dim, device=device)
        x_fake_img = G(z)                          # in [-1,1]
        imgs = (x_fake_img + 1.0) / 2.0
        imgs = torch.clamp(imgs, 0.0, 1.0)

        grid = make_grid(imgs, nrow=int(np.sqrt(n_samples)) or 8, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(grid_np.squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("MNIST samples – after ISL pretraining")
    plt.tight_layout()
    img_pre_path = outdir / f"mnist_isl_pretraining_samples_{pre_suffix}.png"
    plt.savefig(img_pre_path, dpi=200)
    plt.close()
    print(f"Saved ISL-pretrained samples grid to {img_pre_path}")

    # ------------------------------------------------------------------
    # Stage 2: Standard DCGAN training (GAN loss only), init G from pretrain
    # ------------------------------------------------------------------
    print("\n=== Stage 2: DCGAN adversarial training from ISL-pretrained G ===")

    # Reuse G with pretrained weights (already in memory)
    G.train()
    D = DCGANDiscriminatorMNIST(ndf=ndf).to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr_gan_g, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_gan_d, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()

    d_losses: List[float] = []
    g_losses: List[float] = []

    step = 0
    epoch = 0

    while step < gan_steps:
        epoch += 1
        for x, _ in loader:
            if step >= gan_steps:
                break

            x = x.to(device, non_blocking=True)
            B = x.size(0)

            # Rescale real to [-1,1]
            x_real = x * 2.0 - 1.0                 # (B, 1, 28, 28)

            # ---------------------- Update D ---------------------- #
            D.train()
            G.train()

            z = sample_latent(B, noise_dim, device=device)
            with torch.no_grad():
                x_fake = G(z)                      # (B, 1, 28, 28)

            logits_real = D(x_real)
            logits_fake = D(x_fake.detach())

            real_targets = torch.ones_like(logits_real, device=device)
            fake_targets = torch.zeros_like(logits_fake, device=device)

            loss_D_real = bce(logits_real, real_targets)
            loss_D_fake = bce(logits_fake, fake_targets)
            loss_D = loss_D_real + loss_D_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---------------------- Update G (GAN loss only) ---------------------- #
            z = sample_latent(B, noise_dim, device=device)
            x_fake = G(z)
            logits_fake_for_G = D(x_fake)

            adv_targets = torch.ones_like(logits_fake_for_G, device=device)
            loss_G = bce(logits_fake_for_G, adv_targets)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # Logging
            step += 1
            d_losses.append(float(loss_D.item()))
            g_losses.append(float(loss_G.item()))

            if step % log_every == 0 or step == 1 or step == gan_steps:
                print(
                    f"[Stage2 step {step:6d}/{gan_steps}] "
                    f"D_loss={loss_D.item():.4f} | "
                    f"G_loss={loss_G.item():.4f} | epoch={epoch}"
                )

    # Save final checkpoints
    gan_suffix = (
        f"MNIST_dcgan_from_pretrained_z{noise_dim}_ngf{ngf}_ndf{ndf}"
        f"_pre{pretrain_steps}_gan{gan_steps}"
    )
    ckpt_G_gan = outdir / f"generator_gan_{gan_suffix}.pt"
    ckpt_D_gan = outdir / f"discriminator_gan_{gan_suffix}.pt"
    torch.save(G.state_dict(), ckpt_G_gan)
    torch.save(D.state_dict(), ckpt_D_gan)
    print(f"\nSaved GAN-fine-tuned generator checkpoint to {ckpt_G_gan}")
    print(f"Saved discriminator checkpoint to {ckpt_D_gan}")

    # Plot GAN training curves
    steps_arr_gan = np.arange(1, len(d_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(steps_arr_gan, d_losses, label="D loss")
    plt.plot(steps_arr_gan, g_losses, label="G loss")
    plt.xlabel("GAN step")
    plt.ylabel("Loss")
    plt.title("MNIST DCGAN – adversarial training (from ISL-pretrained G)")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    curve_gan_path = outdir / f"mnist_gan_from_pretrained_curve_{gan_suffix}.png"
    plt.savefig(curve_gan_path, dpi=200)
    plt.close()
    print(f"Saved GAN training curves to {curve_gan_path}")

    # Sample grid after GAN fine-tuning
    G.eval()
    with torch.no_grad():
        n_samples = eval_n_samples
        z = sample_latent(n_samples, noise_dim, device=device)
        x_fake = G(z)                              # (N, 1, 28, 28) in [-1,1]
        imgs = (x_fake + 1.0) / 2.0
        imgs = torch.clamp(imgs, 0.0, 1.0)

        grid = make_grid(imgs, nrow=int(np.sqrt(n_samples)) or 8, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(grid_np.squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("MNIST samples – after GAN fine-tuning (from ISL pretrain)")
    plt.tight_layout()
    img_gan_path = outdir / f"mnist_gan_from_pretrained_samples_{gan_suffix}.png"
    plt.savefig(img_gan_path, dpi=200)
    plt.close()
    print(f"Saved GAN-fine-tuned samples grid to {img_gan_path}")


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MNIST DCGAN: pretrain generator with sliced ISL, "
                    "then fine-tune with standard GAN loss."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Directory where MNIST will be downloaded / looked for.",
    )
    parser.add_argument(
        "--pretrain_steps",
        type=int,
        default=20000,
        help="Number of ISL pretraining steps for the generator.",
    )
    parser.add_argument(
        "--gan_steps",
        type=int,
        default=30000,
        help="Number of DCGAN adversarial training steps.",
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
        help="Base number of feature maps in the generator.",
    )
    parser.add_argument(
        "--ndf",
        type=int,
        default=64,
        help="Base number of feature maps in the discriminator.",
    )
    parser.add_argument(
        "--lr_pretrain",
        type=float,
        default=2e-4,
        help="Learning rate for ISL pretraining of G.",
    )
    parser.add_argument(
        "--lr_gan_g",
        type=float,
        default=2e-4,
        help="Learning rate for G during GAN stage.",
    )
    parser.add_argument(
        "--lr_gan_d",
        type=float,
        default=2e-4,
        help="Learning rate for D during GAN stage.",
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
        help="Number of samples for each sample grid.",
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

    device = get_device(prefer_gpu=True)
    print("Using device:", device)

    train_mnist_isl_pretrain_then_gan(
        data_root=args.data_root,
        pretrain_steps=args.pretrain_steps,
        gan_steps=args.gan_steps,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        ngf=args.ngf,
        ndf=args.ndf,
        lr_pretrain=args.lr_pretrain,
        lr_gan_g=args.lr_gan_g,
        lr_gan_d=args.lr_gan_d,
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
