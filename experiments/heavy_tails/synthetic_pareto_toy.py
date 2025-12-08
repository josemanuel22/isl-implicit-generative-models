#!/usr/bin/env python
"""
Synthetic heavy-tailed toy experiment.

We learn a 1D heavy-tailed target distribution with the 1D
Invariant Statistical Loss (ISL), comparing:

  - Different *targets*:
        - Pareto(xm, alpha)
        - Mixture of Cauchys
        - Student-t (low degrees of freedom)
        - Lognormal (large variance)

  - Different *latents*:
        - Gaussian latent  z ~ N(0, I)
        - Generalised Pareto latent  z ~ GPD(ξ, σ)

The generator is an MLP:  G: R^{noise_dim} -> R, trained with isl_1d_soft.

The script:
  1. Generates samples from a chosen heavy-tailed target.
  2. Trains a generator with either Gaussian or GPD latent.
  3. Plots real vs generated histograms (linear + log-y).
  4. Prints tail quantiles and a simple Hill tail-index estimate.
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

from isl.loss_1d import isl_1d_soft
from isl.models import MLPGenerator
from isl.utils import set_seed, get_device, ensure_dir


# ---------------------------------------------------------------------
#  Heavy-tailed *targets*
# ---------------------------------------------------------------------

def sample_true_pareto(
    n: int,
    device: torch.device,
    xm: float = 1.0,
    alpha: float = 2.0,
) -> torch.Tensor:
    """
    Sample from a 1D Pareto(xm, alpha) distribution:

        P(X > x) = (xm / x)^alpha,   x >= xm > 0
    """
    scale = torch.tensor(xm, device=device)
    shape = torch.tensor(alpha, device=device)
    dist = torch.distributions.Pareto(scale=scale, alpha=shape)
    return dist.sample((n,))


def sample_mixture_cauchy(
    n: int,
    device: torch.device,
    locs: tuple[float, ...] = (-3.0, 0.0, 3.0),
    scales: tuple[float, ...] = (0.5, 1.0, 0.5),
    probs: tuple[float, ...] = (0.3, 0.4, 0.3),
) -> torch.Tensor:
    """
    Sample from a 1D mixture of Cauchy distributions.

    X ~ sum_i probs[i] * Cauchy(locs[i], scales[i])
    """
    locs_t = torch.tensor(locs, device=device, dtype=torch.float32)      # (K,)
    scales_t = torch.tensor(scales, device=device, dtype=torch.float32)  # (K,)
    probs_t = torch.tensor(probs, device=device, dtype=torch.float32)    # (K,)
    probs_t = probs_t / probs_t.sum()

    K = locs_t.shape[0]
    cat = torch.distributions.Categorical(probs=probs_t)
    comp_idx = cat.sample((n,))  # (n,)

    base = torch.distributions.Cauchy(
        loc=torch.tensor(0.0, device=device),
        scale=torch.tensor(1.0, device=device),
    )
    z = base.sample((n,))  # (n,)

    chosen_loc = locs_t[comp_idx]
    chosen_scale = scales_t[comp_idx]
    x = chosen_loc + chosen_scale * z
    return x


def sample_student_t(
    n: int,
    device: torch.device,
    df: float = 2.5,
    loc: float = 0.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Sample from a 1D Student-t distribution:

        X = loc + scale * T(df)

    For df < 3, the variance is infinite; for df <= 1, mean is undefined.
    """
    dist = torch.distributions.StudentT(df)
    t = dist.sample((n,))          # on CPU by default
    t = t.to(device=device)
    return loc + scale * t


def sample_lognormal(
    n: int,
    device: torch.device,
    mean: float = 0.0,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Sample from a 1D lognormal distribution:

        X = exp( N(mean, sigma^2) )

    Lognormal has heavy right tail (all positive support).
    """
    dist = torch.distributions.LogNormal(mean, sigma)
    x = dist.sample((n,))          # CPU
    return x.to(device=device)


# ---------------------------------------------------------------------
#  Heavy-tailed *latents*
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
    Hill estimator for Pareto-like tail index alpha based on top-k order stats.

    For Pareto( xm, alpha ) tail, alpha_hat ~ 1 / mean( log(X_i / X_(k)) ).

    For other heavy-tailed distributions (Cauchy mix, Student-t, lognormal)
    this is not exact, but still gives an indication of tail heaviness.
    """
    x = np.asarray(x)
    x = x[x > 0]  # ensure positivity for log
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

def train_heavytail_toy(
    target_type: str = "pareto",     # "pareto", "cauchy_mix", "student_t", "lognormal"
    latent_type: str = "gaussian",   # "gaussian" or "gpd"
    steps: int = 5000,
    batch_size: int = 512,
    noise_dim: int = 4,
    hidden_dims: tuple[int, ...] = (64, 64),
    lr: float = 1e-3,
    K: int = 64,
    cdf_bandwidth: float = 0.15,
    hist_sigma: float = 0.05,
    pareto_xm: float = 1.0,
    pareto_alpha: float = 2.0,
    student_df: float = 2.5,
    student_scale: float = 1.0,
    lognorm_mean: float = 0.0,
    lognorm_sigma: float = 1.0,
    log_every: int = 200,
    device: torch.device | None = None,
    outdir: Path | None = None,
) -> None:
    """
    Train a 1D generator on a heavy-tailed target with ISL and a chosen latent.

    target_type:
      - "pareto"      : Pareto(xm, alpha)
      - "cauchy_mix"  : mixture of Cauchys
      - "student_t"   : Student-t(df, scale)
      - "lognormal"   : Lognormal(mean, sigma)
    """
    if device is None:
        device = get_device(prefer_gpu=True)
    if outdir is None:
        outdir = ROOT / "experiments" / "heavytails_toy"
    ensure_dir(outdir)

    print("=" * 70)
    print("Synthetic heavy-tailed toy with ISL")
    print(f"  target_type    : {target_type}  (pareto | cauchy_mix | student_t | lognormal)")
    print(f"  latent_type    : {latent_type}  (gaussian | gpd)")
    print(f"  steps          : {steps}")
    print(f"  batch_size     : {batch_size}")
    print(f"  noise_dim      : {noise_dim}")
    print(f"  hidden_dims    : {hidden_dims}")
    print(f"  lr             : {lr}")
    print(f"  K              : {K}")
    print(f"  cdf_bandwidth  : {cdf_bandwidth}")
    print(f"  hist_sigma     : {hist_sigma}")
    print(f"  pareto_xm      : {pareto_xm}")
    print(f"  pareto_alpha   : {pareto_alpha}")
    print(f"  student_df     : {student_df}")
    print(f"  student_scale  : {student_scale}")
    print(f"  lognorm_mean   : {lognorm_mean}")
    print(f"  lognorm_sigma  : {lognorm_sigma}")
    print(f"  device         : {device}")
    print(f"  outdir         : {outdir}")
    print("=" * 70)

    torch.set_num_threads(1)

    gen = MLPGenerator(
        noise_dim=noise_dim,
        data_dim=1,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = optim.Adam(gen.parameters(), lr=lr)
    losses: list[float] = []

    # ------------------- training loop ------------------- #
    for step in range(1, steps + 1):
        # Real samples from chosen heavy-tailed target
        if target_type == "pareto":
            x_real = sample_true_pareto(
                batch_size,
                device=device,
                xm=pareto_xm,
                alpha=pareto_alpha,
            )
        elif target_type == "cauchy_mix":
            x_real = sample_mixture_cauchy(
                batch_size,
                device=device,
            )
        elif target_type == "student_t":
            x_real = sample_student_t(
                batch_size,
                device=device,
                df=student_df,
                loc=0.0,
                scale=student_scale,
            )
        elif target_type == "lognormal":
            x_real = sample_lognormal(
                batch_size,
                device=device,
                mean=lognorm_mean,
                sigma=lognorm_sigma,
            )
        else:
            raise ValueError(f"Unknown target_type: {target_type!r}")

        # Latent noise
        z = sample_latent(
            latent_type,
            batch_size,
            noise_dim=noise_dim,
            device=device,
        )

        x_fake = gen(z).view(-1)

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
    suffix = f"target-{target_type}_latent-{latent_type}_K{K}"
    ckpt_path = outdir / f"generator_heavytail_{suffix}.pt"
    torch.save(gen.state_dict(), ckpt_path)
    print(f"\nSaved generator checkpoint to {ckpt_path}")

    # ------------------- Training curve ------------------- #
    steps_arr = np.arange(1, len(losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(steps_arr, losses)
    plt.xlabel("Training step")
    plt.ylabel("ISL loss")
    plt.title(f"Heavy-tail ISL training curve ({target_type}, {latent_type} latent)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    curve_path = outdir / f"heavytail_isl_loss_curve_{suffix}.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved training curve to {curve_path}")

    # ------------------- Evaluation: real vs generated ------------------- #
    gen.eval()
    with torch.no_grad():
        N_eval = 100000
        if target_type == "pareto":
            x_real_eval = sample_true_pareto(
                N_eval,
                device=device,
                xm=pareto_xm,
                alpha=pareto_alpha,
            ).cpu().numpy()
        elif target_type == "cauchy_mix":
            x_real_eval = sample_mixture_cauchy(
                N_eval,
                device=device,
            ).cpu().numpy()
        elif target_type == "student_t":
            x_real_eval = sample_student_t(
                N_eval,
                device=device,
                df=student_df,
                loc=0.0,
                scale=student_scale,
            ).cpu().numpy()
        elif target_type == "lognormal":
            x_real_eval = sample_lognormal(
                N_eval,
                device=device,
                mean=lognorm_mean,
                sigma=lognorm_sigma,
            ).cpu().numpy()
        else:
            raise ValueError(f"Unknown target_type: {target_type!r}")

        z_eval = sample_latent(
            latent_type,
            N_eval,
            noise_dim=noise_dim,
            device=device,
        )
        x_fake_eval = gen(z_eval).view(-1).cpu().numpy()

    print_tail_stats(f"Real ({target_type})", x_real_eval)
    print_tail_stats(f"Generated ({target_type}, {latent_type} latent)", x_fake_eval)

    # Histograms (linear scale)
    plt.figure(figsize=(6, 4))

    # Hard clip visible range for all targets
    xmin, xmax = -200.0, 200.0

    plt.hist(
        x_real_eval,
        bins=200,
        range=(xmin, xmax),  # values outside are ignored
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
    plt.ylim(bottom=0.0)  # keep y-axis sensible
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("density (hist)")
    plt.title(f"Heavy-tailed toy ({target_type}, {latent_type} latent) - linear scale")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    hist_lin_path = outdir / f"heavytail_hist_linear_{suffix}.png"
    plt.savefig(hist_lin_path, dpi=200)
    plt.close()
    print(f"Saved linear-scale histogram to {hist_lin_path}")

    # Histograms (log-y to highlight tails)
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
    plt.ylim(bottom=1e-6)  # avoid crazy huge log scale from outliers
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("density (log scale)")
    plt.title(f"Heavy-tailed toy ({target_type}, {latent_type} latent) - log-y scale")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    hist_log_path = outdir / f"heavytail_hist_logy_{suffix}.png"
    plt.savefig(hist_log_path, dpi=200)
    plt.close()
    print(f"Saved log-y histogram to {hist_log_path}")



# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic heavy-tailed toy experiment with ISL "
                    "(Pareto, mixture of Cauchys, Student-t, Lognormal)."
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="pareto",
        choices=["pareto", "cauchy_mix", "student_t", "lognormal"],
        help="Heavy-tailed target distribution.",
    )
    parser.add_argument(
        "--latent_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "gpd"],
        help="Latent noise type: Gaussian or GPD.",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--noise_dim", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--cdf_bandwidth", type=float, default=0.15)
    parser.add_argument("--hist_sigma", type=float, default=0.05)
    parser.add_argument("--pareto_xm", type=float, default=1.0)
    parser.add_argument("--pareto_alpha", type=float, default=2.0)
    parser.add_argument("--student_df", type=float, default=2.5)
    parser.add_argument("--student_scale", type=float, default=1.0)
    parser.add_argument("--lognorm_mean", type=float, default=0.0)
    parser.add_argument("--lognorm_sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    device = get_device(prefer_gpu=True)

    train_heavytail_toy(
        target_type=args.target_type,
        latent_type=args.latent_type,
        steps=args.steps,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        hidden_dims=(64, 64),
        lr=args.lr,
        K=args.K,
        cdf_bandwidth=args.cdf_bandwidth,
        hist_sigma=args.hist_sigma,
        pareto_xm=args.pareto_xm,
        pareto_alpha=args.pareto_alpha,
        student_df=args.student_df,
        student_scale=args.student_scale,
        lognorm_mean=args.lognorm_mean,
        lognorm_sigma=args.lognorm_sigma,
        log_every=200,
        device=device,
    )


if __name__ == "__main__":
    main()
