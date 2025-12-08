#!/usr/bin/env python
"""
1D ISL experiments: approximation and gradient alignment vs K.

We study how the 1D soft ISL surrogate behaves as a function of the number
of bins K, both in terms of:
  - loss approximation (ISL_K vs ISL_{K_ref}),
  - gradient alignment with respect to a model parameter θ.

The script supports multiple 1D toy problems via modular "example" factories:

  - Gaussian mean shift:
        X_real ~ N(0, 1)
        X_fake ~ N(θ, 1)

  - Symmetric 2-component Gaussian mixture:
        X_real ~ 0.5 N(-δ_true, 1) + 0.5 N(+δ_true, 1)
        X_fake ~ 0.5 N(-θ, 1) + 0.5 N(+θ, 1)

Core experiments (problem-agnostic skeleton):

  - experiment_loss_vs_K:
        uses a sample_fn that returns (x_real, x_fake) for a fixed θ0,
        then computes ISL_K and compares to ISL_{K_ref}.

  - experiment_grad_alignment_vs_K:
        uses a make_loss_fn that returns a closure:
            loss_fn(theta, K) = ISL_K(theta)
        and compares gradients g_K(θ) to g_{K_ref}(θ) on a θ-grid.

All plots are saved into plots_1d_isl/.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
#  Make sure 'isl' is importable from src/ layout
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]   # .../isl-implicit-generative-models
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isl.loss_1d import isl_1d_soft
from isl.utils import set_seed


# ---------------------------------------------------------------------
#  1D data generators for loss-vs-K experiments
# ---------------------------------------------------------------------

def make_samples_gaussian(
    N_samples: int,
    device: torch.device,
    theta0: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1D Gaussian mean-shift example:

        X_real ~ N(0, 1)
        X_fake ~ N(theta0, 1)

    Returns fixed (x_real, x_fake) that can be reused for all K.
    """
    eps_real = torch.randn(N_samples, device=device)
    eps_fake = torch.randn(N_samples, device=device)
    theta0_t = torch.tensor(theta0, device=device, dtype=torch.float32)

    x_real = eps_real
    x_fake = theta0_t + eps_fake
    return x_real, x_fake


def make_samples_mixture(
    N_samples: int,
    device: torch.device,
    theta0: float,
    delta_true: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric 2-component Gaussian mixture example:

    Target (real):
        X_real ~ 0.5 * N(-delta_true, 1) + 0.5 * N(+delta_true, 1).

    Model (fake):
        X_fake(theta0) ~ 0.5 * N(-theta0, 1) + 0.5 * N(+theta0, 1).

    We pre-sample component labels and noise so (x_real, x_fake) are fixed
    for a given (N_samples, device, theta0, delta_true).
    """
    N = N_samples

    # Component signs in {-1, +1}
    s_real = torch.where(
        torch.rand(N, device=device) < 0.5,
        torch.full((N,), -1.0, device=device),
        torch.full((N,), +1.0, device=device),
    )
    s_fake = torch.where(
        torch.rand(N, device=device) < 0.5,
        torch.full((N,), -1.0, device=device),
        torch.full((N,), +1.0, device=device),
    )

    eps_real = torch.randn(N, device=device)
    eps_fake = torch.randn(N, device=device)

    delta_true_t = torch.tensor(delta_true, device=device, dtype=torch.float32)
    theta0_t = torch.tensor(theta0, device=device, dtype=torch.float32)

    x_real = s_real * delta_true_t + eps_real
    x_fake = s_fake * theta0_t + eps_fake
    return x_real, x_fake


# ---------------------------------------------------------------------
#  Generic loss-vs-K experiment (problem-agnostic skeleton)
# ---------------------------------------------------------------------

def experiment_loss_vs_K(
    theta0: float = 1.0,
    K_list: list[int] | None = None,
    K_ref: int = 256,
    N_samples: int = 5000,
    cdf_bandwidth: float = 0.15,
    hist_sigma: float = 0.05,
    device: torch.device | None = None,
    outdir: Path | None = None,
    sample_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = make_samples_gaussian,
    sample_kwargs: dict | None = None,
    experiment_name: str = "Gaussian mean",
    plot_suffix: str = "",
) -> None:
    """
    Generic 1D ISL approximation vs K experiment.

    All the "problem-specific" stuff (Gaussian, mixture, etc.) is encapsulated
    in `sample_fn`, which must have signature:

        x_real, x_fake = sample_fn(N_samples, device, theta0, **sample_kwargs)

    The rest of the skeleton (compute ISL_K, compare to K_ref, plot error vs K)
    is independent of the chosen example.
    """
    if device is None:
        device = torch.device("cpu")
    if outdir is None:
        outdir = Path("plots_1d_isl")
    outdir.mkdir(parents=True, exist_ok=True)

    if K_list is None:
        K_list = [2, 4, 8, 16, 32, 64, 128]
    if sample_kwargs is None:
        sample_kwargs = {}

    print("=" * 70)
    print(f"Experiment 1: 1D ISL approximation vs K ({experiment_name})")
    print(f"  theta0       : {theta0}")
    print(f"  N_samples    : {N_samples}")
    print(f"  K_ref        : {K_ref}")
    print(f"  K_list       : {K_list}")
    print(f"  device       : {device}")
    print(f"  sample_fn    : {sample_fn.__name__}")
    if sample_kwargs:
        print(f"  sample_kwargs: {sample_kwargs}")
    print("=" * 70)

    # --- Problem-specific sampling (modular) ---
    x_real, x_fake = sample_fn(
        N_samples=N_samples,
        device=device,
        theta0=theta0,
        **sample_kwargs,
    )
    # ------------------------------------------

    # Reference loss with large K_ref
    with torch.no_grad():
        loss_ref = isl_1d_soft(
            x_real,
            x_fake,
            K=K_ref,
            cdf_bandwidth=cdf_bandwidth,
            hist_sigma=hist_sigma,
            reduction="mean",
        ).item()

    print(f"Reference ISL (K={K_ref}): {loss_ref:.6e}\n")

    abs_errors = []
    losses_K = []

    for K in K_list:
        with torch.no_grad():
            loss_K = isl_1d_soft(
                x_real,
                x_fake,
                K=K,
                cdf_bandwidth=cdf_bandwidth,
                hist_sigma=hist_sigma,
                reduction="mean",
            ).item()

        losses_K.append(loss_K)
        abs_err = abs(loss_K - loss_ref)
        abs_errors.append(abs_err)
        print(f"K={K:3d}  ISL_K={loss_K:.6e}  |ISL_K - ISL_ref|={abs_err:.3e}")

    # Plot: absolute error vs K (log-log)
    K_array = np.array(K_list, dtype=np.float64)
    err_array = np.array(abs_errors, dtype=np.float64)

    plt.figure(figsize=(6, 4))
    plt.loglog(K_array, err_array, marker="o")
    plt.xlabel("K (number of bins)")
    plt.ylabel(r"|ISL$_K$ - ISL$_{K_{ref}}$|")
    plt.title(f"1D ISL approximation vs K ({experiment_name})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    suffix = f"_{plot_suffix}" if plot_suffix else ""
    fig_path = outdir / f"isl_1d_loss_vs_K{suffix}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"\nSaved loss-vs-K plot to {fig_path}\n")


# ---------------------------------------------------------------------
#  Loss closures for gradient experiments
# ---------------------------------------------------------------------

def make_loss_fn_gaussian(
    N_samples: int,
    device: torch.device,
    cdf_bandwidth: float,
    hist_sigma: float,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """
    Build a loss_fn(theta, K) for the Gaussian mean-shift example:

        X_real ~ N(0, 1)
        X_fake(theta) = theta + eps_fake

    We fix eps_real, eps_fake once to reduce Monte Carlo noise.
    """
    eps_real = torch.randn(N_samples, device=device)
    eps_fake = torch.randn(N_samples, device=device)

    def loss_fn(theta: torch.Tensor, K: int) -> torch.Tensor:
        x_real = eps_real
        x_fake = theta + eps_fake
        loss = isl_1d_soft(
            x_real,
            x_fake,
            K=K,
            cdf_bandwidth=cdf_bandwidth,
            hist_sigma=hist_sigma,
            reduction="mean",
        )
        return loss

    return loss_fn


def make_loss_fn_mixture(
    N_samples: int,
    device: torch.device,
    cdf_bandwidth: float,
    hist_sigma: float,
    delta_true: float = 1.5,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """
    Build a loss_fn(theta, K) for the symmetric 2-component Gaussian mixture:

    Target:
        X_real ~ 0.5 * N(-delta_true, 1) + 0.5 * N(+delta_true, 1).

    Model:
        X_fake(theta) ~ 0.5 * N(-theta, 1) + 0.5 * N(+theta, 1).

    We fix component labels and noise once to reduce Monte Carlo noise.
    """
    N = N_samples

    s_real = torch.where(
        torch.rand(N, device=device) < 0.5,
        torch.full((N,), -1.0, device=device),
        torch.full((N,), +1.0, device=device),
    )
    s_fake = torch.where(
        torch.rand(N, device=device) < 0.5,
        torch.full((N,), -1.0, device=device),
        torch.full((N,), +1.0, device=device),
    )

    eps_real = torch.randn(N, device=device)
    eps_fake = torch.randn(N, device=device)

    delta_true_t = torch.tensor(delta_true, device=device, dtype=torch.float32)
    x_real = s_real * delta_true_t + eps_real

    def loss_fn(theta: torch.Tensor, K: int) -> torch.Tensor:
        x_fake = s_fake * theta + eps_fake
        loss = isl_1d_soft(
            x_real,
            x_fake,
            K=K,
            cdf_bandwidth=cdf_bandwidth,
            hist_sigma=hist_sigma,
            reduction="mean",
        )
        return loss

    return loss_fn


# ---------------------------------------------------------------------
#  Generic gradient-alignment-vs-K experiment (problem-agnostic)
# ---------------------------------------------------------------------

def experiment_grad_alignment_vs_K(
    theta_min: float = -1.0,
    theta_max: float = 1.0,
    n_thetas: int = 11,
    K_list: list[int] | None = None,
    K_ref: int = 256,
    N_samples: int = 3000,
    cdf_bandwidth: float = 0.15,
    hist_sigma: float = 0.05,
    device: torch.device | None = None,
    outdir: Path | None = None,
    relerr_threshold: float = 1e-4,
    make_loss_fn: Callable[..., Callable[[torch.Tensor, int], torch.Tensor]] = make_loss_fn_gaussian,
    make_loss_kwargs: dict | None = None,
    experiment_name: str = "Gaussian mean",
    plot_suffix: str = "",
) -> None:
    """
    Generic gradient alignment vs K experiment.

    We consider a grid of theta values in [theta_min, theta_max], and for each
    K (including K_ref) we estimate:

        g_K(theta) = d/dtheta ISL_K(theta),

    using a loss_fn(theta, K) produced by make_loss_fn.

    We then compare g_K to g_{K_ref} using:
      - cosine similarity (direction),
      - L2 error ||g_K - g_ref||_2,
      - mean relative error over all theta,
      - mean relative error restricted to |g_ref| > relerr_threshold.
    """
    if device is None:
        device = torch.device("cpu")
    if outdir is None:
        outdir = Path("plots_1d_isl")
    outdir.mkdir(parents=True, exist_ok=True)

    if K_list is None:
        K_list = [2, 4, 8, 16, 32, 64, 128]
    if make_loss_kwargs is None:
        make_loss_kwargs = {}

    print("=" * 70)
    print(f"Experiment 2: gradient alignment vs K ({experiment_name})")
    print(f"  theta range        : [{theta_min}, {theta_max}] with {n_thetas} points")
    print(f"  N_samples          : {N_samples}")
    print(f"  K_ref              : {K_ref}")
    print(f"  K_list             : {K_list}")
    print(f"  device             : {device}")
    print(f"  relerr_threshold   : {relerr_threshold}")
    print(f"  make_loss_fn       : {make_loss_fn.__name__}")
    if make_loss_kwargs:
        print(f"  make_loss_kwargs   : {make_loss_kwargs}")
    print("=" * 70)

    # Build loss_fn(theta, K) with fixed underlying randomness
    loss_fn = make_loss_fn(
        N_samples=N_samples,
        device=device,
        cdf_bandwidth=cdf_bandwidth,
        hist_sigma=hist_sigma,
        **make_loss_kwargs,
    )

    # Grid of theta values
    thetas = torch.linspace(theta_min, theta_max, steps=n_thetas, device=device)

    def estimate_grad(theta_val: float, K: int) -> float:
        """
        Estimate d/dtheta ISL_K(theta) at theta_val using autograd and loss_fn.
        """
        theta = torch.tensor(theta_val, device=device, dtype=torch.float32, requires_grad=True)
        loss = loss_fn(theta, K)
        loss.backward()
        grad_val = theta.grad.detach().item()
        return grad_val

    # Reference gradients for K_ref
    grads_ref = []
    for t in thetas:
        g = estimate_grad(float(t.item()), K_ref)
        grads_ref.append(g)
    grads_ref = torch.tensor(grads_ref, device=device, dtype=torch.float32)

    print("Reference gradient vector (K_ref):")
    print(grads_ref)

    cos_sims = []
    mean_rel_errors_all = []
    mean_rel_errors_thresh = []
    l2_errors = []

    for K in K_list:
        grads_K = []
        for t in thetas:
            gk = estimate_grad(float(t.item()), K)
            grads_K.append(gk)
        grads_K = torch.tensor(grads_K, device=device, dtype=torch.float32)

        # Cosine similarity
        ref_norm = torch.norm(grads_ref)
        K_norm = torch.norm(grads_K)
        if ref_norm.item() == 0 or K_norm.item() == 0:
            cos_sim = torch.tensor(0.0, device=device)
        else:
            cos_sim = torch.dot(grads_ref, grads_K) / (ref_norm * K_norm)

        # L2 error
        diff = grads_K - grads_ref
        l2_err = torch.norm(diff)

        # Mean relative error over all entries
        eps = 1e-8
        rel_errors_all = torch.abs(diff) / (torch.abs(grads_ref) + eps)
        mean_rel_all = rel_errors_all.mean()

        # Thresholded mean relative error: only where |g_ref| > relerr_threshold
        mask = torch.abs(grads_ref) > relerr_threshold
        if mask.any():
            rel_errors_th = rel_errors_all[mask]
            mean_rel_th = rel_errors_th.mean()
        else:
            mean_rel_th = torch.tensor(float("nan"), device=device)

        cos_sims.append(float(cos_sim.item()))
        l2_errors.append(float(l2_err.item()))
        mean_rel_errors_all.append(float(mean_rel_all.item()))
        mean_rel_errors_thresh.append(float(mean_rel_th.item()))

        print(
            f"K={K:3d}  cos_sim={cos_sim.item():.4f}  "
            f"L2_err={l2_err.item():.4e}  "
            f"mean_rel_all={mean_rel_all.item():.4f}  "
            f"mean_rel_thr={mean_rel_th.item():.4f}"
        )

    # Convert to numpy for plotting
    K_array = np.array(K_list, dtype=np.float64)
    cos_array = np.array(cos_sims, dtype=np.float64)
    l2_array = np.array(l2_errors, dtype=np.float64)
    mre_thr_array = np.array(mean_rel_errors_thresh, dtype=np.float64)

    suffix = f"_{plot_suffix}" if plot_suffix else ""

    # Plot cosine similarity vs K
    plt.figure(figsize=(6, 4))
    plt.semilogx(K_array, cos_array, marker="o")
    plt.xlabel("K (number of bins)")
    plt.ylabel("cosine similarity (grad_K, grad_ref)")
    plt.title(f"Gradient alignment vs K ({experiment_name})")
    plt.ylim(-0.1, 1.05)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    fig_path_cos = outdir / f"isl_1d_grad_cosine_vs_K{suffix}.png"
    plt.savefig(fig_path_cos, dpi=200)
    plt.close()

    # Plot L2 error vs K
    plt.figure(figsize=(6, 4))
    plt.loglog(K_array, l2_array, marker="o")
    plt.xlabel("K (number of bins)")
    plt.ylabel(r"$\|g_K - g_{ref}\|_2$")
    plt.title(f"Gradient L2 error vs K ({experiment_name})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    fig_path_l2 = outdir / f"isl_1d_grad_L2_vs_K{suffix}.png"
    plt.savefig(fig_path_l2, dpi=200)
    plt.close()

    # Plot thresholded mean relative error vs K
    plt.figure(figsize=(6, 4))
    plt.loglog(K_array, mre_thr_array, marker="o")
    plt.xlabel("K (number of bins)")
    plt.ylabel(f"mean relative error (|g_ref| > {relerr_threshold})")
    plt.title(f"Gradient relative error vs K ({experiment_name}, thresholded)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    fig_path_mre_thr = outdir / f"isl_1d_grad_relerror_thresh_vs_K{suffix}.png"
    plt.savefig(fig_path_mre_thr, dpi=200)
    plt.close()

    print("\nSaved gradient alignment plots to:")
    print(f"  {fig_path_cos}")
    print(f"  {fig_path_l2}")
    print(f"  {fig_path_mre_thr}\n")


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------

def main() -> None:
    # Reproducible seed
    set_seed(42, deterministic=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path("plots_1d_isl")

    # -------------------------------------------------------------
    # 1) Gaussian mean example: loss vs K
    # -------------------------------------------------------------
    experiment_loss_vs_K(
        theta0=1.0,
        K_list=[2, 4, 8, 16, 32, 64, 128],
        K_ref=256,
        N_samples=5000,
        cdf_bandwidth=0.15,
        hist_sigma=0.05,
        device=device,
        outdir=outdir,
        sample_fn=make_samples_gaussian,
        sample_kwargs={},
        experiment_name="Gaussian mean",
        plot_suffix="gaussian",
    )

    # -------------------------------------------------------------
    # 2) Gaussian mean example: gradient alignment vs K
    # -------------------------------------------------------------
    experiment_grad_alignment_vs_K(
        theta_min=-1.0,
        theta_max=1.0,
        n_thetas=11,
        K_list=[2, 4, 8, 16, 32, 64, 128],
        K_ref=256,
        N_samples=3000,
        cdf_bandwidth=0.15,
        hist_sigma=0.05,
        device=device,
        outdir=outdir,
        relerr_threshold=1e-4,
        make_loss_fn=make_loss_fn_gaussian,
        make_loss_kwargs={},
        experiment_name="Gaussian mean",
        plot_suffix="gaussian",
    )

    # -------------------------------------------------------------
    # 3) Mixture-of-Gaussians example: loss vs K
    # -------------------------------------------------------------
    experiment_loss_vs_K(
        theta0=1.0,   # model separation
        K_list=[2, 4, 8, 16, 32, 64, 128],
        K_ref=256,
        N_samples=5000,
        cdf_bandwidth=0.15,
        hist_sigma=0.05,
        device=device,
        outdir=outdir,
        sample_fn=make_samples_mixture,
        sample_kwargs={"delta_true": 1.5},
        experiment_name="Mixture of Gaussians",
        plot_suffix="mixture",
    )

    # -------------------------------------------------------------
    # 4) Mixture-of-Gaussians example: gradient alignment vs K
    # -------------------------------------------------------------
    experiment_grad_alignment_vs_K(
        theta_min=0.5,
        theta_max=2.5,
        n_thetas=11,
        K_list=[2, 4, 8, 16, 32, 64, 128],
        K_ref=256,
        N_samples=3000,
        cdf_bandwidth=0.15,
        hist_sigma=0.05,
        device=device,
        outdir=outdir,
        relerr_threshold=1e-4,
        make_loss_fn=make_loss_fn_mixture,
        make_loss_kwargs={"delta_true": 1.5},
        experiment_name="Mixture of Gaussians",
        plot_suffix="mixture",
    )


if __name__ == "__main__":
    main()
