"""
One-dimensional Invariant Statistical Loss (ISL) and differentiable surrogates.

This module implements the core 1D ISL objective used in our work:
given real samples x_real ~ μ and model samples x_fake ~ ν, we construct a
rank-based histogram on [0,1] and compare it to the uniform distribution.

High-level idea
---------------
If ν = μ, then (roughly) the empirical CDF of x_fake evaluated at x_real,
    u_i = F_fake(x_real[i]),
is approximately uniform on [0,1]. ISL measures deviations from uniformity of
the rank / CDF values {u_i}. The 1D loss can then be extended to the
multidimensional case via slicing.

We provide:

- isl_1d:
    A "hard" ISL implementation that uses the empirical CDF and a hard
    histogram. This is non-differentiable w.r.t. x_fake (and thus not
    suitable for gradient-based training, but useful for evaluation).

- isl_1d_soft:
    A differentiable surrogate that uses a smooth CDF (sigmoid kernel) and
    a soft histogram (Gaussian kernels on bin centers). This can be used
    as a training objective for implicit generative models.

Both functions operate on 1D tensors and are implemented in PyTorch.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .ranks import empirical_cdf, soft_cdf


def isl_1d(
    x_real: Tensor,
    x_fake: Tensor,
    K: int = 32,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """
    Hard 1D ISL loss between real and fake samples.

    This implementation:
      1. Computes the empirical CDF of x_fake at the real points x_real:
           u_i = F_{fake}(x_real[i]) ∈ [0,1].
      2. Builds a hard histogram of {u_i} over K equal-width bins on [0,1].
      3. Normalises the histogram to sum to 1.
      4. Computes an L2-type discrepancy to the uniform histogram 1/K.

    Parameters
    ----------
    x_real : Tensor, shape (N,)
        Real (data) samples.
    x_fake : Tensor, shape (M,)
        Model / generator samples.
    K : int, default=32
        Number of bins in the [0,1] histogram. Larger K increases resolution
        but also variance and cost.
    reduction : {"mean", "sum", "none"}, default="mean"
        Reduction over bins:
        - "mean": return mean squared difference over bins (scalar),
        - "sum":  return sum of squared differences,
        - "none": return per-bin squared differences as a length-K tensor.

    Returns
    -------
    loss : Tensor
        If reduction != "none", a scalar tensor. Otherwise a tensor of shape
        (K,) with per-bin contributions.

    Notes
    -----
    This is non-differentiable w.r.t x_fake because of the hard histogramming.
    It is primarily intended for evaluation / plotting, not for training.
    """
    if x_real.ndim != 1 or x_fake.ndim != 1:
        raise ValueError("isl_1d expects 1D tensors x_real, x_fake")

    if K <= 0:
        raise ValueError("K must be a positive integer")

    # Step 1: empirical CDF of fake at real points -> u in [0,1]
    u = empirical_cdf(x_real, x_fake)  # shape (N,)

    # Step 2: hard histogram over [0,1]
    N = u.shape[0]
    # Map u ∈ [0,1] to bin indices in {0, ..., K-1}
    # Values exactly at 1 are clipped to K-1.
    bin_idx = torch.clamp((u * K).long(), min=0, max=K - 1)  # (N,)
    hist = torch.bincount(bin_idx, minlength=K).to(dtype=x_real.dtype)  # (K,)

    # Step 3: normalise to sum to 1
    if N > 0:
        hist = hist / float(N)
    else:
        # Edge case: no real samples, hist is undefined; return zeros.
        hist = torch.zeros(K, dtype=x_real.dtype, device=x_real.device)

    # Step 4: compare to uniform
    uniform = torch.full_like(hist, 1.0 / K)
    sq_diff = (hist - uniform) ** 2  # (K,)

    if reduction == "none":
        return sq_diff
    elif reduction == "sum":
        return sq_diff.sum()
    elif reduction == "mean":
        return sq_diff.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


def isl_1d_soft(
    x_real: Tensor,
    x_fake: Tensor,
    K: int = 32,
    cdf_bandwidth: float = 0.1,
    hist_sigma: float = 0.05,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """
    Differentiable 1D ISL-like loss between real and fake samples.

    This is the smooth surrogate used for training:
      1. Replace the hard empirical CDF with a smooth version (soft_cdf)
         using a sigmoid kernel:
             F_soft(x_i) ≈ (1/M) ∑_j sigmoid((x_i - x_fake_j)/h).
      2. Replace the hard histogram over [0,1] by a soft histogram using
         Gaussian kernels around bin centers.
      3. Compute an L1 discrepancy between the soft histogram and the
         uniform distribution.

    Parameters
    ----------
    x_real : Tensor, shape (N,)
        Real (data) samples. Gradients do not typically flow here.
    x_fake : Tensor, shape (M,)
        Model / generator samples. Gradients will flow w.r.t. x_fake.
    K : int, default=32
        Number of soft histogram bins in [0,1].
    cdf_bandwidth : float, default=0.1
        Smoothing parameter for the soft CDF. Smaller values make it closer
        to the hard empirical CDF but may lead to vanishing / exploding
        gradients.
    hist_sigma : float, default=0.05
        Smoothing parameter for the Gaussian kernels used in the soft
        histogram on [0,1].
    reduction : {"mean", "sum", "none"}, default="mean"
        Reduction over bins:
        - "mean": mean absolute difference,
        - "sum":  sum of absolute differences,
        - "none": return per-bin absolute differences, shape (K,).

    Returns
    -------
    loss : Tensor
        If reduction != "none", a scalar tensor. Otherwise a tensor of shape
        (K,) with per-bin contributions.

    Notes
    -----
    This function is fully differentiable w.r.t. x_fake, and by extension
    w.r.t. generator parameters if x_fake = G(z, θ). It is a surrogate for
    the hard ISL loss; in practice one can tune (cdf_bandwidth, hist_sigma)
    to trade off bias vs stability.
    """
    if x_real.ndim != 1 or x_fake.ndim != 1:
        raise ValueError("isl_1d_soft expects 1D tensors x_real, x_fake")

    if K <= 0:
        raise ValueError("K must be a positive integer")

    if cdf_bandwidth <= 0.0 or hist_sigma <= 0.0:
        raise ValueError("cdf_bandwidth and hist_sigma must be positive")

    # Step 1: smooth CDF of fake at real points -> u in (0,1)
    u = soft_cdf(x_real, x_fake, bandwidth=cdf_bandwidth)  # (N,)

    # Step 2: soft histogram over [0,1]
    device = x_real.device
    dtype = x_real.dtype

    N = u.shape[0]
    if N == 0:
        # Edge case: no real samples => uniform histogram by default
        hist = torch.full((K,), 1.0 / K, device=device, dtype=dtype)
    else:
        # Bin centers c_k = (k+0.5)/K
        k = torch.arange(K, device=device, dtype=dtype)
        centers = (k + 0.5) / K  # (K,)

        # Assign each u_i to bins softly using Gaussian kernel:
        #   w_{i,k} ∝ exp(-(u_i - c_k)^2 / (2 * hist_sigma^2)).
        u_expanded = u.view(-1, 1)               # (N, 1)
        centers_expanded = centers.view(1, -1)   # (1, K)
        diff_uc = u_expanded - centers_expanded  # (N, K)

        logits = -0.5 * (diff_uc / hist_sigma) ** 2  # (N, K)
        # Softmax across bins ensures each sample contributes total mass 1.
        weights = torch.softmax(logits, dim=1)       # (N, K)

        # Average over samples -> soft histogram on [0,1]
        hist = weights.mean(dim=0)                   # (K,)

    # Step 3: L1 distance to uniform
    uniform = torch.full_like(hist, 1.0 / K)
    abs_diff = torch.abs(hist - uniform)  # (K,)

    if reduction == "none":
        return abs_diff
    elif reduction == "sum":
        return abs_diff.sum()
    elif reduction == "mean":
        return abs_diff.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")
    
__all__ = ["isl_1d", "isl_1d_soft"]
