# src/isl/ranks.py

import torch
from torch import Tensor

"""
Rank and CDF utilities for Invariant Statistical Loss (ISL).

This module provides basic building blocks to work with rank statistics in 1D:
hard ranks, rescaled "uniform" ranks, and (hard or smooth) empirical CDFs.
These are the core ingredients used to construct the 1D ISL objective and its
differentiable surrogates.

Main functions
--------------
- hard_ranks_1d(x):
    Returns integer ranks r ∈ {0, …, N-1} for a 1D tensor x, where r = 0
    corresponds to the smallest (or largest) value depending on the
    `descending` flag.

- uniform_ranks_1d(x):
    Returns rescaled ranks u = (r + 0.5)/N ∈ (0, 1). If x is drawn from a
    continuous distribution, these u values are approximately Uniform[0,1].
    This is the canonical representation used in ISL.

- empirical_cdf(x, y):
    Computes the empirical CDF of y evaluated at each x_i:
        F_y(x_i) = (1/|y|) * sum_j 1{ y_j <= x_i }.
    This is the "hard" CDF, non-differentiable in y.

- soft_cdf(x, y, bandwidth):
    Differentiable surrogate of the empirical CDF, obtained by replacing
    the hard indicator with a sigmoid kernel:
        1{ y_j <= x_i } ≈ sigmoid((x_i - y_j)/h).
    This is used inside the soft ISL loss to propagate gradients through
    generator samples.

All functions assume 1D tensors and are implemented in PyTorch so they can be
seamlessly integrated into training loops for implicit generative models.
"""


def hard_ranks_1d(x: Tensor, descending: bool = False) -> Tensor:
    """
    Compute hard ranks of a 1D tensor.

    Parameters
    ----------
    x : Tensor, shape (N,)
        Input values.
    descending : bool, default=False
        If False, rank 0 corresponds to the smallest value.
        If True,  rank 0 corresponds to the largest value.

    Returns
    -------
    ranks : Tensor, shape (N,)
        Integer ranks in {0, 1, ..., N-1}.
        For simplicity we assume that ties are rare (continuous data).
        If ties occur, their relative order follows torch.argsort's
        stable sort semantics.
    """
    if x.ndim != 1:
        raise ValueError(f"hard_ranks_1d expects a 1D tensor, got shape {x.shape}")

    # argsort gives indices of sorted values
    if descending:
        order = torch.argsort(x, dim=0, descending=True, stable=True)
    else:
        order = torch.argsort(x, dim=0, descending=False, stable=True)

    # ranks[order[i]] = i
    ranks = torch.empty_like(order, dtype=torch.long)
    ranks[order] = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
    return ranks


def uniform_ranks_1d(x: Tensor, descending: bool = False) -> Tensor:
    """
    Compute (approximate) uniform ranks in (0, 1).

    This rescales integer ranks r in {0,...,N-1} to
        u = (r + 0.5) / N  \in (0, 1).

    Parameters
    ----------
    x : Tensor, shape (N,)
        Input values.
    descending : bool, default=False
        If False, smallest x has smallest rank.
        If True,  largest x has smallest rank.

    Returns
    -------
    u : Tensor, shape (N,)
        Values in (0, 1) approximating Uniform[0,1] if x is drawn
        from a continuous distribution.
    """
    N = x.shape[0]
    ranks = hard_ranks_1d(x, descending=descending).to(dtype=x.dtype)
    return (ranks + 0.5) / float(N)


def empirical_cdf(x: Tensor, y: Tensor) -> Tensor:
    """
    Empirical CDF of y evaluated at each point in x:

        F_y(x_i) = (1/|y|) * sum_j 1{ y_j <= x_i }.

    Parameters
    ----------
    x : Tensor, shape (N,)
        Query points where we evaluate the CDF.
    y : Tensor, shape (M,)
        Sample from the reference distribution.

    Returns
    -------
    F : Tensor, shape (N,)
        Empirical CDF values F_y(x_i) in [0,1].
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("empirical_cdf expects 1D tensors x, y")

    # Broadcast compare: (N,1) vs (1,M) -> (N,M)
    xr = x.view(-1, 1)
    yr = y.view(1, -1)
    indicators = (yr <= xr).to(x.dtype)
    F = indicators.mean(dim=1)
    return F


def soft_cdf(x: Tensor,
             y: Tensor,
             bandwidth: float = 0.1) -> Tensor:
    """
    Smooth empirical CDF of y evaluated at each point in x.

    We replace the hard indicator 1{ y_j <= x_i } by a sigmoid kernel:
        1{y_j <= x_i} ≈ sigmoid((x_i - y_j) / h),

    so that everything becomes differentiable w.r.t. y (and any parameters
    upstream of y). This is what we used in the soft ISL surrogate.

    Parameters
    ----------
    x : Tensor, shape (N,)
        Query points where we evaluate the (smooth) CDF.
    y : Tensor, shape (M,)
        Sample from the reference distribution (typically generator samples).
    bandwidth : float, default=0.1
        Smoothing parameter h. Smaller values make the sigmoid steeper
        and the CDF closer to the hard empirical CDF, but may cause
        gradients to vanish or explode.

    Returns
    -------
    F_soft : Tensor, shape (N,)
        Smooth CDF values ~ F_y(x_i) in (0,1).
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("soft_cdf expects 1D tensors x, y")

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    xr = x.view(-1, 1)
    yr = y.view(1, -1)

    diff = (xr - yr) / bandwidth
    soft_ind = torch.sigmoid(diff)  # smooth 1{ y_j <= x_i }
    F_soft = soft_ind.mean(dim=1)
    return F_soft

def project_and_uniform_ranks(x: Tensor, theta: Tensor) -> Tensor:
    """
    Project x in R^d along direction theta in S^{d-1},
    then return uniform ranks in (0,1).
    """
    proj = x @ theta  # (N,)
    return uniform_ranks_1d(proj)

__all__ = [
    "hard_ranks_1d",
    "uniform_ranks_1d",
    "empirical_cdf",
    "soft_cdf",
]
