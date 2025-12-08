"""
Evaluation metrics for ISL-based generative models.

This module provides a few metrics that we use throughout the experiments:

- Kernel Stein Discrepancy (KSD) with an RBF kernel:
    ksd_rbf(samples, score_fn, bandwidth=None)
  where `score_fn` is the score (∇_x log p(x)) of the target distribution.

- Empirical complementary CDF (CCDF) and ACCDF-based error:
    empirical_ccdf(x, thresholds)
    accdf_error(x_real, x_fake, thresholds, p=1)
  useful for assessing heavy-tailed behaviour and the fit of extremes.

- Quantile-based error:
    quantile_error(x_real, x_fake, quantiles, p=1)
  which compares empirical quantiles of real and generated samples.

All functions are implemented in PyTorch and operate on tensors. In most
cases we assume 1D samples for CCDF / quantile metrics, and arbitrary
dimension d ≥ 1 for KSD.
"""

from __future__ import annotations

from typing import Callable, Iterable, Literal, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------
#  Kernel Stein Discrepancy (RBF kernel)
# ---------------------------------------------------------------------

@torch.no_grad()
def ksd_rbf(
    samples: Tensor,
    score_fn: Callable[[Tensor], Tensor],
    bandwidth: Optional[float] = None,
) -> Tensor:
    """
    Kernel Stein Discrepancy (KSD) with an RBF kernel.

    For a target distribution with score function s(x) = ∇_x log p(x),
    and a set of samples {x_i} from a candidate distribution q, the
    KSD is defined (up to a square root) as:

        KSD^2(q, p) = E_{x,x'~q} [ u(x, x') ],

    where, for the RBF kernel k(x,x') = exp(-||x-x'||^2 / (2 h^2)),

        u(x, x') =
            s(x)^T s(x') k(x,x')
          + s(x)^T ∇_{x'} k(x,x')
          + s(x')^T ∇_{x} k(x,x')
          + trace(∇_x∇_{x'} k(x,x')).

    This function computes the standard U-statistic / V-statistic style
    estimator and returns sqrt( KSD^2 ), i.e. a non-negative scalar.

    Parameters
    ----------
    samples : Tensor, shape (n, d)
        Samples from the candidate distribution q.
    score_fn : callable
        Function mapping a tensor of shape (n, d) to the score s(x) with
        the same shape (n, d).
    bandwidth : float, optional
        RBF bandwidth h. If None, we use the median heuristic:
            h^2 = median(||x_i - x_j||^2) / 2.

    Returns
    -------
    ksd : Tensor (scalar)
        Estimated KSD (square root of KSD^2) between q and p.

    Notes
    -----
    This function runs under torch.no_grad() as it is typically used
    only for evaluation. If you need gradients w.r.t. samples, remove
    the decorator and ensure score_fn supports autograd.
    """
    X = samples
    if X.ndim != 2:
        raise ValueError("ksd_rbf expects samples with shape (n, d)")

    n, d = X.shape
    if n < 2:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    score = score_fn(X)  # (n, d)
    if score.shape != X.shape:
        raise ValueError("score_fn must return a tensor of shape (n, d)")

    # Pairwise differences
    diff = X.unsqueeze(1) - X.unsqueeze(0)   # (n, n, d)
    sqdist = (diff ** 2).sum(dim=-1)        # (n, n)

    # Bandwidth via median heuristic if not provided
    if bandwidth is None:
        sqdist_vec = sqdist.flatten()
        sqdist_vec = sqdist_vec[sqdist_vec > 0]  # remove zeros on diagonal
        if sqdist_vec.numel() == 0:
            # All points identical; degenerate case
            return torch.tensor(0.0, device=X.device, dtype=X.dtype)
        med_sq = torch.median(sqdist_vec)
        h2 = med_sq / 2.0
    else:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        h2 = bandwidth ** 2

    k = torch.exp(-sqdist / (2.0 * h2))     # (n, n)

    s_i = score.unsqueeze(1)  # (n, 1, d)
    s_j = score.unsqueeze(0)  # (1, n, d)

    # Terms of u(x_i, x_j)
    term1 = (s_i * s_j).sum(dim=-1) * k
    term2 = (s_i * (-diff) / h2).sum(dim=-1) * k
    term3 = (s_j * (diff) / h2).sum(dim=-1) * k
    term4 = (d / h2 - sqdist / (h2 ** 2)) * k

    u = term1 + term2 + term3 + term4
    ksd2 = u.mean()
    # Numerical safety: ensure non-negative
    ksd2 = torch.clamp(ksd2, min=0.0)
    return torch.sqrt(ksd2)


# ---------------------------------------------------------------------
#  Empirical CCDF and ACCDF-based error
# ---------------------------------------------------------------------

def empirical_ccdf(x: Tensor, thresholds: Tensor) -> Tensor:
    """
    Empirical complementary CDF (CCDF) of a 1D sample x.

    Given thresholds t_k, we define:
        CCDF_x(t_k) = P(X > t_k) ≈ (1/N) ∑_i 1{ x_i > t_k }.

    Parameters
    ----------
    x : Tensor, shape (N,)
        1D sample from a distribution.
    thresholds : Tensor, shape (K,)
        Thresholds t_k at which to evaluate the CCDF.

    Returns
    -------
    ccdf : Tensor, shape (K,)
        Empirical CCDF values at each threshold.
    """
    if x.ndim != 1:
        raise ValueError("empirical_ccdf expects a 1D tensor x")
    if thresholds.ndim != 1:
        raise ValueError("empirical_ccdf expects a 1D tensor thresholds")

    x = x.view(-1)
    t = thresholds.view(-1).to(device=x.device, dtype=x.dtype)

    # Broadcast: (N,1) vs (1,K) -> (N,K)
    xr = x.view(-1, 1)
    tr = t.view(1, -1)
    indicators = (xr > tr).to(x.dtype)  # (N, K)
    ccdf = indicators.mean(dim=0)       # (K,)
    return ccdf


def accdf_error(
    x_real: Tensor,
    x_fake: Tensor,
    thresholds: Tensor,
    p: int | float = 1,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """
    ACCDF-based discrepancy between two 1D samples.

    We compare empirical CCDF curves for real and fake samples at a set
    of thresholds {t_k}. The error is defined as:

        E = || CCDF_real - CCDF_fake ||_p

    with optional reduction over thresholds.

    Parameters
    ----------
    x_real : Tensor, shape (N,)
        Real data samples.
    x_fake : Tensor, shape (M,)
        Model / generator samples.
    thresholds : Tensor, shape (K,)
        Thresholds at which to evaluate CCDF.
    p : int or float, default=1
        L_p norm exponent (p >= 1).
    reduction : {"mean", "sum", "none"}, default="mean"
        - "mean": return mean of |Δ|^p over thresholds, i.e.
              (1/K) ∑_k |Δ_k|^p
        - "sum":  return ∑_k |Δ_k|^p
        - "none": return vector |Δ_k|^p of shape (K,).

    Returns
    -------
    error : Tensor
        Scalar if reduction != "none", else shape (K,).

    Notes
    -----
    This is particularly useful for heavy-tailed settings, where we care
    about matching tail probabilities and extreme events.
    """
    if x_real.ndim != 1 or x_fake.ndim != 1:
        raise ValueError("accdf_error expects 1D tensors x_real, x_fake")
    if p <= 0:
        raise ValueError("p must be >= 1")

    ccdf_real = empirical_ccdf(x_real, thresholds)
    ccdf_fake = empirical_ccdf(x_fake, thresholds.to(device=x_real.device, dtype=x_real.dtype))

    delta = torch.abs(ccdf_real - ccdf_fake) ** p  # (K,)

    if reduction == "none":
        return delta
    elif reduction == "sum":
        return delta.sum()
    elif reduction == "mean":
        return delta.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


# ---------------------------------------------------------------------
#  Quantile-based error
# ---------------------------------------------------------------------

def quantile_error(
    x_real: Tensor,
    x_fake: Tensor,
    quantiles: Iterable[float],
    p: int | float = 1,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """
    Quantile-based discrepancy between two 1D samples.

    Given levels α_j ∈ (0,1), we compare empirical quantiles:

        q_real(α_j), q_fake(α_j),

    and define the error as:

        E = || q_real - q_fake ||_p

    with optional reduction over j.

    Parameters
    ----------
    x_real : Tensor, shape (N,)
        Real data samples.
    x_fake : Tensor, shape (M,)
        Model / generator samples.
    quantiles : iterable of float
        Quantile levels α_j in (0,1).
    p : int or float, default=1
        L_p norm exponent (p >= 1).
    reduction : {"mean", "sum", "none"}, default="mean"
        - "mean": mean of |Δ|^p over quantiles,
        - "sum":  sum of |Δ|^p,
        - "none": vector of |Δ|^p, shape (J,).

    Returns
    -------
    error : Tensor
        Scalar if reduction != "none", else shape (J,).

    Notes
    -----
    This is also useful in heavy-tailed settings, especially when
    quantiles are chosen near the tail (e.g., α ∈ {0.9, 0.95, 0.99}).
    """
    if x_real.ndim != 1 or x_fake.ndim != 1:
        raise ValueError("quantile_error expects 1D tensors x_real, x_fake")
    if p <= 0:
        raise ValueError("p must be >= 1")

    device = x_real.device
    dtype = x_real.dtype

    qs = list(quantiles)
    if len(qs) == 0:
        raise ValueError("quantiles must contain at least one level")

    q_real = torch.quantile(x_real.to(dtype=dtype), torch.tensor(qs, device=device, dtype=dtype))
    q_fake = torch.quantile(x_fake.to(dtype=dtype), torch.tensor(qs, device=device, dtype=dtype))

    delta = torch.abs(q_real - q_fake) ** p  # (J,)

    if reduction == "none":
        return delta
    elif reduction == "sum":
        return delta.sum()
    elif reduction == "mean":
        return delta.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")

__all__ = ["ksd_rbf", "empirical_ccdf", "accdf_error", "quantile_error"]