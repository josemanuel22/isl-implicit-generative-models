"""
Sliced Invariant Statistical Loss (ISL) in R^d.

This module provides utilities to extend the one-dimensional ISL objective
to multivariate data via random projections ("slicing"). The basic idea is:

    1. Sample directions θ_1, …, θ_m on the unit sphere S^{d-1}.
    2. Project real and fake samples onto each direction:
           x_real^k = <x_real, θ_k>,   x_fake^k = <x_fake, θ_k>.
    3. Compute a 1D ISL loss along each projection.
    4. Average the per-direction losses to obtain a sliced ISL.

We implement:

- sample_random_directions(m, d):
    Draw m random unit vectors in R^d (rows of shape (d,)).

- sliced_isl(x_real, x_fake, directions, ...):
    Core routine: given a set of directions, compute the mean 1D ISL over them.

- sliced_isl_random(x_real, x_fake, m, ...):
    Convenience wrapper that samples m random directions and calls sliced_isl.

- select_smart_directions(...):
    "Max-sliced" / greedy heuristic that chooses m_select directions out of
    m_candidates so as to maximise the 1D ISL discrepancy while roughly
    enforcing orthogonality between selected directions.

- sliced_isl_smart(x_real, x_fake, m_select, m_candidates, ...):
    Convenience wrapper: selects smart directions and evaluates sliced ISL
    on those directions.

All functions are implemented in PyTorch and use the differentiable 1D ISL
surrogate isl_1d_soft by default, making them suitable as training losses
for implicit generative models.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .loss_1d import isl_1d_soft


# ---------------------------------------------------------------------
#  Direction sampling
# ---------------------------------------------------------------------

def sample_random_directions(
    m: int,
    d: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Sample m random directions uniformly on the unit sphere S^{d-1}.

    Parameters
    ----------
    m : int
        Number of directions.
    d : int
        Dimension of the ambient space R^d.
    device : torch.device, optional
        Device for the returned tensor. If None, uses the current default.
    dtype : torch.dtype, optional
        Data type for the returned tensor. If None, uses torch.float32.

    Returns
    -------
    directions : Tensor, shape (m, d)
        Each row is a unit vector (L2-normalised).
    """
    if m <= 0:
        raise ValueError("m must be a positive integer")
    if d <= 0:
        raise ValueError("d must be a positive integer")

    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    dirs = torch.randn(m, d, device=device, dtype=dtype)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    return dirs


# ---------------------------------------------------------------------
#  Core sliced ISL
# ---------------------------------------------------------------------

def sliced_isl(
    x_real: Tensor,
    x_fake: Tensor,
    directions: Tensor,
    K: int = 32,
    cdf_bandwidth: float = 0.1,
    hist_sigma: float = 0.05,
) -> Tensor:
    """
    Sliced 1D ISL in R^d: average 1D ISL over given directions.

    For each direction θ in `directions`, we:
        - project real and fake samples onto θ,
        - compute a 1D differentiable ISL surrogate isl_1d_soft,
        - then average across directions.

    Parameters
    ----------
    x_real : Tensor, shape (N, d)
        Real (data) samples.
    x_fake : Tensor, shape (M, d)
        Model / generator samples.
    directions : Tensor, shape (m, d)
        Unit vectors in R^d along which we slice.
    K : int, default=32
        Number of bins in the 1D ISL surrogate.
    cdf_bandwidth : float, default=0.1
        Smoothing parameter for the soft CDF in 1D ISL.
    hist_sigma : float, default=0.05
        Smoothing parameter for the soft histogram in 1D ISL.

    Returns
    -------
    loss : Tensor (scalar)
        Mean 1D ISL loss over all directions.

    Notes
    -----
    This function is differentiable w.r.t. x_fake (and upstream parameters)
    as long as isl_1d_soft is used internally. Gradients flow through the
    projections x_fake @ θ.
    """
    if x_real.ndim != 2 or x_fake.ndim != 2:
        raise ValueError("sliced_isl expects x_real, x_fake with shape (N, d) and (M, d)")

    if directions.ndim != 2:
        raise ValueError("directions must have shape (m, d)")

    N, d_real = x_real.shape
    M, d_fake = x_fake.shape
    m, d_dir = directions.shape

    if d_real != d_fake or d_real != d_dir:
        raise ValueError(
            f"Dimension mismatch: x_real dim={d_real}, x_fake dim={d_fake}, "
            f"directions dim={d_dir}"
        )

    losses = []
    for theta in directions:
        # theta: (d,)
        proj_real = x_real @ theta        # (N,)
        proj_fake = x_fake @ theta        # (M,)

        loss_1d = isl_1d_soft(
            proj_real,
            proj_fake,
            K=K,
            cdf_bandwidth=cdf_bandwidth,
            hist_sigma=hist_sigma,
            reduction="mean",
        )
        losses.append(loss_1d)

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=x_real.device)


def sliced_isl_random(
    x_real: Tensor,
    x_fake: Tensor,
    m: int,
    K: int = 32,
    cdf_bandwidth: float = 0.1,
    hist_sigma: float = 0.05,
) -> Tensor:
    """
    Sliced ISL with random directions.

    Convenience wrapper that:
      1. samples m random directions in R^d,
      2. calls `sliced_isl` with those directions.

    Parameters
    ----------
    x_real : Tensor, shape (N, d)
    x_fake : Tensor, shape (M, d)
    m : int
        Number of random directions.
    K, cdf_bandwidth, hist_sigma :
        Passed to `sliced_isl`.

    Returns
    -------
    loss : Tensor (scalar)
        Mean 1D ISL loss over the m random directions.
    """
    if x_real.ndim != 2:
        raise ValueError("x_real must have shape (N, d)")
    d = x_real.shape[1]
    directions = sample_random_directions(m, d, device=x_real.device, dtype=x_real.dtype)
    return sliced_isl(
        x_real,
        x_fake,
        directions,
        K=K,
        cdf_bandwidth=cdf_bandwidth,
        hist_sigma=hist_sigma,
    )


# ---------------------------------------------------------------------
#  "Smart" / max-sliced directions
# ---------------------------------------------------------------------

def _orth_component(v: Tensor, basis: list[Tensor]) -> Tensor:
    """
    Project v onto span(basis) and subtract, returning the orthogonal component.

    Parameters
    ----------
    v : Tensor, shape (d,)
        Vector to be orthogonalised.
    basis : list of Tensors, each shape (d,)
        Current set of chosen directions.

    Returns
    -------
    v_orth : Tensor, shape (d,)
        Component of v orthogonal to span(basis).
    """
    if len(basis) == 0:
        return v
    B = torch.stack(basis, dim=0)  # (k, d)
    # projection of v onto span(B): proj = (v B^T) B
    proj = (v @ B.T) @ B
    return v - proj


@torch.no_grad()
def select_smart_directions(
    x_real: Tensor,
    x_fake: Tensor,
    m_candidates: int,
    m_select: int,
    K: int = 32,
    cdf_bandwidth: float = 0.1,
    hist_sigma: float = 0.05,
    enforce_orthogonality: bool = True,
) -> Tensor:
    """
    Select "smart" (max-sliced style) directions for sliced ISL.

    We follow a simple greedy heuristic:

      1. Sample m_candidates random directions θ_1,...,θ_{m_candidates}.
      2. For each θ_j, compute the 1D ISL discrepancy L_j along that
         direction (with no gradient tracking).
      3. If `enforce_orthogonality` is False, simply take the indices
         of the top-m_select scores.
      4. If `enforce_orthogonality` is True, greedily build a set of
         directions: pick the highest-score direction first, then iteratively
         add the direction that maximises (score × orthogonality factor)
         with respect to the currently chosen set.

    Parameters
    ----------
    x_real : Tensor, shape (N, d)
        Real samples.
    x_fake : Tensor, shape (M, d)
        Fake / generator samples.
    m_candidates : int
        Number of random candidate directions to evaluate.
    m_select : int
        Number of directions to select (must be <= m_candidates).
    K, cdf_bandwidth, hist_sigma :
        Parameters for the 1D ISL surrogate used to score each direction.
    enforce_orthogonality : bool, default=True
        If True, apply the greedy orthogonal selection heuristic; otherwise,
        simply pick the top-m_select directions by score.

    Returns
    -------
    smart_dirs : Tensor, shape (m_select, d)
        Selected directions, unit-normalised.

    Notes
    -----
    This function is decorated with `torch.no_grad()`, as we only use it
    to select directions; gradients do not propagate through the selection
    procedure.
    """
    if x_real.ndim != 2 or x_fake.ndim != 2:
        raise ValueError("select_smart_directions expects x_real, x_fake with shape (N, d) and (M, d)")

    N, d_real = x_real.shape
    M, d_fake = x_fake.shape
    if d_real != d_fake:
        raise ValueError(f"Dimension mismatch: x_real dim={d_real}, x_fake dim={d_fake}")

    if m_select > m_candidates:
        raise ValueError("m_select must be <= m_candidates")

    device = x_real.device
    dtype = x_real.dtype

    cand_dirs = sample_random_directions(m_candidates, d_real, device=device, dtype=dtype)
    scores = []

    # Score each candidate direction by 1D ISL
    for theta in cand_dirs:
        proj_real = x_real @ theta
        proj_fake = x_fake @ theta
        loss_1d = isl_1d_soft(
            proj_real,
            proj_fake,
            K=K,
            cdf_bandwidth=cdf_bandwidth,
            hist_sigma=hist_sigma,
            reduction="mean",
        )
        scores.append(loss_1d)

    scores = torch.stack(scores)  # (m_candidates,)

    if not enforce_orthogonality:
        # Simple top-k selection by score
        _, top_idx = torch.topk(scores, k=m_select, largest=True)
        smart_dirs = cand_dirs[top_idx]
        return smart_dirs

    # Greedy orthogonal selection
    chosen_indices: list[int] = []
    remaining = list(range(m_candidates))

    # First: take best by score
    first_idx = int(torch.argmax(scores).item())
    chosen_indices.append(first_idx)
    remaining = [i for i in remaining if i != first_idx]

    while len(chosen_indices) < m_select and len(remaining) > 0:
        chosen_dirs = [cand_dirs[i] for i in chosen_indices]
        best_idx = None
        best_eff_score = None

        for idx in remaining:
            v = cand_dirs[idx]
            v_orth = _orth_component(v, chosen_dirs)
            norm_v = v.norm()
            norm_v_orth = v_orth.norm()

            # If almost collinear with existing directions, skip
            if norm_v_orth < 1e-6 or norm_v < 1e-6:
                continue

            # Orthogonality factor: ||v_orth|| / ||v||
            ortho_factor = (norm_v_orth / norm_v).item()
            eff_score = scores[idx] * ortho_factor

            if (best_eff_score is None) or (eff_score > best_eff_score):
                best_eff_score = eff_score
                best_idx = idx

        if best_idx is None:
            # All remaining directions are nearly collinear; stop early.
            break

        chosen_indices.append(best_idx)
        remaining = [i for i in remaining if i != best_idx]

    # If we stopped early due to collinearity, pad by pure score-based choices
    if len(chosen_indices) < m_select and len(remaining) > 0:
        # Take the remaining highest-score directions
        remaining_scores = scores[remaining]
        _, extra_idx = torch.topk(remaining_scores,
                                  k=min(m_select - len(chosen_indices), len(remaining)),
                                  largest=True)
        extra_idx = [remaining[i] for i in extra_idx.tolist()]
        chosen_indices.extend(extra_idx)

    chosen_indices = chosen_indices[:m_select]
    idx_tensor = torch.tensor(chosen_indices, device=device, dtype=torch.long)
    smart_dirs = cand_dirs[idx_tensor]
    return smart_dirs


def sliced_isl_smart(
    x_real: Tensor,
    x_fake: Tensor,
    m_select: int,
    m_candidates: int,
    K: int = 32,
    cdf_bandwidth: float = 0.1,
    hist_sigma: float = 0.05,
    enforce_orthogonality: bool = True,
) -> Tensor:
    """
    Sliced ISL with "smart" (max-sliced style) directions.

    This convenience wrapper:
      1. Calls `select_smart_directions` to obtain m_select directions,
      2. Evaluates sliced ISL on those directions via `sliced_isl`.

    Parameters
    ----------
    x_real : Tensor, shape (N, d)
    x_fake : Tensor, shape (M, d)
    m_select : int
        Number of directions to slice along.
    m_candidates : int
        Number of random candidate directions to consider when selecting
        "smart" directions.
    K, cdf_bandwidth, hist_sigma :
        Passed to both the selection scoring (isl_1d_soft) and final sliced ISL.
    enforce_orthogonality : bool, default=True
        Whether to encourage approximate orthogonality between the selected
        directions during the greedy selection step.

    Returns
    -------
    loss : Tensor (scalar)
        Mean 1D ISL loss over the selected smart directions.
    """
    smart_dirs = select_smart_directions(
        x_real,
        x_fake,
        m_candidates=m_candidates,
        m_select=m_select,
        K=K,
        cdf_bandwidth=cdf_bandwidth,
        hist_sigma=hist_sigma,
        enforce_orthogonality=enforce_orthogonality,
    )
    return sliced_isl(
        x_real,
        x_fake,
        smart_dirs,
        K=K,
        cdf_bandwidth=cdf_bandwidth,
        hist_sigma=hist_sigma,
    )

__all__ = [
    "sample_random_directions",
    "sliced_isl",
    "sliced_isl_random",
    "select_smart_directions",
    "sliced_isl_smart",
]