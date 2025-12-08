"""
General utilities for ISL experiments.

This module provides a small collection of helper functions that are useful
across experiments and scripts:

- set_seed(seed, deterministic=False):
    Set random seeds for Python, NumPy and PyTorch, and optionally enable
    deterministic behaviour in cuDNN (at the cost of speed).

- get_device(prefer_gpu=True, index=0):
    Convenience wrapper to choose a torch.device (CUDA if available, else CPU).

- count_parameters(model, trainable_only=True):
    Count the number of (trainable) parameters in a PyTorch module.

- ensure_dir(path):
    Create a directory (and parents) if it does not exist.

- save_checkpoint(path, model, optimizer=None, extra=None):
    Save model (and optional optimizer + extra metadata) to a checkpoint file.

- load_checkpoint(path, model=None, optimizer=None, map_location=None):
    Load a checkpoint and optionally restore model and optimizer state.

These helpers are intentionally lightweight and do not impose any experiment
framework; they are meant to be imported directly in scripts and notebooks.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch.

    Parameters
    ----------
    seed : int
        Seed value to use for all RNGs.
    deterministic : bool, default=False
        If True, configure PyTorch/cuDNN for deterministic behaviour.
        This may slow down training and disable some optimised kernels,
        but improves reproducibility across runs.

    Notes
    -----
    This function does not set seeds for all possible randomness sources
    (e.g., CUDA operations in some custom kernels, Python hash seed), but
    covers the most common ones used in typical ISL experiments.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Leave PyTorch defaults for performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
#  Device helpers
# ---------------------------------------------------------------------

def get_device(prefer_gpu: bool = True, index: int = 0) -> torch.device:
    """
    Select a torch.device for computation.

    Parameters
    ----------
    prefer_gpu : bool, default=True
        If True and CUDA is available, return a CUDA device; otherwise
        return CPU.
    index : int, default=0
        CUDA device index to use if multiple GPUs are available.

    Returns
    -------
    device : torch.device
        Either torch.device("cuda:index") or torch.device("cpu").
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


# ---------------------------------------------------------------------
#  Model inspection
# ---------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        PyTorch module.
    trainable_only : bool, default=True
        If True, count only parameters with requires_grad=True.

    Returns
    -------
    n_params : int
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------
#  Filesystem helpers
# ---------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """
    Ensure that a directory exists, creating it (and parents) if needed.

    Parameters
    ----------
    path : str or Path
        Directory path.

    Returns
    -------
    path_obj : Path
        pathlib.Path object for the created/existing directory.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


# ---------------------------------------------------------------------
#  Checkpointing
# ---------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint to disk.

    Parameters
    ----------
    path : str or Path
        Destination file path (e.g., "checkpoints/epoch_10.pt").
    model : nn.Module
        Model whose state_dict will be saved.
    optimizer : torch.optim.Optimizer, optional
        Optimizer whose state_dict will be saved (if provided).
    extra : dict, optional
        Additional metadata to store (e.g., epoch, metrics, config).

    Notes
    -----
    The checkpoint is stored as a dict with keys:
        - "model_state"
        - "optimizer_state" (optional)
        - "extra" (optional)
    """
    path = Path(path)
    ensure_dir(path.parent)

    checkpoint: Dict[str, Any] = {
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint from disk and optionally restore state.

    Parameters
    ----------
    path : str or Path
        Path to the checkpoint file.
    model : nn.Module, optional
        If provided, its state_dict will be updated from the checkpoint.
    optimizer : torch.optim.Optimizer, optional
        If provided and present in the checkpoint, its state_dict will be
        updated from the checkpoint.
    map_location : str or torch.device, optional
        Passed to torch.load for device remapping (e.g., "cpu").

    Returns
    -------
    checkpoint : dict
        The full checkpoint dictionary as loaded from disk, including
        keys "model_state", "optimizer_state" (if present) and "extra"
        (if present).

    Notes
    -----
    This function does not assume a specific training loop; you are free
    to interpret and use the "extra" metadata as needed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    if model is not None and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint 