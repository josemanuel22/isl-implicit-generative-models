#!/usr/bin/env python
"""
RNN + sliced ISL for 1-step-ahead prediction on an ETT *multivariate* series
(Gaussian latent, teacher-forcing forecast).

We:
  - load the chosen ETT CSV as a multivariate time series (all numeric
    columns except 'date', or a user-specified subset),
  - optionally log-transform it,
  - NORMALISE each channel: x_norm = (x - mean) / std,
  - train an RNN+MLP generator with Gaussian latent noise z ~ N(0,I),
  - optimise a *sliced* ISL loss on the one-step-ahead conditional
    distribution in R^d (over all channels),
  - evaluate 1-step-ahead distributional fit,
  - and produce a 1-step-ahead forecast plot (teacher forcing) for one
    chosen channel (e.g. 'OT').

All diagnostics & plots are in the ORIGINAL data scale of the plot channel.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import sys
import argparse
from typing import Optional, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
#  Load ETT multivariate series
# ---------------------------------------------------------------------

def load_ett_multivariate(
    path: str | Path,
    feature_cols: Optional[Sequence[str]],
    device: torch.device,
    log_transform: bool = False,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, list[str]]:
    """
    Load an ETT dataset as a multivariate time series.

    Parameters
    ----------
    path : str or Path
        Path to an ETT .csv file (e.g. ETTm1.csv).
    feature_cols : sequence of str or None
        Which columns to keep as features. If None, we take all numeric
        columns except 'date'.
    device : torch.device
        Device for the resulting tensor.
    log_transform : bool, default=False
        If True, apply log(x + eps) elementwise.
    eps : float, default=1e-3
        Small offset for log-transform.

    Returns
    -------
    series : Tensor, shape (N, d)
        Multivariate time series on `device`.
    cols   : list[str]
        List of feature column names (length d).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ETT csv file not found: {path}")

    df = pd.read_csv(path)

    if feature_cols is None:
        # All non-date columns that are numeric
        numeric_cols = [
            c for c in df.columns
            if c != "date" and np.issubdtype(df[c].dtype, np.number)
        ]
        if not numeric_cols:
            raise ValueError("No numeric feature columns found in ETT CSV.")
        feature_cols = numeric_cols
    else:
        for c in feature_cols:
            if c not in df.columns:
                raise ValueError(
                    f"Requested feature column {c!r} not found. "
                    f"Available columns: {list(df.columns)}"
                )

    cols = list(feature_cols)
    arr = df[cols].to_numpy().astype("float32")  # (N, d)

    if log_transform:
        arr = np.log(arr + eps)

    series = torch.tensor(arr, dtype=torch.float32, device=device)
    return series, cols


# ---------------------------------------------------------------------
#  Sliding-window dataset
# ---------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for a (possibly multivariate) time series.

    Given series X[0],...,X[N-1] with X[t] in R^d, construct windows:

        seq_i = (X[s_i], ..., X[s_i + L - 1]),   length L = seq_len,

    where s_i runs over {0, stride, 2*stride, ...} up to N - L.
    """

    def __init__(
        self,
        series: torch.Tensor,  # (N, d) or (N,)
        seq_len: int,
        stride: int = 1,
    ) -> None:
        if series.ndim not in (1, 2):
            raise ValueError("TimeSeriesDataset expects series with shape (N,) or (N, d)")

        self.series = series
        self.seq_len = int(seq_len)
        self.stride = int(stride)

        N = series.shape[0]
        if N < seq_len:
            raise ValueError(f"Series too short (N={N}) for seq_len={seq_len}")

        self.indices = list(range(0, N - seq_len, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = self.indices[idx]
        end = start + self.seq_len
        return self.series[start:end]  # (L,) or (L, d)


# ---------------------------------------------------------------------
#  Gaussian latent
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
#  Tail diagnostics
# ---------------------------------------------------------------------

def hill_tail_index(x: np.ndarray, k: int = 500) -> float:
    x = np.asarray(x)
    x = x[x > 0]
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
    qs = [0.9, 0.95, 0.99, 0.995, 0.999]
    print(f"\n{name} tail diagnostics:")
    for q in qs:
        val = np.quantile(x, q)
        print(f"  q={q:.3f}: {val:.4f}")
    alpha_hat = hill_tail_index(x, k=min(1000, max(100, x.shape[0] // 10)))
    print(f"  Hill alpha_hat ≈ {alpha_hat:.4f}")


# ---------------------------------------------------------------------
#  RNN + decoder model (multivariate)
# ---------------------------------------------------------------------

class RNNISLSlicedModel(nn.Module):
    """
    RNN-based implicit generator for a multivariate ETT series in R^d.

    Forward:

        x_fake_seq = model(x_hist, z_seq)

    x_hist: (B, L-1, d_in), true past (teacher forcing)
    z_seq : (B, L-1, Dz), latent per step
    x_fake_seq: (B, L-1, d_in), generated one-step-ahead
    """

    def __init__(
        self,
        input_dim: int,
        noise_dim: int,
        hidden_dim: int = 64,
        rnn_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x_hist: torch.Tensor, z_seq: torch.Tensor) -> torch.Tensor:
        # x_hist: (B, L-1, d_in), z_seq: (B, L-1, Dz)
        B, Lm1, d_in = x_hist.shape
        Bz, Lm1_z, Dz = z_seq.shape
        if B != Bz or Lm1 != Lm1_z:
            raise ValueError("Batch/time dimension mismatch between x_hist and z_seq")

        h_seq, _ = self.rnn(x_hist)  # (B, L-1, H)
        H = self.hidden_dim

        h_flat = h_seq.reshape(B * Lm1, H)
        z_flat = z_seq.reshape(B * Lm1, Dz)
        dec_in = torch.cat([h_flat, z_flat], dim=1)  # (B*L-1, H+Dz)

        x_fake_flat = self.decoder(dec_in)           # (B*L-1, d_in)
        x_fake_seq = x_fake_flat.view(B, Lm1, d_in)  # (B, L-1, d_in)
        return x_fake_seq


# ---------------------------------------------------------------------
#  Training + evaluation (sliced ISL in R^d)
# ---------------------------------------------------------------------

def train_rnn_sliced_isl_ett(
    data_path: str | Path,
    feature_cols: Optional[Sequence[str]] = None,
    plot_column: str = "OT",
    log_transform: bool = False,
    seq_len: int = 64,
    stride: int = 1,
    steps: int = 10000,
    batch_size: int = 64,
    noise_dim: int = 4,
    hidden_dim: int = 64,
    rnn_layers: int = 1,
    lr: float = 1e-3,
    n_slices: int = 32,
    K: int = 64,
    cdf_bandwidth: float = 0.15,
    hist_sigma: float = 0.05,
    log_every: int = 200,
    forecast_horizon: int = 500,
    device: Optional[torch.device] = None,
    outdir: Optional[Path] = None,
) -> None:
    """
    Train an RNN-based implicit generator on a multivariate ETT series
    with sliced ISL (random directions).

    - latent z ~ N(0, I),
    - per-channel normalisation,
    - sliced ISL over all channels,
    - forecast plot (teacher forcing) for one chosen channel.
    """
    if device is None:
        device = get_device(prefer_gpu=True)
    if outdir is None:
        outdir = ROOT / "experiments" / "ett_rnn_sliced_isl"
    ensure_dir(outdir)

    print("=" * 70)
    print("RNN + sliced ISL on ETT multivariate series "
          "(Gaussian latent, normalised channels)")
    print(f"  data_path        : {data_path}")
    print(f"  feature_cols     : {feature_cols}")
    print(f"  plot_column      : {plot_column}")
    print(f"  log_transform    : {log_transform}")
    print(f"  seq_len          : {seq_len}")
    print(f"  stride           : {stride}")
    print(f"  steps            : {steps}")
    print(f"  batch_size       : {batch_size}")
    print(f"  noise_dim        : {noise_dim}")
    print(f"  hidden_dim       : {hidden_dim}")
    print(f"  rnn_layers       : {rnn_layers}")
    print(f"  lr               : {lr}")
    print(f"  n_slices         : {n_slices}")
    print(f"  K                : {K}")
    print(f"  cdf_bandwidth    : {cdf_bandwidth}")
    print(f"  hist_sigma       : {hist_sigma}")
    print(f"  forecast_horizon : {forecast_horizon}")
    print(f"  device           : {device}")
    print(f"  outdir           : {outdir}")
    print("=" * 70)

    torch.set_num_threads(1)

    # --------- Load and NORMALISE series --------- #
    series_raw, cols = load_ett_multivariate(
        path=data_path,
        feature_cols=feature_cols,
        device=device,
        log_transform=log_transform,
        eps=1e-3,
    )
    N, d = series_raw.shape
    print(f"Loaded ETT multivariate series with N={N}, d={d}.")
    print(f"Feature columns: {cols}")

    if plot_column not in cols:
        raise ValueError(
            f"plot_column {plot_column!r} not in feature columns {cols}"
        )
    plot_idx = cols.index(plot_column)

    mean = series_raw.mean(dim=0)                 # (d,)
    std = series_raw.std(dim=0).clamp_min(1e-6)   # (d,)
    series = (series_raw - mean) / std

    mean_np = mean.detach().cpu().numpy()         # (d,)
    std_np = std.detach().cpu().numpy()

    mean_plot = float(mean_np[plot_idx])
    std_plot = float(std_np[plot_idx])
    print(f"Normalisation (per channel). For plot column {plot_column!r}: "
          f"mean={mean_plot:.4f}, std={std_plot:.4f}")

    dataset = TimeSeriesDataset(series=series, seq_len=seq_len, stride=stride)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    print(f"Number of windows: {len(dataset)}")

    model = RNNISLSlicedModel(
        input_dim=d,
        noise_dim=noise_dim,
        hidden_dim=hidden_dim,
        rnn_layers=rnn_layers,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    step = 0
    epoch = 0

    # ------------------- training loop ------------------- #
    while step < steps:
        epoch += 1
        for batch in loader:
            if step >= steps:
                break

            x_seq = batch.to(device)  # (B, L, d) in NORMALISED scale
            B, L, d_in = x_seq.shape
            if L < 2:
                continue

            x_hist = x_seq[:, :-1, :]  # (B, L-1, d)
            x_tgt = x_seq[:, 1:, :]    # (B, L-1, d)

            n_latent = B * (L - 1)
            z_flat = sample_latent(
                n=n_latent,
                noise_dim=noise_dim,
                device=device,
            )
            z_seq = z_flat.view(B, L - 1, noise_dim)

            x_fake_seq = model(x_hist, z_seq)   # (B, L-1, d)

            x_real_flat = x_tgt.reshape(-1, d_in)      # (B*(L-1), d)
            x_fake_flat = x_fake_seq.reshape(-1, d_in)

            loss = sliced_isl_random(
                x_real_flat,
                x_fake_flat,
                m=n_slices,
                K=K,
                cdf_bandwidth=cdf_bandwidth,
                hist_sigma=hist_sigma,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            losses.append(float(loss.item()))

            if step % log_every == 0 or step == 1 or step == steps:
                print(f"[step {step:6d}/{steps}] sliced ISL loss = {loss.item():.6f}, "
                      f"epoch={epoch}")

    # ------------------- Save checkpoint ------------------- #
    suffix = (
        f"ETT_multi_latent-gaussian_slices{n_slices}_K{K}_L{seq_len}"
        + ("_logx" if log_transform else "")
    )
    ckpt_path = outdir / f"rnn_sliced_isl_ett_model_{suffix}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved model checkpoint to {ckpt_path}")

    # ------------------- Training curve ------------------- #
    steps_arr = np.arange(1, len(losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(steps_arr, losses)
    plt.xlabel("Training step")
    plt.ylabel("sliced ISL loss")
    plt.title(f"ETT-RNN-sliced-ISL training curve (d={d}, slices={n_slices})")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    curve_path = outdir / f"rnn_sliced_isl_ett_loss_curve_{suffix}.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved training curve to {curve_path}")

    # ------------------- Evaluation: marginal 1-step on plot channel ------------------- #
    model.eval()
    real_vals_plot_norm: List[np.ndarray] = []
    fake_vals_plot_norm: List[np.ndarray] = []

    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    with torch.no_grad():
        for batch in eval_loader:
            x_seq = batch.to(device)  # (B, L, d)
            B, L, d_in = x_seq.shape
            if L < 2:
                continue

            x_hist = x_seq[:, :-1, :]
            x_tgt = x_seq[:, 1:, :]

            n_latent = B * (L - 1)
            z_flat = sample_latent(
                n=n_latent,
                noise_dim=noise_dim,
                device=device,
            )
            z_seq = z_flat.view(B, L - 1, noise_dim)

            x_fake_seq = model(x_hist, z_seq)   # (B, L-1, d)

            # Take only the plot channel
            real_vals_plot_norm.append(
                x_tgt[..., plot_idx].detach().cpu().numpy().ravel()
            )
            fake_vals_plot_norm.append(
                x_fake_seq[..., plot_idx].detach().cpu().numpy().ravel()
            )

    x_real_plot_norm = np.concatenate(real_vals_plot_norm, axis=0)
    x_fake_plot_norm = np.concatenate(fake_vals_plot_norm, axis=0)

    # De-normalise for diagnostics
    x_real_eval = x_real_plot_norm * std_plot + mean_plot
    x_fake_eval = x_fake_plot_norm * std_plot + mean_plot

    print_tail_stats(f"ETT real 1-step ({plot_column})", x_real_eval)
    print_tail_stats(f"ETT generated 1-step ({plot_column}, sliced ISL)", x_fake_eval)

    # ------------------- Histograms (linear & log-y) ------------------- #
    xmin, xmax = -200.0, 200.0  # fixed window for comparability

    # Linear scale
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
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.xlabel(plot_column if not log_transform else f"log({plot_column} + eps)")
    plt.ylabel("density (hist)")
    plt.title(f"ETT-RNN-sliced-ISL 1-step ({plot_column}) - linear scale")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    hist_lin_path = outdir / f"rnn_sliced_isl_ett_hist_linear_{suffix}_{plot_column}.png"
    plt.savefig(hist_lin_path, dpi=200)
    plt.close()
    print(f"Saved linear-scale histogram to {hist_lin_path}")

    # Log-y scale
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
    plt.ylim(bottom=1e-6)
    plt.legend()
    plt.xlabel(plot_column if not log_transform else f"log({plot_column} + eps)")
    plt.ylabel("density (log scale)")
    plt.title(f"ETT-RNN-sliced-ISL 1-step ({plot_column}) - log-y scale")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    hist_log_path = outdir / f"rnn_sliced_isl_ett_hist_logy_{suffix}_{plot_column}.png"
    plt.savefig(hist_log_path, dpi=200)
    plt.close()
    print(f"Saved log-y histogram to {hist_log_path}")

    # ------------------- 1-step forecast with teacher forcing (plot channel) ------------------- #
    L = seq_len
    max_horizon = max(1, N - (L - 1))
    horizon = min(forecast_horizon, max_horizon)

    if horizon > 0:
        start_idx = N - horizon
        start_idx = max(start_idx, L - 1)
        horizon = N - start_idx

        preds_plot_norm: List[float] = []
        true_plot_norm: List[float] = []

        with torch.no_grad():
            for j in range(horizon):
                idx = start_idx + j  # index of target X[idx] in R^d

                # window [idx-L+1, ..., idx] in NORMALISED scale
                x_seq_norm = series[idx - L + 1: idx + 1, :]  # (L, d)
                x_hist = x_seq_norm[:-1, :].unsqueeze(0)      # (1, L-1, d)

                n_latent = L - 1
                z_flat = sample_latent(
                    n=n_latent,
                    noise_dim=noise_dim,
                    device=device,
                )
                z_seq = z_flat.view(1, L - 1, noise_dim)

                x_fake_seq = model(x_hist, z_seq)            # (1, L-1, d)
                pred_vec_norm = x_fake_seq[0, -1, :]         # (d,)

                pred_norm = float(pred_vec_norm[plot_idx].item())
                true_norm = float(x_seq_norm[-1, plot_idx].item())

                preds_plot_norm.append(pred_norm)
                true_plot_norm.append(true_norm)

        preds_plot_norm = np.asarray(preds_plot_norm)
        true_plot_norm = np.asarray(true_plot_norm)

        preds = preds_plot_norm * std_plot + mean_plot
        true_vals = true_plot_norm * std_plot + mean_plot

        t_idx = np.arange(horizon)

        plt.figure(figsize=(8, 4))
        plt.plot(t_idx, true_vals, label="real")
        plt.plot(t_idx, preds,
                 label="1-step forecast (sliced ISL, teacher forcing)",
                 alpha=0.8)
        plt.xlabel("Time index (last segment)")
        plt.ylabel(plot_column if not log_transform else f"log({plot_column} + eps)")
        plt.legend()
        plt.title(
            f"1-step-ahead forecast with teacher forcing "
            f"({plot_column}, horizon={horizon}, sliced ISL, d={d})"
        )
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        forecast_path = outdir / f"rnn_sliced_isl_ett_forecast_teacher_{suffix}_{plot_column}.png"
        plt.savefig(forecast_path, dpi=200)
        plt.close()
        print(f"Saved 1-step teacher-forced forecast plot to {forecast_path}")
    else:
        print("Not enough data points to build a forecast plot.")


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RNN-based implicit generative model on a multivariate ETT "
                    "series using sliced Invariant Statistical Loss (ISL) "
                    "with Gaussian latent and per-channel normalisation."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to ETT .csv file (e.g. data/ETTm1.csv).",
    )
    parser.add_argument(
        "--feature_cols",
        type=str,
        nargs="*",
        default=None,
        help="Feature columns to use (default: all numeric except 'date').",
    )
    parser.add_argument(
        "--plot_column",
        type=str,
        default="OT",
        help="Column name used for 1D diagnostics/plots (default: OT).",
    )
    parser.add_argument(
        "--log_transform",
        action="store_true",
        help="Model log(x + eps) instead of x.",
    )
    parser.add_argument("--seq_len", type=int, default=64, help="Window length L.")
    parser.add_argument("--stride", type=int, default=1, help="Stride for windows.")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--noise_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--n_slices",
        type=int,
        default=32,
        help="Number of random directions for sliced ISL.",
    )
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--cdf_bandwidth", type=float, default=0.15)
    parser.add_argument("--hist_sigma", type=float, default=0.05)
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=500,
        help="Number of time indices in the last segment for 1-step forecasting.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    device = get_device(prefer_gpu=True)

    train_rnn_sliced_isl_ett(
        data_path=args.data_path,
        feature_cols=args.feature_cols,
        plot_column=args.plot_column,
        log_transform=args.log_transform,
        seq_len=args.seq_len,
        stride=args.stride,
        steps=args.steps,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        rnn_layers=args.rnn_layers,
        lr=args.lr,
        n_slices=args.n_slices,
        K=args.K,
        cdf_bandwidth=args.cdf_bandwidth,
        hist_sigma=args.hist_sigma,
        log_every=200,
        forecast_horizon=args.forecast_horizon,
        device=device,
    )


if __name__ == "__main__":
    main()
