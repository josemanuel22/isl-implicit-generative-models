"""
Neural network architectures used in ISL experiments.

This module collects a small set of reusable PyTorch models that cover the
main experimental settings of the ISL / Pareto-ISL paper:

- Low-dimensional synthetic data (1D / 2D / R^d):
    * MLPGenerator
    * MLPDiscriminator

- Image data (MNIST, Fashion-MNIST, CelebA-32/64, etc.):
    * DCGANGenerator
    * DCGANDiscriminator

- Time-series forecasting:
    * RNNForecast

The goal is not to be exhaustive, but to provide simple, well-documented
architectures that can be plugged into ISL / sliced-ISL objectives.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import spectral_norm


# ---------------------------------------------------------------------
#  Weight initialisation
# ---------------------------------------------------------------------

def init_weights_dcgan(m: nn.Module) -> None:
    """
    DCGAN-style weight initialisation.

    - Convolutional and transposed convolutional layers:
        weights ~ N(0, 0.02)
    - BatchNorm layers:
        weights ~ N(1, 0.02), biases = 0
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif "BatchNorm" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias.data)


def init_weights_mlp(m: nn.Module) -> None:
    """
    Simple MLP weight initialisation:
    Kaiming normal for Linear layers, zeros for biases.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------
#  MLP-based generator / discriminator (low-dimensional data)
# ---------------------------------------------------------------------

class MLPGenerator(nn.Module):
    """
    Fully-connected generator for low-dimensional data in R^d.

    Architecture:
        z in R^{noise_dim} -> [Linear + ReLU] x L -> Linear -> x in R^{data_dim}

    Parameters
    ----------
    noise_dim : int
        Dimension of the input noise vector.
    data_dim : int
        Dimension of the output data space.
    hidden_dims : iterable of int, default=(128, 128)
        Sizes of hidden layers.
    activation : nn.Module, default=nn.ReLU
        Nonlinearity used between linear layers.
    """

    def __init__(
        self,
        noise_dim: int,
        data_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = noise_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h

        layers.append(nn.Linear(in_dim, data_dim))
        self.net = nn.Sequential(*layers)
        self.apply(init_weights_mlp)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class MLPDiscriminator(nn.Module):
    """
    Simple MLP discriminator / critic for low-dimensional data.

    Architecture:
        x in R^{data_dim} -> [Linear + LeakyReLU] x L -> Linear -> scalar

    Parameters
    ----------
    data_dim : int
        Dimension of input data.
    hidden_dims : iterable of int, default=(128, 128)
        Sizes of hidden layers.
    activation : nn.Module, default=nn.LeakyReLU(0.2)
        Nonlinearity between linear layers.
    spectral_normed : bool, default=False
        If True, apply spectral normalization to linear layers.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: nn.Module = nn.LeakyReLU(0.2),
        spectral_normed: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = data_dim

        for h in hidden_dims:
            lin = nn.Linear(in_dim, h)
            if spectral_normed:
                lin = spectral_norm(lin)
            layers.append(lin)
            layers.append(activation)
            in_dim = h

        lin_out = nn.Linear(in_dim, 1)
        if spectral_normed:
            lin_out = spectral_norm(lin_out)

        layers.append(lin_out)
        self.net = nn.Sequential(*layers)
        self.apply(init_weights_mlp)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).view(-1)


# ---------------------------------------------------------------------
#  DCGAN-style generator and discriminator (image data)
# ---------------------------------------------------------------------

class DCGANGenerator(nn.Module):
    """
    DCGAN-style generator for image data (e.g., MNIST, FMNIST, CelebA).

    This implementation supports image_size 32 or 64 with power-of-two scaling.
    The network upsamples from a latent vector z to an image via ConvTranspose2d.

    Parameters
    ----------
    z_dim : int
        Dimension of the latent noise vector.
    img_channels : int
        Number of output image channels (1 for grayscale, 3 for RGB).
    feature_maps : int, default=64
        Base number of feature maps (64 is standard).
    image_size : int, default=32
        Either 32 or 64. Controls the number of upsampling layers.
    """

    def __init__(
        self,
        z_dim: int = 128,
        img_channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 32,
    ) -> None:
        super().__init__()

        if image_size not in (32, 64):
            raise ValueError("DCGANGenerator currently supports image_size in {32, 64}.")

        self.z_dim = z_dim
        self.img_channels = img_channels
        self.feature_maps = feature_maps
        self.image_size = image_size

        # Start from 4x4 spatial size
        if image_size == 32:
            # 4x4 -> 8x8 -> 16x16 -> 32x32
            self.net = nn.Sequential(
                # Input: z -> (ngf*4) x 4 x 4
                nn.ConvTranspose2d(z_dim, feature_maps * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.ReLU(True),

                # (ngf*4) x 4 x 4 -> (ngf*2) x 8 x 8
                nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.ReLU(True),

                # (ngf*2) x 8 x 8 -> (ngf) x 16 x 16
                nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps),
                nn.ReLU(True),

                # (ngf) x 16 x 16 -> img_channels x 32 x 32
                nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        else:
            # image_size == 64:
            # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
            self.net = nn.Sequential(
                # Input: z -> (ngf*8) x 4 x 4
                nn.ConvTranspose2d(z_dim, feature_maps * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(feature_maps * 8),
                nn.ReLU(True),

                # (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
                nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.ReLU(True),

                # (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
                nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.ReLU(True),

                # (ngf*2) x 16 x 16 -> (ngf) x 32 x 32
                nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps),
                nn.ReLU(True),

                # (ngf) x 32 x 32 -> img_channels x 64 x 64
                nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )

        self.apply(init_weights_dcgan)

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z : Tensor, shape (batch_size, z_dim)

        Returns
        -------
        x : Tensor, shape (batch_size, img_channels, H, W)
            Generated images in [-1, 1] (due to Tanh).
        """
        # Reshape z to (batch_size, z_dim, 1, 1)
        z = z.view(z.size(0), self.z_dim, 1, 1)
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN-style discriminator / critic for image data.

    Parameters
    ----------
    img_channels : int
        Number of image channels (1 for grayscale, 3 for RGB).
    feature_maps : int, default=64
        Base number of feature maps.
    image_size : int, default=32
        Either 32 or 64. Controls number of downsampling layers.
    use_spectral_norm : bool, default=False
        If True, apply spectral normalization to Conv layers (useful for WGAN).
    """

    def __init__(
        self,
        img_channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 32,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        if image_size not in (32, 64):
            raise ValueError("DCGANDiscriminator currently supports image_size in {32, 64}.")

        def conv_sn(conv: nn.Conv2d) -> nn.Conv2d:
            return spectral_norm(conv) if use_spectral_norm else conv

        if image_size == 32:
            # img_channels x 32 x 32 -> 1 scalar
            layers = [
                # (img_channels) x 32 x 32 -> (ndf) x 16 x 16
                conv_sn(nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),

                # (ndf) x 16 x 16 -> (ndf*2) x 8 x 8
                conv_sn(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(feature_maps * 2),
                nn.LeakyReLU(0.2, inplace=True),

                # (ndf*2) x 8 x 8 -> (ndf*4) x 4 x 4
                conv_sn(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(feature_maps * 4),
                nn.LeakyReLU(0.2, inplace=True),

                # (ndf*4) x 4 x 4 -> 1 x 1 x 1
                conv_sn(nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False)),
            ]
        else:
            # image_size == 64
            layers = [
                # (img_channels) x 64 x 64 -> (ndf) x 32 x 32
                conv_sn(nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),

                # (ndf) x 32 x 32 -> (ndf*2) x 16 x 16
                conv_sn(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(feature_maps * 2),
                nn.LeakyReLU(0.2, inplace=True),

                # (ndf*2) x 16 x 16 -> (ndf*4) x 8 x 8
                conv_sn(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(feature_maps * 4),
                nn.LeakyReLU(0.2, inplace=True),

                # (ndf*4) x 8 x 8 -> (ndf*8) x 4 x 4
                conv_sn(nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(feature_maps * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # (ndf*8) x 4 x 4 -> 1 x 1 x 1
                conv_sn(nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False)),
            ]

        self.net = nn.Sequential(*layers)
        self.apply(init_weights_dcgan)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch_size, img_channels, H, W)

        Returns
        -------
        logits : Tensor, shape (batch_size,)
            Raw discriminator scores.
        """
        out = self.net(x)
        return out.view(-1)


# ---------------------------------------------------------------------
#  RNN-based model for time-series forecasting
# ---------------------------------------------------------------------

class RNNForecast(nn.Module):
    """
    Simple RNN/GRU-based forecasting model for time-series.

    Given an input sequence (e.g., x_{1:t}), the model produces a hidden
    representation h_t and a prediction for the next value y_{t+1}. In the
    ISL context, one can treat h_t as a sufficient statistic of the past
    and apply ISL / Pareto-ISL to residuals or conditional distributions.

    Architecture:
        input_dim -> RNN (GRU/LSTM) -> last hidden state -> MLP -> output_dim

    Parameters
    ----------
    input_dim : int
        Dimensionality of the per-timestep input.
    hidden_dim : int
        Hidden size of the RNN.
    output_dim : int
        Dimensionality of the output / prediction.
    rnn_type : {"gru", "lstm", "rnn"}, default="gru"
        Type of recurrent cell.
    num_layers : int, default=1
        Number of stacked RNN layers.
    dropout : float, default=0.0
        Dropout between RNN layers (if num_layers > 1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rnn_type: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        rnn_type = rnn_type.lower()
        if rnn_type not in {"gru", "lstm", "rnn"}:
            raise ValueError("rnn_type must be 'gru', 'lstm' or 'rnn'")

        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:  # "rnn"
            self.rnn = nn.RNN(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                nonlinearity="tanh",
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(init_weights_mlp)

    def forward(
        self,
        x: Tensor,
        h0: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape (batch_size, seq_len, input_dim)
            Input sequence.
        h0 : Tensor, optional
            Initial hidden state (and cell state for LSTM).

        Returns
        -------
        y_pred : Tensor, shape (batch_size, output_dim)
            Prediction from the last time-step hidden state.
        h_last : Tensor, shape (num_layers, batch_size, hidden_dim)
            Last hidden state (for GRU/RNN) or last hidden state h_n
            for LSTM. If you need c_n for LSTM, you can access it
            via self.rnn directly or adapt the code.
        """
        if self.rnn_type == "lstm":
            # h0 is (h_0, c_0) or None
            output, (h_n, c_n) = self.rnn(x, h0)
            h_last = h_n[-1]  # (batch, hidden_dim)
        else:
            output, h_n = self.rnn(x, h0)
            h_last = h_n[-1]  # (batch, hidden_dim)

        y_pred = self.head(h_last)
        return y_pred, h_n
    
__all__ = [
    "MLPGenerator",
    "MLPDiscriminator",
    "DCGANGenerator",
    "DCGANDiscriminator",
    "RNNForecast",
]