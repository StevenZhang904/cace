###########################################################################################
# Radial basis
# modified from mace/mace/modules/radials.py and schnetpack/src/schnetpack/nn/radials.py 
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import numpy as np
import torch
import torch.nn as nn

__all__ = ["BesselRBF", "GaussianRBF", "GaussianRBFCentered", "BesselRBF_SchNet"]

class BesselRBF(nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """

    def __init__(self, cutoff: float, n_rbf=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / cutoff
            * torch.linspace(
                start=1.0,
                end=n_rbf,
                steps=n_rbf,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / cutoff), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., n_rbf]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cutoff={self.cutoff}, n_rbf={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.8, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super().__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cutoff={self.cutoff}, n_rbf={self.rbf}, "
            f"trainable={self.widths.requires_grad})"
        )

class GaussianRBFCentered(nn.Module):
    r"""Gaussian radial basis functions centered at the origin."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 1.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: width of last Gaussian function, :math:`\mu_{N_g}`
            start: width of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths of Gaussian functions
                are adjusted during training process.
        """
        super().__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        widths = torch.linspace(start, cutoff, n_rbf)
        offset = torch.zeros_like(widths)
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cutoff={self.cutoff}, n_rbf={self.rbf}, "
            f"trainable={self.widths.requires_grad})"
        )

class BesselRBF_SchNet(nn.Module):
    """
    Sine for radial basis functions with coulomb decay (0th order bessel).

    References:

    .. [#dimenet] Klicpera, Groß, Günnemann:
       Directional message passing for molecular graphs.
       ICLR 2020
    """

    def __init__(self, n_rbf: int, cutoff: float):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super().__init__()
        self.n_rbf = n_rbf

        freqs = torch.arange(1, n_rbf + 1) * np.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        ax = inputs[..., None] * self.freqs
        sinax = torch.sin(ax)
        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[..., None]
        return y
